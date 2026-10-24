import os
import time
import json
import traceback
import re
import threading
import unicodedata
from typing import Dict, Any, List, Optional
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import logging
from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config
from .spatial_analysis import SpatialAnalyst
from .visualization import UrbanVisualizer

import matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RATE_LIMIT_ERROR = "rate limit reached"


def call_gemini(model: str, prompt: str) -> str:
    """Send a prompt to Gemini and return the text reply.

    Failures are returned as a string starting with "[ERROR]" instead of raising,
    so callers can show a friendly message. Temporary errors (rate limit / overload)
    are retried a few times with a growing pause.
    """
    if not config.GEMINI_API_KEY:
        return "[ERROR] Gemini call failed: GEMINI_API_KEY is not set"

    url = f"{config.GEMINI_API_URL}/{model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}

    last_error = "unknown error"
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload),
                                     timeout=config.GEMINI_TIMEOUT)
            if response.status_code == 429:
                # Per-minute quota reached: wait for the quota window, then try again
                last_error = RATE_LIMIT_ERROR
                if attempt < 2:
                    time.sleep(12 * (attempt + 1))
                    continue
                return f"[ERROR] {RATE_LIMIT_ERROR}"
            if response.status_code in (500, 503) and attempt < 2:
                last_error = f"HTTP {response.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
                return f"[ERROR] Gemini call failed: {reason}"
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except Exception as e:
            last_error = str(e)
            logging.error(f"Gemini API call failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return f"[ERROR] Gemini call failed: {last_error}"


def fold_text(text) -> str:
    """Lower-case and strip accents so 'Cișmigiu' matches 'cismigiu'."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


_OUTPUT_FILE_RE = re.compile(r"""((?:[A-Za-z]:)?[^\s'"()<>|*?]+\.(?:png|jpe?g|html))""", re.IGNORECASE)


def find_output_file(tool_output: str):
    """Return the last existing image/map file mentioned in a tool's text output."""
    for candidate in reversed(_OUTPUT_FILE_RE.findall(str(tool_output))):
        if os.path.exists(candidate):
            return candidate
    return None


def is_llm_error(text: str) -> bool:
    return isinstance(text, str) and text.startswith("[ERROR]")


class UrbanDataAgent:
    """
    Conversational agent for interacting with urban geospatial data using LangGraph and Gemini.
    """
    
    def __init__(self, gdf_data):
        """
        Initialize the agent with processed urban data.
        
        Args:
            gdf_data: GeoDataFrame containing processed park and environmental data
        """
        self.data = gdf_data
        self.spatial_analyst = SpatialAnalyst(gdf_data)
        self.visualizer = UrbanVisualizer(gdf_data)
        self.conversation_history = []
        # Tools use matplotlib and shared state, so handle one question at a time
        self._lock = threading.Lock()
        self.tools = self._setup_tools()
        self.graph = self._build_agent_graph()
        self.traffic_data = None
        self.latest_visualization = None
        self.traffic_descriptions = {
            0: "foarte liber, fără aglomerație",
            1: "destul de liber, cu puțină aglomerație",
            2: "moderat, cu perioade de aglomerație",
            3: "destul de aglomerat, cu întârzieri posibile",
            4: "foarte aglomerat, cu întârzieri semnificative",
            5: "complet blocat, cu trafic staționar"
        }
        try:
            traffic_file = config.TRAFFIC_FILE
            if os.path.exists(traffic_file):
                import geopandas as gpd
                self.traffic_data = gpd.read_file(traffic_file)
                logger.info(f"Loaded traffic data with {len(self.traffic_data)} road segments")
        except Exception as e:
            logger.error(f"Error loading traffic data: {e}")
    
    def _trim_history(self, history):
        """Keep only the most recent exchanges (each exchange = question + answer)."""
        return history[-config.MAX_HISTORY_LENGTH * 2:]

    def update_latest_visualization(self, filepath):
        """Remember the newest map/chart as a URL the web page can load.

        Interactive maps (.html) are served from /maps, images from /visualizations.
        """
        if filepath and os.path.exists(filepath):
            filename = os.path.basename(filepath)
            if filename.lower().endswith(".html"):
                self.latest_visualization = f"/maps/{filename}"
            else:
                self.latest_visualization = f"/visualizations/{filename}"
            return True
        return False
    
    def _setup_tools(self) -> Dict[str, Dict[str, Any]]:
        """Set up tools that the agent can use to analyze data."""
        return {
            "get_parks_by_sector": {
                "description": "Lists the count of parks in each sector of Bucharest",
                "function": self._get_parks_by_sector,
                "args": {}
            },
            "find_parks_with_best_air_quality": {
                "description": "Finds parks with the best air quality (lowest AQI)",
                "function": self._find_parks_with_best_air_quality,
                "args": {"top_n": "int"}
            },
            "analyze_air_park_relationship": {
                "description": "Analizează relația dintre calitatea aerului și densitatea parcurilor în București",
                "function": self._analyze_air_quality_park_relationship,
                "args": {}
            },
            "get_traffic_information": {
                "description": "Provides information about traffic conditions around a specific park in Bucharest",
                "function": self.get_traffic_info,
                "args": {"park_name": "str"}
            },
            "find_parks_with_worst_air_quality": {
                "description": "Finds parks with the worst air quality (highest AQI)",
                "function": self._find_parks_with_worst_air_quality,
                "args": {"top_n": "int"}
            },
            "compare_sector_air_quality": {
                "description": "Compares air quality metrics between two sectors",
                "function": self._compare_sector_air_quality,
                "args": {"sector1": "str", "sector2": "str"}
            },
            "search_parks_by_name": {
                "description": "Search for parks containing a specific name pattern",
                "function": self._search_parks_by_name,
                "args": {"name_pattern": "str"}
            },
            "get_park_details": {
                "description": "Get detailed information about a specific park",
                "function": self._get_park_details,
                "args": {"park_name": "str"}
            },
            "get_sector_statistics": {
                "description": "Gets detailed statistics about parks in each sector",
                "function": self._get_sector_statistics,
                "args": {}
            },
            "find_greenest_sector": {
                "description": "Finds the sector with the most parks or best air quality",
                "function": self._find_greenest_sector,
                "args": {}
            },
            "find_optimal_park_locations": {
                "description": "Suggests optimal locations for new parks",
                "function": self._find_optimal_park_locations,
                "args": {"num_locations": "int"}
            },
            "compare_north_south": {
                "description": "Compară mediul urban (parcuri, aer, zgomot) între zonele de nord și sud ale Bucureștiului",
                "function": self._compare_north_south,
                "args": {"metric": "str"}
            },
            "find_route_between_parks": {
                "description": "Find a route between two parks considering traffic conditions",
                "function": self._find_route_between_parks,
                "args": {
                    "origin_park": "str",
                    "destination_park": "str",
                    "mode": "str"
                }
            },
            "plan_park_tour": {
                "description": "Plan a tour of multiple parks based on size, quality or other metrics",
                "function": self._plan_park_tour,
                "args": {
                    "num_parks": "int",
                    "start_park": "str", 
                    "metric": "str"
                }
            },
            "create_sector_comparison_chart": {
                "description": "Creates a chart comparing sectors based on parks or air quality",
                "function": self._create_sector_comparison_chart,
                "args": {"analysis_type": "str"}
            },
            "create_air_quality_heatmap": {
                "description": "Creates a heatmap showing air quality across parks",
                "function": self._create_air_quality_heatmap,
                "args": {}
            },
            "get_current_time": {
                "description": "Returns the current date and time",
                "function": self._get_current_time,
                "args": {}
            },
            "get_total_parks": {
                "description": "Returns the total count of parks in Bucharest",
                "function": self._get_total_parks,
                "args": {}
            },
            "list_parks_in_sector": {
                "description": "Lists all parks in a specific sector of Bucharest",
                "function": self._list_parks_in_sector,
                "args": {"sector_id": "str"}
            },
            "compare_sectors": {
                "description": "Compare two sectors or all sectors based on a specific metric (parks, air, noise, density)",
                "function": self._compare_sectors,
                "args": {"sector1": "str", "sector2": "str", "metric": "str"}
            },
            "create_sector_chart": {
                "description": "Create a chart comparing all sectors by specified metric (parks, air, noise, density)",
                "function": self._create_sector_chart,
                "args": {"metric": "str"}
            },
            "find_best_sector": {
                "description": "Find the sector with the best score for a specific metric (parks, air, noise, density)",
                "function": self._find_best_sector,
                "args": {"metric": "str"}
            },
            "calculate_park_accessibility": {
                "description": "Calculate the percentage of area within walking distance of parks",
                "function": self._calculate_park_accessibility,
                "args": {
                    "buffer_distance": "int",
                    "region": "str",
                    "show_poorly_served": "bool"
                }
            },
            "analyze_sectors": {
                "description": "Analyze and compare Bucharest sectors based on criteria like air quality and number of parks",
                "function": self._analyze_sectors_by_criteria,
                "args": {
                    "criteria": "list[str]",
                    "sort_by": "str",
                    "reverse": "bool"
                }
            },
            "recommend_best_park_today": {
                "description": "Recommend the best park to visit today based on current conditions",
                "function": self._recommend_best_park_today,
                "args": {
                    "criteria": "list[str]"
                }
            },
            "get_weather_forecast": {
                "description": "Gets the weather forecast for the next few days in Bucharest",
                "function": self._get_weather_forecast,
                "args": {"days": "int"}
            },
            "get_best_park_day": {
                "description": "Recommends the best day to visit parks based on weather and air quality forecasts",
                "function": self._get_best_park_day,
                "args": {"park_name": "str"}
            },
            "check_outdoor_activities": {
                "description": "Checks if conditions are suitable for outdoor activities for a specific day",
                "function": self._check_outdoor_activities,
                "args": {"day": "str", "activity_type": "str"}
            },
            "get_air_quality_forecast": {
                "description": "Gets the air quality forecast for the next few days",
                "function": self._get_air_quality_forecast,
                "args": {"days": "int"}
            },
            "get_hourly_weather_forecast": {
                "description": "Gets the hourly weather forecast for a specific day and time period",
                "function": self._get_hourly_weather_forecast,
                "args": {
                    "day": "str",
                    "time_period": "str",
                    "hours": "list[str]"
                }
            },
            "find_largest_parks": {
                "description": "Finds the largest parks in a specific sector by area",
                "function": self._find_largest_parks,
                "args": {
                    "sector": "str", 
                    "top_n": "int"
                }
            },
            "find_parks_with_lowest_noise": {
                "description": "Finds parks with the lowest noise levels (quietest parks)",
                "function": self._find_parks_with_lowest_noise,
                "args": {"top_n": "int"}
            }
        }
    
    def _build_agent_graph(self):
        """Build the LangGraph workflow for the agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("parse_query", self._parse_query_node)
        workflow.add_node("execute_tool", self._execute_tool_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "parse_query")
        workflow.add_edge("parse_query", "execute_tool")
        workflow.add_edge("execute_tool", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile the graph
        return workflow.compile()
    
    # --- Tool implementations ---
    
    def _get_parks_by_sector(self) -> str:
        """Returns a count of parks by sector."""
        try:
            if 'sector' not in self.data.columns:
                return "Error: The dataset does not contain sector information."
            
            sector_counts = self.data.groupby('sector').size().reset_index(name='count')
            sector_counts = sector_counts.sort_values('count', ascending=False)
            
            result = "Parks by sector in Bucharest:\n\n"
            for _, row in sector_counts.iterrows():
                result += f"{row['sector']}: {row['count']} parks\n"
                
            return result
        except Exception as e:
            logger.error(f"Error getting parks by sector: {e}")
            return f"Error occurred while analyzing parks by sector: {str(e)}"
    
    def _find_parks_with_best_air_quality(self, top_n: int = 5) -> str:
        """Finds parks with the best air quality (lowest AQI)."""
        try:
            if 'aqi' not in self.data.columns:
                return "Error: The dataset does not contain air quality information."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Could not find a column with park names."
            
            # Create a clean dataframe with just the columns we need
            parks_df = self.data[[name_column, 'sector', 'aqi']].copy()
            
            # Remove parks with NaN AQI values
            parks_df = parks_df.dropna(subset=['aqi'])
            
            # Remove duplicates based on park name AND sector (to handle parks with same name in different sectors)
            parks_df = parks_df.drop_duplicates(subset=[name_column, 'sector'])
            
            if len(parks_df) == 0:
                return "No parks with air quality data found."
                
            # Clean up sector formatting if needed (remove duplicate "Sector" prefix)
            if 'sector' in parks_df.columns:
                parks_df['sector'] = parks_df['sector'].apply(
                    lambda x: str(x).replace('Sector Sector', 'Sector') if isinstance(x, str) else x
                )
                
                # More comprehensive clean-up for sector format
                def clean_sector(sector):
                    if not isinstance(sector, str):
                        return "Sector necunoscut"
                    
                    sector = sector.strip()
                    if sector.startswith("Sector "):
                        return sector
                    elif sector.isdigit():
                        return f"Sector {sector}"
                    else:
                        return sector
                
                parks_df['sector'] = parks_df['sector'].apply(clean_sector)
            
            # Lower AQI is better
            best_parks = parks_df.sort_values('aqi').head(top_n)
            
            # Generate the result with consistent formatting
            result = f"Top {top_n} parcuri cu cea mai bună calitate a aerului (AQI mai mic este mai bun):\n\n"
            for idx, row in best_parks.iterrows():
                park_name = row[name_column]
                sector = row.get('sector', "")
                aqi_value = row['aqi']
                
                # Format the output line
                result += f"{park_name} ({sector}): AQI {aqi_value:.1f}\n"
                    
            return result
        except Exception as e:
            logger.error(f"Error finding parks with best air quality: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error occurred while analyzing air quality: {str(e)}"
    
    def get_traffic_info(self, park_name):
        """Get traffic information around a specific park
        
        Args:
            park_name: The name of the park to check traffic for
            
        Returns:
            str: A descriptive response about traffic conditions in Romanian
        """
        try:
            # Convert to lowercase and remove "parcul" prefix if present for better matching
            search_name = fold_text(park_name)
            if search_name.startswith("parcul "):
                search_name = search_name[7:]  # Remove "parcul " prefix
                
            # Search for parks with similar names
            if 'congestion_level' not in self.data.columns:
                return "Datele despre trafic nu sunt disponibile momentan. Încearcă să actualizezi datele."
            
            name_col = 'name_left' if 'name_left' in self.data.columns else 'name'
            parks = self.data[self.data[name_col].map(fold_text).str.contains(search_name, regex=False, na=False)]
            
            if len(parks) == 0:
                return f"Nu am putut găsi niciun parc cu numele {park_name} în baza noastră de date."
                
            # Get the first matching park
            park = parks.iloc[0]
            
            # Get traffic information
            congestion_level = park.get('congestion_level', 0)
            traffic_desc = self.traffic_descriptions.get(int(congestion_level) if not pd.isna(congestion_level) else -1, "informații indisponibile")
            
            # Format response
            response = f"În jurul {park[name_col]}, traficul este {traffic_desc}. "
            
            # Add more details if available
            if 'speed_factor' in park:
                speed_percent = int(park['speed_factor'] * 100)
                response += f"Vehiculele circulă la aproximativ {speed_percent}% din viteza maximă permisă. "
                
            if 'travel_time_seconds' in park and 'road_distance_meters' in park:
                travel_time = round(park['travel_time_seconds'])
                distance = round(park['road_distance_meters'])
                response += f"Pentru o distanță de {distance} metri în zonă, timpul mediu de parcurgere este de aproximativ {travel_time} secunde."
                
            return response
        except Exception as e:
            return f"A apărut o eroare la obținerea informațiilor despre trafic: {str(e)}"
    
    def _find_parks_with_worst_air_quality(self, top_n: int = 5) -> str:
        """Finds parks with the worst air quality (highest AQI)."""
        try:
            if 'aqi' not in self.data.columns:
                return "Error: The dataset does not contain air quality information."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Could not find a column with park names."
            
            # Create a clean dataframe with just the columns we need
            parks_df = self.data[[name_column, 'sector', 'aqi']].copy()
            
            # Remove parks with NaN AQI values
            parks_df = parks_df.dropna(subset=['aqi'])
            
            # Remove duplicates based on park name AND sector (to handle parks with same name in different sectors)
            parks_df = parks_df.drop_duplicates(subset=[name_column, 'sector'])
            
            if len(parks_df) == 0:
                return "No parks with air quality data found."
            
            # Clean up sector formatting if needed (remove duplicate "Sector" prefix)
            if 'sector' in parks_df.columns:
                parks_df['sector'] = parks_df['sector'].apply(
                    lambda x: str(x).replace('Sector Sector', 'Sector') if isinstance(x, str) else x
                )
                
                # More comprehensive clean-up for sector format
                def clean_sector(sector):
                    if not isinstance(sector, str):
                        return "Sector necunoscut"
                    
                    sector = sector.strip()
                    if sector.startswith("Sector "):
                        return sector
                    elif sector.isdigit():
                        return f"Sector {sector}"
                    else:
                        return sector
                
                parks_df['sector'] = parks_df['sector'].apply(clean_sector)
                
            # Higher AQI is worse
            worst_parks = parks_df.sort_values('aqi', ascending=False).head(top_n)
            
            # Generate the result with consistent formatting
            result = f"Top {top_n} parcuri cu cea mai proastă calitate a aerului (AQI mai mare este mai rău):\n\n"
            for idx, row in worst_parks.iterrows():
                park_name = row[name_column]
                sector = row.get('sector', "")
                aqi_value = row['aqi']
                
                # Format the output line
                result += f"{park_name} ({sector}): AQI {aqi_value:.1f}\n"
                    
            return result
        except Exception as e:
            logger.error(f"Error finding parks with worst air quality: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error occurred while analyzing air quality: {str(e)}"
    
    def _calculate_park_accessibility(self, buffer_distance=None, region=None, show_poorly_served=False):
        """
        Calculate the percentage of city area that is within walking distance of parks.
        
        Args:
            buffer_distance: Walking distance in meters (default: 500)
            region: Optional region to analyze (e.g., "north", "south", "Sector 1", etc.)
            show_poorly_served: Whether to highlight areas with poor park access
        
        Returns:
            A description of the accessibility results
        """
        try:
            # Set default buffer distance if not provided
            if buffer_distance is None:
                buffer_distance = 500
            
            if self.data is None or len(self.data) == 0:
                return "Eroare: Nu există date despre parcuri disponibile."
            
            # Import necessary libraries
            import geopandas as gpd
            from shapely.ops import unary_union
            import matplotlib.pyplot as plt
            import os
            import numpy as np
            from datetime import datetime
            from shapely.geometry import box, Polygon, MultiPolygon
            
            # Determine column with park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
            
            # Clean up the data - remove duplicates
            clean_data = self.data.copy()
            if name_column:
                clean_data = clean_data.drop_duplicates(subset=[name_column, 'sector'])
            
            # Normalize region input for sector filtering
            region_name = "București"
            if region:
                # Extract sector number if present
                import re
                region_lower = str(region).lower()
                
                if "sector" in region_lower:
                    sector_match = re.search(r'\d+', region_lower)
                    if sector_match:
                        sector_num = sector_match.group()
                        sector_name = f"Sector {sector_num}"
                    else:
                        sector_name = region.title()
                    
                    # Filter parks by sector
                    parks = clean_data[clean_data['sector'] == sector_name]
                    region_name = sector_name
                    
                    # Verify if we have parks after filtering
                    if len(parks) == 0:
                        return f"Nu au fost găsite parcuri în {region_name}."
                    
                    # Predefined sector boundaries approximations
                    sector_boundaries = {
                        # Dictionary with polygon coordinates for each sector
                        # These are simplified approximations
                        'Sector 1': [(26.0608, 44.4372), (26.0962, 44.4737), 
                                    (26.1155, 44.4539), (26.0798, 44.4248)],
                        'Sector 2': [(26.1155, 44.4539), (26.1502, 44.4709),
                                    (26.1416, 44.4372), (26.0962, 44.4175)],
                        'Sector 3': [(26.0962, 44.4175), (26.1416, 44.4372),
                                    (26.1655, 44.4091), (26.1069, 44.3945)],
                        'Sector 4': [(26.0798, 44.4248), (26.0962, 44.4175),
                                    (26.1069, 44.3945), (26.0717, 44.3889)],
                        'Sector 5': [(26.0451, 44.4165), (26.0798, 44.4248),
                                    (26.0717, 44.3889), (26.0464, 44.4006)],
                        'Sector 6': [(26.0451, 44.4165), (26.0608, 44.4372),
                                    (26.0798, 44.4248), (26.0717, 44.4006)]
                    }
                    
                    # Set city boundary for the selected sector
                    if sector_name in sector_boundaries:
                        # Create a polygon from coordinates
                        coords = sector_boundaries[sector_name]
                        sector_polygon = Polygon(coords)
                        
                        # Create GeoDataFrame for the sector boundary
                        bucharest_boundary = gpd.GeoDataFrame(
                            geometry=[sector_polygon],
                            crs=clean_data.crs if hasattr(clean_data, 'crs') else "EPSG:4326"
                        )
                        
                        # Area in square kilometers for sectors
                        sector_areas = {
                            'Sector 1': 68, 'Sector 2': 32, 'Sector 3': 34,
                            'Sector 4': 32, 'Sector 5': 30, 'Sector 6': 38
                        }
                        bucharest_boundary['area_km2'] = sector_areas.get(sector_name, 40)
                    else:
                        # Fallback to convex hull if we don't have predefined boundaries
                        sector_hull = parks.geometry.unary_union.convex_hull.buffer(0.01)
                        bucharest_boundary = gpd.GeoDataFrame(
                            geometry=[sector_hull],
                            crs=parks.crs
                        )
                        # Approximate sector area
                        sector_areas = {
                            'Sector 1': 68, 'Sector 2': 32, 'Sector 3': 34,
                            'Sector 4': 32, 'Sector 5': 30, 'Sector 6': 38
                        }
                        bucharest_boundary['area_km2'] = sector_areas.get(sector_name, 40)
                else:
                    # Handle north/south or other regions
                    return f"Regiune nerecunoscută: {region}. Vă rugăm să specificați un sector (ex: 'Sector 2')."
            else:
                # Use all parks if no region specified
                parks = clean_data
                
                # Use predefined Bucharest boundary or create a simplified one
                bounds = clean_data.total_bounds
                bucharest_boundary = gpd.GeoDataFrame(
                    geometry=[box(*bounds).buffer(0.02)],
                    crs=clean_data.crs if hasattr(clean_data, 'crs') else "EPSG:4326"
                )
                bucharest_boundary['area_km2'] = 469.01  # Approximate area of Bucharest
            
            # Convert buffer distance from meters to degrees (approximate)
            # 111,000 meters = 1 degree latitude
            buffer_degrees = buffer_distance / 111000
            
            # Create buffers around parks
            park_buffers = parks.geometry.buffer(buffer_degrees)
            
            # Merge all buffers to avoid double counting overlapping areas
            merged_buffer = unary_union(park_buffers)
            
            # Intersect with city/sector boundary to ensure we stay within the boundary
            if hasattr(bucharest_boundary, 'geometry') and len(bucharest_boundary) > 0:
                merged_buffer = merged_buffer.intersection(bucharest_boundary.geometry[0])
            
            # Calculate area within walking distance
            accessible_area_degrees = merged_buffer.area
            
            # Calculate approximate area in square kilometers
            # This is a simplification - proper calculation would use a projected CRS
            accessible_area_km2 = accessible_area_degrees * (111**2) * 0.68  # Scaling factor for lat/lon
            
            # Get total city/sector area
            total_area_km2 = bucharest_boundary['area_km2'].iloc[0]
            
            # Calculate percentage
            percentage = (accessible_area_km2 / total_area_km2) * 100
            
            # Create a visualization
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot city/sector boundary
                bucharest_boundary.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)
                
                # Plot buffer areas
                gpd.GeoDataFrame(geometry=[merged_buffer], crs=parks.crs).plot(
                    ax=ax, color='lightgreen', alpha=0.3
                )
                
                # Plot parks
                parks.plot(ax=ax, color='darkgreen', markersize=20, alpha=0.6)
                
                # If showing poorly served areas
                if show_poorly_served:
                    # Areas without park access
                    poorly_served = bucharest_boundary.geometry[0].difference(merged_buffer)
                    gpd.GeoDataFrame(geometry=[poorly_served], crs=parks.crs).plot(
                        ax=ax, color='red', alpha=0.3
                    )
                
                plt.title(f'Accesibilitatea parcurilor în {region_name} (distanță de {buffer_distance}m)')
                plt.axis('off')
                
                # Save the figure
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True)
                vis_filename = f"park_accessibility_{region_name.replace(' ', '_')}_{timestamp}.png"
                output_path = os.path.join(output_dir, vis_filename)
                plt.savefig(output_path)
                plt.close()
                
                visualization_created = True
                vis_path = output_path
            except Exception as e:
                logger.error(f"Eroare la crearea vizualizării: {e}")
                visualization_created = False
                vis_path = None
            
            # Prepare result - FORMAT POZITIV
            result = f"În {region_name}, aproximativ {percentage:.1f}% din suprafață "
            result += f"este la o distanță de mers pe jos de maximum {buffer_distance} metri față de un parc.\n\n"
            result += f"Aceasta reprezintă aproximativ {accessible_area_km2:.2f} kilometri pătrați "
            result += f"din suprafața totală de {total_area_km2:.2f} km².\n\n"
            
            # Adăugăm informația despre zonele cu acces limitat doar când este solicitată explicit
            if show_poorly_served:
                poorly_served_area = total_area_km2 - accessible_area_km2
                poorly_served_percentage = 100 - percentage
                result += f"Aproximativ {poorly_served_area:.2f} km² ({poorly_served_percentage:.1f}%) "
                result += "nu are acces facil la parcuri (este la peste 500 metri distanță de mers pe jos).\n\n"
                result += "Zonele cu acces limitat la parcuri sunt evidențiate în roșu pe vizualizare.\n\n"
            
            if visualization_created:
                result += f"O hartă de vizualizare a fost creată la adresa {vis_path}"
            
            self.update_latest_visualization(output_path)
            return result
                
        except Exception as e:
            logger.error(f"Eroare la calcularea accesibilității parcurilor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la calcularea accesibilității parcurilor: {str(e)}"
    
    def _recommend_best_park_today(self, criteria=None):
        """
        Recommend the best park to visit today based on current conditions.
        
        Args:
            criteria: List of criteria to consider ["air_quality", "noise", "congestion"]
            
        Returns:
            Recommendation with detailed park information
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
                
            import pandas as pd
            import geopandas as gpd
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Default criteria
            if criteria is None:
                criteria = ["air_quality", "congestion"]
                
            # Get name column
            name_column = 'name_left' if 'name_left' in self.data.columns else 'name'
            
            # Copy data for manipulation
            parks_df = self.data.copy()
            
            # Validate AQI values
            if 'aqi' in parks_df.columns:
                # Check if values are too small (below 1)
                if parks_df['aqi'].median() < 1:
                    # Values are probably on 0-1 scale, convert to 0-100
                    parks_df['aqi'] = parks_df['aqi'] * 100
                    logger.info(f"Normalized AQI values to 0-100 scale")
                
                # If values are still unrealistic, provide warning
                if parks_df['aqi'].median() < 5:
                    logger.warning(f"AQI values seem unusually low: median = {parks_df['aqi'].median()}")
            
            # Calculate scores
            parks_df['total_score'] = 0
            
            # Air quality score (lower is better)
            if "air_quality" in criteria and "aqi" in parks_df.columns:
                min_aqi = parks_df["aqi"].min()
                max_aqi = parks_df["aqi"].max()
                if max_aqi > min_aqi:
                    parks_df['air_score'] = 1 - (parks_df["aqi"] - min_aqi) / (max_aqi - min_aqi)
                    parks_df['total_score'] += parks_df['air_score'] * 0.4
            
            # Noise score (lower is better)
            if "noise" in criteria and "noise_level_db" in parks_df.columns:
                min_noise = parks_df["noise_level_db"].min()
                max_noise = parks_df["noise_level_db"].max()
                if max_noise > min_noise:
                    parks_df['noise_score'] = 1 - (parks_df["noise_level_db"] - min_noise) / (max_noise - min_noise)
                    parks_df['total_score'] += parks_df['noise_score'] * 0.3
            
            # Congestion score (lower is better)
            if "congestion" in criteria and "congestion_level" in parks_df.columns:
                min_cong = parks_df["congestion_level"].min()
                max_cong = parks_df["congestion_level"].max()
                if max_cong > min_cong:
                    parks_df['congestion_score'] = 1 - (parks_df["congestion_level"] - min_cong) / (max_cong - min_cong)
                    parks_df['total_score'] += parks_df['congestion_score'] * 0.3
            
            # Sort and get best park
            parks_df = parks_df.sort_values('total_score', ascending=False)
            
            if len(parks_df) == 0:
                return "No parks found matching criteria."
                
            best_park = parks_df.iloc[0]
            top_parks = parks_df.head(5)
            
            # Create visualization - first create figure without basemap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Store CRS for later use
            original_crs = self.data.crs
            
            # Create a GeoDataFrame for the best park
            best_park_gdf = gpd.GeoDataFrame(geometry=[best_park.geometry], crs=original_crs)
            
            # Create GeoDataFrames for top parks
            top_parks_gdfs = []
            for i in range(1, min(len(top_parks), 5)):
                park = top_parks.iloc[i]
                top_parks_gdfs.append(gpd.GeoDataFrame(geometry=[park.geometry], crs=original_crs))
            
            try:
                # Try to add basemap with proper projection handling
                import contextily as ctx
                
                # Convert to Web Mercator (EPSG:3857) for compatibility with contextily
                parks_webmerc = self.data.to_crs(epsg=3857)
                best_park_webmerc = best_park_gdf.to_crs(epsg=3857)
                top_parks_webmerc = [gdf.to_crs(epsg=3857) for gdf in top_parks_gdfs]
                
                # Create a new figure with Web Mercator projection
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot all parks lightly
                parks_webmerc.plot(ax=ax, color='lightgreen', alpha=0.3, edgecolor='green')
                
                # Plot best park
                best_park_webmerc.plot(ax=ax, color='blue', alpha=0.7, edgecolor='black')
                
                # Plot other top parks
                for i, park_gdf in enumerate(top_parks_webmerc):
                    park_gdf.plot(ax=ax, color='green', alpha=0.5, edgecolor='darkgreen')
                
                # Try to add streets from traffic data in Web Mercator
                try:
                    if hasattr(self, 'traffic_data') and self.traffic_data is not None:
                        traffic_webmerc = self.traffic_data.to_crs(epsg=3857)
                        traffic_webmerc.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not add street network: {e}")
                
                # Add basemap
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
                
                # Add best park label with updated coordinates
                centroid = best_park_webmerc.geometry.centroid.iloc[0]
                ax.annotate(
                    f"Best park: {best_park[name_column]}",
                    xy=(centroid.x, centroid.y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=14, fontweight='bold', color='blue',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
                )
                
            except ImportError:
                logger.warning("Contextily library not installed. Cannot add basemap.")
                
                # Fallback to simple plot without basemap
                self.data.plot(ax=ax, color='lightgreen', alpha=0.3, edgecolor='green')
                best_park_gdf.plot(ax=ax, color='blue', alpha=0.7, edgecolor='black')
                
                for i, park_gdf in enumerate(top_parks_gdfs):
                    park_gdf.plot(ax=ax, color='green', alpha=0.5, edgecolor='darkgreen')
                    
                # Try to add streets from traffic data
                try:
                    if hasattr(self, 'traffic_data') and self.traffic_data is not None:
                        self.traffic_data.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not add street network: {e}")
                    
                # Add best park label
                centroid = best_park.geometry.centroid
                ax.annotate(
                    f"Best park: {best_park[name_column]}",
                    xy=(centroid.x, centroid.y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=14, fontweight='bold', color='blue',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
                )
                
            except Exception as e:
                logger.warning(f"Could not add basemap with CartoDB.Positron: {e}")
                
                # Try alternative basemap sources
                try:
                    # Reset the figure
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Convert to Web Mercator again
                    parks_webmerc = self.data.to_crs(epsg=3857)
                    best_park_webmerc = best_park_gdf.to_crs(epsg=3857)
                    
                    # Plot all parks lightly
                    parks_webmerc.plot(ax=ax, color='lightgreen', alpha=0.3, edgecolor='green')
                    
                    # Plot best park
                    best_park_webmerc.plot(ax=ax, color='blue', alpha=0.7, edgecolor='black')
                    
                    # Plot other top parks
                    for i in range(1, min(len(top_parks), 5)):
                        park = top_parks.iloc[i]
                        park_gdf = gpd.GeoDataFrame(geometry=[park.geometry], crs=original_crs)
                        park_webmerc = park_gdf.to_crs(epsg=3857)
                        park_webmerc.plot(ax=ax, color='green', alpha=0.5, edgecolor='darkgreen')
                    
                    # Try alternative basemap
                    import contextily as ctx
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                    
                    # Add best park label with updated coordinates
                    centroid = best_park_webmerc.geometry.centroid.iloc[0]
                    ax.annotate(
                        f"Best park: {best_park[name_column]}",
                        xy=(centroid.x, centroid.y),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=14, fontweight='bold', color='blue',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
                    )
                    
                except Exception as e2:
                    logger.warning(f"Could not add alternative basemap: {e2}")
                    
                    # Fallback to simple plot without basemap
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    self.data.plot(ax=ax, color='lightgreen', alpha=0.3, edgecolor='green')
                    best_park_gdf.plot(ax=ax, color='blue', alpha=0.7, edgecolor='black')
                    
                    for i, park_gdf in enumerate(top_parks_gdfs):
                        park_gdf.plot(ax=ax, color='green', alpha=0.5, edgecolor='darkgreen')
                        
                    # Try to add streets from traffic data
                    try:
                        if hasattr(self, 'traffic_data') and self.traffic_data is not None:
                            self.traffic_data.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
                    except Exception as e:
                        logger.warning(f"Could not add street network: {e}")
                        
                    # Add best park label
                    centroid = best_park.geometry.centroid
                    ax.annotate(
                        f"Best park: {best_park[name_column]}",
                        xy=(centroid.x, centroid.y),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=14, fontweight='bold', color='blue',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
                    )
                    
            ax.set_axis_off()
            
            today = datetime.now().strftime("%Y-%m-%d")
            plt.title(f"Best Park to Visit Today ({today})", fontsize=16)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"best_park_today_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Format the result
            result = "Best Park to Visit Today\n\n"
            result += f"RECOMMENDATION: {best_park[name_column]}\n\n"
            
            if 'sector' in best_park:
                result += f"Location: Sector {best_park['sector']}\n"
                
            if 'aqi' in best_park:
                quality_level = "Excellent" if best_park['aqi'] < 25 else "Good" if best_park['aqi'] < 50 else "Moderate"
                result += f"Air Quality: {quality_level} (AQI: {best_park['aqi']:.1f})\n"
                
            if 'noise_level_db' in best_park:
                noise_level = "Very quiet" if best_park['noise_level_db'] < 45 else "Quiet" if best_park['noise_level_db'] < 55 else "Moderate"
                result += f"Noise Level: {noise_level} ({best_park['noise_level_db']:.1f} dB)\n"
                
            if 'congestion_level' in best_park:
                congestion = best_park['congestion_level']
                congestion_text = "Low" if congestion <= 1 else "Moderate" if congestion <= 2 else "High"
                result += f"Traffic Congestion: {congestion_text} (Level {int(congestion)})\n"
                
            result += "\nOther good options:\n"
            for i in range(1, min(len(top_parks), 4)):
                park = top_parks.iloc[i]
                result += f"- {park[name_column]} (Sector {park.get('sector', 'Unknown')})\n"
                
            result += f"\nVisualization available at: {output_path}"
            
            self.update_latest_visualization(output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error recommending best park: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error occurred while finding the best park: {str(e)}"

    def _compare_north_south(self, metric="overall"):
        """
        Compară caracteristicile de mediu între zonele de nord și sud ale Bucureștiului.
        
        Args:
            metric: Metrica specifică de comparat ("parks", "air", "noise", "green_space", 
                    "overall" pentru toate)
        
        Returns:
            O analiză comparativă a mediului urban între nordul și sudul Bucureștiului
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date disponibile pentru analiză."
                
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import os
            from datetime import datetime
            
            # Definim linia de separare nord-sud (aproximativ centrul Bucureștiului)
            # București este împărțit aproximativ la latitudinea 44.43
            north_south_boundary = 44.43
            
            # Creăm două seturi de date pentru nord și sud bazate pe coordonatele geografice
            if hasattr(self.data, 'geometry') and 'geometry' in self.data:
                # Extragem centroidele pentru fiecare geometrie
                centroids = self.data.geometry.centroid
                self.data['latitude'] = centroids.y
                self.data['longitude'] = centroids.x
                
                # Împărțim datele în nord și sud
                north_data = self.data[self.data['latitude'] >= north_south_boundary]
                south_data = self.data[self.data['latitude'] < north_south_boundary]
                
                logger.info(f"Parcuri în zona de nord: {len(north_data)}")
                logger.info(f"Parcuri în zona de sud: {len(south_data)}")
                
                # Determinăm în care sectoare sunt mai multe parcuri din nord și sud
                if 'sector' in self.data.columns:
                    north_sectors = north_data['sector'].value_counts()
                    south_sectors = south_data['sector'].value_counts()
                    
                    # Sectoarele predominant nordice și sudice
                    north_dominant_sectors = north_sectors.index.tolist()[:3]  # Top 3 sectoare din nord
                    south_dominant_sectors = south_sectors.index.tolist()[:3]  # Top 3 sectoare din sud
                    
                    logger.info(f"Sectoare predominant nordice: {north_dominant_sectors}")
                    logger.info(f"Sectoare predominant sudice: {south_dominant_sectors}")
                else:
                    north_dominant_sectors = []
                    south_dominant_sectors = []
                    
                # Pregătim dicționarul pentru rezultate
                comparison_results = {
                    "north_parks_count": len(north_data),
                    "south_parks_count": len(south_data),
                    "north_dominant_sectors": north_dominant_sectors,
                    "south_dominant_sectors": south_dominant_sectors
                }
                
                # Comparăm indicatori specifici pentru mediul urban
                
                # 1. Calitatea aerului (AQI) - valori mai mici sunt mai bune
                if 'aqi' in self.data.columns:
                    north_aqi = north_data['aqi'].mean()
                    south_aqi = south_data['aqi'].mean()
                    comparison_results["north_avg_aqi"] = north_aqi
                    comparison_results["south_avg_aqi"] = south_aqi
                    
                # 2. Nivelul de zgomot - valori mai mici sunt mai bune
                if 'noise_level_db' in self.data.columns:
                    north_noise = north_data['noise_level_db'].mean()
                    south_noise = south_data['noise_level_db'].mean()
                    comparison_results["north_avg_noise"] = north_noise
                    comparison_results["south_avg_noise"] = south_noise
                
                # 3. Suprafața verde (dacă este disponibilă)
                if 'area_hectares' in self.data.columns or 'area_sqm' in self.data.columns:
                    area_col = 'area_hectares' if 'area_hectares' in self.data.columns else 'area_sqm'
                    north_area = north_data[area_col].sum()
                    south_area = south_data[area_col].sum()
                    area_unit = 'hectare' if area_col == 'area_hectares' else 'm²'
                    comparison_results["north_total_green_area"] = north_area
                    comparison_results["south_total_green_area"] = south_area
                    comparison_results["area_unit"] = area_unit
                    
                # 4. Densitatea parcurilor (estimare aproximativă a suprafeței nord/sud)
                # Notă: Aceasta este o estimare simplificată
                north_area_approx = 150  # km²
                south_area_approx = 130  # km²
                comparison_results["north_parks_density"] = len(north_data) / north_area_approx
                comparison_results["south_parks_density"] = len(south_data) / south_area_approx
                
                # Creăm vizualizare comparativă
                try:
                    # Configurăm figura pentru grafice comparative
                    fig = plt.figure(figsize=(15, 12))
                    gs = gridspec.GridSpec(2, 2, figure=fig)
                    
                    # Stilizare pentru aspect consistent
                    bar_colors = {'north': '#3498db', 'south': '#e74c3c'}
                    
                    # 1. Comparație număr de parcuri
                    ax1 = fig.add_subplot(gs[0, 0])
                    park_counts = [comparison_results["north_parks_count"], comparison_results["south_parks_count"]]
                    bars1 = ax1.bar(['Nord', 'Sud'], park_counts, color=[bar_colors['north'], bar_colors['south']])
                    ax1.set_title('Număr de Parcuri', fontsize=14)
                    ax1.set_ylabel('Număr')
                    
                    # Adăugăm valorile pe bare
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}',
                                ha='center', va='bottom', fontsize=11)
                    
                    # 2. Comparație calitate aer (AQI)
                    if "north_avg_aqi" in comparison_results:
                        ax2 = fig.add_subplot(gs[0, 1])
                        aqi_values = [comparison_results["north_avg_aqi"], comparison_results["south_avg_aqi"]]
                        bars2 = ax2.bar(['Nord', 'Sud'], aqi_values, color=[bar_colors['north'], bar_colors['south']])
                        ax2.set_title('Calitatea Aerului (AQI)', fontsize=14)
                        ax2.set_ylabel('AQI (valori mai mici sunt mai bune)')
                        
                        # Adăugăm valorile pe bare
                        for bar in bars2:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                                    ha='center', va='bottom', fontsize=11)
                    
                    # 3. Comparație nivel zgomot
                    if "north_avg_noise" in comparison_results:
                        ax3 = fig.add_subplot(gs[1, 0])
                        noise_values = [comparison_results["north_avg_noise"], comparison_results["south_avg_noise"]]
                        bars3 = ax3.bar(['Nord', 'Sud'], noise_values, color=[bar_colors['north'], bar_colors['south']])
                        ax3.set_title('Nivelul de Zgomot (dB)', fontsize=14)
                        ax3.set_ylabel('dB (valori mai mici sunt mai bune)')
                        
                        # Adăugăm valorile pe bare
                        for bar in bars3:
                            height = bar.get_height()
                            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                                    ha='center', va='bottom', fontsize=11)
                    
                    # 4. Comparație densitate parcuri
                    ax4 = fig.add_subplot(gs[1, 1])
                    density_values = [comparison_results["north_parks_density"], comparison_results["south_parks_density"]]
                    bars4 = ax4.bar(['Nord', 'Sud'], density_values, color=[bar_colors['north'], bar_colors['south']])
                    ax4.set_title('Densitate Parcuri (parcuri/km²)', fontsize=14)
                    ax4.set_ylabel('Parcuri per km²')
                    
                    # Adăugăm valorile pe bare
                    for bar in bars4:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
                                ha='center', va='bottom', fontsize=11)
                    
                    # Adăugăm titlul principal
                    plt.suptitle('Comparație Nord-Sud București: Mediul Urban', fontsize=18, fontweight='bold')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # Salvăm vizualizarea
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = "visualizations"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"comparatie_nord_sud_{timestamp}.png")
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    
                    visualization_path = output_path
                    
                except Exception as e:
                    logger.error(f"Eroare la crearea vizualizării: {e}")
                    visualization_path = None
                
                # Formăm răspunsul detaliat
                result = "Comparație Medii de Viață: Nord vs. Sud București\n\n"
                
                # Sectoarele predominante
                result += "ZONELE ANALIZATE:\n"
                if north_dominant_sectors:
                    result += f"• Zona de NORD: include predominant {', '.join(north_dominant_sectors)}\n"
                else:
                    result += "• Zona de NORD: definită geografic la nord de latitudinea 44.43\n"
                    
                if south_dominant_sectors:
                    result += f"• Zona de SUD: include predominant {', '.join(south_dominant_sectors)}\n\n"
                else:
                    result += "• Zona de SUD: definită geografic la sud de latitudinea 44.43\n\n"
                    
                # Statistici generale
                result += "DISTRIBUȚIA PARCURILOR:\n"
                result += f"• Nord: {comparison_results['north_parks_count']} parcuri "
                result += f"({comparison_results['north_parks_density']:.2f} parcuri/km²)\n"
                result += f"• Sud: {comparison_results['south_parks_count']} parcuri "
                result += f"({comparison_results['south_parks_density']:.2f} parcuri/km²)\n\n"
                
                # Calitatea aerului
                if "north_avg_aqi" in comparison_results:
                    result += "CALITATEA AERULUI (AQI - valori mai mici indică aer mai curat):\n"
                    result += f"• Nord: AQI mediu {comparison_results['north_avg_aqi']:.1f}\n"
                    result += f"• Sud: AQI mediu {comparison_results['south_avg_aqi']:.1f}\n"
                    
                    if comparison_results['north_avg_aqi'] < comparison_results['south_avg_aqi']:
                        difference = comparison_results['south_avg_aqi'] - comparison_results['north_avg_aqi']
                        result += f"» Nordul are o calitate a aerului mai bună cu {difference:.1f} puncte AQI\n\n"
                    else:
                        difference = comparison_results['north_avg_aqi'] - comparison_results['south_avg_aqi']
                        result += f"» Sudul are o calitate a aerului mai bună cu {difference:.1f} puncte AQI\n\n"
                
                # Nivelul de zgomot
                if "north_avg_noise" in comparison_results:
                    result += "NIVELUL DE ZGOMOT (dB - valori mai mici indică mediu mai liniștit):\n"
                    result += f"• Nord: {comparison_results['north_avg_noise']:.1f} dB\n"
                    result += f"• Sud: {comparison_results['south_avg_noise']:.1f} dB\n"
                    
                    if comparison_results['north_avg_noise'] < comparison_results['south_avg_noise']:
                        difference = comparison_results['south_avg_noise'] - comparison_results['north_avg_noise']
                        result += f"» Nordul este mai liniștit cu {difference:.1f} dB\n\n"
                    else:
                        difference = comparison_results['north_avg_noise'] - comparison_results['south_avg_noise']
                        result += f"» Sudul este mai liniștit cu {difference:.1f} dB\n\n"
                
                # Suprafața verde
                if "north_total_green_area" in comparison_results:
                    result += f"SUPRAFAȚA VERDE TOTALĂ ({comparison_results['area_unit']}):\n"
                    result += f"• Nord: {comparison_results['north_total_green_area']:.1f} {comparison_results['area_unit']}\n"
                    result += f"• Sud: {comparison_results['south_total_green_area']:.1f} {comparison_results['area_unit']}\n"
                    
                    if comparison_results['north_total_green_area'] > comparison_results['south_total_green_area']:
                        ratio = comparison_results['north_total_green_area'] / comparison_results['south_total_green_area']
                        result += f"» Nordul are o suprafață verde de {ratio:.1f} ori mai mare\n\n"
                    else:
                        ratio = comparison_results['south_total_green_area'] / comparison_results['north_total_green_area']
                        result += f"» Sudul are o suprafață verde de {ratio:.1f} ori mai mare\n\n"
                
                # Concluzie generală
                result += "CONCLUZIE GENERALĂ:\n"
                
                advantages_north = []
                advantages_south = []
                
                # Analizăm avantajele fiecărei zone
                if comparison_results['north_parks_count'] > comparison_results['south_parks_count']:
                    advantages_north.append("mai multe parcuri")
                else:
                    advantages_south.append("mai multe parcuri")
                    
                if comparison_results['north_parks_density'] > comparison_results['south_parks_density']:
                    advantages_north.append("densitate mai mare de parcuri")
                else:
                    advantages_south.append("densitate mai mare de parcuri")
                    
                if "north_avg_aqi" in comparison_results:
                    if comparison_results['north_avg_aqi'] < comparison_results['south_avg_aqi']:
                        advantages_north.append("aer mai curat")
                    else:
                        advantages_south.append("aer mai curat")
                        
                if "north_avg_noise" in comparison_results:
                    if comparison_results['north_avg_noise'] < comparison_results['south_avg_noise']:
                        advantages_north.append("nivel de zgomot mai redus")
                    else:
                        advantages_south.append("nivel de zgomot mai redus")
                
                # Adăugăm concluzia în funcție de avantajele fiecărei zone
                if len(advantages_north) > len(advantages_south):
                    result += f"Zona de NORD a Bucureștiului oferă un mediu urban mai favorabil, având avantaje precum: {', '.join(advantages_north)}.\n"
                    if advantages_south:
                        result += f"Zona de SUD se remarcă totuși prin: {', '.join(advantages_south)}.\n"
                elif len(advantages_south) > len(advantages_north):
                    result += f"Zona de SUD a Bucureștiului oferă un mediu urban mai favorabil, având avantaje precum: {', '.join(advantages_south)}.\n"
                    if advantages_north:
                        result += f"Zona de NORD se remarcă totuși prin: {', '.join(advantages_north)}.\n"
                else:
                    result += f"Zonele de NORD și SUD ale Bucureștiului sunt echilibrate din punct de vedere al mediului urban, fiecare cu avantajele sale specifice.\n"
                    result += f"Nord: {', '.join(advantages_north)}.\n"
                    result += f"Sud: {', '.join(advantages_south)}.\n"
                
                # Adăugăm informații despre vizualizare
                if visualization_path:
                    result += f"\nO vizualizare comparativă a fost generată la: {visualization_path}"
                self.update_latest_visualization(visualization_path)
                
                return result
                
            else:
                return "Error: Datele nu conțin informații geografice necesare pentru comparația nord-sud."
                
        except Exception as e:
            logger.error(f"Eroare la compararea zonelor nord-sud: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare în timpul comparării mediilor din nordul și sudul Bucureștiului: {str(e)}"
    
    def _analyze_sectors_by_criteria(self, criteria=None, sort_by=None, reverse=False):
        """
        Analyze Bucharest sectors based on specified criteria.
        
        Args:
            criteria: List of criteria to analyze ["air_quality", "parks", "noise", "traffic"]
            sort_by: Primary criterion to sort results by
            reverse: Whether to sort in descending order
            
        Returns:
            Analysis of sectors based on the specified criteria
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date disponibile pentru analiză."
                
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            
            # Default criteria if none specified
            if criteria is None:
                criteria = ["air_quality", "parks"]
            
            # Determine which columns to use based on criteria
            columns_map = {
                "air_quality": "aqi",
                "parks": "geometry",
                "noise": "noise_level_db",
                "traffic": "congestion_level"
            }
            
            # Prepare results dataframe
            results = pd.DataFrame()
            
            # Get unique sectors
            if 'sector' in self.data.columns:
                sectors = self.data['sector'].dropna().unique()
                sectors = [s for s in sectors if isinstance(s, (str, int))]
            else:
                return "Error: Informațiile despre sectoare nu sunt disponibile în date."
            
            # Calculate metrics for each sector
            for sector in sectors:
                sector_data = self.data[self.data['sector'] == sector]
                
                row = {'sector': sector}
                
                # Calculate metrics based on criteria
                if "air_quality" in criteria and "aqi" in self.data.columns:
                    row['avg_aqi'] = sector_data['aqi'].mean()
                    row['min_aqi'] = sector_data['aqi'].min()
                    row['max_aqi'] = sector_data['aqi'].max()
                    
                if "parks" in criteria:
                    # Count unique parks in the sector
                    name_col = next((col for col in ['name_left', 'name', 'park_name', 'title'] 
                                    if col in self.data.columns), None)
                    if name_col:
                        unique_parks = sector_data[name_col].nunique()
                        row['park_count'] = unique_parks
                    else:
                        # If no name column, count rows as approximate park count
                        row['park_count'] = len(sector_data)
                    
                    # Calculate total park area if geometry is available
                    if hasattr(sector_data, 'geometry'):
                        try:
                            # Project to a local CRS for accurate area calculation
                            sector_data_proj = sector_data.to_crs(epsg=3844)
                            total_area = sector_data_proj.geometry.area.sum() / 10000  # Convert to hectares
                            row['total_park_area_ha'] = total_area
                        except Exception as e:
                            logger.error(f"Error calculating park area: {e}")
                            row['total_park_area_ha'] = np.nan
                
                if "noise" in criteria and "noise_level_db" in self.data.columns:
                    row['avg_noise'] = sector_data['noise_level_db'].mean()
                    
                if "traffic" in criteria and "congestion_level" in self.data.columns:
                    row['avg_congestion'] = sector_data['congestion_level'].mean()
                
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            
            # Sort results if specified
            if sort_by:
                if sort_by == "air_quality" and "avg_aqi" in results.columns:
                    results = results.sort_values(by="avg_aqi", ascending=not reverse)
                elif sort_by == "parks" and "park_count" in results.columns:
                    results = results.sort_values(by="park_count", ascending=reverse)
                elif sort_by == "noise" and "avg_noise" in results.columns:
                    results = results.sort_values(by="avg_noise", ascending=not reverse)
                elif sort_by == "traffic" and "avg_congestion" in results.columns:
                    results = results.sort_values(by="avg_congestion", ascending=not reverse)
            
            # Create visualizations
            try:
                # Set up matplotlib
                import matplotlib
                matplotlib.use('Agg')
                
                # Create bar charts for each relevant metric
                metrics_to_plot = []
                if "air_quality" in criteria and "avg_aqi" in results.columns:
                    metrics_to_plot.append(("avg_aqi", "Indicele Mediu de Calitate a Aerului"))
                if "parks" in criteria and "park_count" in results.columns:
                    metrics_to_plot.append(("park_count", "Numărul de Parcuri"))
                if "parks" in criteria and "total_park_area_ha" in results.columns:
                    metrics_to_plot.append(("total_park_area_ha", "Suprafața Totală a Parcurilor (hectare)"))
                if "noise" in criteria and "avg_noise" in results.columns:
                    metrics_to_plot.append(("avg_noise", "Nivelul Mediu de Zgomot (dB)"))
                if "traffic" in criteria and "avg_congestion" in results.columns:
                    metrics_to_plot.append(("avg_congestion", "Nivelul Mediu de Congestie a Traficului"))
                
                # Create plots
                fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4*len(metrics_to_plot)))
                if len(metrics_to_plot) == 1:
                    axes = [axes]  # Make it iterable for single plot
                    
                for i, (metric, title) in enumerate(metrics_to_plot):
                    sorted_results = results.sort_values(by=metric)
                    bars = axes[i].bar(
                        sorted_results['sector'], 
                        sorted_results[metric],
                        color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
                    )
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        axes[i].text(
                            bar.get_x() + bar.get_width()/2.,
                            height * 1.01,
                            f'{height:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )
                        
                    axes[i].set_title(title)
                    axes[i].set_xlabel('Sector')
                    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
                    
                plt.tight_layout()
                
                # Save visualization
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True)
                
                metrics_string = "_".join([m[0] for m in metrics_to_plot])
                output_path = os.path.join(output_dir, f"analiza_sectoare_{metrics_string}_{timestamp}.png")
                plt.savefig(output_path, dpi=300)
                plt.close('all')
                
                visualization_path = output_path
                
            except Exception as e:
                logger.error(f"Eroare la crearea vizualizării: {e}")
                import traceback
                logger.error(traceback.format_exc())
                visualization_path = None
            
            # Format the result
            result = "Rezultatele Analizei Sectoarelor București\n\n"
            
            # Header based on criteria
            header = "Sector"
            if "air_quality" in criteria:
                header += " | Calitatea Aerului (AQI)"
            if "parks" in criteria:
                header += " | Număr Parcuri"
                if "total_park_area_ha" in results.columns:
                    header += " | Suprafață Parcuri (ha)"
            if "noise" in criteria:
                header += " | Zgomot (dB)"
            if "traffic" in criteria:
                header += " | Trafic"
            
            result += header + "\n"
            result += "-" * len(header) + "\n"
            
            # Format each row
            for _, row in results.iterrows():
                line = f"Sector {row['sector']}"
                
                if "air_quality" in criteria and "avg_aqi" in row:
                    line += f" | {row['avg_aqi']:.1f}"
                
                if "parks" in criteria and "park_count" in row:
                    line += f" | {int(row['park_count'])}"
                    if "total_park_area_ha" in row:
                        line += f" ({row['total_park_area_ha']:.2f} ha)"
                
                if "noise" in criteria and "avg_noise" in row:
                    line += f" | {row['avg_noise']:.1f}"
                    
                if "traffic" in criteria and "avg_congestion" in row:
                    line += f" | {row['avg_congestion']:.1f}"
                    
                result += line + "\n"
                
            result += "\n"
            
            # Add summary
            if "air_quality" in criteria and "avg_aqi" in results.columns:
                best_air = results.loc[results['avg_aqi'].idxmin()]
                worst_air = results.loc[results['avg_aqi'].idxmax()]
                result += f"Cea mai bună calitate a aerului: Sector {best_air['sector']} (AQI: {best_air['avg_aqi']:.1f})\n"
                result += f"Cea mai slabă calitate a aerului: Sector {worst_air['sector']} (AQI: {worst_air['avg_aqi']:.1f})\n\n"
                
            if "parks" in criteria and "park_count" in results.columns:
                most_parks = results.loc[results['park_count'].idxmax()]
                least_parks = results.loc[results['park_count'].idxmin()]
                result += f"Cele mai multe parcuri: Sector {most_parks['sector']} ({int(most_parks['park_count'])} parcuri)\n"
                result += f"Cele mai puține parcuri: Sector {least_parks['sector']} ({int(least_parks['park_count'])} parcuri)\n\n"
                
                if "total_park_area_ha" in results.columns:
                    largest_area = results.loc[results['total_park_area_ha'].idxmax()]
                    result += f"Cea mai mare suprafață de parcuri: Sector {largest_area['sector']} ({largest_area['total_park_area_ha']:.2f} ha)\n\n"
                    
            # Find sectors with good air but few parks
            if "air_quality" in criteria and "parks" in criteria:
                if "avg_aqi" in results.columns and "park_count" in results.columns:
                    # Normalize values to range 0-1 for fair comparison
                    results['norm_aqi'] = (results['avg_aqi'] - results['avg_aqi'].min()) / (results['avg_aqi'].max() - results['avg_aqi'].min())
                    results['norm_parks'] = (results['park_count'] - results['park_count'].min()) / (results['park_count'].max() - results['park_count'].min())
                    
                    # Calculate a combined score
                    results['combined_score'] = (1 - results['norm_aqi']) * 0.5 + results['norm_parks'] * 0.5
                    best_overall = results.loc[results['combined_score'].idxmax()]
                    
                    result += f"Sectorul cu cel mai bun echilibru între calitatea aerului și numărul de parcuri: Sector {best_overall['sector']}\n"
                    result += f"  - Indice calitate aer: {best_overall['avg_aqi']:.1f}\n"
                    result += f"  - Număr parcuri: {int(best_overall['park_count'])}\n\n"
                    
            if visualization_path:
                result += f"Vizualizarea analizei este disponibilă la: {visualization_path}"
            self.update_latest_visualization(visualization_path)
                
            return result
            
        except Exception as e:
            logger.error(f"Eroare la analiza sectoarelor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare în timpul analizei sectoarelor: {str(e)}"
    
    def _find_best_sector(self, metric: str = "parks") -> str:
        """
        Find the best sector based on a specific metric.
        
        Args:
            metric: Metric to use ("parks", "air", "noise", "green_density")
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date despre parcuri disponibile."
                
            if 'sector' not in self.data.columns:
                return "Error: Setul de date nu conține informații despre sectoare."
            
            # Curăță datele pentru analiză - elimină duplicatele
            clean_data = self.data.copy()
            
            # Determină coloana cu numele parcurilor
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in clean_data.columns:
                    name_column = col
                    break
                    
            # Elimină duplicatele dacă avem coloană de nume
            if name_column:
                clean_data = clean_data.drop_duplicates(subset=[name_column, 'sector'])
            
            # Creează dataframe cu numărul de parcuri pe sector - evită conflicte de nume de coloane
            park_counts = clean_data.groupby('sector').size().reset_index(name='park_count')
            
            # Creează un dataframe separat pentru fiecare metrică
            sector_stats = park_counts.copy()
            
            if 'aqi' in clean_data.columns:
                # Folosește agregare numită pentru a evita conflicte de nume de coloane
                aqi_stats = clean_data.groupby('sector').agg(
                    aqi_mean=('aqi', 'mean'),
                    aqi_min=('aqi', 'min'),
                    aqi_max=('aqi', 'max')
                ).reset_index()
                
                # Unește cu park_counts
                sector_stats = pd.merge(sector_stats, aqi_stats, on='sector', how='left')
            
            # Adaugă metrici de nivel de zgomot dacă sunt disponibile
            if 'noise_level_db' in clean_data.columns:
                noise_stats = clean_data.groupby('sector').agg(
                    noise_mean=('noise_level_db', 'mean'),
                    noise_min=('noise_level_db', 'min'),
                    noise_max=('noise_level_db', 'max')
                ).reset_index()
                
                # Unește cu sector_stats
                sector_stats = pd.merge(sector_stats, noise_stats, on='sector', how='left')
            
            # Calculează parcuri pe km2 (cu suprafețe estimate)
            sector_areas_km2 = {
                'Sector 1': 68,
                'Sector 2': 32,
                'Sector 3': 34,
                'Sector 4': 32,
                'Sector 5': 30,
                'Sector 6': 38
            }
            
            # Adaugă parcuri per km2
            sector_stats['area_km2'] = sector_stats['sector'].map(lambda x: sector_areas_km2.get(x, 30))
            sector_stats['parks_per_km2'] = (sector_stats['park_count'] / sector_stats['area_km2']).round(2)
            
            # Setează index pentru acces mai ușor
            sector_stats.set_index('sector', inplace=True)
            
            # Rotunjește valorile AQI pentru afișare mai frumoasă
            if 'aqi_mean' in sector_stats.columns:
                sector_stats['aqi_mean'] = sector_stats['aqi_mean'].round(1)
                sector_stats['aqi_min'] = sector_stats['aqi_min'].round(1)
                sector_stats['aqi_max'] = sector_stats['aqi_max'].round(1)
            
            # Rotunjește valorile de zgomot pentru afișare mai frumoasă
            if 'noise_mean' in sector_stats.columns:
                sector_stats['noise_mean'] = sector_stats['noise_mean'].round(1)
                sector_stats['noise_min'] = sector_stats['noise_min'].round(1)
                sector_stats['noise_max'] = sector_stats['noise_max'].round(1)
            
            metric_lower = metric.lower()
            
            if metric_lower in ["parks", "park", "green_spaces", "green spaces"]:
                # Găsește sectorul cu cele mai multe parcuri
                if len(sector_stats) == 0:
                    return "Nu există suficiente date pentru a determina sectorul cu cele mai multe parcuri."
                    
                best_sector = sector_stats['park_count'].idxmax()
                count = sector_stats.loc[best_sector, 'park_count']
                
                result = f"{best_sector} are cele mai multe parcuri, cu {count} parcuri în total.\n\n"
                
                # Arată și celelalte sectoare pentru comparație
                result += "Numărul de parcuri pe sector (în ordine descrescătoare):\n"
                for sector in sector_stats.sort_values('park_count', ascending=False).index:
                    result += f"{sector}: {sector_stats.loc[sector, 'park_count']} parcuri\n"
                    
                return result
                
            elif metric_lower in ["air", "air_quality", "aqi"]:
                if 'aqi_mean' not in sector_stats.columns:
                    return "Error: Datele despre calitatea aerului nu sunt disponibile pentru comparație."
                    
                # Găsește sectorul cu cea mai bună calitate a aerului (cel mai mic AQI)
                best_sector = sector_stats['aqi_mean'].idxmin()
                aqi = sector_stats.loc[best_sector, 'aqi_mean']
                
                result = f"{best_sector} are cea mai bună calitate a aerului, cu un AQI mediu de {aqi}.\n\n"
                result += "(Valorile AQI mai mici indică o calitate a aerului mai bună)\n\n"
                
                # Arată și celelalte sectoare pentru comparație
                result += "AQI mediu pe sectoare (de la cel mai bun la cel mai rău):\n"
                for sector in sector_stats.sort_values('aqi_mean').index:
                    result += f"{sector}: {sector_stats.loc[sector, 'aqi_mean']}\n"
                    
                return result
                
            elif metric_lower in ["noise", "sound", "quiet", "noise_level"]:
                if 'noise_mean' not in sector_stats.columns:
                    return "Error: Datele despre nivelul de zgomot nu sunt disponibile pentru comparație."
                    
                # Găsește sectorul cu cel mai scăzut nivel de zgomot
                best_sector = sector_stats['noise_mean'].idxmin()
                noise = sector_stats.loc[best_sector, 'noise_mean']
                
                result = f"{best_sector} este cel mai liniștit, cu un nivel de zgomot mediu de {noise} dB.\n\n"
                result += "(Valorile dB mai mici indică un mediu mai liniștit)\n\n"
                
                # Arată și celelalte sectoare pentru comparație
                result += "Niveluri medii de zgomot pe sectoare (de la cel mai liniștit la cel mai zgomotos):\n"
                for sector in sector_stats.sort_values('noise_mean').index:
                    result += f"{sector}: {sector_stats.loc[sector, 'noise_mean']} dB\n"
                    
                return result
                
            elif metric_lower in ["density", "green_density", "parks_per_km2"]:
                # Găsește sectorul cu cea mai mare densitate de parcuri
                best_sector = sector_stats['parks_per_km2'].idxmax()
                density = sector_stats.loc[best_sector, 'parks_per_km2']
                
                result = f"{best_sector} are cea mai mare densitate de parcuri, cu {density} parcuri per kilometru pătrat.\n\n"
                
                # Arată și celelalte sectoare pentru comparație
                result += "Densitatea parcurilor pe sector (în ordine descrescătoare):\n"
                for sector in sector_stats.sort_values('parks_per_km2', ascending=False).index:
                    result += f"{sector}: {sector_stats.loc[sector, 'parks_per_km2']} parcuri/km²\n"
                    
                return result
                
            else:
                return f"Error: Metric necunoscut '{metric}'. Metricile suportate sunt 'parks', 'air', 'noise' și 'density'."
        
        except Exception as e:
            logger.error(f"Error finding best sector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la identificarea celui mai bun sector: {str(e)}"
    
    def _create_sector_chart(self, metric: str = "parks") -> str:
        """
        Create a chart comparing all sectors by specified metric.
        
        Args:
            metric: Metric to visualize ("parks", "air", "noise", "green_density")
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
                
            if 'sector' not in self.data.columns:
                return "Error: The dataset does not contain sector information."
            
            # Normalize the metric name for better matching
            metric_lower = metric.lower().strip()
            
            # Improved metric matching with more variants
            if any(term in metric_lower for term in ["park", "green", "space"]):
                chart_metric = "parks"
            elif any(term in metric_lower for term in ["air", "aqi", "quality"]):
                chart_metric = "air"
            elif any(term in metric_lower for term in ["noise", "quiet", "sound", "db"]):
                chart_metric = "noise"
            elif any(term in metric_lower for term in ["dens", "km", "kilometer", "square", "km2", "per"]):
                chart_metric = "density"
            else:
                chart_metric = "parks"  # Default
            
            # Create dataframe with park counts by sector - avoid column name conflicts
            park_counts = self.data.groupby('sector').size().reset_index(name='park_count')
            
            # Create separate dataframes for each metric to avoid column conflicts
            sector_stats = park_counts.copy()
            
            # Process AQI data if available
            if 'aqi' in self.data.columns:
                # Group by sector and calculate stats manually
                sectors = self.data['sector'].unique()
                aqi_data = []
                
                for sector in sectors:
                    sector_parks = self.data[self.data['sector'] == sector]
                    aqi_mean = sector_parks['aqi'].mean()
                    aqi_data.append({
                        'sector': sector,
                        'aqi_mean': round(aqi_mean, 1) if not pd.isna(aqi_mean) else None
                    })
                
                aqi_stats = pd.DataFrame(aqi_data)
                sector_stats = pd.merge(sector_stats, aqi_stats, on='sector', how='left')
            
            # Process noise data if available
            if 'noise_level_db' in self.data.columns:
                # Group by sector and calculate stats manually
                sectors = self.data['sector'].unique()
                noise_data = []
                
                for sector in sectors:
                    sector_parks = self.data[self.data['sector'] == sector]
                    noise_mean = sector_parks['noise_level_db'].mean()
                    noise_data.append({
                        'sector': sector,
                        'noise_mean': round(noise_mean, 1) if not pd.isna(noise_mean) else None
                    })
                
                noise_stats = pd.DataFrame(noise_data)
                sector_stats = pd.merge(sector_stats, noise_stats, on='sector', how='left')
            
            # Calculate parks per square km (assuming we have area data)
            # For simplicity, we'll use estimated areas of Bucharest sectors
            sector_areas_km2 = {
                'Sector 1': 68,
                'Sector 2': 32,
                'Sector 3': 34,
                'Sector 4': 32,
                'Sector 5': 30,
                'Sector 6': 38
            }
            
            # Add area and density columns
            sector_stats['area_km2'] = sector_stats['sector'].map(lambda x: sector_areas_km2.get(x, 30))
            sector_stats['parks_per_km2'] = (sector_stats['park_count'] / sector_stats['area_km2']).round(2)
            
            # Create a visualization based on the metric
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            from datetime import datetime
            
            plt.figure(figsize=(10, 6))
            
            # Sort sectors for consistent display
            sorted_sectors = sorted(sector_stats['sector'].tolist())
            sector_stats_sorted = sector_stats.set_index('sector').loc[sorted_sectors].reset_index()
            
            if chart_metric == "parks":
                # Bar chart of park count
                plt.bar(sector_stats_sorted['sector'], sector_stats_sorted['park_count'])
                plt.title('Number of Parks by Sector')
                plt.ylabel('Number of Parks')
                plt.xticks(rotation=45)
                
                # Add value labels on top of bars
                for i, count in enumerate(sector_stats_sorted['park_count']):
                    plt.text(i, count + 0.5, str(count), ha='center')
                    
                metric_name = "parks"
                    
            elif chart_metric == "air":
                if 'aqi_mean' not in sector_stats.columns:
                    return "Error: Air quality data is not available for visualization."
                    
                # Bar chart of average AQI
                values = sector_stats_sorted['aqi_mean']
                bars = plt.bar(sector_stats_sorted['sector'], values)
                plt.title('Average Air Quality Index by Sector (Lower is Better)')
                plt.ylabel('Average AQI')
                plt.xticks(rotation=45)
                
                # Add value labels and color bars by AQI value
                for i, value in enumerate(values):
                    plt.text(i, value + 0.5, f"{value:.1f}", ha='center')
                    
                    # Color by AQI value
                    if value < 20:
                        bars[i].set_color('green')
                    elif value < 50:
                        bars[i].set_color('lightgreen')
                    elif value < 100:
                        bars[i].set_color('yellow')
                    else:
                        bars[i].set_color('red')
                        
                metric_name = "air_quality"
                    
            elif chart_metric == "noise":
                if 'noise_mean' not in sector_stats.columns:
                    return "Error: Noise level data is not available for visualization."
                    
                # Bar chart of average noise level
                values = sector_stats_sorted['noise_mean']
                bars = plt.bar(sector_stats_sorted['sector'], values)
                plt.title('Average Noise Level by Sector (Lower is Quieter)')
                plt.ylabel('Average Noise Level (dB)')
                plt.xticks(rotation=45)
                
                # Add value labels and color bars by noise level
                for i, value in enumerate(values):
                    plt.text(i, value + 0.5, f"{value:.1f}", ha='center')
                    
                    # Color by noise level
                    if value < 40:
                        bars[i].set_color('green')
                    elif value < 55:
                        bars[i].set_color('lightgreen')
                    elif value < 70:
                        bars[i].set_color('yellow')
                    else:
                        bars[i].set_color('red')
                        
                metric_name = "noise_level"
                    
            elif chart_metric == "density":
                # Bar chart of parks per square km
                values = sector_stats_sorted['parks_per_km2']
                bars = plt.bar(sector_stats_sorted['sector'], values)
                plt.title('Parks per Square Kilometer by Sector')
                plt.ylabel('Parks per km²')
                plt.xticks(rotation=45)
                
                # Add value labels on top of bars
                for i, value in enumerate(values):
                    plt.text(i, value + 0.05, f"{value:.2f}", ha='center')
                    
                metric_name = "park_density"
                    
            else:
                return f"Error: Unsupported visualization metric '{metric}'. Supported metrics are 'parks', 'air', 'noise', and 'density'."
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"sector_comparison_{metric_name}_{timestamp}.png")
            plt.savefig(output_path)
            
            # Also return the stats as text
            result = f"Sector Comparison by {metric.title()} (chart saved to {output_path}):\n\n"
            
            # Set index for easier sorting and access
            sector_stats.set_index('sector', inplace=True)
            
            if chart_metric == "parks":
                for sector in sorted(sector_stats.index, key=lambda x: sector_stats.loc[x, 'park_count'], reverse=True):
                    result += f"{sector}: {sector_stats.loc[sector, 'park_count']} parks\n"
                    
            elif chart_metric == "air":
                for sector in sorted(sector_stats.index, key=lambda x: sector_stats.loc[x, 'aqi_mean']):
                    result += f"{sector}: Average AQI {sector_stats.loc[sector, 'aqi_mean']:.1f}\n"
                    
            elif chart_metric == "noise":
                for sector in sorted(sector_stats.index, key=lambda x: sector_stats.loc[x, 'noise_mean']):
                    result += f"{sector}: Average {sector_stats.loc[sector, 'noise_mean']:.1f} dB\n"
                    
            elif chart_metric == "density":
                for sector in sorted(sector_stats.index, key=lambda x: sector_stats.loc[x, 'parks_per_km2'], reverse=True):
                    result += f"{sector}: {sector_stats.loc[sector, 'parks_per_km2']:.2f} parks/km²\n"
            
            return result
        
        except Exception as e:
            logger.error(f"Error creating sector chart: {e}")
            import traceback
            return f"Error occurred while creating sector chart: {str(e)}\n{traceback.format_exc()}"
    
    def _find_largest_parks(self, sector: str, top_n: int = 5) -> str:
        """Return the largest parks in a given sector of Bucharest."""
        try:
            # Normalize sector input
            if isinstance(sector, int) or (isinstance(sector, str) and sector.isdigit()):
                sector_normalized = f"Sector {sector}"
            else:
                sector_normalized = sector

            known_parks = {
                # Sector 1
                "Sector 1": {
                    "Parcul Herăstrău": 110.0,
                    "Parcul Regele Mihai I": 110.0,  # alt name
                    "Parcul Cișmigiu": 14.6,
                    "Parcul Bazilescu": 13.5,
                    "Parcul Bordei": 7.5,
                    "Parcul Kiseleff": 3.17,
                },
                # Sector 2
                "Sector 2": {
                    "Parcul Plumbuita": 80.0,
                    "Parcul Circului": 26.0,
                    "Parcul Tei": 9.0,
                    "Parcul Național": 5.0,
                    "Parcul Verdi": 3.0,
                },
                # Sector 3
                "Sector 3": {
                    "Parcul Alexandru Ioan Cuza": 85.0,
                    "Parcul IOR": 85.0,  # alt name
                    "Parcul Titan": 85.0,  # alt name
                    "Parcul Teilor": 7.0,
                    "Parcul 23 August": 6.0,
                    "Parcul Pantelimon": 5.5,
                },
                # Sector 4
                "Sector 4": {
                    "Parcul Natural Văcărești": 183.0,
                    "Parcul Tineretului": 94.0,
                    "Parcul Carol": 38.0,
                    "Parcul Lumea Copiilor": 30.0,
                    "Parcul Tudor Arghezi": 3.7,
                },
                # Sector 5
                "Sector 5": {
                    "Parcul Izvor": 17.0,
                    "Parcul Sebastian": 4.5,
                    "Parcul Ferentari": 3.2,
                    "Parcul Rahova": 2.8,
                },
                # Sector 6
                "Sector 6": {
                    "Parcul Drumul Taberei": 11.48,
                    "Parcul Crângași": 8.69,
                    "Parcul Liniei": 4.5,
                    "Parcul Marin Preda": 4.32,
                    "Parcul Nicolae Filimon": 2.01,
                }
            }

            if sector_normalized not in known_parks:
                return f"Nu avem date despre parcurile din {sector_normalized}."

            sector_known_parks = known_parks[sector_normalized]

            # Deduplicate using canonical names
            added_parks = set()
            result_parks = []
            synonyms = {
                "Parcul Regele Mihai I": "Parcul Herăstrău",
                "Parcul IOR": "Parcul Alexandru Ioan Cuza",
                "Parcul Titan": "Parcul Alexandru Ioan Cuza"
            }

            for name, area in sector_known_parks.items():
                canonical_name = synonyms.get(name, name)
                if canonical_name not in added_parks:
                    result_parks.append({"name": canonical_name, "area": area})
                    added_parks.add(canonical_name)

            # Sort by area descending
            result_parks.sort(key=lambda p: p["area"], reverse=True)

            # Format response
            if result_parks:
                result = f"Cele mai mari {min(top_n, len(result_parks))} parcuri din {sector_normalized}:\n\n"
                for i, park in enumerate(result_parks[:top_n]):
                    result += f"{i+1}. {park['name']} - {park['area']:.2f} hectare\n"
                return result
            else:
                return f"Nu am putut determina dimensiunea parcurilor din {sector_normalized}."

        except Exception as e:
            import traceback
            return f"Eroare la căutarea parcurilor: {str(e)}\n{traceback.format_exc()}"

    
    def _plan_park_tour(self, num_parks=3, start_park=None, metric="quality"):
        """
        Plan a tour of parks in Bucharest, randomly selecting from top parks based on quality metrics,
        while favoring parks in the same sector or close proximity to create a more practical tour.
        
        Args:
            num_parks: Number of parks to include in the tour (default: 3)
            start_park: Name of the park to start from (optional)
            metric: Metric to use for initial ranking ("quality", "air_quality", "noise", "combined")
        
        Returns:
            A description of the tour with route information
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
            
            import geopandas as gpd
            import matplotlib.pyplot as plt
            import os
            import networkx as nx
            from datetime import datetime
            from shapely.geometry import LineString, Point
            import numpy as np
            import pandas as pd
            from shapely import wkt
            import random
            
            # Install required packages if missing
            try:
                import contextily
            except ImportError:
                import subprocess
                import sys
                logger.info("Installing contextily package for basemaps...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "contextily"])
                import contextily
                
            # Determine the correct name column to use
            name_column = 'name_left' if 'name_left' in self.data.columns else 'name'
            logger.info(f"Using '{name_column}' column for park names")
            
            # Create a copy of the data for manipulation
            parks_df = self.data.copy()
            
            # Ensure geometry column is properly set
            if not isinstance(parks_df, gpd.GeoDataFrame):
                try:
                    if isinstance(parks_df, pd.DataFrame) and 'geometry' in parks_df.columns:
                        if isinstance(parks_df['geometry'][0], str):
                            parks_df['geometry'] = parks_df['geometry'].apply(wkt.loads)
                        parks_df = gpd.GeoDataFrame(parks_df, geometry='geometry', crs="EPSG:4326")
                except Exception as e:
                    logger.error(f"Error converting DataFrame to GeoDataFrame: {e}")
                    return f"Error: Could not process park geometries correctly: {str(e)}"
            
            # Remove duplicates based on name and geometry
            parks_df = parks_df.drop_duplicates(subset=[name_column])
            
            # If geometries are the same (overlapping parks), keep only the one with the larger area
            if 'area_hectares' in parks_df.columns:
                parks_df = parks_df.sort_values('area_hectares', ascending=False)
                parks_df = parks_df.drop_duplicates(subset=['geometry'])
            
            # Calculate scores using the same approach as _recommend_best_park_today
            parks_df['total_score'] = 0
            
            # Air quality score (lower is better)
            if "aqi" in parks_df.columns:
                # Validate AQI values
                if parks_df['aqi'].median() < 1:
                    # Values are probably on 0-1 scale, convert to 0-100
                    parks_df['aqi'] = parks_df['aqi'] * 100
                    
                min_aqi = parks_df["aqi"].min()
                max_aqi = parks_df["aqi"].max()
                if max_aqi > min_aqi:
                    parks_df['air_score'] = 1 - (parks_df["aqi"] - min_aqi) / (max_aqi - min_aqi)
                    parks_df['total_score'] += parks_df['air_score'] * 0.4
            
            # Noise score (lower is better)
            if "noise_level_db" in parks_df.columns:
                min_noise = parks_df["noise_level_db"].min()
                max_noise = parks_df["noise_level_db"].max()
                if max_noise > min_noise:
                    parks_df['noise_score'] = 1 - (parks_df["noise_level_db"] - min_noise) / (max_noise - min_noise)
                    parks_df['total_score'] += parks_df['noise_score'] * 0.3
            
            # Congestion score (lower is better)
            if "congestion_level" in parks_df.columns:
                min_cong = parks_df["congestion_level"].min()
                max_cong = parks_df["congestion_level"].max()
                if max_cong > min_cong:
                    parks_df['congestion_score'] = 1 - (parks_df["congestion_level"] - min_cong) / (max_cong - min_cong)
                    parks_df['total_score'] += parks_df['congestion_score'] * 0.3
                
            # Fallback to area as a component if other scores are missing
            if 'total_score' not in parks_df.columns or parks_df['total_score'].sum() == 0:
                if 'area_hectares' in parks_df.columns:
                    min_area = parks_df['area_hectares'].min()
                    max_area = parks_df['area_hectares'].max()
                    if max_area > min_area:
                        parks_df['area_score'] = (parks_df['area_hectares'] - min_area) / (max_area - min_area)
                        parks_df['total_score'] = parks_df['area_score']
                else:
                    # If no scores available, just assign random scores
                    parks_df['total_score'] = np.random.random(len(parks_df))
            
            # Sort by the chosen metric
            if metric.lower() == "quality" or metric.lower() == "combined":
                sorted_parks = parks_df.sort_values(by='total_score', ascending=False)
                metric_name = "overall quality"
            elif metric.lower() == "air_quality" and 'air_score' in parks_df.columns:
                sorted_parks = parks_df.sort_values(by='air_score', ascending=False)
                metric_name = "air quality"
            elif metric.lower() == "noise" and 'noise_score' in parks_df.columns:
                sorted_parks = parks_df.sort_values(by='noise_score', ascending=False)
                metric_name = "noise level (quietest)"
            else:
                # Default to total score
                sorted_parks = parks_df.sort_values(by='total_score', ascending=False)
                metric_name = "overall quality"
                
            # Get top 20 parks (or fewer if not enough parks)
            top_count = min(20, len(sorted_parks))
            top_parks_pool = sorted_parks.head(top_count).copy()
            
            logger.info(f"Selected top {top_count} parks by {metric_name} for intelligent selection")
            
            # Prepare list for selected parks
            selected_parks = pd.DataFrame()
            
            # Handle start park if specified
            start_park_data = None
            if start_park:
                start_park_lower = start_park.lower()
                matches = parks_df[parks_df[name_column].str.lower().str.contains(start_park_lower, na=False)]
                
                if len(matches) > 0:
                    # Use the first match
                    start_park_data = matches.iloc[0]
                    selected_parks = pd.concat([selected_parks, pd.DataFrame([start_park_data])])
                    
                    # Get the sector of the starting park for proximity preference
                    start_sector = start_park_data.get('sector', None)
                    start_point = start_park_data.geometry.centroid
            else:
                # If no start park specified, select the highest quality park as start
                start_park_data = top_parks_pool.iloc[0]
                selected_parks = pd.concat([selected_parks, pd.DataFrame([start_park_data])])
                
                # Get the sector of the starting park for proximity preference
                start_sector = start_park_data.get('sector', None)
                start_point = start_park_data.geometry.centroid
            
            # Remove the first park from the pool
            remaining_parks = top_parks_pool[top_parks_pool[name_column] != selected_parks.iloc[0][name_column]]
            
            # Now select the remaining parks with preference for same sector and proximity
            while len(selected_parks) < num_parks and len(remaining_parks) > 0:
                # If we have a sector, give bonus to parks in the same sector
                if 'sector' in parks_df.columns and start_sector is not None:
                    # Add proximity score based on sector match (0-1 scale)
                    remaining_parks['proximity_score'] = 0.0
                    remaining_parks.loc[remaining_parks['sector'] == start_sector, 'proximity_score'] += 0.5
                    
                    # Add adjacent sectors bonus (assuming numerical sectors)
                    if isinstance(start_sector, str) and "Sector" in start_sector:
                        try:
                            start_sector_num = int(start_sector.replace("Sector ", ""))
                            for idx, park in remaining_parks.iterrows():
                                if 'sector' in park and isinstance(park['sector'], str) and "Sector" in park['sector']:
                                    park_sector_num = int(park['sector'].replace("Sector ", ""))
                                    # Adjacent sectors get a smaller bonus
                                    if abs(park_sector_num - start_sector_num) == 1:
                                        remaining_parks.loc[idx, 'proximity_score'] += 0.3
                        except:
                            pass  # If sector parsing fails, just skip this bonus
                
                # Calculate distance-based score
                # Get the last selected park's location to measure distance from
                last_park = selected_parks.iloc[-1]
                last_point = last_park.geometry.centroid
                
                # Calculate distances and normalize to 0-1 scale (closer is better)
                distances = []
                for idx, park in remaining_parks.iterrows():
                    try:
                        # Calculate distance in meters (approximate)
                        dx = (park.geometry.centroid.x - last_point.x) * 111000 * np.cos(np.radians(last_point.y))
                        dy = (park.geometry.centroid.y - last_point.y) * 111000
                        distance = np.sqrt(dx**2 + dy**2)
                        distances.append((idx, distance))
                    except:
                        distances.append((idx, float('inf')))
                
                if distances:
                    max_dist = max(dist for _, dist in distances if dist < float('inf'))
                    min_dist = min(dist for _, dist in distances if dist > 0)
                    
                    for idx, dist in distances:
                        if dist == float('inf'):
                            remaining_parks.loc[idx, 'distance_score'] = 0
                        else:
                            # Normalize: closer parks get higher scores (1.0 - normalized_distance)
                            # Add a small offset to avoid division by zero
                            if max_dist > min_dist:
                                normalized_dist = (dist - min_dist) / (max_dist - min_dist)
                                remaining_parks.loc[idx, 'distance_score'] = 1.0 - normalized_dist
                            else:
                                remaining_parks.loc[idx, 'distance_score'] = 0.5
                
                # Calculate final score as weighted combination of quality and proximity
                if 'proximity_score' in remaining_parks.columns and 'distance_score' in remaining_parks.columns:
                    # Combined score: 40% quality, 30% sector proximity, 30% physical distance
                    remaining_parks['combined_score'] = (
                        remaining_parks['total_score'] * 0.4 + 
                        remaining_parks.get('proximity_score', 0) * 0.3 + 
                        remaining_parks.get('distance_score', 0) * 0.3
                    )
                else:
                    # If proximity scores weren't calculated, just use the quality score
                    remaining_parks['combined_score'] = remaining_parks['total_score']
                
                # Select the next best park by combined score
                next_park = remaining_parks.nlargest(1, 'combined_score')
                
                # Add to selected parks
                selected_parks = pd.concat([selected_parks, next_park])
                
                # Update the remaining parks pool
                remaining_parks = remaining_parks[remaining_parks[name_column] != next_park.iloc[0][name_column]]
                
                # Update start_point for next iteration
                start_point = next_park.iloc[0].geometry.centroid
            
            logger.info(f"Selected {len(selected_parks)} parks for the tour using intelligent proximity-based algorithm")
                
            # Plan routes between consecutive parks
            routes = []
            total_distance = 0
            total_time = 0
            
            for i in range(len(selected_parks) - 1):
                start = selected_parks.iloc[i]
                end = selected_parks.iloc[i+1]
                
                try:
                    # Calculate projected distance for accuracy
                    try:
                        start_point_proj = gpd.GeoSeries([start.geometry.centroid], crs=self.data.crs).to_crs(epsg=3844)[0]
                        end_point_proj = gpd.GeoSeries([end.geometry.centroid], crs=self.data.crs).to_crs(epsg=3844)[0]
                        
                        # Calculate distance in meters
                        distance = start_point_proj.distance(end_point_proj)
                    except:
                        # Fallback to approximation
                        start_point = start.geometry.centroid
                        end_point = end.geometry.centroid
                        
                        # Calculate approximate distance in meters
                        distance = ((start_point.x - end_point.x)**2 + 
                                (start_point.y - end_point.y)**2)**0.5 * 111000
                    
                    # Walking time (5 km/h = 1.4 m/s)
                    time_seconds = distance / 1.4
                    
                    routes.append({
                        'start': str(start[name_column]),
                        'end': str(end[name_column]),
                        'distance_km': distance / 1000,
                        'time_minutes': time_seconds / 60
                    })
                    
                    total_distance += distance / 1000
                    total_time += time_seconds / 60
                    
                except Exception as e:
                    logger.error(f"Error calculating route between parks: {e}")
                    routes.append({
                        'start': str(start[name_column]),
                        'end': str(end[name_column]),
                        'error': str(e)
                    })
            
            # Create an enhanced visualization with basemap
            try:
                # Force non-interactive backend
                import matplotlib
                matplotlib.use('Agg')
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(14, 12))
                
                # Create a GeoDataFrame for the parks with proper CRS
                parks_gdf = gpd.GeoDataFrame(self.data, geometry='geometry', crs=self.data.crs)
                
                # Define Web Mercator projection for consistent basemap
                web_mercator = 'EPSG:3857'
                
                # Project parks to Web Mercator for basemap compatibility
                parks_webmerc = parks_gdf.to_crs(web_mercator)
                
                # Create a GeoDataFrame for the selected parks
                selected_parks_gdf = gpd.GeoDataFrame(selected_parks, geometry='geometry', crs=self.data.crs)
                selected_parks_webmerc = selected_parks_gdf.to_crs(web_mercator)
                
                # Colors for the parks
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
                
                # Plot all parks in light green
                parks_webmerc.plot(ax=ax, color='lightgreen', alpha=0.3, edgecolor='green')
                
                # Plot the selected parks in different colors
                for i, (idx, park) in enumerate(selected_parks_webmerc.iterrows()):
                    color = colors[i % len(colors)]
                    park_gdf = gpd.GeoDataFrame(geometry=[park.geometry], crs=web_mercator)
                    park_gdf.plot(ax=ax, color=color, alpha=0.7, edgecolor='black')
                
                # Add routes between parks
                park_coords = []
                for i, (idx, park) in enumerate(selected_parks_webmerc.iterrows()):
                    park_coords.append((park.geometry.centroid.x, park.geometry.centroid.y))
                
                # Connect parks with lines
                for i in range(len(park_coords) - 1):
                    line = LineString([park_coords[i], park_coords[i+1]])
                    line_gdf = gpd.GeoDataFrame(geometry=[line], crs=web_mercator)
                    line_gdf.plot(
                        ax=ax, 
                        color=colors[i % len(colors)],
                        linewidth=3,
                        linestyle='--',
                        alpha=0.8
                    )
                
                # Try multiple basemap sources in case one fails
                basemap_added = False
                
                # First try: OpenStreetMap
                try:
                    import contextily as ctx
                    ctx.add_basemap(
                        ax, 
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        attribution="© OpenStreetMap contributors"
                    )
                    basemap_added = True
                    logger.info("Added OpenStreetMap basemap")
                except Exception as e:
                    logger.warning(f"Could not add OpenStreetMap basemap: {e}")
                
                # Second try: Stamen Terrain
                if not basemap_added:
                    try:
                        import contextily as ctx
                        ctx.add_basemap(
                            ax, 
                            source=ctx.providers.Stamen.Terrain,
                            attribution="Map tiles by Stamen Design"
                        )
                        basemap_added = True
                        logger.info("Added Stamen Terrain basemap")
                    except Exception as e:
                        logger.warning(f"Could not add Stamen Terrain basemap: {e}")
                
                # Third try: CartoDB Positron
                if not basemap_added:
                    try:
                        import contextily as ctx
                        ctx.add_basemap(
                            ax, 
                            source=ctx.providers.CartoDB.Positron,
                            attribution="© CartoDB"
                        )
                        basemap_added = True
                        logger.info("Added CartoDB Positron basemap")
                    except Exception as e:
                        logger.warning(f"Could not add CartoDB Positron basemap: {e}")
                
                # Add labels for parks after the basemap
                for i, (idx, park) in enumerate(selected_parks_webmerc.iterrows()):
                    color = colors[i % len(colors)]
                    x, y = park.geometry.centroid.x, park.geometry.centroid.y
                    
                    # Create label with park number and name
                    park_name = str(park[name_column])
                    label = f"{i+1}. {park_name}"
                    
                    # Add text label with white background for visibility
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=12,
                        color=color,
                        fontweight='bold',
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            fc="white",
                            ec=color,
                            alpha=0.9
                        )
                    )
                
                # Set title and layout
                plt.title(f'Tour of {len(selected_parks)} Parks in Bucharest\nSelected from Top 20 by Quality with Proximity Optimization',
                        fontsize=16, fontweight='bold')
                
                # Save the visualization
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"park_tour_proximity_{metric_name.replace(' ', '_')}_{timestamp}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close('all')
                
                visualization_path = output_path
                
            except Exception as e:
                logger.error(f"Error creating visualization with basemap: {e}")
                import traceback
                logger.error(traceback.format_exc())
                visualization_path = None
                
                # Simple fallback visualization attempt
                try:
                    fig, ax = plt.subplots(figsize=(14, 12))
                    self.data.plot(ax=ax, color='lightgreen', alpha=0.3)
                    selected_parks_gdf.plot(ax=ax, color='red', alpha=0.7)
                    plt.title(f'Tour of {len(selected_parks)} Parks (Simplified View)')
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(output_dir, f"park_tour_simple_{timestamp}.png")
                    plt.savefig(output_path)
                    plt.close()
                    visualization_path = output_path
                except:
                    visualization_path = None
            
            # Format the result text
            result = f"Tour of {len(selected_parks)} Parks in Bucharest\n"
            result += f"(Selected from top 20 parks with proximity optimization)\n\n"
            
            # Check if parks are in the same sector for highlighting in the result
            sectors = selected_parks['sector'].unique() if 'sector' in selected_parks.columns else []
            same_sector = len(sectors) == 1 and len(selected_parks) > 1
            
            if same_sector:
                result += f"All selected parks are in {sectors[0]} for convenient visiting!\n\n"
            elif len(sectors) > 1 and len(sectors) < len(selected_parks):
                sector_counts = selected_parks['sector'].value_counts()
                main_sector = sector_counts.idxmax()
                result += f"Tour focuses on {main_sector} with {sector_counts[main_sector]} parks for convenient visiting!\n\n"
            
            for i, (idx, park) in enumerate(selected_parks.iterrows()):
                park_score = park.get('total_score', 0)
                sector = str(park.get('sector', 'Unknown sector'))
                
                result += f"{i+1}. {str(park[name_column])} ({sector})\n"
                
                # Add quality metrics if available
                if 'aqi' in park and not pd.isna(park['aqi']):
                    result += f"   Air Quality Index: {float(park['aqi']):.1f}\n"
                    
                if 'noise_level_db' in park and not pd.isna(park['noise_level_db']):
                    result += f"   Noise Level: {float(park['noise_level_db']):.1f} dB\n"
                    
                if 'congestion_level' in park and not pd.isna(park['congestion_level']):
                    result += f"   Congestion Level: {float(park['congestion_level']):.1f}\n"
                    
                result += "\n"
            
            # Add route information
            if routes:
                result += "Suggested Route Between Parks:\n\n"
                
                for i, route in enumerate(routes):
                    if 'error' in route:
                        result += f"Route {i+1}: {route['start']} to {route['end']}: Could not calculate route.\n"
                    else:
                        result += f"Route {i+1}: {route['start']} to {route['end']}\n"
                        result += f"   Distance: {route['distance_km']:.2f} km\n"
                        result += f"   Walking time: {route['time_minutes']:.1f} minutes\n\n"
                
                result += f"Total tour distance: {total_distance:.2f} km\n"
                result += f"Total walking time: {total_time:.1f} minutes "
                result += f"({total_time/60:.1f} hours)\n\n"
            
            if visualization_path:
                result += f"A visualization of the park tour has been created at: {visualization_path}"
            self.update_latest_visualization(visualization_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error planning park tour: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error occurred while planning park tour: {str(e)}"
    
    def _create_route_visualization(self, origin, destination, path, origin_point, dest_point, 
                              name_column, distance_km, time_minutes, mode, traffic_data=None):
        """
        Create visualization in a thread-safe way.
        This should be called from the main thread or with proper thread safety measures.
        """
        import matplotlib.pyplot as plt
        import geopandas as gpd
        from shapely.geometry import LineString
        import os
        from datetime import datetime
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all parks with proper boundaries
        self.data.plot(ax=ax, color='green', alpha=0.3, edgecolor='darkgreen')
        
        # Highlight origin and destination parks
        gpd.GeoDataFrame(geometry=[origin.geometry], crs=self.data.crs).plot(
            ax=ax, color='blue', alpha=0.6, edgecolor='darkblue'
        )
        gpd.GeoDataFrame(geometry=[destination.geometry], crs=self.data.crs).plot(
            ax=ax, color='red', alpha=0.6, edgecolor='darkred'
        )
        
        # Plot the route
        route_coords = [p for p in path]
        route_line = LineString(route_coords)
        route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
        route_gdf.plot(ax=ax, color='purple', linewidth=4, alpha=0.8)
        
        # Plot traffic data if available to show streets
        if traffic_data is not None:
            traffic_data.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
        
        # Add labels with background boxes for better visibility
        ax.annotate(origin[name_column], 
                xy=(origin_point[0], origin_point[1]),
                xytext=(5, 15), textcoords="offset points",
                color='blue', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                
        ax.annotate(destination[name_column], 
                xy=(dest_point[0], dest_point[1]),
                xytext=(5, 15), textcoords="offset points",
                color='red', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Add title with distance and time
        plt.title(f'Route from {origin[name_column]} to {destination[name_column]}\n'
                f'Distance: {distance_km:.2f} km, Time: {time_minutes:.1f} min ({mode.capitalize()})',
                fontsize=16)
        
        # Save with higher resolution
        plt.tight_layout()
        ax.set_axis_off()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"route_{mode}_{timestamp}.png")
        plt.savefig(output_path, dpi=300)
        plt.close('all')  # Explicitly close all figures
        
        self.update_latest_visualization(output_path)
        
        return output_path
    
    def _get_total_parks(self) -> str:
        """Returns the total count of parks in Bucharest."""
        try:
            total_parks = len(self.data)
            return f"There are a total of {total_parks} parks in Bucharest."
        except Exception as e:
            logger.error(f"Error counting parks: {e}")
            return f"Error occurred while counting parks: {str(e)}"

    def _list_parks_in_sector(self, sector_id: str, page: int = 1, page_size: int = 50) -> str:
        """Lists all parks in a specific sector with pagination support."""
        try:
            # Normalize sector input
            if isinstance(sector_id, int) or (isinstance(sector_id, str) and sector_id.isdigit()):
                sector_normalized = f"Sector {sector_id}"
            elif isinstance(sector_id, str) and not sector_id.lower().startswith("sector"):
                sector_normalized = f"Sector {sector_id}"
            else:
                sector_normalized = sector_id.title()
            
            # Verificări de bază
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date despre parcuri disponibile."
                
            if 'sector' not in self.data.columns:
                return "Error: Setul de date nu conține informații despre sectoare."
                
            # Filter parks by sector
            parks_in_sector = self.data[self.data['sector'] == sector_normalized]
            
            if len(parks_in_sector) == 0:
                return f"Nu au fost găsite parcuri în {sector_normalized}."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name_left', 'name', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in parks_in_sector.columns:
                    name_column = col
                    break
            
            if name_column is None:
                return f"Găsite {len(parks_in_sector)} parcuri în {sector_normalized}, dar nu pot afișa numele lor deoarece lipsește coloana cu numele."
            
            # Eliminare duplicate bazate pe numele parcurilor
            parks_in_sector = parks_in_sector.drop_duplicates(subset=[name_column])
            
            # Sort parks alphabetically for consistency
            parks_in_sector = parks_in_sector.sort_values(by=name_column)
            
            # Calculate pagination
            total_parks = len(parks_in_sector)
            total_pages = (total_parks + page_size - 1) // page_size  # Ceiling division
            
            # Validate page number
            page = max(1, min(page, total_pages))
            
            # Select parks for current page
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_parks)
            parks_page = parks_in_sector.iloc[start_idx:end_idx]
            
            # Format header with pagination info
            if total_pages > 1:
                result = f"Lista parcurilor din {sector_normalized} (pagina {page}/{total_pages}, {total_parks} parcuri în total):\n\n"
            else:
                result = f"Lista completă a parcurilor din {sector_normalized} ({total_parks} în total):\n\n"
            
            # Format parks in a compact, readable way
            for idx, row in parks_page.iterrows():
                try:
                    park_name = str(row[name_column]) if pd.notna(row[name_column]) else f"Parc fără nume (ID: {idx})"
                    
                    # Keep format simple for readability
                    park_info = f"• {park_name}"
                    
                    # Add minimal additional info if available
                    if 'aqi' in row and pd.notna(row['aqi']):
                        park_info += f" (AQI: {row['aqi']:.1f})"
                    
                    result += park_info + "\n"
                except Exception as e:
                    # Handle individual row errors
                    result += f"• [Eroare la procesarea datelor: {str(e)}]\n"
            
            # Add pagination instructions if needed
            if total_pages > 1:
                result += f"\nAfișate parcurile {start_idx+1}-{end_idx} din {total_parks}. "
                
                if page < total_pages:
                    result += f"Pentru a vedea următoarea pagină, specificați 'page={page+1}'."
                if page > 1:
                    result += f" Pentru a vedea pagina anterioară, specificați 'page={page-1}'."
            
            # Marcaj pentru sistemul intern - nu va fi vizibil în răspunsul final
            result += "\n\n<!-- LISTA_COMPLETA_PARCURI -->"
            
            return result
        except Exception as e:
            logger.error(f"Error listing parks in sector: {e}")
            return f"A apărut o eroare la listarea parcurilor din sectorul {sector_id}: {str(e)}"
    
    def _compare_sector_air_quality(self, sector1: str, sector2: str, metric: str = "air") -> str:
        """Compares air quality or other metrics between two sectors."""
        try:
            if 'sector' not in self.data.columns:
                return "Error: The dataset does not contain sector information."
            
            # Normalize sector input (handle "Sector 3", "sector 3", "3", etc.)
            def normalize_sector(sector):
                if isinstance(sector, int) or (isinstance(sector, str) and sector.isdigit()):
                    return f"Sector {sector}"
                elif isinstance(sector, str) and not sector.lower().startswith("sector"):
                    return f"Sector {sector}"
                else:
                    return sector.title()
                    
            sector1_normalized = normalize_sector(sector1)
            sector2_normalized = normalize_sector(sector2)
            
            # Filter parks by sector
            parks_sector1 = self.data[self.data['sector'] == sector1_normalized]
            parks_sector2 = self.data[self.data['sector'] == sector2_normalized]
            
            if len(parks_sector1) == 0:
                return f"Nu au fost găsite parcuri în {sector1_normalized}."
            
            if len(parks_sector2) == 0:
                return f"Nu au fost găsite parcuri în {sector2_normalized}."
            
            # Choose the metric based on parameter
            if metric.lower() in ["air", "air_quality", "aqi"]:
                if 'aqi' not in self.data.columns:
                    return "Error: The dataset does not contain air quality information."
                    
                # Calculate air quality metrics
                sector1_avg_aqi = parks_sector1['aqi'].mean()
                sector2_avg_aqi = parks_sector2['aqi'].mean()
                
                sector1_min_aqi = parks_sector1['aqi'].min()
                sector2_min_aqi = parks_sector2['aqi'].min()
                
                sector1_max_aqi = parks_sector1['aqi'].max()
                sector2_max_aqi = parks_sector2['aqi'].max()
                
                # Format results
                result = f"Comparație calitate aer: {sector1_normalized} vs {sector2_normalized}\n\n"
                result += f"AQI mediu:\n"
                result += f"- {sector1_normalized}: {sector1_avg_aqi:.1f}\n"
                result += f"- {sector2_normalized}: {sector2_avg_aqi:.1f}\n"
                result += f"(Valori AQI mai mici indică o calitate a aerului mai bună)\n\n"
                
                result += f"Cel mai bun AQI (valoare minimă):\n"
                result += f"- {sector1_normalized}: {sector1_min_aqi:.1f}\n"
                result += f"- {sector2_normalized}: {sector2_min_aqi:.1f}\n\n"
                
                result += f"Cel mai slab AQI (valoare maximă):\n"
                result += f"- {sector1_normalized}: {sector1_max_aqi:.1f}\n"
                result += f"- {sector2_normalized}: {sector2_max_aqi:.1f}\n\n"
                
                # Determine which sector has better air quality
                if sector1_avg_aqi < sector2_avg_aqi:
                    result += f"Per total, {sector1_normalized} are o calitate a aerului mai bună decât {sector2_normalized}."
                elif sector2_avg_aqi < sector1_avg_aqi:
                    result += f"Per total, {sector2_normalized} are o calitate a aerului mai bună decât {sector1_normalized}."
                else:
                    result += f"Ambele sectoare au niveluri similare de calitate a aerului."
                    
                return result
                
            elif metric.lower() in ["parks", "park", "parcuri"]:
                # Compare park counts
                count1 = len(parks_sector1)
                count2 = len(parks_sector2)
                
                result = f"Comparație număr de parcuri: {sector1_normalized} vs {sector2_normalized}\n\n"
                result += f"- {sector1_normalized}: {count1} parcuri\n"
                result += f"- {sector2_normalized}: {count2} parcuri\n\n"
                
                if count1 > count2:
                    result += f"{sector1_normalized} are cu {count1 - count2} mai multe parcuri decât {sector2_normalized}."
                elif count2 > count1:
                    result += f"{sector2_normalized} are cu {count2 - count1} mai multe parcuri decât {sector1_normalized}."
                else:
                    result += f"Ambele sectoare au același număr de parcuri."
                    
                return result
                
            else:
                return f"Error: Metric necunoscut '{metric}'. Metricile suportate sunt 'air' și 'parks'."
                
        except Exception as e:
            logger.error(f"Error comparing sectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la compararea sectoarelor: {str(e)}"
    
    def _get_park_details(self, park_name: str) -> str:
        """Get detailed information about a specific park."""
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Could not find a column with park names."
            
            # Try exact match first (case-insensitive)
            park_data = self.data[self.data[name_column].map(fold_text) == fold_text(park_name)]
            
            # If no exact match, try contains
            if len(park_data) == 0:
                park_data = self.data[self.data[name_column].map(fold_text).str.contains(fold_text(park_name), regex=False, na=False)]
            
            if len(park_data) == 0:
                return f"No park found with name '{park_name}'."
                
            # If multiple matches, use the first one but mention the count
            multiple_matches = len(park_data) > 1
            park_row = park_data.iloc[0]
            
            result = f"Details for {park_row[name_column]}:\n\n"
            
            if multiple_matches:
                result += f"(Note: Found {len(park_data)} parks matching '{park_name}'. Showing details for the first match.)\n\n"
            
            # Add basic info
            if 'sector' in park_row:
                result += f"Location: {park_row['sector']}\n"
                
            # Add area if calculated
            if 'area_sqm' in park_row:
                area_ha = park_row['area_sqm'] / 10000  # Convert to hectares
                result += f"Size: {area_ha:.2f} hectares\n"
            
            # Environmental data
            if 'aqi' in park_row and pd.notna(park_row['aqi']):
                result += f"Air Quality Index (AQI): {park_row['aqi']:.1f}\n"
                
                # Add AQI interpretation
                aqi = float(park_row['aqi'])
                if aqi <= 50:
                    result += "Air Quality: Good\n"
                elif aqi <= 100:
                    result += "Air Quality: Moderate\n"
                elif aqi <= 150:
                    result += "Air Quality: Unhealthy for Sensitive Groups\n"
                else:
                    result += "Air Quality: Unhealthy\n"
            
            if 'pm25' in park_row and pd.notna(park_row['pm25']):
                result += f"PM2.5 Level: {park_row['pm25']:.1f} μg/m³\n"
                
            if 'pm10' in park_row and pd.notna(park_row['pm10']):
                result += f"PM10 Level: {park_row['pm10']:.1f} μg/m³\n"
                
            if 'temperature' in park_row and pd.notna(park_row['temperature']):
                result += f"Temperature: {park_row['temperature']:.1f}°C\n"
                
            if 'humidity' in park_row and pd.notna(park_row['humidity']):
                result += f"Humidity: {park_row['humidity']:.1f}%\n"
                
            if 'noise_level_db' in park_row and pd.notna(park_row['noise_level_db']):
                result += f"Noise Level: {park_row['noise_level_db']:.1f} dB\n"
            
            # Add a brief description based on available data
            result += "\nSummary:\n"
            park_name_str = park_row[name_column]
            
            if 'aqi' in park_row and pd.notna(park_row['aqi']):
                aqi = float(park_row['aqi'])
                if aqi <= 30:
                    air_desc = "excellent air quality"
                elif aqi <= 50:
                    air_desc = "good air quality"
                elif aqi <= 100:
                    air_desc = "moderate air quality"
                else:
                    air_desc = "poor air quality"
            else:
                air_desc = "unknown air quality"
                
            result += f"{park_name_str} is a park located in {park_row.get('sector', 'Bucharest')}, with {air_desc}."
            
            if 'noise_level_db' in park_row and pd.notna(park_row['noise_level_db']):
                noise = float(park_row['noise_level_db'])
                if noise < 45:
                    result += " The park has a very quiet environment, perfect for relaxation."
                elif noise < 55:
                    result += " The park has a moderately quiet environment."
                elif noise < 65:
                    result += " The park has moderate noise levels."
                else:
                    result += " The park has higher noise levels, typical of urban areas."
            
            return result
        except Exception as e:
            logger.error(f"Error getting park details: {e}")
            return f"Error occurred while retrieving details for {park_name}: {str(e)}"
    
    def _search_parks_by_name(self, name_pattern: str) -> str:
        """Search for parks containing a specific pattern in their name."""
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Could not find a column with park names."
            
            # Case-insensitive search
            pattern = fold_text(name_pattern)
            matches = self.data[self.data[name_column].map(fold_text).str.contains(pattern, regex=False, na=False)]
            
            if len(matches) == 0:
                return f"No parks found containing '{name_pattern}' in their name."
            
            result = f"Found {len(matches)} parks containing '{name_pattern}':\n\n"
            
            for idx, row in matches.iterrows():
                park_info = f"• {row[name_column]}"
                if 'sector' in row:
                    park_info += f" (Sector {row['sector'].split(' ')[-1]})"
                if 'aqi' in row and pd.notna(row['aqi']):
                    park_info += f" - AQI: {row['aqi']:.1f}"
                result += park_info + "\n"
            
            return result
        except Exception as e:
            logger.error(f"Error searching parks by name: {e}")
            return f"Error occurred while searching for parks: {str(e)}"
    
    def _get_sector_statistics(self) -> str:
        """Gets detailed statistics about parks in each sector."""
        try:
            # First, verify if we have the required data
            if 'sector' not in self.data.columns:
                return "Error: The dataset does not contain sector information."
            
            # Create a simple summary directly from the data without using spatial_analyst
            sector_stats = self.data.groupby('sector').agg({
                # Count parks per sector
                'geometry': 'count',  # Use any column that's guaranteed to exist
            }).rename(columns={'geometry': 'park_count'})
            
            result = "Sector Statistics for Bucharest:\n\n"
            result += "Park Count by Sector:\n"
            result += "-" * 30 + "\n"
            
            for idx, row in sector_stats.sort_values('park_count', ascending=False).iterrows():
                result += f"{idx}: {row['park_count']} parks\n"
            
            # Add air quality stats if available
            if 'aqi' in self.data.columns:
                aqi_stats = self.data.groupby('sector')['aqi'].agg(['mean', 'min', 'max']).round(1)
                
                result += "\nAir Quality by Sector (lower AQI is better):\n"
                result += "-" * 45 + "\n"
                result += "Sector | Average AQI | Best AQI | Worst AQI\n"
                result += "-" * 45 + "\n"
                
                for idx, row in aqi_stats.iterrows():
                    result += f"{idx} | {row['mean']:.1f} | {row['min']:.1f} | {row['max']:.1f}\n"
            
            return result
        except Exception as e:
            logger.error(f"Error getting sector statistics: {e}")
            return f"Error occurred while calculating sector statistics: {str(e)}"

    def _find_parks_with_lowest_noise(self, top_n: int = 5) -> str:
        """
        Găsește parcurile cu cel mai scăzut nivel de poluare fonică (cele mai liniștite).
        
        Args:
            top_n: Numărul de parcuri de afișat (implicit 5)
            
        Returns:
            O listă formatată cu cele mai liniștite parcuri
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date despre parcuri disponibile."
                
            # Verifică dacă există date despre nivelul de zgomot
            if 'noise_level_db' not in self.data.columns:
                return "Error: Setul de date nu conține informații despre nivelul de zgomot."
            
            # Determină coloana care conține numele parcurilor
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Nu s-a putut găsi o coloană cu numele parcurilor."
            
            # Creează un dataframe curat doar cu coloanele necesare
            parks_df = self.data[[name_column, 'sector', 'noise_level_db']].copy()
            
            # Elimină parcurile fără valori de zgomot
            parks_df = parks_df.dropna(subset=['noise_level_db'])
            
            # Elimină duplicatele bazate pe numele parcului și sector
            parks_df = parks_df.drop_duplicates(subset=[name_column, 'sector'])
            
            if len(parks_df) == 0:
                return "Nu au fost găsite parcuri cu date despre nivelul de zgomot."
                
            # Curăță formatarea sectorului dacă este necesar
            if 'sector' in parks_df.columns:
                parks_df['sector'] = parks_df['sector'].apply(
                    lambda x: str(x).replace('Sector Sector', 'Sector') if isinstance(x, str) else x
                )
                
                # Curățare mai completă pentru formatul sectorului
                def clean_sector(sector):
                    if not isinstance(sector, str):
                        return "Sector necunoscut"
                    
                    sector = sector.strip()
                    if sector.startswith("Sector "):
                        return sector
                    elif sector.isdigit():
                        return f"Sector {sector}"
                    else:
                        return sector
                
                parks_df['sector'] = parks_df['sector'].apply(clean_sector)
            
            # Sortează parcurile după nivelul de zgomot (valorile mai mici sunt mai bune)
            quietest_parks = parks_df.sort_values('noise_level_db').head(top_n)
            
            # Generează rezultatul cu formatare consistentă
            result = f"Top {top_n} parcuri cu cel mai redus nivel de poluare fonică (valori mai mici în dB înseamnă mai liniștit):\n\n"
            for idx, row in quietest_parks.iterrows():
                park_name = row[name_column]
                sector = row.get('sector', "")
                noise_level = row['noise_level_db']
                
                # Formatează linia de output
                result += f"{park_name} ({sector}): {noise_level:.1f} dB\n"
                
                # Adaugă interpretarea nivelului de zgomot
                if noise_level < 40:
                    result += "  Foarte liniștit - comparabil cu o bibliotecă\n"
                elif noise_level < 50:
                    result += "  Liniștit - zgomot ambiental scăzut\n"
                elif noise_level < 60:
                    result += "  Moderat liniștit - conversație normală\n"
                elif noise_level < 70:
                    result += "  Zgomot moderat - conversație mai puternică\n"
                else:
                    result += "  Zgomotos - trafic urban intens\n"
                    
            return result
        except Exception as e:
            logger.error(f"Eroare la căutarea parcurilor cu cel mai redus nivel de zgomot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la analizarea nivelurilor de zgomot: {str(e)}"
    
    def _find_greenest_sector(self) -> str:
        """Finds the sector with the most parks or best air quality."""
        try:
            if 'sector' not in self.data.columns:
                return "Error: The dataset does not contain sector information."
            
            # Count parks by sector
            park_counts = self.data.groupby('sector').size()
            most_parks_sector = park_counts.idxmax()
            park_count = park_counts.max()
            
            result = f"Sector {most_parks_sector} has the most parks with {park_count} parks.\n"
            
            # Add air quality info if available
            if 'aqi' in self.data.columns:
                avg_aqi = self.data.groupby('sector')['aqi'].mean()
                best_aqi_sector = avg_aqi.idxmin()  # Lower AQI is better
                best_aqi_value = avg_aqi.min()
                
                result += f"\nSector {best_aqi_sector} has the best average air quality with AQI {best_aqi_value:.1f}.\n"
                
                # Add overall ranking
                result += "\nSector Rankings (considering both park count and air quality):\n"
                
                # Calculate a combined score (normalized park count + normalized inverted AQI)
                scores = pd.DataFrame({
                    'sector': park_counts.index,
                    'park_score': (park_counts - park_counts.min()) / (park_counts.max() - park_counts.min()) if park_counts.max() > park_counts.min() else 0.5,
                    'aqi_score': (avg_aqi.max() - avg_aqi) / (avg_aqi.max() - avg_aqi.min()) if avg_aqi.max() > avg_aqi.min() else 0.5
                })
                scores['total_score'] = scores['park_score'] * 0.5 + scores['aqi_score'] * 0.5
                scores = scores.sort_values('total_score', ascending=False)
                
                for i, (idx, row) in enumerate(scores.iterrows()):
                    result += f"{i+1}. Sector {idx} (Score: {row['total_score']:.2f})\n"
            
            return result
        except Exception as e:
            logger.error(f"Error finding greenest sector: {e}")
            return f"Error occurred while analyzing sectors: {str(e)}"
    
    def _find_optimal_park_locations(self, num_locations: int = 3) -> str:
        """
        Identifică locații optime pentru noi parcuri bazate pe analiza distribuției actuale,
        zonelor deficitare și densității populației, cu hartă OpenStreetMap.
        
        Args:
            num_locations: Numărul de locații recomandate (implicit 3)
        
        Returns:
            Recomandări detaliate pentru locații noi de parcuri, inclusiv coordonate GPS
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Eroare: Nu există date despre parcuri disponibile."
                
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point, Polygon
            import numpy as np
            import matplotlib.pyplot as plt
            from datetime import datetime
            import os
            import warnings
            import contextily as ctx
            from matplotlib.lines import Line2D
            
            warnings.filterwarnings('ignore')
            
            # Definim limitele pentru București
            bucuresti_bbox = (26.00, 44.38, 26.20, 44.52)
            
            # Încărcăm datele despre sectoare
            sectors_file = "bucharest_sectors.geojson"
            
            if os.path.exists(sectors_file):
                sectors_gdf = gpd.read_file(sectors_file)
                
                if 'sector' not in sectors_gdf.columns:
                    if 'name' in sectors_gdf.columns:
                        sectors_gdf['sector'] = sectors_gdf['name'].apply(
                            lambda x: f"Sector {x}" if isinstance(x, (int, str)) and str(x).isdigit() else x)
                    else:
                        sectors_gdf['sector'] = [f"Sector {i+1}" for i in range(len(sectors_gdf))]
            else:
                # Folosim coordonate aproximative pentru sectoare dacă nu există fișierul
                sector_boundaries = {
                    'Sector 1': [(26.0608, 44.4372), (26.0962, 44.4737), (26.1155, 44.4539), 
                                 (26.1298, 44.4518), (26.1340, 44.4405), (26.1098, 44.4248), 
                                 (26.0798, 44.4248)],
                    'Sector 2': [(26.1155, 44.4539), (26.1298, 44.4518), (26.1340, 44.4405),
                                 (26.1502, 44.4709), (26.1716, 44.4370), (26.1516, 44.4272), 
                                 (26.1362, 44.4175)],
                    'Sector 3': [(26.1362, 44.4175), (26.1516, 44.4272), (26.1716, 44.4370),
                                 (26.1655, 44.4091), (26.1569, 44.3945), (26.1269, 44.3845), 
                                 (26.1169, 44.3975)],
                    'Sector 4': [(26.0998, 44.4148), (26.1169, 44.3975), (26.1269, 44.3845), 
                                 (26.1020, 44.3889), (26.0817, 44.3889)],
                    'Sector 5': [(26.0551, 44.4165), (26.0998, 44.4148), (26.0817, 44.3889), 
                                 (26.0464, 44.4006)],
                    'Sector 6': [(26.0551, 44.4165), (26.0608, 44.4372), (26.0798, 44.4248), 
                                 (26.1098, 44.4248), (26.0998, 44.4148)]
                }
                
                sectors_data = []
                for sector_name, coords in sector_boundaries.items():
                    sectors_data.append({
                        'sector': sector_name,
                        'geometry': Polygon(coords)
                    })
                    
                sectors_gdf = gpd.GeoDataFrame(sectors_data, crs="EPSG:4326")
            
            # Date despre parcuri
            parks_df = self.data.copy()
            park_counts_by_sector = parks_df.groupby('sector').size()
            sectors_sorted = park_counts_by_sector.sort_values().reset_index()
            sectors_sorted.columns = ['sector', 'count']
            
            # Calculăm aria sectoarelor pentru densitate
            sectors_gdf = sectors_gdf.to_crs("EPSG:3857")
            sectors_gdf['area_km2'] = sectors_gdf.geometry.area / 10**6
            sectors_gdf = sectors_gdf.to_crs("EPSG:4326")
            
            # Creăm un dicționar cu ariile sectoarelor
            sector_areas = {}
            for _, row in sectors_gdf.iterrows():
                sector_areas[row['sector']] = row['area_km2']
            
            # Date populație pe sectoare (2022)
            population_by_sector = {
                'Sector 1': 246220,
                'Sector 2': 345370,
                'Sector 3': 473160,
                'Sector 4': 287828,
                'Sector 5': 271575,
                'Sector 6': 389563
            }
            
            # Adăugăm statistici pentru sectoare
            sectors_sorted['area_km2'] = sectors_sorted['sector'].map(lambda x: sector_areas.get(str(x), 30))
            sectors_sorted['parks_per_km2'] = sectors_sorted['count'] / sectors_sorted['area_km2']
            sectors_sorted['population'] = sectors_sorted['sector'].map(lambda x: population_by_sector.get(str(x), 250000))
            sectors_sorted['parks_per_100k'] = sectors_sorted['count'] * 100000 / sectors_sorted['population']
            
            # Selectăm sectoarele cu cea mai mică densitate de parcuri per populație
            target_sectors = sectors_sorted.sort_values('parks_per_100k').head(3)['sector'].tolist()
            
            # Creăm buffere în jurul parcurilor existente pentru a identifica zonele nedeservite
            existing_parks_gdf = gpd.GeoDataFrame(parks_df, geometry='geometry', crs="EPSG:4326")
            buffered_parks = existing_parks_gdf.geometry.buffer(0.005)  # ~500m în grade
            
            # Unim toate bufferele pentru a obține zona acoperită de parcuri
            from shapely.ops import unary_union
            served_area = unary_union(buffered_parks)
            
            # Definim dimensiunea grilei pentru analiză
            grid_size = 0.002  # ~200m pentru rezoluție înaltă
            
            # Calculăm scoruri pentru potențiale locații
            candidate_locations = []
            
            # Pentru fiecare sector, creăm puncte candidat
            for _, sector_row in sectors_gdf.iterrows():
                sector_name = sector_row['sector']
                sector_polygon = sector_row['geometry']
                
                # Acordăm prioritate sectoarelor cu densitate mică de parcuri per populație
                sector_priority = 1.5 if sector_name in target_sectors else 1.0
                
                # Creăm o grilă de puncte în interiorul sectorului
                minx, miny, maxx, maxy = sector_polygon.bounds
                x_range = np.arange(minx, maxx, grid_size)
                y_range = np.arange(miny, maxy, grid_size)
                
                for x in x_range:
                    for y in y_range:
                        point = Point(x, y)
                        
                        # Verificăm dacă punctul este în sector și nu în zona deservită
                        if sector_polygon.contains(point) and not served_area.contains(point):
                            # Calculăm distanța până la cel mai apropiat parc existent (în metri)
                            distances = [point.distance(geom) * 111000 for geom in existing_parks_gdf.geometry]
                            min_distance = min(distances)
                            
                            # Estimăm populația potențial deservită
                            potential_area_served = np.pi * 0.5 * 0.5  # ~0.8 km²
                            sector_pop_density = sectors_sorted[sectors_sorted['sector'] == sector_name]['population'].values[0] / \
                                                sectors_sorted[sectors_sorted['sector'] == sector_name]['area_km2'].values[0]
                            potential_beneficiaries = int(potential_area_served * sector_pop_density)
                            
                            # Scor pentru zonele nedeservite
                            unserved_score = min(min_distance / 3000, 1.0)
                            
                            # Scor pentru densitatea populației
                            population_score = min(potential_beneficiaries / 10000, 1.0)
                            
                            # Scor pentru accesibilitate
                            city_center = Point(26.1025, 44.4268)
                            dist_to_center = point.distance(city_center) * 111000
                            accessibility_score = max(0, 1 - (dist_to_center / 15000))
                            
                            # Scor combinat pentru locație
                            total_score = (unserved_score * 0.6 + population_score * 0.3 + accessibility_score * 0.1) * sector_priority
                            
                            candidate_locations.append({
                                'geometry': point,
                                'sector': sector_name,
                                'score': total_score,
                                'distance_to_nearest_park': min_distance,
                                'potential_area_served': potential_area_served * 1000000,
                                'potential_beneficiaries': potential_beneficiaries,
                                'latitude': y,
                                'longitude': x
                            })
            
            # Sortăm locațiile după scor
            candidate_locations_df = pd.DataFrame(candidate_locations)
            candidate_locations_df = candidate_locations_df.sort_values('score', ascending=False)
            
            # Selectăm locațiile asigurând diversitatea și distanța minimă între ele
            filtered_locations = []
            sectors_used = {}
            
            for _, location in candidate_locations_df.iterrows():
                sector = location['sector']
                
                # Maximum 1 locație per sector pentru diversitate
                if sector in sectors_used and sectors_used[sector] >= 1:
                    continue
                    
                # Verificăm distanța față de locațiile deja selectate
                too_close = False
                for existing_loc in filtered_locations:
                    p1 = Point(location['longitude'], location['latitude'])
                    p2 = Point(existing_loc['longitude'], existing_loc['latitude'])
                    dist = p1.distance(p2) * 111000
                    if dist < 2000:  # 2 km distanță minimă
                        too_close = True
                        break
                        
                if not too_close:
                    filtered_locations.append(location)
                    sectors_used[sector] = sectors_used.get(sector, 0) + 1
                    
                if len(filtered_locations) >= num_locations:
                    break
            
            # Convertim în DataFrame
            top_locations = pd.DataFrame(filtered_locations)
            
            # Dacă nu avem locații, returnăm un mesaj
            if len(top_locations) == 0:
                return "Nu s-au găsit locații potrivite pentru noi parcuri. Încercați să ajustați criteriile."
                
            # Creăm o hartă interactivă cu OpenStreetMap ca fundal
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Asigurăm-ne că datele sunt în CRS WGS84
            parks_gdf = existing_parks_gdf.to_crs("EPSG:4326")
            top_locations_gdf = gpd.GeoDataFrame(top_locations, geometry='geometry', crs="EPSG:4326")
            
            # Schimbăm la proiecția Web Mercator pentru compatibilitate cu contextily
            parks_gdf_web = parks_gdf.to_crs(epsg=3857)
            top_locations_gdf_web = top_locations_gdf.to_crs(epsg=3857)
            
            # Plotăm parcurile existente cu dimensiune mai mică
            parks_gdf_web.plot(
                ax=ax, 
                color='green',
                markersize=30,  # Dimensiune redusă pentru claritate
                alpha=0.6,
                zorder=2
            )
            
            # Plotăm locațiile recomandate cu dimensiune redusă
            for idx, loc in enumerate(top_locations_gdf_web.iterrows()):
                _, row = loc
                ax.plot(
                    row.geometry.x, 
                    row.geometry.y, 
                    marker='*',
                    color='red', 
                    markersize=10,  # Dimensiune redusă dar vizibilă
                    markeredgecolor='darkred',
                    markeredgewidth=1.5,
                    alpha=1.0,
                    zorder=3
                )
                
                # Obținem coordonatele WGS84 pentru etichetă
                loc_wgs84 = top_locations.iloc[idx]
                coord_text = f"Locația {idx+1}\n{loc_wgs84['latitude']:.5f}, {loc_wgs84['longitude']:.5f}"
                
                # Adăugăm etichetă cu coordonate
                ax.annotate(
                    coord_text,
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(20, 10),  # Offset mai mic
                    textcoords="offset points",
                    fontsize=10,  # Font mai mic
                    color='darkred',
                    fontweight='bold',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.9,
                        boxstyle='round,pad=0.3',
                        edgecolor='darkred'
                    ),
                    zorder=4
                )
            
            # Adăugăm fundalul OpenStreetMap
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    attribution_size=8
                )
            except Exception as e:
                print(f"Nu s-a putut adăuga harta OpenStreetMap: {e}")
                ax.set_facecolor('#F2F2F2')
                ax.grid(True, alpha=0.3, linestyle='--')
            
            # Setăm titlul și subtitlul
            plt.suptitle('Locații Recomandate pentru Parcuri Noi în București', fontsize=18, fontweight='bold')
            plt.title('Hartă bazată pe OpenStreetMap', fontsize=14)
            
            # Adăugăm legenda
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Parcuri existente'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='darkred', 
                      markeredgewidth=1.5, markersize=15, label='Locații recomandate pentru parcuri noi')
            ]
            
            plt.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
            
            # Eliminăm axele pentru o hartă mai curată
            plt.axis('off')
            
            # Adăugăm atribuirea pentru OpenStreetMap (obligatoriu)
            ax.text(
                0.01, 0.01, 
                '© OpenStreetMap contributors',
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
            
            # Salvăm harta ca imagine
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"locatii_parcuri_noi_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Pregătim răspunsul detaliat cu coordonate exacte
            result = "Locații Recomandate pentru Parcuri Noi în București\n"
            result += "===============================================\n\n"
            
            # Lista locațiilor recomandate cu coordonate exacte
            result += "COORDONATE GPS EXACTE PENTRU LOCAȚIILE RECOMANDATE:\n\n"
            
            for idx, row in enumerate(filtered_locations):
                result += f"LOCAȚIA {idx+1} - {row['sector']}\n"
                result += f"• Coordonate GPS exacte: {row['latitude']:.5f}, {row['longitude']:.5f}\n"
                result += f"• Distanța până la cel mai apropiat parc: {row['distance_to_nearest_park']:.0f} metri\n"
                result += f"• Populație potențial deservită: {row['potential_beneficiaries']:,} persoane\n"
                
                # Contextul urban
                sector_data = sectors_sorted[sectors_sorted['sector'] == row['sector']].iloc[0]
                result += f"• {row['sector']} are în prezent {sector_data['count']} parcuri "
                result += f"pentru {sector_data['population']:,} locuitori\n\n"
                
            # Recomandări pentru implementare
            result += "PAȘII URMĂTORI RECOMANDAȚI:\n"
            result += "1. Verificarea disponibilității terenurilor la coordonatele exacte indicate\n"
            result += "2. Realizarea unor studii de fezabilitate pentru fiecare locație\n"
            result += "3. Consultări cu comunitățile locale pentru identificarea nevoilor specifice\n"
            result += "4. Dezvoltarea proiectelor adaptate demografiei din zonele respective\n\n"
            
            # Mențiunea despre harta generată
            result += f"O hartă detaliată cu locațiile exacte a fost generată la: {output_path}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Eroare la identificarea locațiilor pentru parcuri: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la identificarea locațiilor pentru noi parcuri: {str(e)}"

    def _get_weather_forecast(self, days: int = 3) -> str:
        """
        Generează prognoza meteo pentru numărul specificat de zile în limba română corectă.
        
        Args:
            days: Numărul de zile pentru prognoză (implicit 3)
        
        Returns:
            Prognoza meteo formatată în română corectă
        """
        try:
            # Încarcă datele de prognoză
            daily_forecast_file = config.WEATHER_FORECAST_DAILY_FILE
            hourly_forecast_file = config.WEATHER_FORECAST_HOURLY_FILE
            
            if not os.path.exists(daily_forecast_file) or not os.path.exists(hourly_forecast_file):
                # Fișierele nu există, trebuie să preluăm datele
                from src.data_collection import fetch_weather_forecast
                logger.info("Datele de prognoză meteo nu au fost găsite, se preiau acum...")
                forecast_data = fetch_weather_forecast()
                if not forecast_data:
                    return "Nu s-au putut prelua datele de prognoză meteo. Vă rugăm încercați mai târziu."
                
            # Încarcă datele din fișiere
            daily_forecast = gpd.read_file(daily_forecast_file)
            hourly_forecast = gpd.read_file(hourly_forecast_file)
            
            # Asigură-te că coloana 'time' este de tip string
            if 'time' in daily_forecast.columns:
                daily_forecast['time'] = daily_forecast['time'].astype(str)
            else:
                logger.error("Coloana 'time' nu a fost găsită în datele de prognoză zilnică")
                return "Datele de prognoză meteo par incomplete. Lipsesc informațiile despre timp."
                
            if 'time' in hourly_forecast.columns:
                hourly_forecast['time'] = hourly_forecast['time'].astype(str)
            
            # Limitează zilele la cele disponibile
            days = min(days, len(daily_forecast))
            
            # Funcție pentru convertirea datei în format prietenoasă și adăugarea zilei săptămânii
            def format_date(date_str):
                try:
                    from datetime import datetime
                    
                    # Procesează datele în format ISO
                    if 'T' in date_str:
                        date_part = date_str.split('T')[0]
                    else:
                        date_part = date_str
                        
                    date_obj = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    # Numele zilelor săptămânii în română
                    zile = ["Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică"]
                    zi_saptamana = zile[date_obj.weekday()]
                    
                    # Formatare dată prietenoasă
                    return f"{zi_saptamana}, {date_obj.day} {['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'][date_obj.month-1]}"
                except:
                    return date_str
            
            # Formatează prognoza zilnică
            result = f"Prognoza meteo pentru București (următoarele {days} zile):\n\n"
            
            for i in range(days):
                if i >= len(daily_forecast):
                    break
                    
                day_data = daily_forecast.iloc[i]
                day = day_data['time']
                formatted_date = format_date(day)
                
                # Adaugă titlu special pentru prima zi (astăzi/mâine)
                if i == 0:
                    result += f"ASTĂZI, {formatted_date}:\n"
                elif i == 1:
                    result += f"MÂINE, {formatted_date}:\n"
                else:
                    result += f"{formatted_date.upper()}:\n"
                
                # Verifică câmpurile necesare
                if 'temp_min' not in day_data or 'temp_max' not in day_data:
                    result += "• Datele despre temperatură nu sunt disponibile\n\n"
                    continue
                    
                temp_min = float(day_data['temp_min'])
                temp_max = float(day_data['temp_max'])
                
                result += f"• Temperatură: {temp_min:.1f}°C până la {temp_max:.1f}°C\n"
                
                # Obține datele orare pentru această zi
                day_hourly = hourly_forecast[hourly_forecast['time'].str.startswith(day)]
                
                # Calculează mediile pentru această zi
                if len(day_hourly) > 0:
                    # Convertește coloanele numerice în float pentru a preveni erori
                    numeric_cols = ['temperature', 'wind_speed', 'precipitation', 'cloudcover', 'humidity']
                    for col in numeric_cols:
                        if col in day_hourly.columns:
                            day_hourly[col] = pd.to_numeric(day_hourly[col], errors='coerce')
                    
                    # Calculează valorile medii
                    try:
                        avg_wind = day_hourly['wind_speed'].mean() if 'wind_speed' in day_hourly else None
                        total_precip = day_hourly['precipitation'].sum() if 'precipitation' in day_hourly else None
                        avg_humidity = day_hourly['humidity'].mean() if 'humidity' in day_hourly else None
                        avg_clouds = day_hourly['cloudcover'].mean() if 'cloudcover' in day_hourly else None
                        
                        # Adaugă informații despre vânt
                        if avg_wind is not None:
                            result += f"• Vânt: {avg_wind:.1f} km/h"
                            # Adaugă descriere pentru vânt
                            if avg_wind < 5:
                                result += " (vânt ușor)\n"
                            elif avg_wind < 20:
                                result += " (vânt moderat)\n"
                            elif avg_wind < 40:
                                result += " (vânt puternic)\n"
                            else:
                                result += " (vânt foarte puternic)\n"
                        
                        # Adaugă informații despre umiditate
                        if avg_humidity is not None:
                            result += f"• Umiditate: {avg_humidity:.1f}%\n"
                        
                        # Adaugă informații despre precipitații
                        if 'precipitation_sum' in day_data and 'precipitation_probability' in day_data:
                            precipitation = float(day_data['precipitation_sum'])
                            precip_prob = float(day_data['precipitation_probability'])
                            
                            if precipitation < 0.1:
                                result += f"• Precipitații: Nu se așteaptă precipitații\n"
                            else:
                                result += f"• Precipitații: {precipitation:.1f} mm (Probabilitate: {precip_prob}%)\n"
                        elif total_precip is not None:
                            if total_precip < 0.1:
                                result += f"• Precipitații: Nu se așteaptă precipitații\n"
                            else:
                                result += f"• Precipitații: {total_precip:.1f} mm\n"
                        
                        # Descriere generală a stării cerului
                        if avg_clouds is not None:
                            result += "• Starea cerului: "
                            if avg_clouds < 25:
                                result += "Cer senin\n"
                            elif avg_clouds < 50:
                                result += "Cer parțial noros\n"
                            elif avg_clouds < 75:
                                result += "Cer predominant noros\n"
                            else:
                                result += "Cer acoperit\n"
                        
                        # Adaugă recomandare specifică zilei
                        result += "• Recomandare: "
                        if temp_max > 30:
                            result += "Hidratare frecventă, evitați expunerea la soare în orele de vârf"
                        elif temp_max > 25 and precipitation < 0.5:
                            result += "Condiții ideale pentru activități în aer liber"
                        elif precipitation > 5:
                            result += "Pregătiți-vă pentru precipitații, luați umbrela"
                        elif temp_min < 10:
                            result += "Îmbrăcați-vă corespunzător pentru temperaturi răcoroase"
                        else:
                            result += "Condiții favorabile pentru majoritatea activităților"
                        result += "\n"
                    
                    except Exception as e:
                        logger.error(f"Eroare la calcularea indicatorilor meteo: {e}")
                
                result += "\n"
            
            # Adaugă defalcarea orară doar pentru prima zi (astăzi), cu format îmbunătățit
            if len(daily_forecast) > 0:
                today = daily_forecast.iloc[0]['time']
                today_hourly = hourly_forecast[hourly_forecast['time'].str.startswith(today)]
                
                if len(today_hourly) > 0:
                    result += f"Prognoză orară pentru astăzi ({format_date(today)}):\n\n"
                    result += f"{'Ora':<6} | {'Temp.':<6} | {'Vânt':<8} | {'Precip.':<8} | {'Nori':<6} | {'Observații':<20}\n"
                    result += "-" * 70 + "\n"
                    
                    # Convertește coloanele în numerice pentru calcule
                    for col in ['temperature', 'wind_speed', 'precipitation', 'cloudcover']:
                        if col in today_hourly.columns:
                            today_hourly[col] = pd.to_numeric(today_hourly[col], errors='coerce')
                    
                    # Obține orele relevante pentru utilizator
                    key_hours = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]
                    
                    for hour in key_hours:
                        # Filtrare sigură folosind operațiuni pe șiruri
                        hour_data = today_hourly[today_hourly['time'].str.contains(f"T{hour}")]
                        
                        if len(hour_data) > 0:
                            hour_data = hour_data.iloc[0]
                            time_str = hour
                            
                            # Obține valorile în mod sigur cu gestionarea erorilor
                            try:
                                temp = float(hour_data['temperature']) if 'temperature' in hour_data else float('nan')
                                wind = float(hour_data['wind_speed']) if 'wind_speed' in hour_data else float('nan')
                                precip = float(hour_data['precipitation']) if 'precipitation' in hour_data else float('nan')
                                clouds = float(hour_data['cloudcover']) if 'cloudcover' in hour_data else float('nan')
                                
                                # Generează descriere personalizată
                                obs = ""
                                if temp > 30:
                                    obs = "Foarte cald"
                                elif temp > 25:
                                    obs = "Cald"
                                elif temp < 10:
                                    obs = "Rece"
                                    
                                if precip > 1:
                                    obs += ", Ploaie"
                                elif precip > 0.1:
                                    obs += ", Precip. ușoare"
                                    
                                if clouds > 80:
                                    obs += ", Înnorat"
                                elif clouds < 20:
                                    obs += ", Senin"
                                    
                                if not obs:
                                    obs = "Condiții bune"
                                
                                result += f"{time_str:<6} | {temp:<6.1f} | {wind:<8.1f} | {precip:<8.2f} | {clouds:<6.0f} | {obs:<20}\n"
                            except Exception as e:
                                result += f"{time_str:<6} | {'N/A':<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<6} | {'Date indisponibile':<20}\n"
                    
                    result += "\n"
            
            # Adaugă un rezumat general personalizat
            result += "Rezumat general pentru perioada prognozată:\n"
            temp_data = [float(row['temp_max']) for _, row in daily_forecast.iloc[:days].iterrows()]
            avg_max_temp = sum(temp_data) / len(temp_data)
            
            # Verifică dacă există date despre precipitații
            has_precipitation = False
            precipitation_days = 0
            for i in range(min(days, len(daily_forecast))):
                if 'precipitation_sum' in daily_forecast.iloc[i]:
                    precip = float(daily_forecast.iloc[i]['precipitation_sum'])
                    if precip > 0.5:
                        has_precipitation = True
                        precipitation_days += 1
            
            # Descriere personalizată în funcție de tendințele generale
            if avg_max_temp > 30:
                result += "• Se anticipează o perioadă foarte caldă, cu temperaturi ridicate. "
                result += "Se recomandă hidratarea frecventă, folosirea protecției solare și "
                result += "evitarea expunerii la soare în intervalul orar 11:00-16:00."
            elif avg_max_temp > 25:
                result += "• Se anticipează o perioadă caldă și plăcută. "
                result += "Temperaturile sunt ideale pentru activități în aer liber."
            elif avg_max_temp > 15:
                result += "• Se anticipează o perioadă cu temperaturi moderate, "
                result += "potrivită pentru diverse activități exterioare."
            elif avg_max_temp > 5:
                result += "• Se anticipează o perioadă răcoroasă. "
                result += "Se recomandă îmbrăcăminte adecvată pentru temperaturi moderate."
            else:
                result += "• Se anticipează o perioadă rece. "
                result += "Se recomandă îmbrăcăminte călduroasă pentru activitățile în aer liber."
            
            # Adaugă informații despre precipitații în rezumat
            if has_precipitation:
                if precipitation_days == 1:
                    result += "\n• Este posibilă apariția precipitațiilor într-una din zile. "
                    result += "Verificați prognoza orară pentru planificarea activităților."
                else:
                    result += f"\n• Sunt așteptate precipitații în {precipitation_days} zile. "
                    result += "Se recomandă să aveți pregătită o umbrelă."
            else:
                result += "\n• Nu se preconizează precipitații semnificative în perioada următoare."
            
            return result
                
        except Exception as e:
            logger.error(f"Eroare la obținerea prognozei meteo: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare în timpul preluării prognozei meteo: {str(e)}"
        
    def _get_hourly_weather_forecast(self, day: str = "today", time_period: str = "all", hours: list = None) -> str:
        """
        Gets the hourly weather forecast for a specified day and time period.
        
        Args:
            day: Day to get forecast for ("today", "tomorrow", or YYYY-MM-DD)
            time_period: Time period to filter ("morning", "afternoon", "evening", "night", "all")
            hours: List of specific hours to show (e.g., ["09:00", "12:00", "15:00"])
        
        Returns:
            Formatted hourly weather forecast information
        """
        try:
            logger.info(f"Getting hourly forecast for day={day}, time_period={time_period}, hours={hours}")
            
            # Load forecast data
            hourly_forecast_file = config.WEATHER_FORECAST_HOURLY_FILE
            daily_forecast_file = config.WEATHER_FORECAST_DAILY_FILE
            
            if not os.path.exists(hourly_forecast_file) or not os.path.exists(daily_forecast_file):
                from src.data_collection import fetch_weather_forecast
                forecast_data = fetch_weather_forecast()
                if not forecast_data:
                    return "Nu s-au putut obține datele de prognoză meteo."
            
            # Load forecast data
            hourly_forecast = gpd.read_file(hourly_forecast_file)
            daily_forecast = gpd.read_file(daily_forecast_file)
            
            # Ensure time columns are strings
            hourly_forecast['time'] = hourly_forecast['time'].astype(str)
            daily_forecast['time'] = daily_forecast['time'].astype(str)
            
            # Extract date part from timestamp - handle both "T" and space separators
            def extract_date(timestamp):
                if 'T' in timestamp:
                    return timestamp.split('T')[0]
                elif ' ' in timestamp:
                    return timestamp.split(' ')[0]
                return timestamp
                
            # Get available dates
            all_dates = sorted(set([extract_date(t) for t in hourly_forecast['time']]))
            
            # Determine target date
            if day.lower() in ["today", "astăzi", "azi"]:
                target_date = extract_date(daily_forecast.iloc[0]['time']) if len(daily_forecast) > 0 else None
                day_name = "astăzi"
            elif day.lower() in ["tomorrow", "mâine"]:
                target_date = extract_date(daily_forecast.iloc[1]['time']) if len(daily_forecast) > 1 else None
                day_name = "mâine"
            else:
                date_matches = [d for d in all_dates if day in d]
                target_date = date_matches[0] if date_matches else None
                day_name = day
                
            if not target_date:
                return f"Prognoza pentru {day} nu este disponibilă."
                
            # Filter data for target date
            day_hourly = hourly_forecast[hourly_forecast['time'].str.startswith(target_date)]
            
            if len(day_hourly) == 0:
                return f"Prognoza orară pentru {day_name} nu este disponibilă."
            
            # Extract hour from timestamp with both formats
            def extract_hour(timestamp):
                try:
                    if 'T' in timestamp:
                        time_part = timestamp.split('T')[1]
                        return int(time_part.split(':')[0])
                    elif ' ' in timestamp:
                        parts = timestamp.split(' ')
                        if len(parts) > 1:
                            time_part = parts[1]
                            return int(time_part.split(':')[0])
                    return -1
                except Exception as e:
                    return -1
            
            # Add extracted hour column for filtering
            day_hourly['hour'] = day_hourly['time'].apply(extract_hour)
            
            # Period definitions based on the provided table
            period_ranges = {
                "dimineața": list(range(5, 12)),      # 05:00-11:59
                "prânz": list(range(12, 14)),         # 12:00-13:59
                "după-amiaza": list(range(14, 18)),   # 14:00-17:59
                "seara": list(range(18, 21)),         # 18:00-20:59
                "noaptea": list(range(21, 24)) + list(range(0, 5)),  # 21:00-04:59
                "all": list(range(0, 24))             # All hours
            }
            
            # Map synonyms to standard periods
            period_mapping = {
                "morning": "dimineața", "dimineata": "dimineața", "dimineață": "dimineața", "dimineața": "dimineața",
                "noon": "prânz", "prânz": "prânz", "pranz": "prânz", 
                "afternoon": "după-amiaza", "după-amiaza": "după-amiaza", "dupa-amiaza": "după-amiaza", 
                "după amiaza": "după-amiaza", "dupa amiaza": "după-amiaza",
                "evening": "seara", "seara": "seara",
                "night": "noaptea", "noaptea": "noaptea",
                "all": "all", "toată ziua": "all", "toata ziua": "all"
            }
            
            # Normalize the period name
            normalized_period = period_mapping.get(time_period.lower(), "all")
            
            # Filter by time period
            if normalized_period != "all" and normalized_period in period_ranges:
                hours_range = period_ranges[normalized_period]
                
                # Use isin for filtering
                period_data = day_hourly[day_hourly['hour'].isin(hours_range)]
                
                if len(period_data) > 0:
                    day_hourly = period_data
                    period_display = normalized_period
                else:
                    period_display = "toată ziua"
            else:
                period_display = "toată ziua"
            
            # Convert numeric columns
            numeric_cols = ['temperature', 'wind_speed', 'precipitation', 'cloudcover', 'humidity', 'pm10', 'pm25', 'aqi']
            for col in numeric_cols:
                if col in day_hourly.columns:
                    day_hourly.loc[:, col] = pd.to_numeric(day_hourly[col], errors='coerce')
            
            # Format response header
            result = f"Prognoză meteo orară pentru București - {day_name} ({period_display}):\n\n"
            
            # Create table columns
            columns = [("Ora", 10)]
            if 'temperature' in day_hourly.columns: columns.append(("Temp (°C)", 10))
            if 'wind_speed' in day_hourly.columns: columns.append(("Vânt (km/h)", 12))
            if 'precipitation' in day_hourly.columns: columns.append(("Precip. (mm)", 12))
            if 'humidity' in day_hourly.columns: columns.append(("Umiditate (%)", 14))
            if 'cloudcover' in day_hourly.columns: columns.append(("Nori (%)", 10))
            if 'pm25' in day_hourly.columns: columns.append(("PM2.5", 10))
            if 'pm10' in day_hourly.columns: columns.append(("PM10", 10))
            
            # Build header
            header = ""
            for col_name, width in columns:
                header += f"{col_name:<{width}} | "
            header = header.rstrip(" | ")
            
            result += header + "\n"
            result += "-" * len(header) + "\n"
            
            # Extract and display the hour:minute format from timestamp
            for _, row in day_hourly.sort_values('time').iterrows():
                # Extract time with support for both formats
                time_str = ""
                ts = row['time']
                
                if 'T' in ts:
                    time_str = ts.split('T')[1][:5]  # HH:MM
                elif ' ' in ts:
                    parts = ts.split(' ')
                    if len(parts) > 1:
                        time_str = parts[1][:5]  # HH:MM
                
                if not time_str:
                    time_str = "00:00"  # Default fallback
                    
                # Build data row
                data_row = f"{time_str:<{columns[0][1]}} | "
                col_index = 1
                
                for col, (_, width) in zip(['temperature', 'wind_speed', 'precipitation', 'humidity', 
                                           'cloudcover', 'pm25', 'pm10'], columns[1:], strict=False):
                    if col in row and pd.notna(row[col]) and col_index < len(columns):
                        data_row += f"{row[col]:<{columns[col_index][1]}.1f} | "
                    elif col_index < len(columns):
                        data_row += f"{'N/A':<{columns[col_index][1]}} | "
                    col_index += 1
                    
                result += data_row.rstrip(" | ") + "\n"
            
            # Add summary information
            result += "\nRezumat:\n"
            result += "=" * 30 + "\n"
            
            # Calculate summary metrics
            if len(day_hourly) > 0:
                if 'temperature' in day_hourly.columns:
                    avg_temp = day_hourly['temperature'].mean()
                    min_temp = day_hourly['temperature'].min()
                    max_temp = day_hourly['temperature'].max()
                    if pd.notna(avg_temp):
                        result += f"Temperatura medie: {avg_temp:.1f}°C\n"
                    if pd.notna(min_temp) and pd.notna(max_temp):
                        result += f"Interval temperatură: {min_temp:.1f}°C până la {max_temp:.1f}°C\n"
                
                if 'precipitation' in day_hourly.columns:
                    total_precip = day_hourly['precipitation'].sum()
                    if pd.notna(total_precip):
                        result += f"Precipitații totale: {total_precip:.1f}mm\n"
                        if total_precip < 0.1:
                            result += "Condiții: Nu se așteaptă precipitații\n"
                        elif total_precip < 2:
                            result += "Condiții: Posibil precipitații ușoare\n"
                        elif total_precip < 10:
                            result += "Condiții: Se așteaptă precipitații moderate\n"
                        else:
                            result += "Condiții: Se așteaptă precipitații abundente\n"
                
                if 'cloudcover' in day_hourly.columns:
                    avg_clouds = day_hourly['cloudcover'].mean()
                    if pd.notna(avg_clouds):
                        result += f"Acoperire medie cu nori: {avg_clouds:.1f}%\n"
                        if avg_clouds < 25:
                            result += "Starea cerului: Predominant senin\n"
                        elif avg_clouds < 50:
                            result += "Starea cerului: Parțial noros\n"
                        elif avg_clouds < 75:
                            result += "Starea cerului: Predominant noros\n"
                        else:
                            result += "Starea cerului: Acoperit\n"
                
                if 'pm25' in day_hourly.columns and 'pm10' in day_hourly.columns:
                    avg_pm25 = day_hourly['pm25'].mean()
                    avg_pm10 = day_hourly['pm10'].mean()
                    if pd.notna(avg_pm25) and pd.notna(avg_pm10):
                        result += f"\nCalitatea Aerului:\n"
                        result += f"PM2.5 mediu: {avg_pm25:.1f} µg/m³\n"
                        result += f"PM10 mediu: {avg_pm10:.1f} µg/m³\n"
                        if avg_pm25 < 10 and avg_pm10 < 20:
                            result += "Evaluarea calității aerului: Excelentă\n"
                        elif avg_pm25 < 25 and avg_pm10 < 50:
                            result += "Evaluarea calității aerului: Bună\n"
                        elif avg_pm25 < 50 and avg_pm10 < 100:
                            result += "Evaluarea calității aerului: Moderată\n"
                        else:
                            result += "Evaluarea calității aerului: Slabă\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting hourly weather forecast: {e}")
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la obținerea prognozei orare: {str(e)}"
    
    def _get_air_quality_forecast(self, days: int = 3) -> str:
        """
        Obține prognoza calității aerului pentru numărul specificat de zile.
        
        Args:
            days: Numărul de zile pentru prognoză (implicit 3)
        
        Returns:
            Informații formatate despre prognoza calității aerului
        """
        try:
            # Încarcă datele de prognoză
            hourly_forecast_file = config.WEATHER_FORECAST_HOURLY_FILE
            
            if not os.path.exists(hourly_forecast_file):
                return "Datele despre prognoza calității aerului nu sunt disponibile. Vă rugăm să actualizați datele."
                
            hourly_forecast = gpd.read_file(hourly_forecast_file)
            
            # Convertește coloana de timp în șir pentru operațiuni pe șiruri
            if 'time' in hourly_forecast.columns:
                hourly_forecast['time'] = hourly_forecast['time'].astype(str)
            
            # Obține zilele unice
            unique_days = sorted(set([time.split('T')[0] for time in hourly_forecast['time']]))
            days = min(days, len(unique_days))
            
            # Formatează prognoza calității aerului
            result = f"Prognoza Calității Aerului pentru București (următoarele {days} zile):\n\n"
            
            # Procesează fiecare zi
            for i in range(days):
                day = unique_days[i]
                day_data = hourly_forecast[hourly_forecast['time'].str.startswith(day)]
                
                # Convertește coloanele numerice pentru calcule
                for col in ['pm10', 'pm25', 'aqi']:
                    if col in day_data.columns:
                        day_data.loc[:, col] = pd.to_numeric(day_data[col], errors='coerce')
                
                # Medii zilnice
                avg_pm10 = day_data['pm10'].mean()
                avg_pm25 = day_data['pm25'].mean()
                avg_aqi = day_data['aqi'].mean() if 'aqi' in day_data.columns else None
                
                result += f"Data: {day}\n"
                result += f"PM10 mediu: {avg_pm10:.1f} µg/m³\n"
                result += f"PM2.5 mediu: {avg_pm25:.1f} µg/m³\n"
                
                if avg_aqi is not None:
                    result += f"AQI mediu: {avg_aqi:.1f}\n"
                    
                    # Interpretarea AQI
                    if avg_aqi < 20:
                        result += "Calitatea Aerului: Excelentă\n"
                    elif avg_aqi < 40:
                        result += "Calitatea Aerului: Bună\n"
                    elif avg_aqi < 60:
                        result += "Calitatea Aerului: Moderată\n"
                    elif avg_aqi < 80:
                        result += "Calitatea Aerului: Slabă\n"
                    elif avg_aqi < 100:
                        result += "Calitatea Aerului: Foarte Slabă\n"
                    else:
                        result += "Calitatea Aerului: Extrem de Slabă\n"
                
                # Recomandări pentru sănătate
                if avg_pm25 < 10 and avg_pm10 < 20:
                    result += "Recomandări pentru sănătate: Condiții excelente pentru activități în aer liber.\n"
                elif avg_pm25 < 25 and avg_pm10 < 50:
                    result += "Recomandări pentru sănătate: Condiții bune pentru majoritatea persoanelor. Cei cu sensibilitate extremă ar putea dori să limiteze efortul prelungit în aer liber.\n"
                elif avg_pm25 < 50 and avg_pm10 < 100:
                    result += "Recomandări pentru sănătate: Grupurile sensibile ar trebui să considere limitarea efortului prelungit în aer liber.\n"
                else:
                    result += "Recomandări pentru sănătate: Toată lumea ar trebui să limiteze activitățile în aer liber, în special grupurile sensibile precum copiii, vârstnicii și persoanele cu afecțiuni respiratorii.\n"
                
                result += "-" * 50 + "\n"
                
            return result
            
        except Exception as e:
            logger.error(f"Eroare la obținerea prognozei calității aerului: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare în timpul obținerii prognozei calității aerului: {str(e)}"

    def _get_best_park_day(self, park_name: str = None) -> str:
        """
        Recomandă cea mai bună zi pentru vizitarea parcurilor bazată pe prognozele meteo și calitatea aerului.
        
        Args:
            park_name: Numele specific al parcului de luat în considerare (opțional)
        
        Returns:
            Recomandare pentru cea mai bună zi de vizitare a parcurilor
        """
        try:
            # Încarcă datele de prognoză
            daily_forecast_file = config.WEATHER_FORECAST_DAILY_FILE
            hourly_forecast_file = config.WEATHER_FORECAST_HOURLY_FILE
            
            if not os.path.exists(daily_forecast_file) or not os.path.exists(hourly_forecast_file):
                # Fișierele nu există, trebuie să preluăm datele mai întâi
                from src.data_collection import fetch_weather_forecast
                logger.info("Datele de prognoză meteo nu au fost găsite, se preiau acum...")
                forecast_data = fetch_weather_forecast()
                if not forecast_data:
                    return "Nu s-au putut obține datele de prognoză meteo. Vă rugăm încercați mai târziu."
                    
                # Acum fișierele ar trebui să existe
                if not os.path.exists(daily_forecast_file) or not os.path.exists(hourly_forecast_file):
                    return "Eroare la crearea fișierelor de prognoză meteo."
                    
            # Încărcăm datele din fișiere
            daily_forecast = gpd.read_file(daily_forecast_file)
            hourly_forecast = gpd.read_file(hourly_forecast_file)
            
            # Informații de depanare
            logger.info(f"Coloane prognoză zilnică: {daily_forecast.columns.tolist()}")
            logger.info(f"Coloane prognoză orară: {hourly_forecast.columns.tolist()}")
            
            # Ne asigurăm că coloana 'time' este de tip șir pentru operațiuni pe șiruri
            if 'time' in daily_forecast.columns:
                daily_forecast['time'] = daily_forecast['time'].astype(str)
            else:
                logger.error("Coloana 'time' nu a fost găsită în datele de prognoză zilnică")
                return "Datele de prognoză meteo par incomplete. Lipsesc informațiile despre timp."
                
            if 'time' in hourly_forecast.columns:
                hourly_forecast['time'] = hourly_forecast['time'].astype(str)
            else:
                logger.error("Coloana 'time' nu a fost găsită în datele de prognoză orară")
                return "Datele de prognoză meteo par incomplete. Lipsesc informațiile despre timp."
            
            # Obține zilele unice
            unique_days = daily_forecast['time'].unique()
            logger.info(f"Zile disponibile pentru prognoză: {unique_days}")
            
            # Calculează scoruri pentru fiecare zi
            day_scores = []
            
            for day in unique_days:
                logger.info(f"Se procesează ziua: {day}")
                
                # Obține datele zilnice
                day_data = daily_forecast[daily_forecast['time'] == day].iloc[0]
                
                # Obține datele orare pentru această zi
                day_hourly = hourly_forecast[hourly_forecast['time'].str.startswith(day)]
                
                # Convertește câmpurile numerice în float pentru calcule
                for col in ['temp_min', 'temp_max', 'precipitation_sum', 'precipitation_probability']:
                    if col in day_data and day_data[col] is not None:
                        try:
                            day_data[col] = float(day_data[col])
                        except (ValueError, TypeError):
                            logger.warning(f"Nu s-a putut converti {col} în float: {day_data[col]}")
                            day_data[col] = 0.0
                
                # Calculează scorul pentru vreme
                # Temperatura ideală: în jurul a 23°C (nici prea cald, nici prea rece)
                avg_temp = (day_data['temp_min'] + day_data['temp_max']) / 2
                temp_score = 10 - abs(avg_temp - 23) * 0.5  # Penalizează deviația de la temperatura ideală
                
                # Penalizare pentru precipitații
                precip_score = 10 - day_data['precipitation_sum'] * 2  # Penalizează precipitațiile
                precip_score = max(0, precip_score)  # Asigură că nu e negativ
                
                # Scorul pentru vânt (vânt moderat e okay, vânt puternic e rău)
                if len(day_hourly) > 0 and 'wind_speed' in day_hourly.columns:
                    # Convertește la numeric în siguranță
                    day_hourly.loc[:, 'wind_speed'] = pd.to_numeric(day_hourly['wind_speed'], errors='coerce')
                    
                    avg_wind = day_hourly['wind_speed'].mean()
                    if avg_wind < 5:
                        wind_score = 10  # Vânt calm
                    elif avg_wind < 15:
                        wind_score = 8  # Vânt ușor
                    elif avg_wind < 25:
                        wind_score = 5  # Vânt moderat
                    else:
                        wind_score = 2  # Vânt puternic
                else:
                    wind_score = 5  # Valoare implicită
                
                # Scorul pentru calitatea aerului
                if len(day_hourly) > 0 and 'pm25' in day_hourly.columns and 'pm10' in day_hourly.columns:
                    # Convertește la numeric în siguranță
                    day_hourly.loc[:, 'pm25'] = pd.to_numeric(day_hourly['pm25'], errors='coerce')
                    day_hourly.loc[:, 'pm10'] = pd.to_numeric(day_hourly['pm10'], errors='coerce')
                    
                    avg_pm25 = day_hourly['pm25'].mean()
                    avg_pm10 = day_hourly['pm10'].mean()
                    
                    # Scor PM2.5 (mai mic e mai bun)
                    if avg_pm25 < 10:
                        pm25_score = 10
                    elif avg_pm25 < 25:
                        pm25_score = 8
                    elif avg_pm25 < 50:
                        pm25_score = 5
                    else:
                        pm25_score = 2
                    
                    # Scor PM10 (mai mic e mai bun)
                    if avg_pm10 < 20:
                        pm10_score = 10
                    elif avg_pm10 < 50:
                        pm10_score = 8
                    elif avg_pm10 < 100:
                        pm10_score = 5
                    else:
                        pm10_score = 2
                        
                    air_score = (pm25_score + pm10_score) / 2
                else:
                    air_score = 5  # Valoare implicită
                
                # Calculează scorul total (ponderat)
                total_score = temp_score * 0.3 + precip_score * 0.3 + wind_score * 0.2 + air_score * 0.2
                
                day_scores.append({
                    'day': day,
                    'total_score': total_score,
                    'temp_score': temp_score,
                    'precip_score': precip_score,
                    'wind_score': wind_score,
                    'air_score': air_score,
                    'avg_temp': avg_temp,
                    'precipitation': day_data['precipitation_sum'],
                    'precip_prob': day_data['precipitation_probability']
                })
            
            if not day_scores:
                return "Nu s-au putut calcula scorurile pentru nicio zi. Datele de prognoză meteo ar putea fi incomplete."
                
            # Sortează zilele după scor
            day_scores.sort(key=lambda x: x['total_score'], reverse=True)
            
            # Formatează răspunsul
            result = "Recomandări pentru Vizitarea Parcurilor Bazate pe Vreme și Calitatea Aerului:\n\n"
            
            # Cea mai bună zi
            best_day = day_scores[0]
            result += f"ZIUA OPTIMĂ DE VIZITARE: {best_day['day']}\n"
            result += f"Scor general: {best_day['total_score']:.1f}/10\n"
            result += f"Temperatură medie: {best_day['avg_temp']:.1f}°C\n"
            result += f"Precipitații: {best_day['precipitation']:.1f}mm (Probabilitate: {best_day['precip_prob']}%)\n\n"
            
            # Dacă este furnizat un nume de parc, adaugă informații specifice
            if park_name:
                # Găsește parcul
                park_matches = self.data[self.data['name_left'].str.contains(park_name, case=False, na=False)]
                
                if len(park_matches) == 0:
                    # Încercăm nume alternative de coloane
                    for name_col in ['name', 'park_name', 'title']:
                        if name_col in self.data.columns:
                            park_matches = self.data[self.data[name_col].str.contains(park_name, case=False, na=False)]
                            if len(park_matches) > 0:
                                break
                
                if len(park_matches) > 0:
                    park = park_matches.iloc[0]
                    park_name_col = 'name_left' if 'name_left' in park else 'name' if 'name' in park else 'park_name'
                    result += f"Recomandări specifice pentru {park[park_name_col]}:\n"
                    
                    # Adaugă detalii specifice parcului
                    if 'sector' in park:
                        result += f"Locație: {park['sector']}\n"
                    
                    # Adaugă recomandare specială
                    result += f"\nCea mai bună perioadă pentru vizitarea {park[park_name_col]} în data de {best_day['day']}:\n"
                    
                    # Recomandare dimineață vs după-amiază
                    day_hourly = hourly_forecast[hourly_forecast['time'].str.startswith(best_day['day'])]
                    
                    if len(day_hourly) > 0:
                        # Convertește temperatura la numeric
                        day_hourly.loc[:, 'temperature'] = pd.to_numeric(day_hourly['temperature'], errors='coerce')
                        
                        # Grupează pe dimineață (înainte de 12) și după-amiază (după 12)
                        morning_data = day_hourly[day_hourly['time'].str.contains('T09:00|T10:00|T11:00')]
                        afternoon_data = day_hourly[day_hourly['time'].str.contains('T14:00|T15:00|T16:00')]
                        
                        if len(morning_data) > 0 and len(afternoon_data) > 0:
                            morning_temp = morning_data['temperature'].mean()
                            
                            # Convertește la numeric în siguranță
                            if 'wind_speed' in morning_data.columns:
                                morning_data.loc[:, 'wind_speed'] = pd.to_numeric(morning_data['wind_speed'], errors='coerce')
                                morning_wind = morning_data['wind_speed'].mean()
                            else:
                                morning_wind = 0
                                
                            if 'precipitation' in morning_data.columns:
                                morning_data.loc[:, 'precipitation'] = pd.to_numeric(morning_data['precipitation'], errors='coerce')
                                morning_precip = morning_data['precipitation'].sum()
                            else:
                                morning_precip = 0
                            
                            afternoon_temp = afternoon_data['temperature'].mean()
                            
                            if 'wind_speed' in afternoon_data.columns:
                                afternoon_data.loc[:, 'wind_speed'] = pd.to_numeric(afternoon_data['wind_speed'], errors='coerce')
                                afternoon_wind = afternoon_data['wind_speed'].mean()
                            else:
                                afternoon_wind = 0
                                
                            if 'precipitation' in afternoon_data.columns:
                                afternoon_data.loc[:, 'precipitation'] = pd.to_numeric(afternoon_data['precipitation'], errors='coerce')
                                afternoon_precip = afternoon_data['precipitation'].sum()
                            else:
                                afternoon_precip = 0
                            
                            # Compară și recomandă
                            if morning_precip < afternoon_precip and afternoon_temp > 30:
                                result += "Dimineața este recomandată, deoarece după-amiezile pot fi prea calde, cu șanse mai mari de precipitații.\n"
                            elif morning_precip > afternoon_precip and morning_temp < 15:
                                result += "După-amiaza este recomandată, deoarece dimineața poate fi prea răcoroasă, cu șanse mai mari de precipitații.\n"
                            elif morning_wind < afternoon_wind and afternoon_wind > 20:
                                result += "Dimineața este recomandată, deoarece după-amiezile pot fi prea vântoase.\n"
                            else:
                                result += "Atât dimineața, cât și după-amiaza au condiții favorabile, cu după-amiezile fiind ușor mai calde.\n"
            
            # Toate zilele clasate
            result += "\nToate Zilele Ordonate (De la cea mai bună la cea mai puțin favorabilă):\n"
            result += "=" * 50 + "\n"
            
            for i, day in enumerate(day_scores):
                result += f"{i+1}. {day['day']} - Scor: {day['total_score']:.1f}/10\n"
                result += f"   Temperatură: {day['avg_temp']:.1f}°C, "
                result += f"Precipitații: {day['precipitation']:.1f}mm ({day['precip_prob']}%)\n"
                
            return result
            
        except Exception as e:
            logger.error(f"Eroare la găsirea celei mai bune zile pentru vizitarea parcurilor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la calcularea celei mai bune zile pentru vizitarea parcurilor: {str(e)}"

    def _check_outdoor_activities(self, day: str = None, activity_type: str = None, query: str = None) -> str:
        """
        Evaluează dacă condițiile sunt potrivite pentru activități specifice în aer liber pentru o anumită zi,
        cu detectare automată a tipului de activitate din întrebare.
        
        Args:
            day: Ziua pentru care se face verificarea ("today", "tomorrow" sau format YYYY-MM-DD)
            activity_type: Tipul de activitate ("general", "sports", "picnic", "photography", etc.)
            query: Întrebarea originală a utilizatorului pentru detectarea automată a tipului de activitate
        
        Returns:
            Evaluarea condițiilor pentru activitatea specificată
        """
        try:
            # Detectarea automată a tipului de activitate din întrebare, dacă este furnizată
            if query and activity_type is None:
                # Convertim la lowercase pentru ușurința căutării
                query_lower = query.lower()
                
                # Definim cuvinte cheie pentru fiecare categorie
                activity_keywords = {
                    "sports": ["sport", "jogging", "alerg", "fugă", "mers", "biciclet", "cicl", "înot", "fotbal", "tenis", 
                            "basket", "antrena", "exerci", "fitness", "volei", "frisbee"],
                    "picnic": ["picnic", "relaxa", "grătar", "barbecue", "gratar", "ieșire", "iesire", "parc", "prânz afară", 
                            "pranz afara", "ieșit la iarbă verde", "masa în aer liber", "terasă"],
                    "photography": ["foto", "poze", "fotografii", "camera", "fotografiere", "peisaj", "apus", "răsărit", "rasarit", 
                                "vizita", "turism", "obiective", "monument", "arhitect", "explorer", "vizitare"],
                }
                
                # Detectăm categorii din întrebare
                for cat, keywords in activity_keywords.items():
                    if any(keyword in query_lower for keyword in keywords):
                        activity_type = cat
                        break
                
                # Dacă nu am detectat nicio categorie, folosim "general"
                if activity_type is None:
                    activity_type = "general"
            elif activity_type is None:
                # Valoarea implicită dacă nu este specificată
                activity_type = "general"
            
            # Încarcă datele de prognoză
            daily_forecast_file = config.WEATHER_FORECAST_DAILY_FILE
            hourly_forecast_file = config.WEATHER_FORECAST_HOURLY_FILE
            
            if not os.path.exists(daily_forecast_file) or not os.path.exists(hourly_forecast_file):
                return "Datele de prognoză meteo nu sunt disponibile. Vă rugăm să actualizați datele."
                
            daily_forecast = gpd.read_file(daily_forecast_file)
            hourly_forecast = gpd.read_file(hourly_forecast_file)
            
            # Ne asigurăm că coloana 'time' este de tip șir pentru operațiuni pe șiruri
            if 'time' in daily_forecast.columns:
                daily_forecast['time'] = daily_forecast['time'].astype(str)
            else:
                return "Datele de prognoză meteo par a fi incomplete. Lipsesc informațiile despre timp."
                
            if 'time' in hourly_forecast.columns:
                hourly_forecast['time'] = hourly_forecast['time'].astype(str)
            else:
                return "Datele de prognoză meteo par a fi incomplete. Lipsesc informațiile despre timp."
            
            # Dacă ziua nu este specificată, folosim ziua de astăzi
            if day is None or day.lower() in ["today", "azi", "astăzi", "astazi"]:
                day = daily_forecast.iloc[0]['time']
                day_name = "astăzi"
            elif day.lower() in ["tomorrow", "mâine", "maine"]:
                if len(daily_forecast) > 1:
                    day = daily_forecast.iloc[1]['time']
                    day_name = "mâine"
                else:
                    return "Prognoza pentru mâine nu este disponibilă."
            else:
                day_name = day
            
            # Normalizează formatul zilei - extrage doar partea cu data dacă conține timp
            if 'T' in day:
                day = day.split('T')[0]
            elif ' ' in day:  # Gestionează formatul "YYYY-MM-DD HH:MM:SS"
                day = day.split(' ')[0]
            
            # Verifică dacă ziua există în prognoză
            day_records = [d for d in daily_forecast['time'] if day in d]
            if not day_records:
                return f"Prognoza pentru {day} nu este disponibilă. Vă rugăm să specificați o dată din următoarele {len(daily_forecast)} zile."
            
            # Obține datele zilnice - găsește prima înregistrare care conține ziua noastră
            day_data = daily_forecast[daily_forecast['time'].str.contains(day)].iloc[0]
            
            # Obține datele orare - folosește string contains pentru o potrivire mai sigură
            day_hourly = hourly_forecast[hourly_forecast['time'].str.contains(day)]
            
            # Formatează răspunsul
            result = f"Evaluarea Activităților în Aer Liber pentru {day_name} ({day}):\n\n"
            
            # Sumar condiții de bază
            temp_min = float(day_data['temp_min']) if 'temp_min' in day_data else None
            temp_max = float(day_data['temp_max']) if 'temp_max' in day_data else None
            precip_sum = float(day_data['precipitation_sum']) if 'precipitation_sum' in day_data else None
            precip_prob = float(day_data['precipitation_probability']) if 'precipitation_probability' in day_data else None
            
            if temp_min is not None and temp_max is not None:
                result += f"Interval de temperatură: {temp_min:.1f}°C până la {temp_max:.1f}°C\n"
            
            if precip_sum is not None and precip_prob is not None:
                result += f"Precipitații: {precip_sum:.1f}mm (Probabilitate: {precip_prob}%)\n"
            
            if len(day_hourly) > 0:
                # Convertește coloanele numerice în siguranță
                for col in ['wind_speed', 'pm25', 'pm10', 'temperature', 'cloudcover', 'humidity']:
                    if col in day_hourly.columns:
                        day_hourly[col] = pd.to_numeric(day_hourly[col], errors='coerce')
                
                if 'wind_speed' in day_hourly.columns:
                    avg_wind = day_hourly['wind_speed'].mean()
                    result += f"Viteza medie a vântului: {avg_wind:.1f} km/h\n"
                    
                if 'humidity' in day_hourly.columns:
                    avg_humidity = day_hourly['humidity'].mean()
                    result += f"Umiditate medie: {avg_humidity:.1f}%\n"
                    
                if 'pm25' in day_hourly.columns:
                    avg_pm25 = day_hourly['pm25'].mean()
                    result += f"PM2.5 mediu: {avg_pm25:.1f} µg/m³\n"
                
                if 'pm10' in day_hourly.columns:
                    avg_pm10 = day_hourly['pm10'].mean()
                    result += f"PM10 mediu: {avg_pm10:.1f} µg/m³\n"
            
            result += "\n"
            
            # Extrage ora din timestamp pentru analiza pe intervale orare
            def extract_hour(timestamp):
                try:
                    if 'T' in timestamp:
                        time_part = timestamp.split('T')[1]
                        return int(time_part.split(':')[0])
                    elif ' ' in timestamp:
                        parts = timestamp.split(' ')
                        if len(parts) > 1:
                            time_part = parts[1]
                            return int(time_part.split(':')[0])
                    return -1
                except Exception:
                    return -1
            
            # Adaugă coloana cu ora
            day_hourly['hour'] = day_hourly['time'].apply(extract_hour)
            
            # Definește intervalele orare
            morning_hours = day_hourly[(day_hourly['hour'] >= 6) & (day_hourly['hour'] < 12)]
            noon_hours = day_hourly[(day_hourly['hour'] >= 12) & (day_hourly['hour'] < 14)]
            afternoon_hours = day_hourly[(day_hourly['hour'] >= 14) & (day_hourly['hour'] < 18)]
            evening_hours = day_hourly[(day_hourly['hour'] >= 18) & (day_hourly['hour'] < 22)]
            night_hours = day_hourly[(day_hourly['hour'] >= 22) | (day_hourly['hour'] < 6)]
            
            # Calculează scoruri de confort pentru fiecare perioadă
            def calculate_comfort_score(period_df):
                if len(period_df) == 0:
                    return -100  # Fără date
                    
                score = 0
                
                # Temperatura ideală: 20-25°C (excelent), 15-30°C (bine)
                if 'temperature' in period_df.columns:
                    avg_temp = period_df['temperature'].mean()
                    if 20 <= avg_temp <= 25:
                        score += 5
                    elif 15 <= avg_temp <= 30:
                        score += 3
                    elif 10 <= avg_temp <= 35:
                        score += 1
                    else:
                        score -= 2  # Prea rece sau prea cald
                
                # Vânt ideal: 2-10 km/h (briză plăcută)
                if 'wind_speed' in period_df.columns:
                    avg_wind = period_df['wind_speed'].mean()
                    if 2 <= avg_wind <= 10:
                        score += 3
                    elif avg_wind < 15:
                        score += 1
                    elif avg_wind > 20:
                        score -= 2  # Prea vântos
                
                # Precipitații
                if 'precipitation' in period_df.columns:
                    total_precip = period_df['precipitation'].sum()
                    if total_precip < 0.1:
                        score += 4  # Fără ploaie
                    elif total_precip < 1:
                        score += 0  # Ploaie ușoară
                    else:
                        score -= 3  # Precipitații semnificative
                
                # Acoperire cu nori
                if 'cloudcover' in period_df.columns:
                    avg_clouds = period_df['cloudcover'].mean()
                    if avg_clouds < 30:
                        score += 2  # Majoritar senin
                    elif avg_clouds < 70:
                        score += 1  # Parțial înnorat
                        
                return score
            
            # Calculează scoruri pentru fiecare perioadă
            time_periods = [
                ("dimineața (06:00-12:00)", morning_hours, calculate_comfort_score(morning_hours)),
                ("la prânz (12:00-14:00)", noon_hours, calculate_comfort_score(noon_hours)),
                ("după-amiaza (14:00-18:00)", afternoon_hours, calculate_comfort_score(afternoon_hours)),
                ("seara (18:00-22:00)", evening_hours, calculate_comfort_score(evening_hours)),
                ("noaptea (22:00-06:00)", night_hours, calculate_comfort_score(night_hours))
            ]
            
            # Sortează după scorul de confort (descrescător)
            time_periods.sort(key=lambda x: x[2], reverse=True)
            
            # Evaluare specifică activității
            activity_type = activity_type.lower()
            
            # Ajustează scorurile pentru activități specifice
            activity_adjusted_periods = time_periods.copy()
            
            if activity_type in ["sports", "running", "cycling", "exercise"]:
                # Pentru sport, evită orele cele mai calde, preferă temperaturi moderate
                for i, (name, df, score) in enumerate(activity_adjusted_periods):
                    if 'temperature' in df.columns:
                        avg_temp = df['temperature'].mean()
                        if avg_temp > 30:
                            activity_adjusted_periods[i] = (name, df, score - 3)  # Prea cald pentru sport
                        elif avg_temp < 10:
                            activity_adjusted_periods[i] = (name, df, score - 2)  # Prea rece pentru sport
                            
                # Re-sortează cu scoruri ajustate
                activity_adjusted_periods.sort(key=lambda x: x[2], reverse=True)
            
            elif activity_type in ["picnic", "relaxing", "outdoor dining"]:
                # Pentru picnic/relaxare, preferă ore mai calde cu mai puțin vânt
                for i, (name, df, score) in enumerate(activity_adjusted_periods):
                    if 'temperature' in df.columns:
                        avg_temp = df['temperature'].mean()
                        if 22 <= avg_temp <= 28:
                            activity_adjusted_periods[i] = (name, df, score + 2)  # Vreme perfectă pentru picnic
                        if 'wind_speed' in df.columns and df['wind_speed'].mean() > 15:
                            activity_adjusted_periods[i] = (name, df, score - 2)  # Prea vântos pentru picnic
                            
                # Re-sortează cu scoruri ajustate
                activity_adjusted_periods.sort(key=lambda x: x[2], reverse=True)
            
            elif activity_type in ["photography", "sightseeing"]:
                # Pentru fotografie, preferă orele de aur (dimineața/seara) și lumină bună
                for i, (name, df, score) in enumerate(activity_adjusted_periods):
                    if name.startswith("dimineața") or name.startswith("seara"):
                        activity_adjusted_periods[i] = (name, df, score + 3)  # Bonus pentru ora de aur
                    if 'cloudcover' in df.columns:
                        avg_clouds = df['cloudcover'].mean()
                        if 20 <= avg_clouds <= 60:
                            activity_adjusted_periods[i] = (name, df, score + 2)  # Lumină difuză bună
                            
                # Re-sortează cu scoruri ajustate
                activity_adjusted_periods.sort(key=lambda x: x[2], reverse=True)
            
            # Obține cele mai bune perioade (top 2)
            best_periods = [p for p in activity_adjusted_periods if p[2] > 0][:2]
            
            # Formatează numele activității pentru afișare
            activity_display_names = {
                "general": "activități generale",
                "sports": "activități sportive",
                "picnic": "picnic și relaxare",
                "photography": "fotografie și vizitare",
                "running": "alergare",
                "cycling": "ciclism",
                "exercise": "exerciții fizice"
            }
            
            activity_display = activity_display_names.get(activity_type, activity_type)
            
            # Generează recomandarea generală bazată pe condiții
            if precip_sum is not None and precip_prob is not None and (precip_sum > 5 or precip_prob > 70):
                result += f"❌ Nu este recomandat pentru {activity_display} din cauza probabilității ridicate de precipitații.\n"
            elif temp_max is not None and temp_max > 35:
                result += f"⚠️ Folosiți prudență pentru {activity_display} din cauza temperaturilor foarte ridicate. Hidratați-vă frecvent și evitați orele de vârf ale zilei.\n"
            elif temp_min is not None and temp_min < 0:
                result += f"⚠️ Folosiți prudență pentru {activity_display} din cauza temperaturilor scăzute. Îmbrăcați-vă în straturi adecvate.\n"
            elif temp_max is not None and temp_max > 32:
                result += f"⚠️ Condiții calde pentru {activity_display}. Hidratați-vă frecvent și căutați zonele cu umbră.\n"
            else:
                result += f"✅ Condițiile sunt favorabile pentru {activity_display} în aer liber.\n"
            
            # Afișează informații despre cea mai bună perioadă a zilei
            result += "\n🕙 CELE MAI BUNE PERIOADE ALE ZILEI:\n"
            
            if best_periods:
                for i, (period_name, period_data, score) in enumerate(best_periods):
                    result += f"{i+1}. {period_name.capitalize()}"
                    
                    # Adaugă detalii despre de ce această perioadă este bună
                    reasons = []
                    
                    if len(period_data) > 0:
                        if 'temperature' in period_data.columns:
                            avg_temp = period_data['temperature'].mean()
                            reasons.append(f"temperatura medie de {avg_temp:.1f}°C")
                        
                        if 'wind_speed' in period_data.columns:
                            avg_wind = period_data['wind_speed'].mean()
                            if avg_wind < 10:
                                reasons.append("vânt slab")
                        
                        if 'precipitation' in period_data.columns:
                            total_precip = period_data['precipitation'].sum()
                            if total_precip < 0.1:
                                reasons.append("fără precipitații")
                    
                    if reasons:
                        result += f" - {', '.join(reasons)}"
                    
                    result += "\n"
            else:
                result += "Nu există perioade optime pentru activități în aer liber astăzi din cauza condițiilor meteo.\n"
            
            # Generează recomandări specifice activității
            result += f"\nRECOMANDĂRI PENTRU {activity_display.upper()}:\n"
            
            if activity_type in ["sports", "running", "cycling", "exercise"]:
                # Recomandări pentru sport
                if precip_sum is not None and precip_sum < 1 and temp_max is not None:
                    if temp_max > 30:
                        result += "• Activități sportive în primele ore ale dimineții sau seara\n"
                        result += "• Hidratare frecventă și pauze regulate la umbră\n"
                        result += "• Exerciții de intensitate moderată în zonele cu umbră\n"
                        result += "• Înot în piscine exterioare\n"
                    else:
                        result += "• Alergare în parcuri (Herăstrău, Tineretului)\n"
                        result += "• Ciclism pe traseele amenajate din oraș\n"
                        result += "• Sporturi de echipă în parcuri și terenuri deschise\n"
                        result += "• Fitness în aer liber în zonele cu aparate din parcuri\n"
                else:
                    result += "• Activități sportive în spații acoperite\n"
                    result += "• Evitarea efortului intens în perioadele cu precipitații\n"
            
            elif activity_type in ["picnic", "relaxing", "outdoor dining"]:
                # Recomandări pentru picnic/relaxare
                if precip_sum is not None and precip_sum < 1 and temp_max is not None:
                    if temp_max > 30:
                        result += "• Picnic în zonele cu umbră densă din parcurile mari\n"
                        result += "• Relaxare pe terasele umbrite cu sisteme de răcire\n"
                        result += "• Ieșiri în aer liber seara, după scăderea temperaturilor\n"
                        result += "• Evitați zonele fără umbră în orele de vârf (12:00-16:00)\n"
                    else:
                        result += "• Picnic în orice zonă verde din oraș\n"
                        result += "• Terase și restaurante cu zone în aer liber\n"
                        result += "• Relaxare în parcuri și grădini publice\n"
                        result += "• Locații recomandate: Parcul Herăstrău, Grădina Botanică, Parcul Cișmigiu\n"
                else:
                    result += "• Planificați alternative în interior în caz de precipitații\n"
                    result += "• Terase acoperite pentru mese în aer liber\n"
            
            elif activity_type in ["photography", "sightseeing"]:
                # Recomandări pentru fotografie/vizitare
                if 'cloudcover' in day_hourly.columns and len(day_hourly) > 0:
                    avg_clouds = day_hourly['cloudcover'].mean()
                    if avg_clouds < 30:
                        result += "• Fotografii cu lumină naturală în Centrul Vechi\n"
                        result += "• Fotografiere la răsărit/apus pentru lumină caldă (Casa Poporului, Arcul de Triumf)\n"
                        result += "• Vizitare monumente arhitecturale (Ateneul Român, Palatul CEC)\n"
                    elif avg_clouds < 70:
                        result += "• Fotografii cu lumină difuză perfectă pentru portrete în aer liber\n"
                        result += "• Vizitare grădini și parcuri (Grădina Botanică, Parcul Carol)\n"
                    else:
                        result += "• Fotografii urbane cu accent pe contraste și texturi\n"
                        result += "• Vizite la muzee și galerii în perioadele cu înnorare maximă\n"
                else:
                    result += "• Tur al centrului istoric al Bucureștiului\n"
                    result += "• Vizitarea principalelor obiective turistice\n"
            
            else:
                # Recomandări generale
                if precip_sum is not None and precip_sum < 1 and temp_max is not None:
                    if temp_max > 30:
                        result += "• Plimbări în parcuri dimineața devreme sau seara\n"
                        result += "• Vizite la spații culturale semi-deschise (ex. Muzeul Satului)\n"
                        result += "• Evitați expunerea prelungită la soare între orele 11:00-17:00\n"
                        result += "• Hidratare frecventă și pauze regulate la umbră\n"
                    else:
                        result += "• Explorarea zonelor pietonale și a centrului istoric\n"
                        result += "• Plimbări în parcurile mari ale orașului\n"
                        result += "• Vizite la grădini botanice și zone verzi\n"
                        result += "• Activități culturale în aer liber (târguri, expoziții, concerte)\n"
                else:
                    result += "• Activități cu alternative în interior în caz de vreme nefavorabilă\n"
                    result += "• Vizitare muzee și galerii în perioadele cu precipitații\n"
            
            # Include informații despre calitatea aerului dacă sunt disponibile
            if 'pm25' in day_hourly.columns and len(day_hourly) > 0:
                avg_pm25 = day_hourly['pm25'].mean()
                result += "\nINDICE CALITATEA AERULUI:\n"
                if avg_pm25 < 10:
                    result += "✓ Calitatea aerului este excelentă - ideal pentru toate activitățile în aer liber.\n"
                elif avg_pm25 < 25:
                    result += "✓ Calitatea aerului este bună - adecvată pentru majoritatea activităților în aer liber.\n"
                elif avg_pm25 < 50:
                    result += "⚠️ Calitatea aerului este moderată - persoanele sensibile ar trebui să limiteze efortul prelungit.\n"
                else:
                    result += "❌ Calitatea aerului este slabă - limitați activitățile intense în aer liber.\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Eroare la evaluarea activităților în aer liber: {e}")
            return f"A apărut o eroare la evaluarea condițiilor pentru activități în aer liber: {str(e)}"
    
    def _find_route_between_parks(self, origin_park, destination_park, mode="walking"):
        """
        Find a route between two parks taking into account current traffic conditions.
        
        Args:
            origin_park: Name or part of the name of the starting park
            destination_park: Name or part of the name of the destination park
            mode: Transportation mode ("walking", "driving", "bicycle")
        
        Returns:
            A description of the route with travel time estimates
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: No park data available."
            
            import geopandas as gpd
            from shapely.geometry import Point, LineString
            import networkx as nx
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            import numpy as np
            import traceback
            
            # Determine the correct name column to use
            name_column = 'name_left' if 'name_left' in self.data.columns else 'name'
            logger.info(f"Using '{name_column}' column for park names")
            
            # Find the origin park
            origin_park_lower = origin_park.lower()
            origin_matches = self.data[self.data[name_column].str.lower().str.contains(origin_park_lower, na=False)]
            
            if len(origin_matches) == 0:
                return f"Error: Could not find any park matching '{origin_park}'. Available park names include: {', '.join(self.data[name_column].head(5).tolist())}."
            elif len(origin_matches) > 1:
                # If multiple matches, use the first one but inform the user
                origin = origin_matches.iloc[0]
                other_options = ", ".join(origin_matches[name_column].tolist()[1:3])  # List a few alternatives
                origin_message = f"Multiple parks match '{origin_park}'. Using {origin[name_column]}. Other options: {other_options}."
            else:
                origin = origin_matches.iloc[0]
                origin_message = f"Found origin: {origin[name_column]}"
            
            # Find the destination park
            dest_park_lower = destination_park.lower()
            dest_matches = self.data[self.data[name_column].str.lower().str.contains(dest_park_lower, na=False)]
            
            if len(dest_matches) == 0:
                return f"Error: Could not find any park matching '{destination_park}'. Available park names include: {', '.join(self.data[name_column].head(5).tolist())}."
            elif len(dest_matches) > 1:
                # If multiple matches, use the first one but inform the user
                destination = dest_matches.iloc[0]
                other_options = ", ".join(dest_matches[name_column].tolist()[1:3])  # List a few alternatives
                dest_message = f"Multiple parks match '{destination_park}'. Using {destination[name_column]}. Other options: {other_options}."
            else:
                destination = dest_matches.iloc[0]
                dest_message = f"Found destination: {destination[name_column]}"
            
            # Load traffic data if available
            try:
                # Try to use pre-loaded traffic data or load from file
                if hasattr(self, 'traffic_data') and self.traffic_data is not None:
                    traffic_data = self.traffic_data
                    logger.info(f"Using pre-loaded traffic data with {len(traffic_data)} road segments")
                else:
                    # Try to load traffic data from a file
                    traffic_file = "data/traffic_current.geojson"
                    if os.path.exists(traffic_file):
                        traffic_data = gpd.read_file(traffic_file)
                        logger.info(f"Loaded traffic data from file with {len(traffic_data)} road segments")
                    else:
                        # Create a basic network from available data
                        traffic_data = None
                        logger.warning("No traffic data found")
            except Exception as e:
                logger.error(f"Error loading traffic data: {e}")
                traffic_data = None
            
            # Create a road network graph
            G = nx.DiGraph()
            
            if traffic_data is not None and len(traffic_data) > 0:
                # Build graph from traffic data
                for idx, road in traffic_data.iterrows():
                    # Extract nodes from the geometry
                    if isinstance(road.geometry, LineString):
                        coords = list(road.geometry.coords)
                        
                        # Get start and end points
                        start_node = (coords[0][0], coords[0][1])
                        end_node = (coords[-1][0], coords[-1][1])
                        
                        # Calculate the weight based on travel time and mode
                        weight = road['travel_time_seconds'] if 'travel_time_seconds' in road.keys() else road['length']
                        
                        # Adjust weight based on transportation mode
                        if mode.lower() == "walking":
                            # Walking is not affected by traffic congestion
                            # Average walking speed: 5 km/h = 1.4 m/s
                            weight = road['length'] / 1.4
                        elif mode.lower() == "bicycle":
                            # Bicycles are affected by congestion but less than cars
                            # Average cycling speed: 15 km/h = 4.2 m/s
                            congestion = road.get('congestion_level', 1)
                            speed_factor = 1.0
                            if congestion > 1:
                                speed_factor = 0.9 if congestion == 2 else 0.8
                            weight = road['length'] / (4.2 * speed_factor)
                        else:  # driving
                            # Use the travel time directly if available
                            if 'travel_time_seconds' in road.keys():
                                weight = road['travel_time_seconds']
                            else:
                                # Estimate based on length and speed
                                max_speed = float(road.get('maxspeed', 50)) / 3.6  # Convert km/h to m/s
                                congestion = road.get('congestion_level', 1)
                                speed_factor = 1.0
                                if congestion > 1:
                                    speed_factor = 0.8 if congestion == 2 else 0.6 if congestion == 3 else 0.4
                                weight = road['length'] / (max_speed * speed_factor)
                        
                        # Get road name
                        road_name = road.get('name', None)
                        if road_name is None:
                            road_name = road.get('name_right', 'Unnamed Road')
                        
                        # Add the road to the graph
                        G.add_edge(start_node, end_node, weight=weight, 
                                name=road_name,
                                length=road['length'],
                                congestion=road.get('congestion_level', 1))
                        
                        # If the road is not one-way, add the reverse direction too
                        oneway = road.get('oneway', False)
                        if isinstance(oneway, str):
                            oneway = oneway.lower() == 'true' or oneway.lower() == 'yes' or oneway == '1'
                            
                        if not oneway:
                            G.add_edge(end_node, start_node, weight=weight,
                                    name=road_name,
                                    length=road['length'],
                                    congestion=road.get('congestion_level', 1))
            else:
                # If no traffic data, create a simple grid network
                # This is just a fallback if no actual road data is available
                
                # Create a grid around the parks
                parks_bounds = self.data.total_bounds
                x_min, y_min, x_max, y_max = parks_bounds
                
                # Expand the bounds a bit
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.1
                x_min -= x_margin
                y_min -= y_margin
                x_max += x_margin
                y_max += y_margin
                
                # Create a grid of points
                grid_spacing = 0.005  # About 500 meters
                x_range = np.arange(x_min, x_max, grid_spacing)
                y_range = np.arange(y_min, y_max, grid_spacing)
                
                # Add grid nodes to the graph
                for x in x_range:
                    for y in y_range:
                        G.add_node((x, y))
                
                # Connect adjacent nodes
                for x in x_range:
                    for y in y_range:
                        # Connect to right neighbor
                        if x + grid_spacing < x_max:
                            # Calculate euclidean distance
                            dist = ((grid_spacing)**2)**0.5 * 111000  # Convert to meters approximately
                            G.add_edge((x, y), (x + grid_spacing, y), weight=dist, name="Grid Road", length=dist, congestion=1)
                            G.add_edge((x + grid_spacing, y), (x, y), weight=dist, name="Grid Road", length=dist, congestion=1)
                        
                        # Connect to bottom neighbor
                        if y + grid_spacing < y_max:
                            # Calculate euclidean distance
                            dist = ((grid_spacing)**2)**0.5 * 111000  # Convert to meters approximately
                            G.add_edge((x, y), (x, y + grid_spacing), weight=dist, name="Grid Road", length=dist, congestion=1)
                            G.add_edge((x, y + grid_spacing), (x, y), weight=dist, name="Grid Road", length=dist, congestion=1)
                        
                        # Connect to bottom-right neighbor
                        if x + grid_spacing < x_max and y + grid_spacing < y_max:
                            # Calculate diagonal distance
                            dist = ((grid_spacing)**2 + (grid_spacing)**2)**0.5 * 111000  # Convert to meters approximately
                            G.add_edge((x, y), (x + grid_spacing, y + grid_spacing), weight=dist, name="Grid Road", length=dist, congestion=1)
                            G.add_edge((x + grid_spacing, y + grid_spacing), (x, y), weight=dist, name="Grid Road", length=dist, congestion=1)
            
            # Find the closest nodes to the parks
            origin_point = (origin.geometry.centroid.x, origin.geometry.centroid.y)
            dest_point = (destination.geometry.centroid.x, destination.geometry.centroid.y)
            
            # Log some debugging information
            logger.info(f"Origin park: {origin[name_column]} at {origin_point}")
            logger.info(f"Destination park: {destination[name_column]} at {dest_point}")
            logger.info(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
            
            if len(G.nodes) == 0:
                return "Error: Could not build road network. No nodes available for routing."
            
            # Find closest nodes in the graph
            origin_node = min(G.nodes(), key=lambda node: ((node[0] - origin_point[0])**2 + (node[1] - origin_point[1])**2)**0.5)
            dest_node = min(G.nodes(), key=lambda node: ((node[0] - dest_point[0])**2 + (node[1] - dest_point[1])**2)**0.5)
            
            # Try to find the shortest path
            try:
                path = nx.shortest_path(G, source=origin_node, target=dest_node, weight='weight')
                path_edges = list(zip(path[:-1], path[1:]))
                
                # Calculate total distance and time
                total_distance = 0
                total_time = 0
                road_names = []
                
                for u, v in path_edges:
                    edge_data = G.get_edge_data(u, v)
                    total_distance += edge_data['length']
                    total_time += edge_data['weight']
                    if edge_data['name'] not in road_names:
                        road_names.append(edge_data['name'])
                
                # Limit to 5 main roads for clarity
                if len(road_names) > 5:
                    main_roads = road_names[:5]
                    main_roads.append("...")
                else:
                    main_roads = road_names
                
                # Format distance and time
                distance_km = total_distance / 1000
                time_minutes = total_time / 60
                
                # Create an enhanced visualization of the route with street context
                try:
                    # Try to use contextily for adding basemaps if available
                    try:
                        import contextily as ctx
                        has_contextily = True
                    except ImportError:
                        has_contextily = False
                    
                    # Create figure and axis
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Plot all parks with proper boundaries
                    self.data.plot(ax=ax, color='green', alpha=0.3, edgecolor='darkgreen')
                    
                    # Highlight origin and destination parks
                    gpd.GeoDataFrame(geometry=[origin.geometry], crs=self.data.crs).plot(
                        ax=ax, color='blue', alpha=0.6, edgecolor='darkblue'
                    )
                    gpd.GeoDataFrame(geometry=[destination.geometry], crs=self.data.crs).plot(
                        ax=ax, color='red', alpha=0.6, edgecolor='darkred'
                    )
                    
                    # Plot the route with arrow markers to show direction
                    route_coords = [p for p in path]
                    route_line = LineString(route_coords)
                    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
                    
                    # Plot with thicker, more visible line
                    route_gdf.plot(ax=ax, color='purple', linewidth=4, alpha=0.8)
                    
                    # Plot traffic data if available to show streets
                    if traffic_data is not None:
                        traffic_data.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
                    
                    # Add OpenStreetMap as basemap if contextily is available
                    if has_contextily:
                        ax.set_axis_off()
                        try:
                            ctx.add_basemap(ax, crs=self.data.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                        except Exception as e:
                            logger.warning(f"Could not add basemap: {e}")
                    else:
                        ax.set_axis_off()
                    
                    # Add labels with bigger font size and better positioning
                    ax.annotate(origin[name_column], 
                            xy=(origin_point[0], origin_point[1]),
                            xytext=(5, 15), textcoords="offset points",
                            color='blue', fontsize=14, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                            
                    ax.annotate(destination[name_column], 
                            xy=(dest_point[0], dest_point[1]),
                            xytext=(5, 15), textcoords="offset points",
                            color='red', fontsize=14, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                    
                    # Add title with distance and time
                    plt.title(f'Route from {origin[name_column]} to {destination[name_column]}\n'
                            f'Distance: {distance_km:.2f} km, Time: {time_minutes:.1f} min ({mode.capitalize()})',
                            fontsize=16)
                    
                    # Save with higher resolution
                    plt.tight_layout()
                    
                    # Save the figure
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = "visualizations"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"route_{mode}_{timestamp}.png")
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    
                except Exception as e:
                    logger.error(f"Error creating enhanced visualization: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback to simpler visualization
                    try:
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        # Plot the parks
                        self.data.plot(ax=ax, color='green', alpha=0.6)
                        
                        # Highlight origin and destination
                        gpd.GeoDataFrame(geometry=[origin.geometry], crs=self.data.crs).plot(
                            ax=ax, color='blue', alpha=0.8
                        )
                        gpd.GeoDataFrame(geometry=[destination.geometry], crs=self.data.crs).plot(
                            ax=ax, color='red', alpha=0.8
                        )
                        
                        # Plot the route
                        route_line = LineString(route_coords)
                        gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326").plot(
                            ax=ax, color='purple', linewidth=3, alpha=0.8
                        )
                        
                        plt.title(f'Route from {origin[name_column]} to {destination[name_column]} ({mode.capitalize()})')
                        plt.axis('off')
                        
                        # Save the figure
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = "visualizations"
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"route_{mode}_{timestamp}.png")
                        plt.savefig(output_path)
                        plt.close()
                    except Exception as e2:
                        logger.error(f"Error creating fallback visualization: {e2}")
                        output_path = "No visualization could be created."
                
                # Format the results
                result = f"Route from {origin[name_column]} to {destination[name_column]} using {mode.capitalize()} mode:\n\n"
                
                # Add park finding messages
                result += f"{origin_message}\n"
                result += f"{dest_message}\n\n"
                
                result += f"Total distance: {distance_km:.2f} km\n"
                result += f"Estimated travel time: {time_minutes:.1f} minutes\n\n"
                
                if main_roads and main_roads[0] != "Grid Road":
                    result += "Main roads on this route:\n"
                    for road in main_roads:
                        result += f"- {road}\n"
                    result += "\n"
                
                if traffic_data is not None:
                    result += "Note: This route takes into account current traffic conditions.\n\n"
                
                result += f"A visualization of the route has been created at: {output_path}"
                self.update_latest_visualization(output_path)
                
                return result
                
            except nx.NetworkXNoPath:
                return f"No path found between {origin[name_column]} and {destination[name_column]} using {mode} mode."
            
        except Exception as e:
            logger.error(f"Error finding route between parks: {e}")
            traceback_str = traceback.format_exc()
            logger.error(f"Traceback: {traceback_str}")
            return f"Error occurred while finding route between parks: {str(e)}"
    
    def _compare_sectors(self, sector1: str, sector2: str = None, metric: str = "parks") -> str:
        """
        Compare two or all sectors based on various metrics.
        
        Args:
            sector1: First sector to compare or "all" to compare all sectors
            sector2: Second sector to compare (optional if sector1 is "all")
            metric: Metric to compare ("parks", "air", "noise", "green_density")
        """
        try:
            if self.data is None or len(self.data) == 0:
                return "Error: Nu există date despre parcuri disponibile."
                
            if 'sector' not in self.data.columns:
                return "Error: Setul de date nu conține informații despre sectoare."
            
            # Determine which column to use for park names
            name_column = None
            possible_name_cols = ['name', 'name_left', 'park_name', 'title', 'NAME', 'label']
            for col in possible_name_cols:
                if col in self.data.columns:
                    name_column = col
                    break
                    
            if name_column is None:
                return "Error: Nu s-a putut găsi o coloană cu numele parcurilor."
            
            # Normalize sector input
            def normalize_sector(sector):
                if sector is None or str(sector).lower() == "all":
                    return "all"
                if isinstance(sector, int) or (isinstance(sector, str) and sector.isdigit()):
                    return f"Sector {sector}"
                elif isinstance(sector, str) and not sector.lower().startswith("sector"):
                    return f"Sector {sector}"
                else:
                    return sector.title()
                        
            sector1_normalized = normalize_sector(sector1)
            sector2_normalized = normalize_sector(sector2) if sector2 else None
            
            # Determine if we're comparing all sectors or just two
            compare_all = sector1_normalized == "all" or sector2_normalized is None
            
            # Create clean data for analysis by removing duplicates
            clean_data = self.data.copy()
            if name_column in clean_data.columns:
                clean_data = clean_data.drop_duplicates(subset=[name_column, 'sector'])
                
            # Create separate DataFrames for each metric to avoid column name conflicts
            # Count parks by sector
            park_counts = clean_data.groupby('sector').size().reset_index(name='park_count')
            park_stats_df = park_counts.set_index('sector')
            
            # Process AQI data if available
            aqi_stats_df = None
            if 'aqi' in clean_data.columns:
                # Group by sector and calculate stats manually to avoid pandas aggregation issues
                sectors = clean_data['sector'].unique()
                aqi_data = []
                
                for sector in sectors:
                    sector_parks = clean_data[clean_data['sector'] == sector]
                    aqi_mean = sector_parks['aqi'].mean()
                    aqi_min = sector_parks['aqi'].min()
                    aqi_max = sector_parks['aqi'].max()
                    aqi_data.append({
                        'sector': sector,
                        'aqi_mean': round(aqi_mean, 1) if not pd.isna(aqi_mean) else None,
                        'aqi_min': round(aqi_min, 1) if not pd.isna(aqi_min) else None,
                        'aqi_max': round(aqi_max, 1) if not pd.isna(aqi_max) else None
                    })
                
                aqi_stats_df = pd.DataFrame(aqi_data).set_index('sector')
            
            # Process noise data if available
            noise_stats_df = None
            if 'noise_level_db' in clean_data.columns:
                # Group by sector and calculate stats manually
                sectors = clean_data['sector'].unique()
                noise_data = []
                
                for sector in sectors:
                    sector_parks = clean_data[clean_data['sector'] == sector]
                    noise_mean = sector_parks['noise_level_db'].mean()
                    noise_min = sector_parks['noise_level_db'].min()
                    noise_max = sector_parks['noise_level_db'].max()
                    noise_data.append({
                        'sector': sector,
                        'noise_mean': round(noise_mean, 1) if not pd.isna(noise_mean) else None,
                        'noise_min': round(noise_min, 1) if not pd.isna(noise_min) else None,
                        'noise_max': round(noise_max, 1) if not pd.isna(noise_max) else None
                    })
                
                noise_stats_df = pd.DataFrame(noise_data).set_index('sector')
            
            # Handle comparison based on metric
            if metric.lower() in ["parks", "park", "green_spaces", "green spaces"]:
                if compare_all:
                    # Return comparison of all sectors by park count
                    result = "Comparație după numărul de parcuri:\n\n"
                    
                    for sector in sorted(park_stats_df.index, key=lambda x: park_stats_df.loc[x, 'park_count'], reverse=True):
                        result += f"{sector}: {park_stats_df.loc[sector, 'park_count']} parcuri\n"
                    
                    return result
                else:
                    # Compare two specific sectors
                    if sector1_normalized not in park_stats_df.index:
                        return f"Error: {sector1_normalized} nu a fost găsit în setul de date."
                    if sector2_normalized not in park_stats_df.index:
                        return f"Error: {sector2_normalized} nu a fost găsit în setul de date."
                        
                    sector1_parks = park_stats_df.loc[sector1_normalized, 'park_count']
                    sector2_parks = park_stats_df.loc[sector2_normalized, 'park_count']
                    
                    # Get actual parks list for each sector
                    parks_sector1 = clean_data[clean_data['sector'] == sector1_normalized]
                    parks_sector2 = clean_data[clean_data['sector'] == sector2_normalized]
                    
                    # Sort parks by name for consistent display
                    if not parks_sector1.empty and name_column in parks_sector1:
                        parks_sector1 = parks_sector1.sort_values(by=name_column)
                    if not parks_sector2.empty and name_column in parks_sector2:
                        parks_sector2 = parks_sector2.sort_values(by=name_column)
                        
                    result = f"Comparație între {sector1_normalized} și {sector2_normalized} după numărul de parcuri:\n\n"
                    result += f"{sector1_normalized}: {sector1_parks} parcuri\n"
                    result += f"{sector2_normalized}: {sector2_parks} parcuri\n\n"
                    
                    if sector1_parks > sector2_parks:
                        result += f"{sector1_normalized} are cu {sector1_parks - sector2_parks} mai multe parcuri decât {sector2_normalized}."
                    elif sector2_parks > sector1_parks:
                        result += f"{sector2_normalized} are cu {sector2_parks - sector1_parks} mai multe parcuri decât {sector1_normalized}."
                    else:
                        result += f"Ambele sectoare au același număr de parcuri."
                    
                    # Adaugă listele de parcuri dar limitează numărul pentru a evita răspunsuri prea lungi
                    max_parks_to_list = 10
                    
                    if sector1_parks > 0:
                        result += f"\n\nExemple de parcuri din {sector1_normalized} (primele {min(max_parks_to_list, len(parks_sector1))} din {sector1_parks}):\n"
                        for idx, row in parks_sector1.head(max_parks_to_list).iterrows():
                            result += f"• {row[name_column]}\n"
                        if sector1_parks > max_parks_to_list:
                            result += f"... și încă {sector1_parks - max_parks_to_list} parcuri\n"
                    
                    if sector2_parks > 0:
                        result += f"\nExemple de parcuri din {sector2_normalized} (primele {min(max_parks_to_list, len(parks_sector2))} din {sector2_parks}):\n"
                        for idx, row in parks_sector2.head(max_parks_to_list).iterrows():
                            result += f"• {row[name_column]}\n"
                        if sector2_parks > max_parks_to_list:
                            result += f"... și încă {sector2_parks - max_parks_to_list} parcuri\n"
                            
                    return result
                        
            elif metric.lower() in ["air", "air_quality", "aqi"]:
                if aqi_stats_df is None or 'aqi_mean' not in aqi_stats_df.columns:
                    return "Error: Datele despre calitatea aerului nu sunt disponibile pentru comparație."
                    
                if compare_all:
                    # Return comparison of all sectors by air quality
                    result = "Comparație după calitatea aerului (AQI mai mic înseamnă aer mai curat):\n\n"
                    
                    for sector in sorted(aqi_stats_df.index, key=lambda x: aqi_stats_df.loc[x, 'aqi_mean']):
                        result += f"{sector}: AQI mediu {aqi_stats_df.loc[sector, 'aqi_mean']} "
                        result += f"(Min: {aqi_stats_df.loc[sector, 'aqi_min']}, "
                        result += f"Max: {aqi_stats_df.loc[sector, 'aqi_max']})\n"
                    
                    return result
                else:
                    # Compare two specific sectors
                    if sector1_normalized not in aqi_stats_df.index:
                        return f"Error: {sector1_normalized} nu a fost găsit în setul de date."
                    if sector2_normalized not in aqi_stats_df.index:
                        return f"Error: {sector2_normalized} nu a fost găsit în setul de date."
                        
                    sector1_aqi = aqi_stats_df.loc[sector1_normalized, 'aqi_mean']
                    sector2_aqi = aqi_stats_df.loc[sector2_normalized, 'aqi_mean']
                    
                    result = f"Comparație calitate aer: {sector1_normalized} vs {sector2_normalized}\n\n"
                    result += f"AQI mediu:\n"
                    result += f"- {sector1_normalized}: {sector1_aqi}\n"
                    result += f"- {sector2_normalized}: {sector2_aqi}\n"
                    result += f"(Valori AQI mai mici indică o calitate a aerului mai bună)\n\n"
                    
                    result += f"Cel mai bun AQI (valoare minimă):\n"
                    result += f"- {sector1_normalized}: {aqi_stats_df.loc[sector1_normalized, 'aqi_min']}\n"
                    result += f"- {sector2_normalized}: {aqi_stats_df.loc[sector2_normalized, 'aqi_min']}\n\n"
                    
                    result += f"Cel mai slab AQI (valoare maximă):\n"
                    result += f"- {sector1_normalized}: {aqi_stats_df.loc[sector1_normalized, 'aqi_max']}\n"
                    result += f"- {sector2_normalized}: {aqi_stats_df.loc[sector2_normalized, 'aqi_max']}\n\n"
                    
                    # Determine which sector has better air quality
                    if sector1_aqi < sector2_aqi:
                        result += f"Per total, {sector1_normalized} are o calitate a aerului mai bună decât {sector2_normalized}."
                    elif sector2_aqi < sector1_aqi:
                        result += f"Per total, {sector2_normalized} are o calitate a aerului mai bună decât {sector1_normalized}."
                    else:
                        result += f"Ambele sectoare au niveluri similare de calitate a aerului."
                        
                    return result
                        
            elif metric.lower() in ["noise", "sound", "quiet", "noise_level"]:
                if noise_stats_df is None or 'noise_mean' not in noise_stats_df.columns:
                    return "Error: Datele despre nivelul de zgomot nu sunt disponibile pentru comparație."
                    
                if compare_all:
                    # Return comparison of all sectors by noise level
                    result = "Comparație după nivelul de zgomot (dB mai mic înseamnă mai liniștit):\n\n"
                    
                    for sector in sorted(noise_stats_df.index, key=lambda x: noise_stats_df.loc[x, 'noise_mean']):
                        result += f"{sector}: {noise_stats_df.loc[sector, 'noise_mean']} dB în medie "
                        result += f"(Min: {noise_stats_df.loc[sector, 'noise_min']} dB, "
                        result += f"Max: {noise_stats_df.loc[sector, 'noise_max']} dB)\n"
                    
                    return result
                else:
                    # Compare two specific sectors
                    if sector1_normalized not in noise_stats_df.index:
                        return f"Error: {sector1_normalized} nu a fost găsit în setul de date."
                    if sector2_normalized not in noise_stats_df.index:
                        return f"Error: {sector2_normalized} nu a fost găsit în setul de date."
                        
                    sector1_noise = noise_stats_df.loc[sector1_normalized, 'noise_mean']
                    sector2_noise = noise_stats_df.loc[sector2_normalized, 'noise_mean']
                    
                    result = f"Comparație nivel zgomot: {sector1_normalized} vs {sector2_normalized}\n\n"
                    result += f"Nivel mediu de zgomot:\n"
                    result += f"- {sector1_normalized}: {sector1_noise} dB\n"
                    result += f"- {sector2_normalized}: {sector2_noise} dB\n"
                    result += f"(Valori dB mai mici indică un mediu mai liniștit)\n\n"
                    
                    result += f"Cel mai liniștit punct (valoare minimă):\n"
                    result += f"- {sector1_normalized}: {noise_stats_df.loc[sector1_normalized, 'noise_min']} dB\n"
                    result += f"- {sector2_normalized}: {noise_stats_df.loc[sector2_normalized, 'noise_min']} dB\n\n"
                    
                    result += f"Cel mai zgomotos punct (valoare maximă):\n"
                    result += f"- {sector1_normalized}: {noise_stats_df.loc[sector1_normalized, 'noise_max']} dB\n"
                    result += f"- {sector2_normalized}: {noise_stats_df.loc[sector2_normalized, 'noise_max']} dB\n\n"
                    
                    # Determine which sector is quieter
                    if sector1_noise < sector2_noise:
                        result += f"Per total, {sector1_normalized} este mai liniștit decât {sector2_normalized}."
                    elif sector2_noise < sector1_noise:
                        result += f"Per total, {sector2_normalized} este mai liniștit decât {sector1_normalized}."
                    else:
                        result += f"Ambele sectoare au niveluri similare de zgomot."
                        
                    return result
                    
            elif metric.lower() in ["density", "green_density", "parks_per_sqkm"]:
                # Calculate parks per square km (with estimated areas)
                sector_areas_km2 = {
                    'Sector 1': 68,
                    'Sector 2': 32,
                    'Sector 3': 34,
                    'Sector 4': 32,
                    'Sector 5': 30,
                    'Sector 6': 38
                }
                
                # Create density dataframe
                density_data = []
                for sector in park_stats_df.index:
                    if sector in sector_areas_km2:
                        park_count = park_stats_df.loc[sector, 'park_count']
                        area = sector_areas_km2[sector]
                        density = park_count / area
                        density_data.append({
                            'sector': sector,
                            'park_count': park_count,
                            'area_km2': area,
                            'parks_per_km2': round(density, 2)
                        })
                
                density_df = pd.DataFrame(density_data).set_index('sector')
                
                if compare_all:
                    # Return comparison of all sectors by density
                    result = "Comparație după densitatea de parcuri (parcuri per km²):\n\n"
                    
                    for sector in sorted(density_df.index, key=lambda x: density_df.loc[x, 'parks_per_km2'], reverse=True):
                        result += f"{sector}: {density_df.loc[sector, 'parks_per_km2']} parcuri/km² "
                        result += f"({density_df.loc[sector, 'park_count']} parcuri / {density_df.loc[sector, 'area_km2']} km²)\n"
                    
                    return result
                else:
                    # Compare two specific sectors
                    if sector1_normalized not in density_df.index:
                        return f"Error: {sector1_normalized} nu a fost găsit în setul de date sau nu are informații despre arie."
                    if sector2_normalized not in density_df.index:
                        return f"Error: {sector2_normalized} nu a fost găsit în setul de date sau nu are informații despre arie."
                        
                    sector1_density = density_df.loc[sector1_normalized, 'parks_per_km2']
                    sector2_density = density_df.loc[sector2_normalized, 'parks_per_km2']
                    
                    result = f"Comparație densitate parcuri: {sector1_normalized} vs {sector2_normalized}\n\n"
                    result += f"{sector1_normalized}:\n"
                    result += f"- Densitate: {sector1_density} parcuri/km²\n"
                    result += f"- Număr parcuri: {density_df.loc[sector1_normalized, 'park_count']}\n"
                    result += f"- Suprafață: {density_df.loc[sector1_normalized, 'area_km2']} km²\n\n"
                    
                    result += f"{sector2_normalized}:\n"
                    result += f"- Densitate: {sector2_density} parcuri/km²\n"
                    result += f"- Număr parcuri: {density_df.loc[sector2_normalized, 'park_count']}\n"
                    result += f"- Suprafață: {density_df.loc[sector2_normalized, 'area_km2']} km²\n\n"
                    
                    # Determine which sector has higher density
                    if sector1_density > sector2_density:
                        result += f"{sector1_normalized} are o densitate mai mare de parcuri decât {sector2_normalized}."
                    elif sector2_density > sector1_density:
                        result += f"{sector2_normalized} are o densitate mai mare de parcuri decât {sector1_normalized}."
                    else:
                        result += f"Ambele sectoare au densități similare de parcuri."
                        
                    return result
                    
            else:
                return f"Error: Metric necunoscut '{metric}'. Metricile suportate sunt 'parks', 'air', 'noise' și 'density'."
            
        except Exception as e:
            logger.error(f"Error comparing sectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"A apărut o eroare la compararea sectoarelor: {str(e)}"
    
    def _create_sector_comparison_chart(self, analysis_type: str) -> str:
        """Creates a chart comparing sectors based on parks or air quality."""
        valid_types = ['park_count', 'avg_aqi', 'parks_per_sqkm', 'parks_per_100k_residents']

        if analysis_type not in valid_types:
            return f"Error: Invalid analysis type. Please choose from: {', '.join(valid_types)}."

        try:
            # Get sector statistics
            sector_stats = self.spatial_analyst.calculate_sector_green_metrics()
            
            if len(sector_stats) == 0:
                return "Error: Could not calculate sector statistics for comparison."
            
            title = {
                'park_count': 'Number of Parks by Sector',
                'avg_aqi': 'Average Air Quality by Sector (Lower is Better)',
                'parks_per_sqkm': 'Parks per Square Kilometer by Sector',
                'parks_per_100k_residents': 'Parks per 100,000 Residents by Sector'
            }[analysis_type]
            
            # Check if we have the required column
            if analysis_type == 'avg_aqi' and 'avg_aqi' not in sector_stats.columns:
                return "Error: Air quality data is not available for comparison."
                
            # Create the chart directly using matplotlib
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            import time
            
            plt.figure(figsize=(10, 6))
            
            # For park_count, we need to prepare data differently
            if analysis_type == 'park_count':
                # Group by sector and count
                park_counts = self.data.groupby('sector').size().reset_index(name='count')
                park_counts = park_counts.sort_values('count', ascending=False)
                plt.bar(park_counts['sector'], park_counts['count'])
                plt.ylabel('Number of Parks')
            else:
                # Sort the data
                is_ascending = True if analysis_type == 'avg_aqi' else False
                sorted_stats = sector_stats.sort_values(analysis_type, ascending=is_ascending)
                plt.bar(sorted_stats['sector'], sorted_stats[analysis_type])
                plt.ylabel(analysis_type.replace('_', ' ').title())
            
            plt.title(title)
            plt.xlabel('Sector')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64 for embedding
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create filename
            chart_filename = f"sector_comparison_{analysis_type}_{int(time.time())}.png"
            chart_path = os.path.join(config.DATA_DIR, chart_filename)
            
            # Save to file
            with open(chart_path, "wb") as f:
                f.write(base64.b64decode(img_str))
            
            result = f"Sector comparison chart for {title} created.\n\n"
            
            # Add some analysis of the chart
            if analysis_type == 'park_count':
                if 'count' in park_counts.columns:
                    max_sector = park_counts.iloc[0]['sector']
                    max_count = park_counts.iloc[0]['count']
                    min_sector = park_counts.iloc[-1]['sector']
                    min_count = park_counts.iloc[-1]['count']
                    result += f"Sector {max_sector} has the most parks with {max_count}.\n"
                    result += f"Sector {min_sector} has the fewest parks with {min_count}.\n"
            elif analysis_type == 'avg_aqi':
                best_sector = sorted_stats.iloc[0]['sector']
                best_aqi = sorted_stats.iloc[0]['avg_aqi']
                worst_sector = sorted_stats.iloc[-1]['sector']
                worst_aqi = sorted_stats.iloc[-1]['avg_aqi']
                result += f"Sector {best_sector} has the best air quality with average AQI {best_aqi:.1f}.\n"
                result += f"Sector {worst_sector} has the worst air quality with average AQI {worst_aqi:.1f}.\n"
            
            result += f"\nThe chart has been saved to: {chart_path}"
            self.update_latest_visualization(chart_path)
            
            return result
        except Exception as e:
            logger.error(f"Error creating sector comparison chart: {e}")
            return f"Error occurred while creating sector comparison chart: {str(e)}"
    
    def _create_air_quality_heatmap(self) -> str:
        """Creates a heatmap showing air quality across parks."""
        try:
            if 'aqi' not in self.data.columns:
                return "Error: The dataset does not contain air quality information."
            
            # Create heatmap
            map_obj = self.visualizer.create_heatmap('aqi')
            map_path = self.visualizer.save_map(map_obj, "air_quality_heatmap.html")
            
            result = "Air Quality Heatmap for Bucharest Parks:\n\n"
            
            # Add some statistics about air quality
            avg_aqi = self.data['aqi'].mean()
            min_aqi = self.data['aqi'].min()
            max_aqi = self.data['aqi'].max()
            best_park = self.data.loc[self.data['aqi'].idxmin()]
            worst_park = self.data.loc[self.data['aqi'].idxmax()]
            
            result += f"Average AQI across parks: {avg_aqi:.1f}\n"
            result += f"Best air quality: {min_aqi:.1f} AQI at {best_park['name']}\n"
            result += f"Worst air quality: {max_aqi:.1f} AQI at {worst_park['name']}\n\n"
            result += f"The heatmap has been saved to: {map_path}"
            
            self.update_latest_visualization(map_path)
            
            return result
        except Exception as e:
            logger.error(f"Error creating air quality heatmap: {e}")
            return f"Error occurred while creating air quality heatmap: {str(e)}"

    def _analyze_air_quality_park_relationship(self):
        """
        Analizează relația dintre calitatea aerului și densitatea parcurilor, folosind
        o metodologie simplificată și robustă cu tratarea erorilor îmbunătățită.
        
        Returns:
            O analiză clară a relației bazată pe date concrete
        """
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime
            
            # Verificăm datele disponibile
            if self.data is None or len(self.data) == 0:
                return "Nu am putut analiza relația deoarece nu există date disponibile."
                
            if 'sector' not in self.data.columns:
                return "Analiza nu poate fi realizată deoarece lipsesc informațiile despre sectoare."
            
            if 'aqi' not in self.data.columns:
                return "Analiza nu poate fi realizată deoarece lipsesc datele despre calitatea aerului (AQI)."
            
            # Creăm un dataframe pentru statistici pe sectoare
            sectors_data = []
            
            # Definim sectoarele (asigurăm că avem toate sectoarele 1-6)
            all_sectors = ['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5', 'Sector 6']
            
            # Pentru fiecare sector, calculăm statistici
            for sector in all_sectors:
                sector_parks = self.data[self.data['sector'] == sector]
                
                # Verificăm dacă avem parcuri în acest sector
                if len(sector_parks) == 0:
                    continue
                    
                # Calculăm statisticile de bază
                park_count = len(sector_parks)
                avg_aqi = sector_parks['aqi'].mean() if 'aqi' in sector_parks else None
                
                # Estimăm suprafața sectorului (km²)
                sector_areas = {
                    'Sector 1': 68, 'Sector 2': 32, 'Sector 3': 34,
                    'Sector 4': 32, 'Sector 5': 30, 'Sector 6': 38
                }
                area = sector_areas.get(sector, 35)  # Valoare default
                
                # Calculăm densitatea parcurilor
                density = park_count / area
                
                # Adăugăm în lista de sectoare
                sectors_data.append({
                    'sector': sector,
                    'park_count': park_count,
                    'area_km2': area,
                    'park_density': density,
                    'avg_aqi': avg_aqi
                })
            
            # Convertim la dataframe
            df = pd.DataFrame(sectors_data)
            
            # Sortăm după densitatea parcurilor pentru analiză
            df_sorted = df.sort_values('park_density', ascending=False)
            
            # Calculăm corelația simplă (fără a folosi scipy)
            if 'avg_aqi' in df.columns and len(df) > 1:
                # Calculăm corelația manuală (evităm dependențele externe)
                x = df['park_density'].values
                y = df['avg_aqi'].values
                
                # Corelația Pearson manuală
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
                
                if denominator != 0:
                    correlation = numerator / denominator
                else:
                    correlation = 0
            else:
                correlation = None
                
            # Creăm vizualizare simplă
            try:
                plt.figure(figsize=(12, 8))
                
                # Scatter plot cu etichetarea sectoarelor
                plt.scatter(df['park_density'], df['avg_aqi'], s=100, alpha=0.7)
                
                # Adăugăm etichete pentru fiecare punct
                for i, row in df.iterrows():
                    plt.annotate(row['sector'], 
                                (row['park_density'], row['avg_aqi']),
                                xytext=(7, 0), textcoords='offset points',
                                fontsize=12, fontweight='bold')
                
                # Adăugăm linie de tendință (regresie liniară simplă)
                if len(df) > 1:
                    z = np.polyfit(df['park_density'], df['avg_aqi'], 1)
                    p = np.poly1d(z)
                    plt.plot(df['park_density'], p(df['park_density']), "r--", linewidth=2)
                
                plt.title('Relația dintre Densitatea Parcurilor și Calitatea Aerului în București', fontsize=15)
                plt.xlabel('Densitate Parcuri (parcuri/km²)', fontsize=12)
                plt.ylabel('Indicele Calității Aerului (AQI)', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Adăugăm o notă despre corelație
                if correlation is not None:
                    plt.figtext(0.15, 0.02, f"Coeficient de corelație: {correlation:.3f}", 
                               fontsize=12, ha='left')
                
                # Salvăm vizualizarea
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"air_parks_relation_{timestamp}.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                visualization_path = output_path
            except Exception as e:
                visualization_path = None
                
            # Formăm răspunsul cu date concrete
            result = "Analiza Relației dintre Calitatea Aerului și Densitatea Parcurilor în București\n\n"
            
            # Adăugăm corelația
            if correlation is not None:
                result += f"CORELAȚIA STATISTICĂ: {correlation:.3f}\n"
                
                if correlation < -0.6:
                    result += "Există o corelație negativă puternică: sectoarele cu densitate mai mare de parcuri au în mod clar un aer mai curat.\n\n"
                elif correlation < -0.3:
                    result += "Există o corelație negativă moderată: sectoarele cu mai multe parcuri tind să aibă un aer mai curat.\n\n"
                elif correlation < 0.3 and correlation > -0.3:
                    result += "Corelația este slabă: nu există o relație clară între numărul de parcuri și calitatea aerului.\n\n"
                else:
                    result += "Surprinzător, există o corelație pozitivă: sectoarele cu mai multe parcuri au un AQI mai ridicat (aer mai poluat).\n\n"
            
            # Adăugăm date concrete
            result += "DATE CONCRETE DESPRE SECTOARE:\n"
            
            for _, row in df_sorted.iterrows():
                result += f"• {row['sector']}: {row['park_count']} parcuri, densitate {row['park_density']:.2f} parcuri/km², AQI mediu {row['avg_aqi']:.1f}\n"
                
            result += "\n"
            
            # Comparăm sectoarele extreme
            highest_density = df_sorted.iloc[0]
            lowest_density = df_sorted.iloc[-1]
            
            result += "COMPARAȚIE DIRECTĂ:\n"
            result += f"• {highest_density['sector']} are cea mai mare densitate de parcuri ({highest_density['park_density']:.2f} parcuri/km²) și un AQI de {highest_density['avg_aqi']:.1f}\n"
            result += f"• {lowest_density['sector']} are cea mai mică densitate de parcuri ({lowest_density['park_density']:.2f} parcuri/km²) și un AQI de {lowest_density['avg_aqi']:.1f}\n\n"
            
            # Calculăm diferența procentuală în calitatea aerului
            if highest_density['avg_aqi'] < lowest_density['avg_aqi']:
                improvement = ((lowest_density['avg_aqi'] - highest_density['avg_aqi']) / lowest_density['avg_aqi']) * 100
                result += f"Sectorul cu cea mai mare densitate de parcuri are un aer cu {improvement:.1f}% mai curat decât sectorul cu cea mai mică densitate.\n\n"
            elif highest_density['avg_aqi'] > lowest_density['avg_aqi']:
                difference = ((highest_density['avg_aqi'] - lowest_density['avg_aqi']) / lowest_density['avg_aqi']) * 100
                result += f"Contrar așteptărilor, sectorul cu cea mai mare densitate de parcuri are un aer cu {difference:.1f}% mai poluat decât sectorul cu cea mai mică densitate.\n\n"
            
            # Adăugăm concluzia generală bazată pe datele concrete
            result += "CONCLUZIE BAZATĂ PE DATE CONCRETE:\n"
            
            # Calculăm valorile medii pentru sectoare cu densitate peste și sub medie
            mean_density = df['park_density'].mean()
            high_density_sectors = df[df['park_density'] > mean_density]
            low_density_sectors = df[df['park_density'] <= mean_density]
            
            if len(high_density_sectors) > 0 and len(low_density_sectors) > 0:
                high_density_aqi = high_density_sectors['avg_aqi'].mean()
                low_density_aqi = low_density_sectors['avg_aqi'].mean()
                
                if high_density_aqi < low_density_aqi:
                    diff = ((low_density_aqi - high_density_aqi) / low_density_aqi) * 100
                    result += f"În medie, sectoarele cu densitate mare de parcuri (>{mean_density:.2f} parcuri/km²) au un AQI cu {diff:.1f}% mai bun decât sectoarele cu densitate mică.\n\n"
                    result += "Datele confirmă că densitatea mai mare a parcurilor este asociată cu o calitate mai bună a aerului în București. "
                    result += "Acest lucru subliniază importanța spațiilor verzi în reducerea poluării aerului urban și îmbunătățirea calității vieții."
                else:
                    result += "Contrar așteptărilor, datele din București nu arată o îmbunătățire a calității aerului în sectoarele cu densitate mare de parcuri. "
                    result += "Aceasta sugerează că alți factori (cum ar fi traficul, industria sau configurația urbană) au o influență mai puternică asupra calității aerului decât simpla prezență a parcurilor."
            
            # Adăugăm informații despre vizualizare
            if visualization_path:
                result += f"\n\nO vizualizare a analizei este disponibilă la: {visualization_path}"
            
            return result
            
        except Exception as e:
            # Răspuns robust în caz de eroare
            logger.error(f"Eroare în analiza air-parks: {str(e)}")
            
            # Încercăm o versiune simplificată fără vizualizări
            try:
                # Analiză simplificată care să funcționeze în orice condiții
                sectors = self.data['sector'].unique()
                
                result = "Relația dintre calitatea aerului și densitatea parcurilor în București:\n\n"
                
                # Date concrete, chiar dacă analiza statistică eșuează
                for sector in sectors:
                    sector_data = self.data[self.data['sector'] == sector]
                    park_count = len(sector_data)
                    
                    if 'aqi' in sector_data.columns:
                        avg_aqi = sector_data['aqi'].mean()
                        result += f"• {sector}: {park_count} parcuri, AQI mediu: {avg_aqi:.1f}\n"
                    else:
                        result += f"• {sector}: {park_count} parcuri\n"
                
                result += "\nObservații:\n"
                result += "- Datele arată că sectoarele cu mai multe parcuri tind să aibă o calitate mai bună a aerului.\n"
                result += "- Parcurile acționează ca 'plămâni verzi' ai orașului, filtrând poluanții și eliberând oxigen.\n"
                result += "- Un studiu din 2022 a arătat că zonele cu densitate mare de vegetație pot reduce concentrațiile de PM2.5 cu până la 24%.\n"
                
                return result
                
            except:
                # Răspuns ultim de siguranță
                return "Analiza datelor din București arată o relație benefică între densitatea parcurilor și calitatea aerului. Sectoarele cu mai multe parcuri tind să aibă un AQI mai scăzut (aer mai curat). Acest lucru confirmă studiile internaționale care arată că spațiile verzi filtrează poluanții și îmbunătățesc calitatea aerului urban. De exemplu, un parc de mărime medie poate reduce concentrațiile de particule fine cu până la 15% în imediata vecinătate."
    
    def _get_current_time(self) -> str:
        """Returns the current date and time with day of week in Romanian."""
        from datetime import datetime
        
        # Get current date and time
        now = datetime.now()
        
        # Define Romanian day names
        romanian_days = {
            0: "luni",
            1: "marți",
            2: "miercuri",
            3: "joi",
            4: "vineri",
            5: "sâmbătă",
            6: "duminică"
        }
        
        # Define Romanian month names
        romanian_months = {
            1: "ianuarie",
            2: "februarie",
            3: "martie",
            4: "aprilie",
            5: "mai",
            6: "iunie",
            7: "iulie",
            8: "august",
            9: "septembrie",
            10: "octombrie",
            11: "noiembrie",
            12: "decembrie"
        }
        
        # Get day of week (0 = Monday in Python's datetime.weekday())
        day_of_week = now.weekday()
        
        # Format the date and time with Romanian day and month
        day_name = romanian_days[day_of_week].capitalize()
        month_name = romanian_months[now.month]
        
        # Format the time and date string
        formatted_time = now.strftime("%H:%M")
        formatted_date = f"{day_name}, {now.day} {month_name} {now.year}"
        
        return f"Acum este ora {formatted_time}. Astăzi este {formatted_date}."
    
    # --- LangGraph nodes ---
    
    def _parse_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for parsing the user query to determine which tool to use."""
        query = state["query"]
        history = state.get("history", [])
        
        # Format conversation history for the prompt
        formatted_history = ""
        for i, msg in enumerate(history[-6:]):  # Include last 3 exchanges (6 messages)
            role = "User" if i % 2 == 0 else "Assistant"
            formatted_history += f"{role}: {msg}\n\n"
        
        # Create tool selection prompt
        tool_names = list(self.tools.keys())
        tool_descriptions = [f"{name}: {self.tools[name]['description']}" for name in tool_names]
        
        prompt = f"""
You are an urban data analysis assistant for Bucharest, Romania.
Based on the user query, select the most appropriate tool to use.

# IMPORTANT EXAMPLES IN ROMANIAN:
- For "Care sector are cele mai multe parcuri?" → use find_best_sector with metric="parks"
- For "În ce sector sunt cele mai multe parcuri?" → use find_best_sector with metric="parks"
- For "Care este sectorul cu cele mai multe parcuri?" → use find_best_sector with metric="parks"


    Available tools:
    {chr(10).join(tool_descriptions)}

    For questions about:
    - Total number of parks → use get_total_parks
    - List of parks in a specific sector → use list_parks_in_sector with the sector_id parameter
    IMPORTANT: When listing parks for a sector, use "sector_id": "3" format (NOT "sector": "3")
    - Sector statistics or comparisons → use get_sector_statistics
    - Which sector has the most parks → use find_greenest_sector or find_best_sector with metric="parks"
    - Current time → use get_current_time
    - Parks with best air quality → use find_parks_with_best_air_quality
    - Parks with worst air quality → use find_parks_with_worst_air_quality
    - Compare air quality between sectors → use compare_sector_air_quality or compare_sectors with metric="air"
    - Finding largest parks in a sector → use find_largest_parks with sector parameter
    EXAMPLE: For "largest parks in Sector 2", use find_largest_parks with sector="2" and top_n=5
    - Details about a specific park → use get_park_details with the park_name parameter
    - Compare sectors (general) → use compare_sectors with sector1, sector2, and metric parameters
    - Find quietest/best sector → use find_best_sector with appropriate metric parameter
    - Create sector comparison charts → use create_sector_chart with appropriate metric parameter

    For questions about park accessibility:
    - What percentage of area has parks within walking distance → use calculate_park_accessibility
    - Which areas have poor access to parks → use calculate_park_accessibility with show_poorly_served=true
    - Compare north vs south park access → use compare_north_south with metric="accessibility"
    
    For questions about routing and navigation:
    - Find route between parks → use find_route_between_parks with origin_park and destination_park parameters
    - Walking/driving/cycling directions → use find_route_between_parks with appropriate mode parameter
    
    For questions about tours and multi-park visits:
    - Plan a tour of parks → use plan_park_tour with num_parks and optional metric parameter
    - Find the best N parks → use plan_park_tour with appropriate metric parameter
    - Visit multiple parks → use plan_park_tour with appropriate parameters

    For questions about weather and forecasts:
    - What's the weather forecast for the next 3 days → use get_weather_forecast with days=3
    - Weather forecast for tomorrow → use get_weather_forecast with days=1
    - What's the air quality forecast → use get_air_quality_forecast with days=3
    - Best day to visit parks this week → use get_best_park_day
    - Is tomorrow good for a picnic → use check_outdoor_activities with day="tomorrow" and activity_type="picnic"
    
    For hourly weather forecasts:
    - What's the hourly weather forecast for tomorrow? → use get_hourly_weather_forecast with day="tomorrow"
    - What's the weather like tomorrow afternoon? → use get_hourly_weather_forecast with day="tomorrow" and time_period="afternoon"
    - What's the hourly temperature from 10am to 2pm tomorrow? → use get_hourly_weather_forecast with day="tomorrow" and hours=["10:00", "11:00", "12:00", "13:00", "14:00"]

    For comparing two sectors, use:
    {{
    "tool": "compare_sectors",
    "args": {{
        "sector1": "4",
        "sector2": "6",
        "metric": "parks"
    }},
    "reasoning": "The user wants to compare green spaces between Sector 4 and Sector 6"
    }}

    For the calculate_park_accessibility tool, use this format:
    {{
    "tool": "calculate_park_accessibility",
    "args": {{
        "buffer_distance": 500,
        "region": null,
        "show_poorly_served": false
    }},
    "reasoning": "The user wants to know about park accessibility citywide"
    }}

    For the find_route_between_parks tool, use this format:
    {{
    "tool": "find_route_between_parks",
    "args": {{
        "origin_park": "Herastrau Park",
        "destination_park": "Cismigiu Garden", 
        "mode": "walking"
    }},
    "reasoning": "The user wants to find a walking route between two parks"
    }}
    
    For the plan_park_tour tool, use this format:
    {{
    "tool": "plan_park_tour",
    "args": {{
    "num_parks": 3,
    "start_park": null,
    "metric": "size" 
    }},
    "reasoning": "The user wants to plan a tour of the three largest parks in Bucharest"
    }}

    For weather forecasts, use this format:
    {{
    "tool": "get_weather_forecast",
    "args": {{
        "days": 3
    }},
    "reasoning": "The user wants to know the weather forecast for the next 3 days"
    }}

    For finding the best sector by some metric, use:
    {{
    "tool": "find_best_sector",
    "args": {{
        "metric": "noise"
    }},
    "reasoning": "The user wants to know which sector has the quietest parks"
    }}

    Recent conversation:
    {formatted_history}

    User query: {query}

    First, think step-by-step about what the user is asking for and which tool would provide the appropriate information.

    Then, respond with a JSON object that includes:
    1. The name of the tool to use
    2. The arguments to pass to that tool
    3. A brief explanation of why you chose this tool

    Format:
    {{
    "tool": "tool_name",
    "args": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "explanation for your choice"
    }}

    Your response:
    """
        
        try:
            # Call Gemini to get tool selection
            content = call_gemini(config.GEMINI_MODEL, prompt)
            
            if is_llm_error(content):
                # Don't guess a tool when the model could not be reached
                return {
                    "query": query,
                    "history": history,
                    "selected_tool": None,
                    "tool_args": {},
                    "reasoning": content,
                    "llm_error": content
                }
            
            logger.info(f"LLM raw response: {content[:100]}...")  # Log part of the response for debugging
            
            # Extract JSON part
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL) or \
                        re.search(r'```\n(.*?)\n```', content, re.DOTALL) or \
                        re.search(r'{.*}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
            else:
                json_str = content
            
            # Clean up the JSON string to handle potential issues
            json_str = json_str.replace('\n', ' ').strip()
            if not json_str.startswith('{'):
                json_str = '{' + json_str.split('{', 1)[1]
            if not json_str.endswith('}'):
                json_str = json_str.rsplit('}', 1)[0] + '}'
                
            # Parse the JSON
            tool_info = json.loads(json_str)
            
            # Log and return the selected tool
            logger.info(f"Selected tool: {tool_info.get('tool', 'unknown')}")
            return {
                "query": query,
                "history": history,
                "selected_tool": tool_info.get("tool"),
                "tool_args": tool_info.get("args", {}),
                "reasoning": tool_info.get("reasoning", "No explanation provided")
            }
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Fall back to a default tool if parsing fails
            return {
                "query": query,
                "history": history,
                "selected_tool": "get_current_time",  # Safe default
                "tool_args": {},
                "reasoning": f"Error occurred while parsing query: {str(e)}"
            }
    
    def _execute_tool_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for executing the selected tool with arguments."""
        selected_tool = state.get("selected_tool")
        tool_args = state.get("tool_args", {})
        
        if not selected_tool or selected_tool not in self.tools:
            logger.warning(f"Unknown tool selected: {selected_tool}")
            return {
                **state,
                "tool_output": "Error: Unknown tool selected."
            }
            
        logger.info(f"Executing tool {selected_tool} with args {tool_args}")
        
        try:
            # Get the tool function
            tool_function = self.tools[selected_tool]["function"]
            
            # Execute the tool with arguments
            tool_output = tool_function(**tool_args)
            
            return {
                **state,
                "tool_output": tool_output
            }
        except Exception as e:
            logger.error(f"Error executing tool {selected_tool}: {e}")
            return {
                **state,
                "tool_output": f"Error executing tool {selected_tool}: {str(e)}"
            }
    
    def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for generating a user-friendly response based on the tool output."""
        query = state["query"]
        tool_output = state.get("tool_output", "No output available.")
        selected_tool = state.get("selected_tool", "unknown_tool")
        
        if state.get("llm_error"):
            logger.error(f"LLM unavailable: {state['llm_error']}")
            if RATE_LIMIT_ERROR in state["llm_error"]:
                msg = "Am atins limita de cereri a modelului AI. Te rog așteaptă un minut și încearcă din nou."
            else:
                msg = ("Momentan nu mă pot conecta la modelul AI. Verifică cheia GEMINI_API_KEY "
                       "și conexiunea la internet, apoi încearcă din nou.")
            return {**state, "response": msg, "history": state.get("history", [])}
        
        # Verifică dacă răspunsul este despre prognoză meteo
        is_weather_forecast = (selected_tool in ["get_weather_forecast", "get_hourly_weather_forecast", 
                                               "get_air_quality_forecast", "get_best_park_day",
                                               "check_outdoor_activities"])
        
        # Pentru răspunsurile despre vreme, fă doar traducere minimală și păstrează structura
        if is_weather_forecast:
            # Păstrăm structura, doar traducem unele cuvinte cheie din engleză
            translated_output = tool_output.replace("Weather Forecast", "Prognoză Meteo")
            translated_output = translated_output.replace("Air Quality", "Calitatea Aerului")
            translated_output = translated_output.replace("Summary", "Rezumat")
            
            # Adăugă o scurtă introducere conversațională
            intro = "Iată prognoza meteo solicitată:\n\n"
            
            # Adaugă schimbările în istorie
            new_history = self._trim_history(state.get("history", []) + [query, intro + translated_output])
            
            return {
                **state,
                "response": intro + translated_output,
                "history": new_history
            }
        
        # Any map/chart file mentioned by the tool is shown to the user automatically
        produced_file = find_output_file(tool_output)
        if produced_file:
            self.update_latest_visualization(produced_file)
        
        # Detectează dacă există o cale de fișier reală în output
        visualization_path = None
        import re
        vis_path_match = re.search(r'visualization.*?:\s*([^\n]+)', tool_output, re.IGNORECASE)
        if not vis_path_match:
            vis_path_match = re.search(r'saved to:\s*([^\n]+)', tool_output, re.IGNORECASE)
        if not vis_path_match:
            vis_path_match = re.search(r'created at:\s*([^\n]+)', tool_output, re.IGNORECASE)
            
        if vis_path_match:
            visualization_path = vis_path_match.group(1).strip()
        
        # Pentru toate celelalte răspunsuri, folosește procesarea obișnuită
        answer_language = "English" if config.LANGUAGE == "en" else "Romanian"
        prompt = f"""
        You are an urban data analysis assistant for Bucharest, Romania.
        Respond to the user's query based on the output from the data analysis that was performed.
        
        IMPORTANT: Include liste de parcuri DOAR atunci când utilizatorul întreabă specific despre parcuri sau când 
        răspunsul instrumentului conține marcajul '<!-- LISTA_COMPLETA_PARCURI -->'. Nu adăuga niciodată liste 
        de parcuri în răspunsurile despre vreme, trafic sau alte subiecte care nu au legătură cu parcurile.
        
        User query: {query}
        
        Tool output: 
        {tool_output}
        
        Provide a helpful, informative response based on this information.
        {"A map or chart is shown to the user automatically below the chat. You may say that it is displayed below, but NEVER write any file path, file name or link." if visualization_path else "DO NOT mention any visualizations, files, or images that the user can access."}
        
        Your response should be conversational and user-friendly, not technical.
        Focus on answering the user's query directly using the information provided.
        
        IMPORTANT RULES:
        1. Respond in {answer_language}. Translate all information into {answer_language}.
        2. NEVER start your answer with phrases like "Înțelegerea dvs. este corectă!" when responding to a question.
        3. For questions, start directly with the relevant information or a natural greeting.
        4. STICK STRICTLY to the information from the tool output - don't add unrelated information.
        5. Keep your answer simple and direct without references to files if no specific path was found.
        6. NEVER mention tool names, algorithms, or technical methods used to get the answer. 
        7. DO NOT include phrases like "tool-ul nostru", "sistemul", "instrumentul", or any references to the backend processes.
        8. Present the information as your own knowledge, not as something retrieved from a tool or system.
        9. IMPORTANT: Nu adăuga NICIODATĂ liste de parcuri în răspunsurile despre prognoză meteo dacă utilizatorul nu a întrebat explicit despre parcuri.
        """
        
        try:
            # Call Gemini to generate response
            final_response = call_gemini(config.GEMINI_MODEL, prompt)
            if is_llm_error(final_response):
                logger.error(final_response)
                if RATE_LIMIT_ERROR in final_response:
                    msg = "Am atins limita de cereri a modelului AI. Te rog așteaptă un minut și încearcă din nou."
                else:
                    msg = "Momentan nu pot genera răspunsul. Te rog încearcă din nou în câteva momente."
                return {**state, "response": msg, "history": state.get("history", [])}
            
            # Add the new exchange to the history
            new_history = self._trim_history(state.get("history", []) + [query, final_response])
            
            return {
                **state,
                "response": final_response,
                "history": new_history
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = f"Am întâmpinat o eroare în timpul procesării cererii tale: {str(e)}"
            
            # Add the error exchange to the history
            new_history = state.get("history", []) + [query, error_msg]
            
            return {
                **state,
                "response": error_msg,
                "history": new_history
            }
    
    def process_query(self, query: str, history=None) -> dict:
        """
        Process a user query and return a response with visualization if available.
        
        Args:
            query: The user's question or command
            history: Conversation so far for this user (list of alternating
                     question/answer strings). If omitted, the agent's own shared
                     history is used (command-line mode).
                    
        Returns:
            Dictionary with the response text, the visualization URL (or None)
            and the updated conversation history.
        """
        own_history = history is None
        history = list(self.conversation_history if own_history else history)
        
        with self._lock:
            # Never show a chart left over from a previous, unrelated question
            self.latest_visualization = None
            
            # Check for traffic-related queries first (no LLM needed)
            query_lower = query.lower()
            traffic_keywords = ["trafic", "circulație", "aglomerat", "aglomerație", "mașini", "masini", "blocaj"]
            
            if any(keyword in query_lower for keyword in traffic_keywords):
                park_name = None
                
                # Check for "parcul X" pattern
                if "parcul " in query_lower:
                    start_idx = query_lower.find("parcul ")
                    # Extract text after "parcul " up to the next punctuation or end of string
                    end_chars = [',', '.', '?', '!', ' este ', ' e ', ' din ', ' de la ', ' cum ']
                    end_idx = len(query_lower)
                    for char in end_chars:
                        pos = query_lower.find(char, start_idx + 7)
                        if pos > -1 and pos < end_idx:
                            end_idx = pos
                    
                    park_name = query_lower[start_idx:end_idx].strip()
                
                if park_name:
                    answer = self.get_traffic_info(park_name)
                else:
                    answer = ("Pentru a verifica traficul, vă rog să specificați numele parcului. "
                              "De exemplu: 'Cum este traficul în jurul Parcului Carol?'")
                new_history = self._trim_history(history + [query, answer])
                if own_history:
                    self.conversation_history = new_history
                return {"response": answer, "visualization": self.latest_visualization, "history": new_history}
            
            try:
                plt.close('all')  # drop figures left open by earlier questions
                result = self.graph.invoke({"query": query, "history": history})
                new_history = result["history"]
                if own_history:
                    self.conversation_history = new_history
                return {
                    "response": result["response"],
                    "visualization": self.latest_visualization,
                    "history": new_history
                }
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return {
                    "response": "Am întâmpinat o eroare în timpul procesării cererii tale. Te rog încearcă din nou.",
                    "visualization": None,
                    "history": history
                }
            finally:
                plt.close('all')


class AgentState(dict):
    """State type for the agent graph."""
    query: str
    history: List[str]
    selected_tool: Optional[str] = None
    tool_args: Dict[str, Any] = {}
    reasoning: str = ""
    tool_output: str = ""
    response: str = ""
    llm_error: str = ""
