import os
import time
import json
import traceback
import re
import threading
import unicodedata
from typing import Dict, Any, List, Optional, TypedDict
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import logging
from langgraph.graph import StateGraph
from langgraph.graph import END, START

from . import config
from .spatial_analysis import SpatialAnalyst
from .visualization import UrbanVisualizer
from .contracts import validate_tool_arguments, selection_schema
from html import escape
from .recommendations import rank_parks
from .routing import RouteService
from . import sectors
from .forecast import load_forecast

import matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_PLOT_LOCK = threading.RLock()

RATE_LIMIT_ERROR = "rate limit reached"


def call_gemini(model: str, prompt: str, json_mode=False) -> str:
    """Send a prompt to Gemini and return the text reply.

    Failures are returned as a string starting with "[ERROR]" instead of raising,
    so callers can show a friendly message. Temporary errors (rate limit / overload)
    are retried a few times with a growing pause.
    """
    if not config.GEMINI_API_KEY:
        return "[ERROR] Gemini call failed: GEMINI_API_KEY is not set"

    url = f"{config.GEMINI_API_URL}/{model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
        if isinstance(json_mode, dict):
            payload["generationConfig"]["responseSchema"] = json_mode
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}

    last_error = "unknown error"
    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload),
                                     timeout=config.GEMINI_TIMEOUT)
            if response.status_code == 429:
                # Per-minute quota reached: wait for the quota window, then try again
                last_error = RATE_LIMIT_ERROR
                return f"[ERROR] {RATE_LIMIT_ERROR}"
            if response.status_code in (500, 503) and attempt < 1:
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
            if attempt < 1:
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
        self.data = gdf_data.copy()
        # Existing descriptive tools operate on points; footprints remain in the analyst.
        self.data.geometry = self.data.geometry.representative_point()
        self.route_service = RouteService(gdf_data)
        self.spatial_analyst = SpatialAnalyst(gdf_data)
        self.visualizer = UrbanVisualizer(gdf_data)
        self.conversation_history = []
        # Tools use matplotlib and shared state, so handle one question at a time
        self._lock = _PLOT_LOCK
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
            if config.ENABLE_SIMULATED_DATA and os.path.exists(traffic_file):
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
                "description": "Estimate straight-line geometric coverage around parks; not walking distance or population coverage",
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
        try:
            result = self.spatial_analyst.calculate_park_accessibility(buffer_distance or 500, region=region)
            if result.get('error'):
                return "Analiza nu este disponibilă: " + result['error']
            self.latest_evidence = {'method': 'straight_line_buffer', 'distance_meters': buffer_distance or 500,
                                    'limitations': 'Acoperire geometrică; nu distanță de mers și nu procent din populație.'}
            return (f"Acoperire geometrică la {buffer_distance or 500} m: {result['percentage_accessible']:.2f}% din suprafață "
                    f"({result['accessible_area_sqkm']:.2f} din {result['city_area_sqkm']:.2f} km²). "
                    "Estimare în linie dreaptă; nu ține cont de intrări, bariere sau trasee pietonale și nu măsoară populația deservită.")
        except Exception:
            logger.exception("Accessibility failed")
            return "Analiza acoperirii nu este disponibilă."

    def _recommend_best_park_today(self, criteria=None):
        try:
            ranked = rank_parks(self.data, criteria, config.ENABLE_SIMULATED_DATA).head(3)
        except ValueError as exc:
            return str(exc)
        import folium
        from pathlib import Path
        from uuid import uuid4
        m = folium.Map(location=[config.BUCHAREST_LAT, config.BUCHAREST_LON], zoom_start=12)
        lines = ["Parcuri cu măsurători recente în apropiere. Clasamentul este relativ, nu o garanție a condițiilor locale."]
        evidence = []
        for _, row in ranked.iterrows():
            point = row.geometry.representative_point()
            folium.Marker([point.y, point.x], tooltip=escape(str(row['name']))).add_to(m)
            lines.append(f"{row['name']} ({row['sector']}): Airly CAQI {row['aqi']:.1f}; senzor la {row['sensor_distance_meters']:.0f} m; observație {row['air_observed_at']}.")
            evidence.append({'park': str(row['name']), 'source': 'Airly', 'index': 'AIRLY_CAQI',
                             'observed_at': str(row['air_observed_at']), 'sensor_distance_meters': round(float(row['sensor_distance_meters']))})
        if config.ENABLE_SIMULATED_DATA:
            lines.insert(0, "MOD DEMONSTRAȚIE: traficul și zgomotul sunt simulate.")
        folder = Path(config.MAPS_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / f"recommendation_{uuid4().hex}.html"
        m.save(str(artifact))
        self.update_latest_visualization(str(artifact))
        self.latest_evidence = {'recommendations': evidence, 'limitations': 'Măsurători din apropiere, nu din interiorul fiecărui parc.'}
        return "\n\n".join(lines)

    def _compare_north_south(self, metric="overall"):
        boundary_latitude = 44.43
        points = self.data.geometry.representative_point()
        groups = {'Nord': self.data[points.y >= boundary_latitude], 'Sud': self.data[points.y < boundary_latitude]}
        lines = [f"Comparație descriptivă, separată la latitudinea {boundary_latitude}:"]
        for label, data in groups.items():
            text = f"{label}: {len(data)} înregistrări de parc."
            if 'aqi' in data and data.aqi.notna().any():
                text += f" Airly CAQI mediu: {data.aqi.mean():.1f} ({int(data.aqi.notna().sum())} parcuri cu date)."
            if 'area_hectares' in data and data.area_hectares.notna().any():
                text += f" Suprafața însumată a poligoanelor disponibile: {data.area_hectares.sum():.1f} ha."
            lines.append(text)
        lines.append("Nu este o împărțire administrativă. Fără suprafețe de zonă sau populație verificate nu calculăm densități. Poligoanele pot avea suprapuneri, iar senzorii pot fi comuni mai multor parcuri.")
        return "\n".join(lines)

    def _analyze_sectors_by_criteria(self, criteria=None, sort_by=None, reverse=False):
        criteria = criteria or ['parks', 'air']
        try:
            table = sectors.sector_metrics(self.data)
            columns = [sectors.column_for(metric) for metric in criteria]
            order = sectors.column_for(sort_by or criteria[0])
            if any(column not in table or not table[column].notna().any() for column in columns + [order]):
                return "Lipsesc date verificate pentru unele criterii. Încearcă doar criteriul parks."
            ascending = order in ('avg_aqi', 'avg_noise')
            table = table.sort_values(order, ascending=not ascending if reverse else ascending)
            lines = ["Comparație pe criterii separate, fără scor compozit. Valorile lipsă sunt necunoscute."]
            for _, row in table.iterrows():
                values = '; '.join(f"{column}: {row[column]:.2f}" if pd.notna(row[column]) else f"{column}: necunoscut" for column in columns)
                lines.append(f"{row['sector']}: {values}")
            return "\n".join(lines)
        except ValueError as exc:
            return str(exc)

    def _find_best_sector(self, metric: str = 'parks') -> str:
        try:
            table, _ = sectors.select_metrics(self.data, metric)
            return f"Primul sector după criteriul {metric}: {table.iloc[0]['sector']}.\n" + sectors.report(self.data, metric)
        except ValueError as exc:
            return str(exc)

    def _create_sector_chart(self, metric: str = 'parks') -> str:
        try:
            path = sectors.chart(self.data, metric)
            self.update_latest_visualization(path)
            return sectors.report(self.data, metric)
        except ValueError as exc:
            return str(exc)

    def _find_largest_parks(self, sector: str, top_n: int = 5) -> str:
        if 'area_hectares' not in self.data:
            return "Nu există suprafețe verificate în cache. Actualizează datele pentru geometria parcurilor."
        number = re.search(r'[1-6]', str(sector))
        if not number:
            return "Precizează un sector de la 1 la 6."
        parks = self.data[self.data.sector == f"Sector {number.group()}"]
        parks = parks.dropna(subset=['area_hectares']).sort_values('area_hectares', ascending=False).head(top_n)
        if parks.empty:
            return "Nu există suprafețe măsurate pentru acest sector."
        return "Suprafețe calculate din poligoanele OpenStreetMap (hectare):\n" + "\n".join(
            f"{row['name']}: {row['area_hectares']:.2f} ha" for _, row in parks.iterrows())

    def _plan_park_tour(self, num_parks=3, start_park=None, metric="quality"):
        try:
            if metric in ('quality', 'air', 'air_quality'):
                pool = rank_parks(self.data, simulation=False)
            elif metric in ('size', 'area'):
                if 'area_hectares' not in self.data:
                    return "Lipsesc suprafețele măsurate. Actualizează geometria parcurilor."
                pool = self.data.dropna(subset=['area_hectares']).sort_values('area_hectares', ascending=False)
            else:
                return "Alege criteriul quality sau size."
            if len(pool) < num_parks:
                return f"Există doar {len(pool)} parcuri cu date suficiente pentru acest criteriu."
            pool = pool.head(min(len(pool), num_parks * 3)).copy()
            first = self.route_service.match_park(start_park) if start_park else pool.iloc[0]
            selected = [first]
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            while len(selected) < num_parks:
                remaining = pool[~pool.index.isin([row.name for row in selected])]
                if remaining.empty:
                    return "Nu există suficiente parcuri distincte pentru tur."
                point = selected[-1].geometry.representative_point()
                distances = remaining.geometry.representative_point().map(lambda p: abs(geod.inv(point.x, point.y, p.x, p.y)[2]))
                selected.append(remaining.loc[distances.idxmin()])
            legs = [self.route_service.route(str(a['name']), str(b['name']), 'walking') for a, b in zip(selected, selected[1:])]
            import folium
            from pathlib import Path
            from uuid import uuid4
            coordinates = [point for leg in legs for point in leg['coordinates']]
            map_obj = folium.Map(location=coordinates[0], zoom_start=13)
            for leg in legs:
                folium.PolyLine(leg['coordinates'], color='#176b50', weight=5).add_to(map_obj)
            for row in selected:
                point = row.geometry.representative_point()
                folium.Marker([point.y, point.x], tooltip=escape(str(row['name']))).add_to(map_obj)
            map_obj.fit_bounds(coordinates)
            folder = Path(config.MAPS_DIR)
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / f"tour_{uuid4().hex}.html"
            map_obj.save(str(artifact))
            self.update_latest_visualization(str(artifact))
            self.latest_evidence = {'source': 'OpenStreetMap', 'limitations': 'Ordine euristică, nu tur optim. Durata include doar mersul între nodurile stradale, fără vizite și fără verificarea intrărilor.'}
            lines = ["Tur pietonal pe rețeaua OpenStreetMap:", " → ".join(str(row['name']) for row in selected)]
            for leg in legs:
                lines.append(f"{leg['origin']} → {leg['destination']}: {leg['distance_meters']} m, aproximativ {leg['duration_minutes']} minute.")
            lines.append(f"Total pe rețea: {sum(leg['distance_meters'] for leg in legs)} m; {sum(leg['duration_minutes'] for leg in legs):.1f} minute de mers, fără timpul petrecut în parcuri.")
            lines.append(self.latest_evidence['limitations'])
            return "\n".join(lines)
        except ValueError as exc:
            return str(exc)
        except Exception:
            logger.exception('Park tour routing failed')
            return "Nu pot construi întregul tur pe rețeaua pietonală reală. Încearcă din nou mai târziu."

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
        try:
            row = self.route_service.match_park(park_name)
            row = self.data.loc[row.name]
        except ValueError as exc:
            return str(exc)
        lines = [f"{row['name']} ({row['sector']})"]
        if 'area_hectares' in row and pd.notna(row['area_hectares']):
            lines.append(f"Suprafață din geometria OSM: {row['area_hectares']:.2f} ha.")
        if 'aqi' in row and pd.notna(row['aqi']):
            lines.append(f"Airly CAQI: {row['aqi']:.1f}. Acest indice nu este US AQI.")
            for key in ('air_observed_at', 'sensor_distance_meters', 'sensor_id'):
                if key in row and pd.notna(row[key]):
                    lines.append(f"{key}: {row[key]}")
            for pollutant in ('pm25', 'pm10'):
                if pollutant in row and pd.notna(row[pollutant]):
                    lines.append(f"{pollutant}: {row[pollutant]:.1f} μg/m³")
            lines.append("Măsurare din apropiere, nu din interiorul parcului.")
        else:
            lines.append("Nu există măsurători de aer recente și suficient de apropiate în setul disponibil.")
        if config.ENABLE_SIMULATED_DATA:
            lines.append("Mod demonstrație: zgomotul și traficul sunt simulate.")
        return "\n".join(lines)

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
        result = self.spatial_analyst.find_optimal_new_park_locations(num_locations)
        if result is None or result.empty or 'score' not in result:
            return "Nu pot calcula zone candidate cu datele disponibile."
        lines = ["Zone candidate exploratorii, calculate după golurile de acoperire geometrică. Nu sunt amplasamente validate: proprietatea, utilizarea terenului, populația și fezabilitatea nu au fost verificate."]
        for _, row in result.iterrows():
            lines.append(f"{row.geometry.y:.5f}, {row.geometry.x:.5f}; suprafață suplimentară potențial acoperită: {row['potential_area_served'] / 10000:.1f} ha.")
        return "\n".join(lines)

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
            daily_forecast = load_forecast(daily_forecast_file)
            hourly_forecast = load_forecast(hourly_forecast_file)
            
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
            hourly_forecast = load_forecast(hourly_forecast_file)
            daily_forecast = load_forecast(daily_forecast_file)
            
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
        try:
            forecast = load_forecast(config.WEATHER_FORECAST_HOURLY_FILE)
            forecast['date'] = forecast['time'].str.slice(0, 10)
            grouped = forecast.groupby('date')[['pm10', 'pm25', 'aqi']].mean().head(days)
            lines = ["Prognoză modelată Open-Meteo pentru București; medii zilnice, nu măsurători în parcuri. Indicele este European AQI, diferit de Airly CAQI."]
            for day, row in grouped.iterrows():
                lines.append(f"{day}: PM10 {row.pm10:.1f} μg/m³; PM2.5 {row.pm25:.1f} μg/m³; European AQI {row.aqi:.1f}.")
            return "\n".join(lines)
        except Exception:
            logger.exception('Air forecast unavailable')
            return "Prognoza actuală a aerului nu este disponibilă. Actualizează datele."

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
            daily_forecast = load_forecast(daily_forecast_file)
            hourly_forecast = load_forecast(hourly_forecast_file)
            
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
                
            daily_forecast = load_forecast(daily_forecast_file)
            hourly_forecast = load_forecast(hourly_forecast_file)
            
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
        try:
            result = self.route_service.route(origin_park, destination_park, mode)
            self.update_latest_visualization(result['artifact'])
            self.latest_evidence = {key: value for key, value in result.items() if key not in ('artifact', 'coordinates')}
            duration = f"Durată estimată: {result['duration_minutes']} minute.\n" if result['duration_minutes'] is not None else "Durata auto nu este disponibilă fără viteze verificate.\n"
            return (f"{result['origin']} → {result['destination']} ({result['mode']}):\n"
                    f"Distanță pe rețea: {result['distance_meters']} m.\n" + duration + result['limitations'])
        except ValueError as exc:
            return str(exc)
        except Exception:
            logger.exception("Real-network routing unavailable")
            return "Ruta nu este disponibilă pe rețeaua stradală reală. Încearcă din nou mai târziu."

    def _compare_sectors(self, sector1: str, sector2: str = None, metric: str = 'parks') -> str:
        try:
            selection = None if sector1.casefold() in ('all', 'toate', 'all sectors') else [sector1] + ([sector2] if sector2 else [])
            return sectors.report(self.data, metric, selection)
        except ValueError as exc:
            return str(exc)

    def _create_sector_comparison_chart(self, analysis_type: str) -> str:
        return self._create_sector_chart(analysis_type)

    def _create_air_quality_heatmap(self) -> str:
        """Creates a heatmap showing air quality across parks."""
        try:
            if 'aqi' not in self.data.columns:
                return "Error: The dataset does not contain air quality information."
            
            # Create heatmap
            map_obj = self.visualizer.create_heatmap('aqi')
            from uuid import uuid4
            map_path = self.visualizer.save_map(map_obj, f"air_quality_heatmap_{uuid4().hex}.html")
            
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
        if 'aqi' not in self.data or self.data['aqi'].notna().sum() < 2:
            return "Nu există suficiente măsurători recente pentru această analiză."
        grouped = self.data.groupby('sector').agg(park_count=('name', 'size'), avg_caqi=('aqi', 'mean'), measured_parks=('aqi', 'count'))
        lines = ["Comparație descriptivă pe sectoare, din senzorii Airly disponibili:"]
        for sector, row in grouped.iterrows():
            value = f"{row.avg_caqi:.1f}" if pd.notna(row.avg_caqi) else "indisponibil"
            lines.append(f"{sector}: {int(row.park_count)} parcuri; CAQI mediu {value}; {int(row.measured_parks)} parcuri cu date utilizabile.")
        lines.append("Numărul de parcuri nu este suprafața verde. Mai multe parcuri pot împărți același senzor. Această comparație nu demonstrează că parcurile reduc poluarea și nu stabilește o relație cauzală.")
        return "\n".join(lines)

    def _get_current_time(self) -> str:
        """Returns the current date and time with day of week in Romanian."""
        from datetime import datetime
        
        # Get current date and time
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo('Europe/Bucharest'))
        
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
            content = call_gemini(config.GEMINI_MODEL, prompt, json_mode=selection_schema(self.tools))
            
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
                "selected_tool": None,
                "validation_error": "Nu am putut interpreta cererea. Precizează parcul, sectorul sau comparația dorită.",
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
            validate_tool_arguments(tool_function, tool_args)
            if selected_tool in ('get_traffic_information', 'find_parks_with_lowest_noise') and not config.ENABLE_SIMULATED_DATA:
                tool_output = "Nu există măsurători reale pentru trafic sau zgomot. Datele simulate sunt dezactivate."
            else:
                tool_output = tool_function(**tool_args)
            if not isinstance(tool_output, str):
                raise ValueError('Analysis returned an invalid result')
            if fold_text(tool_output).startswith(('error', 'eroare', 'a aparut o eroare', '[error]')):
                message = 'Nu pot calcula acest rezultat cu datele disponibile. Actualizează datele sau precizează criteriile.'
                return {**state, 'tool_output': message, 'validation_error': message}
            if config.ENABLE_SIMULATED_DATA:
                tool_output = "MOD DEMONSTRAȚIE: traficul și zgomotul sunt simulate.\n" + str(tool_output)
            
            return {
                **state,
                "tool_output": tool_output
            }
        except Exception as e:
            logger.error(f"Error executing tool {selected_tool}: {e}")
            return {
                **state,
                "tool_output": "Cererea nu poate fi executată cu aceste date sau argumente. Precizează parcul și criteriile.",
                "validation_error": "Cererea nu poate fi executată cu aceste date sau argumente. Precizează parcul și criteriile."
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
        
        if state.get('validation_error') or not selected_tool or selected_tool not in self.tools:
            msg = state.get('validation_error', 'Nu am identificat analiza dorită. Precizează parcul, sectorul sau criteriul.')
            return {**state, 'response': msg, 'history': self._trim_history(state.get('history', []) + [query, msg])}

        # Preserve exact measured facts and limitations in these evidence-rich answers.
        if selected_tool in ('recommend_best_park_today', 'find_route_between_parks',
                             'calculate_park_accessibility', 'find_optimal_park_locations', 'plan_park_tour',
                             'analyze_air_park_relationship', 'get_park_details', 'find_largest_parks'):
            return {**state, 'response': tool_output,
                    'history': self._trim_history(state.get('history', []) + [query, tool_output])}

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
            intro = "Prognoză Open-Meteo pentru București, la nivel de oraș. Este o predicție, nu o măsurare în parc.\n\n"
            
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
        8. Preserve source attribution, observation times, uncertainty and limitations. Never imply a measurement was made inside a park if it came from a nearby sensor. Never turn correlation into causation.
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
            self.latest_evidence = {}
            # A snapshot can outlive the freshness window after provider failures.
            if 'air_observed_at' in self.data:
                observed = pd.to_datetime(self.data.air_observed_at, utc=True, errors='coerce', format='mixed')
                age = (pd.Timestamp.now(tz='UTC') - observed).dt.total_seconds()
                stale = age.isna() | (age < 0) | (age > config.MAX_AIR_AGE_HOURS * 3600)
            else:
                stale = pd.Series(True, index=self.data.index)
            for column in ('aqi', 'pm25', 'pm10', 'quality_score', 'aqi_score'):
                if column in self.data:
                    self.data.loc[stale, column] = float('nan')
            self.visualizer.data = self.data.copy()
            
            try:
                plt.close('all')  # drop figures left open by earlier questions
                result = self.graph.invoke({"query": query, "history": history})
                new_history = result["history"]
                if own_history:
                    self.conversation_history = new_history
                return {
                    "response": result["response"],
                    "visualization": self.latest_visualization,
                    "evidence": self.latest_evidence,
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


class AgentState(TypedDict, total=False):
    """State type for the agent graph."""
    query: str
    history: List[str]
    selected_tool: Optional[str] = None
    tool_args: Dict[str, Any] = {}
    reasoning: str = ""
    tool_output: str = ""
    response: str = ""
    llm_error: str = ""
    validation_error: str
