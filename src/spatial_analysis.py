import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union, nearest_points
from shapely.geometry import Point, Polygon, LineString
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from . import config

class SpatialAnalyst:
    """
    Class for advanced spatial analysis on urban data.
    
    This class provides methods for performing complex geospatial analyses
    on Bucharest's urban data, including park accessibility, optimal location
    finding, sector statistics, and route planning between parks.
    """
    
    def __init__(self, gdf_data):
        """
        Initialize with processed urban data.
        
        Takes a pre-processed GeoDataFrame containing parks data with their
        locations, attributes, and quality scores, and prepares it for
        spatial analysis operations.
        
        Args:
            gdf_data: GeoDataFrame containing processed park data
        """
        self.data = gdf_data
        self._cached_bucharest_boundary = None
        self._cached_road_network = None
    
    @property
    def bucharest_boundary(self):
        """
        Lazy-load and cache the Bucharest city boundary.
        
        This property retrieves the administrative boundary of Bucharest
        from OpenStreetMap the first time it's accessed, then caches it
        for future use to improve performance. An unavailable boundary is
        reported explicitly rather than replaced by an invented polygon.
        
        Returns:
            GeoDataFrame containing Bucharest's boundary polygon
        """
        if self._cached_bucharest_boundary is None:
            try:
                # Reuse verified sector boundaries when available, including offline.
                from pathlib import Path
                if Path(config.CITY_BOUNDARY_FILE).exists():
                    self._cached_bucharest_boundary = gpd.read_file(config.CITY_BOUNDARY_FILE)
                elif Path(config.SECTORS_FILE).exists():
                    sectors = gpd.read_file(config.SECTORS_FILE)
                    self._cached_bucharest_boundary = sectors.dissolve()
                else:
                    from .osm_source import city_boundary
                    self._cached_bucharest_boundary = city_boundary()
            except Exception as e:
                print(f"Could not fetch Bucharest boundary: {e}")
                raise ValueError("Limita administrativă a Bucureștiului nu este disponibilă.") from e
        return self._cached_bucharest_boundary
    
    @property
    def road_network(self):
        """
        Lazy-load and cache the Bucharest road network.
        
        This property retrieves the complete road network of Bucharest from
        OpenStreetMap the first time it's accessed, then caches it for future use.
        The network is represented as a NetworkX graph where nodes are intersections
        and edges are road segments.
        
        Returns:
            NetworkX MultiDiGraph of Bucharest's road network
        """
        if self._cached_road_network is None:
            try:
                # Get drivable road network for Bucharest using OSMnx
                self._cached_road_network = ox.graph_from_place(
                    "București, Romania", 
                    network_type="walk"
                )
            except Exception as e:
                print(f"Could not fetch Bucharest road network: {e}")
                self._cached_road_network = nx.Graph()  # Empty graph as fallback
        return self._cached_road_network
    
    def calculate_park_accessibility(self, distance_m=500, region=None):
        """
        Estimate geometric coverage around parks, not pedestrian travel distance.
        
        This method identifies areas of the city that are within a specified
        walking distance of any park. It creates buffers around each park,
        merges them, and calculates what percentage of the city is covered.
        
        Args:
            distance_m: Buffer distance in meters representing walking distance (default: 500m)
            
        Returns:
            Dictionary with accessibility metrics and GeoDataFrame with accessible areas:
            - percentage_accessible: Percentage of city area within walking distance of parks
            - accessible_area_sqkm: Total accessible area in square kilometers
            - city_area_sqkm: Total city area in square kilometers
            - gdf_accessible: GeoDataFrame with geometry of accessible areas
        """
        if len(self.data) == 0:
            return {
                "error": "No park data available",
                "percentage_accessible": 0,
                "accessible_area_sqkm": 0,
                "city_area_sqkm": 0
            }
        
        try:
            # Project to a meter-based CRS for accurate buffer measurements
            # UTM 35N is suitable for local distance and area measurement.
            parks_proj = self.data.to_crs(config.METRIC_CRS)
            city_proj = self.bucharest_boundary.to_crs(config.METRIC_CRS)
            if region:
                import re
                match = re.search(r'[1-6]', str(region))
                if not match or not __import__('os').path.exists(config.SECTORS_FILE):
                    raise ValueError('Limita administrativă verificată a sectorului nu este disponibilă.')
                sectors = gpd.read_file(config.SECTORS_FILE)
                city_proj = sectors[sectors.sector == f'Sector {match.group()}'].to_crs(config.METRIC_CRS)
                if city_proj.empty:
                    raise ValueError('Limita sectorului nu este disponibilă.')
            
            # Create circular buffers around parks representing walking distance
            parks_proj['buffer'] = parks_proj.geometry.buffer(distance_m)
            
            # Dissolve all buffers to get total coverage (union operation)
            # This prevents double-counting where buffers overlap
            all_buffers = parks_proj['buffer'].unary_union
            
            # Clip to city boundary to only include accessible areas within Bucharest
            city_area = city_proj.unary_union
            accessible_area = all_buffers.intersection(city_area)
            
            # Calculate areas in square meters
            accessible_area_sqm = accessible_area.area
            city_area_sqm = city_area.area
            
            # Convert to square kilometers for more readable values
            accessible_area_sqkm = accessible_area_sqm / 1_000_000
            city_area_sqkm = city_area_sqm / 1_000_000
            
            # Calculate percentage coverage
            percentage = (accessible_area_sqm / city_area_sqm) * 100 if city_area_sqm > 0 else 0
            
            # Create GeoDataFrame for visualization of accessible areas
            gdf_accessible = gpd.GeoDataFrame(
                geometry=[accessible_area],
                crs=config.METRIC_CRS
            ).to_crs("EPSG:4326")  # Convert back to WGS84 for standard representation
            
            return {
                "percentage_accessible": round(percentage, 2),
                "accessible_area_sqkm": round(accessible_area_sqkm, 2),
                "city_area_sqkm": round(city_area_sqkm, 2),
                "gdf_accessible": gdf_accessible
            }
            
        except Exception as e:
            print(f"Error calculating park accessibility: {e}")
            return {
                "error": f"Calculation failed: {e}",
                "percentage_accessible": 0,
                "accessible_area_sqkm": 0,
                "city_area_sqkm": 0
            }
    
    def find_optimal_new_park_locations(self, num_locations=3, min_distance_m=1000):
        """
        Find optimal locations for new parks based on maximizing coverage
        in currently underserved areas.
        
        This method identifies areas that are not currently within walking distance
        of existing parks, creates a grid of potential locations, and scores each
        location based on:
        1. Distance from existing parks (farther is better)
        2. Potential area served (larger underserved area is better)
        
        The algorithm ensures suggested locations are spread out to maximize
        overall city coverage.
        
        Args:
            num_locations: Number of suggested locations to return (default: 3)
            min_distance_m: Minimum distance between suggested locations in meters (default: 1km)
            
        Returns:
            GeoDataFrame with suggested new park locations, ranked by suitability,
            including columns for score, potential area served, and distance to
            nearest existing park
        """
        try:
            # First get current park accessibility to identify underserved areas
            accessibility = self.calculate_park_accessibility(500)
            
            if "error" in accessibility:
                return gpd.GeoDataFrame(
                    {"error": [accessibility["error"]]},
                    geometry=[Point(config.BUCHAREST_LON, config.BUCHAREST_LAT)],
                    crs="EPSG:4326"
                )
            
            # Project to a meter-based CRS for accurate spatial operations
            city_proj = self.bucharest_boundary.to_crs(config.METRIC_CRS)
            accessible_area = accessibility["gdf_accessible"].to_crs(config.METRIC_CRS).unary_union
            
            # Find areas not currently served by parks (the "gap areas")
            # This uses the difference operation: city area minus accessible areas
            underserved = city_proj.unary_union.difference(accessible_area)
            
            # If there are no underserved areas, return a message
            if underserved.is_empty:
                return gpd.GeoDataFrame(
                    {"message": ["All areas are already within walking distance of a park"]},
                    geometry=[Point(config.BUCHAREST_LON, config.BUCHAREST_LAT)],
                    crs="EPSG:4326"
                )
                
            # Create a grid of points covering the city to evaluate as potential park locations
            from shapely.geometry import box
            city_bounds = city_proj.total_bounds  # Gets the bounding box of the city
            # Create 50x50 grid (2500 points) for evaluation
            x_range = np.linspace(city_bounds[0], city_bounds[2], 50)
            y_range = np.linspace(city_bounds[1], city_bounds[3], 50)
            
            # Filter points to only include those in underserved areas
            candidate_points = []
            for x in x_range:
                for y in y_range:
                    point = Point(x, y)
                    if point.within(underserved):
                        candidate_points.append(point)
            
            if not candidate_points:
                return gpd.GeoDataFrame(
                    {"message": ["Could not find suitable locations for new parks"]},
                    geometry=[Point(config.BUCHAREST_LON, config.BUCHAREST_LAT)],
                    crs="EPSG:4326"
                )
            
            # Calculate a score for each candidate point
            # Higher score = better location for a new park
            results = []
            
            parks_proj = self.data.to_crs(config.METRIC_CRS)
            for point in candidate_points:
                # Measure distance to existing parks (farther is better)
                min_distance_to_park = min(parks_proj.geometry.distance(point))
                
                # Calculate potential population served by creating a buffer and
                # measuring how much underserved area it covers
                buffer = point.buffer(500)  # 500m walking radius
                potential_area_served = buffer.intersection(underserved).area
                
                # Calculate score using a weighted combination of factors
                # 30% weight on distance from existing parks
                # 70% weight on amount of underserved area covered
                score = 0.3 * min(min_distance_to_park / 3000, 1) + 0.7 * potential_area_served / (np.pi * 500 ** 2)
                
                results.append({
                    "geometry": point,
                    "distance_to_nearest_park": min_distance_to_park,
                    "potential_area_served": potential_area_served,
                    "score": score
                })
            
            # Convert to GeoDataFrame
            gdf_candidates = gpd.GeoDataFrame(results, crs=config.METRIC_CRS)
            
            # Sort by score (higher is better)
            gdf_candidates = gdf_candidates.sort_values("score", ascending=False)
            
            # Select top candidates with minimum distance between them
            # to ensure good distribution across the city
            selected_candidates = []
            
            for _, candidate in gdf_candidates.iterrows():
                # If this is the first candidate, add it automatically
                if not selected_candidates:
                    selected_candidates.append(candidate)
                    continue
                
                # Check distance to all already selected candidates
                too_close = False
                for selected in selected_candidates:
                    if candidate.geometry.distance(selected.geometry) < min_distance_m:
                        too_close = True
                        break
                
                # If not too close to any selected candidate, add it
                if not too_close:
                    selected_candidates.append(candidate)
                
                # Stop if we have enough candidates
                if len(selected_candidates) >= num_locations:
                    break
            
            # Convert to GeoDataFrame
            gdf_selected = gpd.GeoDataFrame(selected_candidates, crs=config.METRIC_CRS)
            
            # Add rank column (1 = highest score/best location)
            gdf_selected['rank'] = range(1, len(gdf_selected) + 1)
            
            # Convert back to WGS84 for standard geographic coordinates
            gdf_selected = gdf_selected.to_crs("EPSG:4326")
            
            # Extract lat/lon as separate columns for easy access
            gdf_selected['latitude'] = gdf_selected.geometry.y
            gdf_selected['longitude'] = gdf_selected.geometry.x
            
            return gdf_selected
            
        except Exception as e:
            print(f"Error finding optimal park locations: {e}")
            return gpd.GeoDataFrame(
                {"error": [f"Analysis failed: {e}"]},
                geometry=[Point(config.BUCHAREST_LON, config.BUCHAREST_LAT)],
                crs="EPSG:4326"
            )
    
    def calculate_sector_green_metrics(self):
        from .sectors import sector_metrics
        return sector_metrics(self.data)

    def find_path_between_parks(self, start_park_name, end_park_name):
        """
        Find the shortest path between two parks using the street network.
        
        This method uses the road network to calculate the optimal walking
        route between two parks, including distance and estimated walking time.
        It performs fuzzy matching on park names to find the closest match
        to the provided names.
        
        Args:
            start_park_name: Name of the starting park (partial match supported)
            end_park_name: Name of the destination park (partial match supported)
            
        Returns:
            Dictionary with path details and GeoDataFrame of the route:
            - start_park: Full name of starting park
            - end_park: Full name of destination park
            - distance_meters: Total route distance in meters
            - walking_time_seconds: Estimated walking time in seconds
            - walking_time_formatted: Human-readable walking time (hours, minutes, seconds)
            - route_gdf: GeoDataFrame with the route geometry
        """
        try:
            # Find the parks using case-insensitive partial matching
            start_park = self.data[self.data['name'].str.contains(start_park_name, case=False)]
            end_park = self.data[self.data['name'].str.contains(end_park_name, case=False)]
            
            if len(start_park) == 0:
                return {"error": f"Could not find park with name '{start_park_name}'"}
            if len(end_park) == 0:
                return {"error": f"Could not find park with name '{end_park_name}'"}
            
            # Get the first match if multiple parks match the search term
            start_park = start_park.iloc[0]
            end_park = end_park.iloc[0]
            
            # Get coordinates (latitude, longitude) for each park
            start_point = (start_park.geometry.y, start_park.geometry.x)
            end_point = (end_park.geometry.y, end_park.geometry.x)
            
            # Find the nearest nodes in the road network to each park
            # These will be our starting and ending points for navigation
            orig_node = ox.nearest_nodes(self.road_network, start_point[1], start_point[0])
            dest_node = ox.nearest_nodes(self.road_network, end_point[1], end_point[0])
            
            # Calculate the shortest path using Dijkstra's algorithm (implemented in NetworkX)
            try:
                # Get the sequence of nodes that form the shortest path
                route = nx.shortest_path(self.road_network, orig_node, dest_node, weight='length')
                # Calculate the total length of this path
                route_length = nx.shortest_path_length(self.road_network, orig_node, dest_node, weight='length')
            except nx.NetworkXNoPath:
                return {"error": "No path found between the specified parks"}
            
            # Create GeoDataFrame of the route by extracting the geometry of each edge
            route_edges = []
            for u, v in zip(route[:-1], route[1:]):
                # Get the data for this edge from the graph
                edge_data = self.road_network.get_edge_data(u, v, 0)
                # Use the geometry if available, or create a straight line if not
                geometry = edge_data.get('geometry', LineString([
                    (self.road_network.nodes[u]['x'], self.road_network.nodes[u]['y']),
                    (self.road_network.nodes[v]['x'], self.road_network.nodes[v]['y'])
                ]))
                
                route_edges.append({
                    'u': u,
                    'v': v,
                    'geometry': geometry
                })
            
            # Create GeoDataFrame with the route geometry
            gdf_route = gpd.GeoDataFrame(route_edges, crs=self.road_network.graph['crs'])
            
            # Calculate walking time assuming average walking speed of 5 km/h
            walking_speed_km_h = 5
            walking_speed_m_s = walking_speed_km_h * 1000 / 3600  # Convert to meters/second
            walking_time_seconds = route_length / walking_speed_m_s
            
            # Format walking time into a human-readable string
            hours = int(walking_time_seconds // 3600)
            minutes = int((walking_time_seconds % 3600) // 60)
            seconds = int(walking_time_seconds % 60)
            walking_time_formatted = ""
            if hours > 0:
                walking_time_formatted += f"{hours} hour{'s' if hours != 1 else ''} "
            if minutes > 0 or hours > 0:
                walking_time_formatted += f"{minutes} minute{'s' if minutes != 1 else ''} "
            walking_time_formatted += f"{seconds} second{'s' if seconds != 1 else ''}"
            
            return {
                "start_park": start_park['name'],
                "end_park": end_park['name'],
                "distance_meters": round(route_length, 2),
                "walking_time_seconds": walking_time_seconds,
                "walking_time_formatted": walking_time_formatted,
                "route_gdf": gdf_route
            }
            
        except Exception as e:
            print(f"Error finding path between parks: {e}")
            return {"error": f"Analysis failed: {e}"}
