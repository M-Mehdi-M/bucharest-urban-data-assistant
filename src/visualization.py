from html import escape
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import io
import base64
from . import config
import logging

class UrbanVisualizer:
    """
    Generates various visualizations for urban data analysis.
    
    This class provides methods to create interactive maps, charts, and other
    visualizations for exploring and analyzing urban data about Bucharest,
    particularly focused on parks, air quality, and other environmental metrics.
    """

    def __init__(self, gdf_data=None):
        """
        Initialize the visualizer with optional GeoDataFrame.
        
        Args:
            gdf_data: GeoDataFrame containing urban data with geometries
        """
        self.data = gdf_data.copy() if gdf_data is not None else None
        if self.data is not None:
            self.data.geometry = self.data.geometry.representative_point()

    def create_simplified_map(self):
        """
        Create an interactive map with basic visualization of all data points.
        
        This method generates a Folium map with all locations in the dataset,
        intelligently determining which attributes to display in popups and
        how to color markers based on available data (air quality, noise levels).
        
        Returns:
            folium.Map: Interactive map object with all locations
        """
        data = self.data
        if data is None or len(data) == 0:
            # If no data is available, return an empty map centered on Bucharest
            m = folium.Map(location=[44.4268, 26.1025], zoom_start=12)
            return m

        logging.info(f"Available columns in data: {data.columns.tolist()}")

        # Create base map centered on Bucharest
        m = folium.Map(location=[44.4268, 26.1025], zoom_start=12)

        # Compute min/max values for color scaling of markers
        aqi_min, aqi_max = None, None
        noise_min, noise_max = None, None
        if 'aqi' in data.columns:
            aqi_min = data['aqi'].min(skipna=True)
            aqi_max = data['aqi'].max(skipna=True)
        
        # Check for noise columns with different possible naming conventions
        noise_col = None
        for col in ['noise_level_db', 'noise_level']:
            if col in data.columns:
                noise_col = col
                noise_min = data[col].min(skipna=True)
                noise_max = data[col].max(skipna=True)
                break

        # Add each location as a marker on the map
        for idx, row in data.iterrows():
            # Extract coordinates from geometry
            point = None
            coords = None
            if hasattr(row.geometry, 'centroid'):
                point = row.geometry.representative_point()
                coords = (point.y, point.x)
            else:
                try:
                    coords = (row.geometry.y, row.geometry.x)
                except Exception:
                    continue  # Skip entries with invalid geometry

            # Build popup content with available attributes
            popup_items = []
            
            # Try to find a name for the location in various possible columns
            name = None
            name_columns = [
                'name_left',  # processed park name from your geojson
                'name', 'park_name', 'location_name', 'title', 'label', 'denumire',
                'NAME', 'PARK_NAME', 'DENUMIRE', 'TITLE', 'poi_name'
            ]
            for name_col in name_columns:
                if name_col in row and pd.notna(row[name_col]) and str(row[name_col]).strip():
                    name = row[name_col]
                    break
            popup_items.append(f"<b>name</b>: {escape(str(name or 'Unnamed Location'))}")

            # Add all available attributes to popup
            if 'sector' in row and pd.notna(row['sector']):
                popup_items.append(f"<b>sector</b>: {escape(str(row['sector']))}")
            if 'id' in row and pd.notna(row['id']):
                popup_items.append(f"<b>id</b>: {escape(str(row['id']))}")
            lat = point.y if point else coords[0]
            lng = point.x if point else coords[1]
            popup_items.append(f"<b>latitude</b>: {lat:.6f}")
            popup_items.append(f"<b>longitude</b>: {lng:.6f}")

            # Try to find address information in different possible column names
            for attr in ['street', 'address', 'strada', 'adresa']:
                if attr in row and pd.notna(row[attr]):
                    popup_items.append(f"<b>street</b>: {escape(str(row[attr]))}")
                    break

            # Add city and country info (default to Bucharest/Romania if not specified)
            if 'city' in row and pd.notna(row['city']):
                popup_items.append(f"<b>city</b>: {escape(str(row['city']))}")
            else:
                popup_items.append("<b>city</b>: Bucuresti")
            if 'country' in row and pd.notna(row['country']):
                popup_items.append(f"<b>country</b>: {escape(str(row['country']))}")
            else:
                popup_items.append("<b>country</b>: Romania")

            # Add air quality information if available
            for attr in ['aqi', 'pm1', 'pm25', 'pm10']:
                if attr in row and pd.notna(row[attr]):
                    popup_items.append(f"<b>{attr}</b>: {row[attr]:.2f}")
            for attr in ['air_source', 'air_index', 'air_observed_at', 'sensor_distance_meters', 'data_mode']:
                if attr in row and pd.notna(row[attr]):
                    popup_items.append(f"<b>{attr}</b>: {escape(str(row[attr]))}")

            # Add noise information if available
            for attr in ['noise_level', 'noise_day', 'noise_night', 'noise_level_db']:
                if attr in row and pd.notna(row[attr]):
                    popup_items.append(f"<b>{attr}</b>: {row[attr]:.2f} dB")

            # Add weather information if available
            for attr in ['pressure', 'humidity', 'temperature']:
                if attr in row and pd.notna(row[attr]):
                    value = row[attr]
                    unit = "°C" if attr == 'temperature' else "%" if attr == 'humidity' else "hPa"
                    popup_items.append(f"<b>{attr}</b>: {value:.2f} {unit}")

            # Add traffic information if available
            if 'traffic_congestion' in row and pd.notna(row['traffic_congestion']):
                popup_items.append(f"<b>traffic_congestion</b>: {row['traffic_congestion']:.2f}")

            popup_content = "<br>".join(popup_items)

            # Determine marker color based on available data
            # Default color is green if no quality metrics are available
            color = '#7b8790'  # Unknown measurements must not look healthy.
            
            # Priority for coloring: AQI (air quality), then noise
            if 'aqi' in row and pd.notna(row['aqi']) and aqi_min is not None and aqi_max is not None and aqi_max > aqi_min:
                # Linear gradient: green (good air quality) to red (poor air quality)
                norm = (row['aqi'] - aqi_min) / (aqi_max - aqi_min)
                # RGB interpolation from green (0, 200, 0) to red (200, 0, 0)
                r = int(0 + norm * (200 - 0))
                g = int(200 - norm * 200)
                b = 0
                color = f'#{r:02x}{g:02x}{b:02x}'
            elif noise_col and noise_col in row and pd.notna(row[noise_col]) and noise_min is not None and noise_max is not None and noise_max > noise_min:
                # Blue (quiet) to orange (loud): (0, 100, 255) to (255, 140, 0)
                norm = (row[noise_col] - noise_min) / (noise_max - noise_min)
                r = int(0 + norm * (255 - 0))
                g = int(100 + norm * (140 - 100))
                b = int(255 - norm * 255)
                color = f'#{r:02x}{g:02x}{b:02x}'

            # Add the marker to the map
            folium.CircleMarker(
                location=coords,
                radius=8,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        return m
    
    def set_data(self, gdf_data):
        """
        Set or update the data for visualization.
        
        This allows updating the dataset without creating a new visualizer instance.
        
        Args:
            gdf_data: GeoDataFrame with urban data to visualize
        """
        self.data = gdf_data
    
    def create_static_chart(self, chart_type, x_column=None, y_column=None, title=None, 
                           ascending=False, top_n=None, figsize=(10, 6), **kwargs):
        """
        Create a static chart and return it as a base64 encoded string.
        
        This method generates various types of charts (bar, scatter, pie, etc.)
        based on the provided data and parameters. The resulting chart is
        returned as a base64-encoded PNG image that can be embedded in HTML.
        
        Args:
            chart_type: Type of chart ('bar', 'scatter', 'pie', etc.)
            x_column: Column to use for x-axis or categories
            y_column: Column to use for y-axis (optional for bar/pie)
            title: Chart title
            ascending: Sort data ascending (default False)
            top_n: Limit to top N items
            figsize: Figure size tuple (width, height in inches)
            **kwargs: Additional arguments for matplotlib plotting functions
            
        Returns:
            str: Base64 encoded PNG image of the chart
        """
        if self.data is None or len(self.data) == 0:
            return "No data available for visualization"
            
        plt.figure(figsize=figsize)
        
        # Create a copy to avoid modifying the original data
        plot_data = self.data.copy()
        
        if chart_type == 'bar':
            # For bar charts without y_column, use count of occurrences
            if y_column is None:
                data = plot_data.groupby(x_column).size()
                if ascending is not None:
                    data = data.sort_values(ascending=ascending)
                if top_n:
                    data = data.nlargest(top_n) if not ascending else data.nsmallest(top_n)
                data.plot(kind='barh', color='green', **kwargs)
            else:
                # With y_column, use mean or other aggregation
                data = plot_data.groupby(x_column)[y_column].mean()
                if ascending is not None:
                    data = data.sort_values(ascending=ascending)
                if top_n:
                    data = data.nlargest(top_n) if not ascending else data.nsmallest(top_n)
                data.plot(kind='barh', **kwargs)
                
        elif chart_type == 'scatter':
            # Create a scatter plot of two variables
            if x_column and y_column:
                plt.scatter(plot_data[x_column], plot_data[y_column], alpha=0.6, **kwargs)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                return "Scatter plot requires x_column and y_column"
                
        elif chart_type == 'pie':
            # Create a pie chart showing proportions
            if x_column:
                data = plot_data.groupby(x_column).size()
                if top_n and len(data) > top_n:
                    # Keep top N categories and group the rest as "Others"
                    top_data = data.nlargest(top_n)
                    others_sum = data.sum() - top_data.sum()
                    top_data['Others'] = others_sum
                    data = top_data
                data.plot(kind='pie', autopct='%1.1f%%', **kwargs)
            else:
                return "Pie chart requires x_column"
                
        elif chart_type == 'histogram':
            # Create a histogram of a single variable
            if x_column:
                plt.hist(plot_data[x_column], bins=kwargs.get('bins', 10), **kwargs)
                plt.xlabel(x_column)
                plt.ylabel('Frequency')
            else:
                return "Histogram requires x_column"
                
        elif chart_type == 'box':
            # Create a box plot showing distribution by group
            if x_column and y_column:
                plot_data.boxplot(column=y_column, by=x_column, vert=False, **kwargs)
            else:
                return "Box plot requires x_column (grouping) and y_column"
        else:
            return f"Unsupported chart type: {chart_type}"
            
        if title:
            plt.title(title)
            
        plt.tight_layout()
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Encode as base64 for embedding in HTML
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        return f"data:image/png;base64,{img_str}"
    
    def create_park_map(self, column_to_visualize=None, title="Parks in Bucharest"):
        """
        Create an interactive folium map of parks with clustering.
        
        This method creates a more specialized map focused on parks,
        with optional color coding based on a selected attribute and
        clustering to handle many markers efficiently.
        
        Args:
            column_to_visualize: Column to use for coloring markers
            title: Map title
            
        Returns:
            folium.Map: Interactive map object with parks and clusters
        """
        if self.data is None or len(self.data) == 0:
            # Create empty map centered on Bucharest if no data
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12,
                tiles='CartoDB positron'
            )
            folium.Marker(
                [config.BUCHAREST_LAT, config.BUCHAREST_LON],
                popup="No data available",
                icon=folium.Icon(color="red")
            ).add_to(m)
            return m
        
        # Create map centered on the mean coordinates of all parks
        center_lat = self.data.geometry.y.mean()
        center_lon = self.data.geometry.x.mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add title to the map
        title_html = f'''
            <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create marker cluster for better performance with many markers
        marker_cluster = MarkerCluster().add_to(m)
        
        # Determine color scale if visualizing a specific column
        if column_to_visualize and column_to_visualize in self.data.columns:
            # Create a colormap based on min/max values
            min_val = self.data[column_to_visualize].min()
            max_val = self.data[column_to_visualize].max()
            range_val = max_val - min_val
            
            # Function to map values to colors on a green-to-red scale
            def get_color(value):
                if column_to_visualize == 'aqi' or column_to_visualize == 'noise_level_db':
                    # For AQI and noise, lower is better (green), higher is worse (red)
                    normalized = (value - min_val) / range_val if range_val > 0 else 0.5
                    return f'#{int(255 * normalized):02x}{int(255 * (1-normalized)):02x}00'
                else:
                    # For most other metrics, higher is better (green)
                    normalized = (value - min_val) / range_val if range_val > 0 else 0.5
                    return f'#{int(255 * (1-normalized)):02x}{int(255 * normalized):02x}00'
            
            # Add markers with colors based on the selected column
            for idx, row in self.data.iterrows():
                value = row[column_to_visualize]
                color = get_color(value)
                
                # Create popup content with key information
                popup_content = f"<b>{row['name']}</b><br>"
                popup_content += f"Sector: {row.get('sector', 'N/A')}<br>"
                popup_content += f"{column_to_visualize}: {value}"
                
                # Add additional metrics if available
                for col in ['aqi', 'noise_level_db', 'quality_score', 'congestion_level']:
                    if col in row and col != column_to_visualize:
                        popup_content += f"<br>{col}: {row[col]}"
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=200)
                ).add_to(marker_cluster)
        else:
            # Simple markers without color coding if no column is specified
            for idx, row in self.data.iterrows():
                # Create popup content with name and sector
                popup_content = f"<b>{row['name']}</b><br>"
                popup_content += f"Sector: {row.get('sector', 'N/A')}"
                
                # Add additional metrics if available
                for col in ['aqi', 'noise_level_db', 'quality_score', 'congestion_level']:
                    if col in row:
                        popup_content += f"<br>{col}: {row[col]}"
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=7,
                    color='green',
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=200)
                ).add_to(marker_cluster)
        
        # Add sector boundaries as a reference layer if available
        try:
            sectors = gpd.read_file(config.PARKS_SECTORS_FILE)
            if 'sector' in sectors.columns:
                folium.GeoJson(
                    sectors,
                    name='Sectors',
                    style_function=lambda x: {
                        'fillColor': '#cccccc',
                        'color': '#000000',
                        'fillOpacity': 0.2,
                        'weight': 1
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['sector'])
                ).add_to(m)
        except Exception as e:
            print(f"Could not add sector boundaries: {e}")
        
        # Add layer control to toggle features
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_heatmap(self, column=None):
        """
        Create a heatmap of park distribution or a specified metric.
        
        This visualization shows the concentration of parks or the intensity of
        a specific attribute across the city using a heat map, which is useful
        for identifying patterns and hotspots.
        
        Args:
            column: Column to use for heatmap intensity (None = just density)
            
        Returns:
            folium.Map: Interactive map with heatmap layer
        """
        if self.data is None or len(self.data) == 0:
            # Create empty map if no data
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12,
                tiles='CartoDB dark_matter'
            )
            return m
        
        # Create base map with dark background for better heatmap visibility
        m = folium.Map(
            location=[self.data.geometry.y.mean(), self.data.geometry.x.mean()],
            zoom_start=12,
            tiles='CartoDB dark_matter'
        )
        
        # Create heatmap data points with intensities
        heat_data = []
        
        if column and column in self.data.columns:
            # Use specified column for intensity
            for idx, row in self.data.iterrows():
                # Skip if value is missing
                if pd.isna(row[column]):
                    continue
                    
                # For AQI or noise, invert the values since lower is better
                if column in ['aqi', 'noise_level_db']:
                    # Get min/max for normalization
                    min_val = self.data[column].min()
                    max_val = self.data[column].max()
                    
                    # Invert scale: subtract from max so lower values (better) get higher intensity
                    intensity = max_val - row[column]
                    # Normalize to [0,1] range
                    intensity = (intensity - 0) / (max_val - min_val) if max_val > min_val else 0.5
                    
                    # Scale to reasonable heatmap weight
                    intensity = intensity * 2  # Adjust as needed for good visualization
                else:
                    # For other metrics, use directly (higher is better/more intense)
                    intensity = row[column]
                    
                heat_data.append([row.geometry.y, row.geometry.x, intensity])
        else:
            # Simple density heatmap (all points equal weight)
            for idx, row in self.data.iterrows():
                heat_data.append([row.geometry.y, row.geometry.x, 1])
        
        # Create appropriate title based on what's being visualized
        title = "Park Density Heatmap"
        if column:
            if column in ['aqi', 'noise_level_db']:
                title = f"Park Distribution by Best {column.upper()}"
            else:
                title = f"Park Distribution by {column}"
        
        # Add title to map
        title_html = f'''
            <h3 align="center" style="font-size:16px; color:white"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add heatmap layer with customized parameters
        HeatMap(
            heat_data,
            radius=15,  # Size of each point in the heatmap
            blur=10,    # Amount of blur
            max_zoom=13,  # Maximum zoom level for the heatmap
        ).add_to(m)
        
        return m
    
    def visualize_route(self, route_data):
        """
        Visualize a route between two parks on an interactive map.
        
        This method creates a map showing the shortest path between two parks
        along the road network, with start/end markers and route information.
        
        Args:
            route_data: Dictionary with route information from spatial_analysis
            
        Returns:
            folium.Map: Interactive map with route visualization
        """
        if 'error' in route_data:
            # Create an error map if route calculation failed
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12
            )
            folium.Marker(
                [config.BUCHAREST_LAT, config.BUCHAREST_LON],
                popup=f"Error: {route_data['error']}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            return m
            
        if 'route_gdf' not in route_data or self.data is None:
            # Missing required data
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12
            )
            folium.Marker(
                [config.BUCHAREST_LAT, config.BUCHAREST_LON],
                popup="Missing route or park data",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            return m
            
        # Find the start and end parks in our dataset
        start_park = self.data[self.data['name'].str.contains(route_data['start_park'], case=False)]
        end_park = self.data[self.data['name'].str.contains(route_data['end_park'], case=False)]
        
        if len(start_park) == 0 or len(end_park) == 0:
            # Parks not found in our dataset
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12
            )
            folium.Marker(
                [config.BUCHAREST_LAT, config.BUCHAREST_LON],
                popup="Could not find start or end park",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            return m
            
        # Calculate map center based on the route bounds
        route_gdf = route_data['route_gdf']
        bounds = route_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map centered on the route
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles='CartoDB positron'
        )
        
        # Add the route as a blue line
        folium.GeoJson(
            route_gdf,
            name='Route',
            style_function=lambda x: {
                'color': '#3388ff',
                'weight': 5,
                'opacity': 0.7
            }
        ).add_to(m)
        
        # Add markers for start and end points
        start_park = start_park.iloc[0]
        end_park = end_park.iloc[0]
        
        # Start marker (green play icon)
        folium.Marker(
            location=[start_park.geometry.y, start_park.geometry.x],
            popup=f"Start: {start_park['name']}",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        
        # End marker (red flag icon)
        folium.Marker(
            location=[end_park.geometry.y, end_park.geometry.x],
            popup=f"End: {end_park['name']}",
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(m)
        
        # Add route information panel
        distance_km = route_data['distance_meters'] / 1000
        walking_time = route_data['walking_time_formatted']
        
        info_html = f'''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                        background-color: white; padding: 10px; border-radius: 5px; 
                        box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                <h4>Route Information</h4>
                <p><b>From:</b> {start_park['name']}</p>
                <p><b>To:</b> {end_park['name']}</p>
                <p><b>Distance:</b> {distance_km:.2f} km</p>
                <p><b>Walking time:</b> {walking_time}</p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(info_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def visualize_optimal_park_locations(self, optimal_locations_gdf):
        """
        Visualize optimal locations for new parks on an interactive map.
        
        This method creates a map showing recommended locations for new parks
        based on analysis of underserved areas, with existing parks shown for
        context and service area buffers around suggested locations.
        
        Args:
            optimal_locations_gdf: GeoDataFrame with optimal locations from spatial_analysis
            
        Returns:
            folium.Map: Interactive map with suggested new park locations
        """
        if optimal_locations_gdf is None or len(optimal_locations_gdf) == 0:
            # Create empty map if no data
            m = folium.Map(
                location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
                zoom_start=12
            )
            folium.Marker(
                [config.BUCHAREST_LAT, config.BUCHAREST_LON],
                popup="No optimal locations data available",
                icon=folium.Icon(color="red")
            ).add_to(m)
            return m
            
        # Create base map centered on Bucharest
        m = folium.Map(
            location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
            zoom_start=12
        )
        
        # Add title to map
        title_html = f'''
            <h3 align="center" style="font-size:16px">
                <b>Suggested Optimal Locations for New Parks</b>
            </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add existing parks as small green circles for context
        if self.data is not None and len(self.data) > 0:
            for idx, row in self.data.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color='green',
                    fill=True,
                    fill_opacity=0.4,
                    popup=row['name']
                ).add_to(m)
        
        # Add each suggested optimal location with its rank
        for idx, row in optimal_locations_gdf.iterrows():
            # Check if there's an error message
            if 'error' in row:
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=f"Error: {row['error']}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
                continue
                
            # Create popup content with location details
            popup_content = f"<b>Suggested Park Location #{row.get('rank', idx+1)}</b><br>"
            if 'potential_area_served' in row:
                popup_content += f"Potential area served: {row['potential_area_served']/1000000:.2f} km²<br>"
            if 'distance_to_nearest_park' in row:
                popup_content += f"Distance to nearest existing park: {row['distance_to_nearest_park']:.0f} m<br>"
            if 'score' in row:
                popup_content += f"Suitability score: {row['score']:.2f}"
            
            # Use star icon with orange color for suggested locations
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=popup_content,
                icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)
            
            # Add 500m buffer to visualize potential service area
            folium.Circle(
                location=[row.geometry.y, row.geometry.x],
                radius=500,  # 500m walking distance radius
                color='orange',
                fill=True,
                fill_opacity=0.2,
                weight=2,
                dash_array='5, 5'  # Dashed line for the buffer
            ).add_to(m)
        
        # Add sector boundaries as reference if available
        try:
            sectors = gpd.read_file(config.PARKS_SECTORS_FILE)
            folium.GeoJson(
                sectors,
                name='Sectors',
                style_function=lambda x: {
                    'fillColor': '#cccccc',
                    'color': '#000000',
                    'fillOpacity': 0.1,
                    'weight': 1
                }
            ).add_to(m)
        except Exception as e:
            print(f"Could not add sector boundaries: {e}")
        
        # Add layer control to toggle features
        folium.LayerControl().add_to(m)
        
        return m
    
    def visualize_park_accessibility(self, accessibility_data):
        """
        Visualize areas with access to parks within walking distance.
        
        This method creates a map showing which parts of the city are within a
        specified walking distance (typically 500m) of any park, highlighting
        coverage and gaps in park accessibility.
        
        Args:
            accessibility_data: Dictionary with accessibility data from spatial_analysis
            
        Returns:
            folium.Map: Interactive map with accessibility visualization
        """
        # Create base map centered on Bucharest
        m = folium.Map(
            location=[config.BUCHAREST_LAT, config.BUCHAREST_LON],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add info panel with accessibility statistics
        title_html = f'''
            <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; 
                      background-color: white; padding: 10px; border-radius: 5px; 
                      box-shadow: 0 0 15px rgba(0,0,0,0.2); max-width: 300px;">
                <h4>Park Accessibility Analysis</h4>
                <p><b>{accessibility_data.get('percentage_accessible', 0):.1f}%</b> of Bucharest 
                   is within 500m walking distance of a park.</p>
                <p>Accessible area: <b>{accessibility_data.get('accessible_area_sqkm', 0):.2f} km²</b></p>
                <p>City area: <b>{accessibility_data.get('city_area_sqkm', 0):.2f} km²</b></p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add areas that are within walking distance of parks (green overlay)
        if 'gdf_accessible' in accessibility_data:
            folium.GeoJson(
                accessibility_data['gdf_accessible'],
                name='Accessible Areas',
                style_function=lambda x: {
                    'fillColor': '#00ff00',
                    'color': '#00aa00',
                    'fillOpacity': 0.3,
                    'weight': 1
                }
            ).add_to(m)
            
        # Add existing parks as markers
        if self.data is not None and len(self.data) > 0:
            for idx, row in self.data.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color='darkgreen',
                    fill=True,
                    fill_opacity=0.7,
                    popup=row['name']
                ).add_to(m)
                
        # Add sector boundaries for context
        try:
            sectors = gpd.read_file(config.PARKS_SECTORS_FILE)
            folium.GeoJson(
                sectors,
                name='Sectors',
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': '#000000',
                    'weight': 1
                }
            ).add_to(m)
        except Exception as e:
            print(f"Could not add sector boundaries: {e}")
            
        # Add layer control to toggle features
        folium.LayerControl().add_to(m)
        
        return m
    
    def save_map(self, map_obj, filename=None):
        """
        Save a Folium map to an HTML file.
        
        This utility method saves an interactive map to a file that can be
        opened in a web browser, with an automatically generated filename
        if none is provided.
        
        Args:
            map_obj: Folium map object to save
            filename: Filename to save to (default creates one based on current time)
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            import time
            filename = f"bucharest_map_{int(time.time())}.html"
            
        # Ensure the filename has .html extension
        if not filename.endswith('.html'):
            filename += '.html'
            
        # Create full path in the data directory
        import os
        os.makedirs(config.MAPS_DIR, exist_ok=True)
        filepath = os.path.join(config.MAPS_DIR, os.path.basename(filename))
        
        # Save map to file
        map_obj.save(filepath)
        
        return filepath
