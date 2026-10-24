import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from . import config

def _drop_tie_duplicates(gdf):
    """
    Remove duplicate park rows created by nearest-neighbour joins.

    sjoin_nearest keeps the park's original index and adds one extra row for every
    sensor / grid point / road segment that is exactly equally close. Keeping the
    first row per index gives one row per park again.
    """
    gdf = gdf[~gdf.index.duplicated(keep='first')]
    # sjoin_nearest leaves an 'index_right' helper column that would block the next join
    return gdf.drop(columns=[c for c in ('index_right',) if c in gdf.columns])


def process_and_integrate_data(data_dict):
    """
    Process and integrate all collected data sources into a single dataset.
    
    This function combines data from multiple sources (parks, air quality, traffic, noise)
    into a unified dataset for analysis. It performs spatial joins, calculates distances,
    and computes quality metrics for each park in Bucharest.
    
    Args:
        data_dict: Dictionary of GeoDataFrames from collect_all_data()
        
    Returns:
        Processed GeoDataFrame with all integrated data and calculated metrics
    """
    print("Processing and integrating urban data...")
    
    # Extract individual datasets from the input dictionary
    # Each represents a different aspect of the urban environment
    parks_sectors = data_dict.get('parks_sectors')
    air_quality = data_dict.get('air_quality')
    traffic = data_dict.get('traffic')
    noise = data_dict.get('noise')
    
    # Verify we have the fundamental data needed (parks and their sectors)
    if parks_sectors is None or len(parks_sectors) == 0:
        print("✗ ERROR: No parks and sectors data available. Cannot proceed with processing.")
        return None
    
    # --- 1. Clean and prepare parks data ---
    # Create a copy to avoid modifying the original data
    gdf_parks = parks_sectors.copy()
    gdf_parks = gdf_parks[gdf_parks['name'].notna()]  # Remove any entries without names
    
    # --- Remove duplicate parks and irrelevant entries ---
    print(" → Removing duplicate park entries...")
    # Count original parks for reporting
    original_park_count = len(gdf_parks)
    
    # Remove entries with "insula" in name as these are island features, not parks
    # Uses case-insensitive matching
    gdf_parks = gdf_parks[~gdf_parks['name'].str.lower().str.contains('insula')]
    
    # For parks with the same name but different locations, keep only the first entry
    # This avoids duplicate entries in the final dataset
    gdf_parks = gdf_parks.drop_duplicates(subset=['name'], keep='first')
    
    # Report results of cleaning process
    removed_parks_count = original_park_count - len(gdf_parks)
    print(f"   • Removed {removed_parks_count} parks (duplicates or with 'insula' in name)")
    print(f"   • {len(gdf_parks)} unique parks remain after cleaning")
    
    # --- 2. Process and integrate air quality data ---
    if air_quality is not None and len(air_quality) > 0:
        print(" → Integrating air quality data...")
        
        # Project both datasets to a metric coordinate system (EPSG:3857)
        # This is necessary for accurate distance calculations in meters
        gdf_parks_proj = gdf_parks.to_crs("EPSG:3857")
        gdf_air_quality_proj = air_quality.to_crs("EPSG:3857")
        
        # Fill missing values in air quality measurements with the mean
        # to avoid gaps in the data
        for col in ['aqi', 'pm25', 'pm10', 'temperature', 'humidity']:
            if col in gdf_air_quality_proj.columns:
                gdf_air_quality_proj[col] = gdf_air_quality_proj[col].fillna(gdf_air_quality_proj[col].mean())
        
        # Perform a nearest neighbor spatial join to associate each park
        # with its closest air quality sensor
        gdf_parks_with_aqi = gpd.sjoin_nearest(
            gdf_parks_proj,
            gdf_air_quality_proj,
            how="left",
            distance_col="sensor_distance_meters"
        )
        
        # Keep only the relevant columns to avoid cluttering the dataset
        keep_cols = ['name', 'sector', 'geometry', 'aqi', 'pm25', 'pm10', 
                     'temperature', 'humidity', 'sensor_distance_meters']
        gdf_parks_with_aqi = _drop_tie_duplicates(gdf_parks_with_aqi)
        gdf_parks_with_aqi = gdf_parks_with_aqi[
            [col for col in keep_cols if col in gdf_parks_with_aqi.columns]
        ]
        
        # Convert back to WGS84 (standard geographic coordinates)
        gdf_parks = gdf_parks_with_aqi.to_crs("EPSG:4326")
    
    # --- 3. Process and integrate noise data ---
    if noise is not None and len(noise) > 0:
        print(" → Integrating noise level data...")
        
        # Project to metric coordinate system for spatial operations
        gdf_parks_proj = gdf_parks.to_crs("EPSG:3857")
        gdf_noise_proj = noise.to_crs("EPSG:3857")
        
        # Find the nearest noise measurement point for each park
        # and calculate the distance to that measurement
        gdf_parks_with_noise = gpd.sjoin_nearest(
            gdf_parks_proj,
            gdf_noise_proj,
            how="left",
            distance_col="noise_distance_meters"
        )
        
        # Only update the parks GeoDataFrame if noise data was successfully joined
        if 'noise_level_db' in gdf_parks_with_noise.columns:
            gdf_parks = _drop_tie_duplicates(gdf_parks_with_noise).to_crs("EPSG:4326")
    
    # --- 4. Process and integrate traffic data ---
    if traffic is not None and len(traffic) > 0:
        print(" → Calculating traffic congestion metrics...")
        
        # Project to metric coordinate system for buffer operations
        gdf_parks_proj = gdf_parks.to_crs("EPSG:3857")
        
        # Define a 500-meter buffer around each park to analyze nearby roads
        # This represents a reasonable walking distance from the park
        buffer_distance = 500  # meters
        gdf_parks_proj['geometry_buffer'] = gdf_parks_proj.geometry.buffer(buffer_distance)
        
        # Project traffic data to the same coordinate system
        traffic_proj = traffic.to_crs("EPSG:3857")
        
        # Find the nearest road segment to each park
        # In a more advanced implementation, this could analyze all roads within the buffer
        parks_with_traffic = gpd.sjoin_nearest(
            gdf_parks_proj,
            traffic_proj,
            how="left",
            distance_col="road_distance_meters"
        )
        
        # Update the parks GeoDataFrame if traffic data was successfully joined
        if 'congestion_level' in parks_with_traffic.columns:
            # Remove the temporary buffer column and convert back to WGS84
            parks_with_traffic = _drop_tie_duplicates(parks_with_traffic)
            gdf_parks = parks_with_traffic.drop(columns=['geometry_buffer']).to_crs("EPSG:4326")
    
    # --- 5. Calculate additional metrics and quality scores ---
    print(" → Calculating additional metrics and scores...")
    
    # Initialize the overall park quality score
    gdf_parks['quality_score'] = 0
    
    # Incorporate air quality into the score if available
    if 'aqi' in gdf_parks.columns:
        # Air Quality Index: lower is better, so invert the scale
        # Find the range of AQI values in the dataset
        aqi_min = gdf_parks['aqi'].min()
        aqi_max = gdf_parks['aqi'].max()
        aqi_range = aqi_max - aqi_min
        
        if aqi_range > 0:  # Avoid division by zero
            # Convert to 0-10 scale, where 10 is best (lowest AQI)
            gdf_parks['aqi_score'] = 10 - ((gdf_parks['aqi'] - aqi_min) / aqi_range * 10)
            gdf_parks['quality_score'] += gdf_parks['aqi_score']
    
    # Incorporate noise levels into the score if available
    if 'noise_level_db' in gdf_parks.columns:
        # Noise level: lower is better
        noise_min = gdf_parks['noise_level_db'].min()
        noise_max = gdf_parks['noise_level_db'].max()
        noise_range = noise_max - noise_min
        
        if noise_range > 0:  # Avoid division by zero
            # Convert to 0-10 scale, where 10 is best (lowest noise)
            gdf_parks['noise_score'] = 10 - ((gdf_parks['noise_level_db'] - noise_min) / noise_range * 10)
            gdf_parks['quality_score'] += gdf_parks['noise_score']
    
    # Incorporate traffic congestion into the score if available
    if 'congestion_level' in gdf_parks.columns:
        # Traffic congestion is on a scale of 1-5, where 1 is best
        # Convert to 0-10 scale, where 10 is best (lowest congestion)
        gdf_parks['traffic_score'] = 10 - (gdf_parks['congestion_level'] * 2)
        gdf_parks['quality_score'] += gdf_parks['traffic_score']
    
    # Average over the metrics that actually contributed to the score
    num_metrics = sum(col in gdf_parks.columns for col in ('aqi_score', 'noise_score', 'traffic_score'))
    
    # Normalize the final quality score to a 0-10 scale
    # This makes it comparable across different parks regardless of available data
    if num_metrics > 0:
        gdf_parks['quality_score'] = gdf_parks['quality_score'] / num_metrics
    
    # Joins may rename the park name column to 'name_left'; other code looks for either
    # spelling, so make sure both always exist
    if 'name' not in gdf_parks.columns and 'name_left' in gdf_parks.columns:
        gdf_parks['name'] = gdf_parks['name_left']
    if 'name_left' not in gdf_parks.columns:
        gdf_parks['name_left'] = gdf_parks['name']
    
    # --- 6. Save processed data to file for future use ---
    gdf_parks.to_file(config.PROCESSED_DATA_FILE, driver="GeoJSON")
    print(f"✓ Processed data saved to '{config.PROCESSED_DATA_FILE}'")
    
    return gdf_parks
