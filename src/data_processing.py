import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from . import config
from .evidence import fresh_air_mask

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
    if air_quality is not None and len(air_quality):
        # An offline sensor must not hide a nearby functioning sensor.
        air_quality = air_quality[fresh_air_mask(air_quality)].copy()
    traffic = data_dict.get('traffic') if config.ENABLE_SIMULATED_DATA else None
    noise = data_dict.get('noise') if config.ENABLE_SIMULATED_DATA else None
    
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
    
    # Names are not identities: distinct parks can share names.
    
    # Preserve same-name parks with distinct stable identities or geometries.
    gdf_parks = gdf_parks.drop_duplicates(subset=['park_id'], keep='first') if 'park_id' in gdf_parks else gdf_parks.drop_duplicates(subset=['name', 'geometry'], keep='first')
    
    # Report results of cleaning process
    removed_parks_count = original_park_count - len(gdf_parks)
    print(f"   • Removed {removed_parks_count} parks (duplicate identities)")
    print(f"   • {len(gdf_parks)} unique parks remain after cleaning")
    
    # --- 2. Process and integrate air quality data ---
    if air_quality is not None and len(air_quality) > 0:
        print(" → Integrating air quality data...")
        
        # Project both datasets to a metric coordinate system (local UTM)
        # This is necessary for accurate distance calculations in meters
        gdf_parks_proj = gdf_parks.to_crs(config.METRIC_CRS)
        gdf_air_quality_proj = air_quality.to_crs(config.METRIC_CRS)

        # Keep missing measurements missing; carry source identity and observation time.
        air_columns = [c for c in ('aqi', 'pm25', 'pm10', 'temperature', 'humidity',
                                  'air_observed_at', 'air_fetched_at', 'air_source', 'air_index', 'geometry')
                       if c in gdf_air_quality_proj]
        if 'id' in gdf_air_quality_proj:
            gdf_air_quality_proj['sensor_id'] = gdf_air_quality_proj['id']
            air_columns.append('sensor_id')
        gdf_air_quality_proj = gdf_air_quality_proj[air_columns]
        # Perform a nearest neighbor spatial join to associate each park
        # with its closest air quality sensor
        gdf_parks_with_aqi = gpd.sjoin_nearest(
            gdf_parks_proj,
            gdf_air_quality_proj,
            how="left",
            distance_col="sensor_distance_meters"
        )
        
        # Keep only the relevant columns to avoid cluttering the dataset
        gdf_parks_with_aqi = _drop_tie_duplicates(gdf_parks_with_aqi)
        if 'air_observed_at' in gdf_parks_with_aqi:
            observed = pd.to_datetime(gdf_parks_with_aqi.air_observed_at, utc=True, errors='coerce', format='mixed')
            age_seconds = (pd.Timestamp.now(tz='UTC') - observed).dt.total_seconds()
            valid = ((age_seconds >= 0) &
                     (age_seconds <= config.MAX_AIR_AGE_HOURS * 3600) &
                     (gdf_parks_with_aqi.sensor_distance_meters <= config.MAX_SENSOR_DISTANCE_METERS))
        else:
            valid = pd.Series(False, index=gdf_parks_with_aqi.index)
        gdf_parks_with_aqi['air_usable'] = valid
        for column in ('aqi', 'pm25', 'pm10', 'temperature', 'humidity'):
            if column in gdf_parks_with_aqi:
                gdf_parks_with_aqi.loc[~valid, column] = np.nan
                gdf_parks_with_aqi.loc[gdf_parks_with_aqi[column] < 0, column] = np.nan

        # Convert back to WGS84 (standard geographic coordinates)
        gdf_parks = gdf_parks_with_aqi.to_crs("EPSG:4326")
    
    # --- 3. Process and integrate noise data ---
    if noise is not None and len(noise) > 0:
        print(" → Integrating noise level data...")
        
        # Project to metric coordinate system for spatial operations
        gdf_parks_proj = gdf_parks.to_crs(config.METRIC_CRS)
        gdf_noise_proj = noise.to_crs(config.METRIC_CRS)
        
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
        gdf_parks_proj = gdf_parks.to_crs(config.METRIC_CRS)
        
        # Define a 500-meter buffer around each park to analyze nearby roads
        # This represents a reasonable walking distance from the park
        buffer_distance = 500  # meters
        gdf_parks_proj['geometry_buffer'] = gdf_parks_proj.geometry.buffer(buffer_distance)
        
        # Project traffic data to the same coordinate system
        traffic_proj = traffic.to_crs(config.METRIC_CRS)
        traffic_proj = traffic_proj[[c for c in ('geometry', 'congestion_level', 'speed_factor', 'travel_time_seconds') if c in traffic_proj]]
        
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
    if 'aqi' not in gdf_parks:
        gdf_parks['aqi'] = np.nan
        gdf_parks['air_usable'] = False

    # Scores are relative comparisons, never absolute environmental grades.
    score_columns = []
    for measurement, score_name in [('aqi', 'aqi_score')]:
        if measurement in gdf_parks:
            values = pd.to_numeric(gdf_parks[measurement], errors='coerce')
            span = values.max() - values.min()
            gdf_parks[score_name] = 10 - 10 * (values - values.min()) / span if span > 0 else values.where(values.isna(), 5.0)
            score_columns.append(score_name)
    gdf_parks['quality_score'] = gdf_parks[score_columns].mean(axis=1) if score_columns else np.nan
    gdf_parks['data_mode'] = 'simulation' if config.ENABLE_SIMULATED_DATA else 'measured'

    # Joins may rename the park name column to 'name_left'; other code looks for either
    # spelling, so make sure both always exist
    if 'name' not in gdf_parks.columns and 'name_left' in gdf_parks.columns:
        gdf_parks['name'] = gdf_parks['name_left']
    if 'name_left' not in gdf_parks.columns:
        gdf_parks['name_left'] = gdf_parks['name']
    
    # --- 6. Save processed data to file for future use ---
    from pathlib import Path
    from uuid import uuid4
    target = Path(config.PROCESSED_DATA_FILE)
    temporary = target.with_name(f".{target.stem}-{uuid4().hex}.geojson")
    try:
        gdf_parks.to_file(temporary, driver="GeoJSON")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"✓ Processed data saved to '{config.PROCESSED_DATA_FILE}'")
    
    return gdf_parks
