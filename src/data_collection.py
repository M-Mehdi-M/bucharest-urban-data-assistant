import os
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Point
from datetime import datetime, timedelta
import time
import random

# Import configuration settings from the config module
from . import config

# --- Static Data Collection: Parks and Sectors ---

def fetch_parks_and_sectors(force_update=False):
    """
    Fetches park locations and sector boundaries from OpenStreetMap,
    combines them, and saves the data. Only updates if necessary or forced.
    
    This function uses OpenStreetMap to collect spatial data about:
    1. All parks in Bucharest with their names and locations
    2. Administrative sector boundaries for Bucharest
    3. Performs a spatial join to determine which sector each park belongs to
    
    Args:
        force_update: Boolean to force update even if file is current
        
    Returns:
        GeoDataFrame with parks and sectors data
    """
    print("Checking for parks and sectors data...")
    
    # Check if data exists and is still valid (rarely changes)
    if not force_update and os.path.exists(config.PARKS_SECTORS_FILE):
        print("✓ Parks & sectors data already exists. Using cached version.")
        return gpd.read_file(config.PARKS_SECTORS_FILE)

    print("Fetching parks and sectors data from OpenStreetMap...")
    try:
        # --- Fetching Sector Boundaries using osmnx ---
        # Retrieve administrative boundaries for Bucharest sectors (admin_level 9)
        print(" → Fetching administrative boundaries for sectors...")
        gdf_sectors = ox.features_from_place("București, România", tags={"admin_level": "9"})
        gdf_sectors = gdf_sectors[['name', 'geometry']].rename(columns={'name': 'sector'})
        print(f"✓ Found {len(gdf_sectors)} sectors.")

        # --- Fetching Park Locations using Overpass API ---
        # Use Overpass API to get all parks with names within the bounding box
        print(" → Fetching park locations...")
        bbox = config.BUCHAREST_BBOX
        parks_query = f"""
        [out:json][timeout:25];
        (
            node["leisure"="park"]["name"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
            way["leisure"="park"]["name"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
            relation["leisure"="park"]["name"]({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});
        );
        out center;
        """
        # Overpass rejects requests without a descriptive User-Agent (HTTP 406)
        response = None
        for attempt in range(3):
            response = requests.post(
                config.OVERPASS_API_URL,
                data={"data": parks_query},
                headers={"User-Agent": "BucharestUrbanDataAssistant/1.0"},
                timeout=60,
            )
            # 429 (rate limit) and 504 (server busy) are temporary - wait and retry
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"  Overpass busy (HTTP {response.status_code}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            break
        response.raise_for_status()
        park_elements = response.json()["elements"]

        # Process parks into GeoDataFrame with name and point geometry for each park
        park_features = [
            {"name": park.get("tags", {}).get("name"), 
             "geometry": Point(park.get("center", {}).get("lon"), park.get("center", {}).get("lat"))}
            for park in park_elements if park.get("center") and park.get("tags", {}).get("name")
        ]
        gdf_parks = gpd.GeoDataFrame(park_features, crs="EPSG:4326")
        print(f"✓ Found {len(gdf_parks)} parks with names.")
        
        # --- Spatial Join and Save ---
        # Join parks with sectors to determine which sector each park belongs to
        print(" → Combining parks and sectors...")
        gdf_final = gpd.sjoin(gdf_parks, gdf_sectors, how="inner", predicate="within")
        gdf_final = gdf_final[['name', 'sector', 'geometry']]
        
        # Save to GeoJSON file for future use
        gdf_final.to_file(config.PARKS_SECTORS_FILE, driver="GeoJSON")
        print(f"✓ Parks and sectors data saved to '{config.PARKS_SECTORS_FILE}'")
        
        return gdf_final

    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Could not connect to Overpass API. {e}")
        # If file exists, return existing data as fallback
        if os.path.exists(config.PARKS_SECTORS_FILE):
            print("  Using cached data as fallback.")
            return gpd.read_file(config.PARKS_SECTORS_FILE)
        return None
    except Exception as e:
        print(f"✗ ERROR: An unexpected error occurred during park/sector fetching: {e}")
        if os.path.exists(config.PARKS_SECTORS_FILE):
            print("  Using cached data as fallback.")
            return gpd.read_file(config.PARKS_SECTORS_FILE)
        return None


# --- Daily Update Data: Air Quality ---

def fetch_air_quality_data(force_update=False):
    """
    Fetches air quality data from Airly API. Updates daily or when forced.
    
    This function retrieves real-world air quality measurements from sensors
    around Bucharest, providing data on pollutants like PM2.5, PM10, as well
    as temperature and humidity values at sensor locations.
    
    Args:
        force_update: Boolean to force update regardless of file age
        
    Returns:
        GeoDataFrame with air quality data
    """
    # Check if air quality data needs to be updated (based on date)
    should_update = config.should_update_file(config.AIR_QUALITY_FILE, 'daily')
    
    if not force_update and not should_update:
        print("✓ Air quality data is up-to-date for today.")
        return gpd.read_file(config.AIR_QUALITY_FILE)
        
    print("Fetching air quality data from Airly API...")
    
    if not config.AIRLY_API_KEY:
        print("✗ ERROR: Airly API key not found. Set AIRLY_API_KEY in your .env file.")
        if os.path.exists(config.AIR_QUALITY_FILE):
            print("  Using cached air quality data instead.")
            return gpd.read_file(config.AIR_QUALITY_FILE)
        # Return empty data if no key and nothing cached
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        # Find sensor installations near Bucharest within specified radius
        installations = _get_airly_installations(config.AIRLY_API_KEY)
        if not installations:
            print("No sensor installations found.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
        # Get measurements from each installation
        measurements = _get_airly_measurements(installations, config.AIRLY_API_KEY)
        if not measurements:
            print("No measurement data collected.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Convert to GeoDataFrame with Point geometry for each sensor
        df = pd.DataFrame(measurements)
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Save to GeoJSON file with current date
        gdf.to_file(config.AIR_QUALITY_FILE, driver="GeoJSON")
        print(f"✓ Air quality data saved to '{config.AIR_QUALITY_FILE}'")
        
        return gdf

    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Could not connect to Airly API. {e}")
        # Fall back to the previously cached file if there is one
        if os.path.exists(config.AIR_QUALITY_FILE):
            print("  Using cached air quality data as fallback.")
            return gpd.read_file(config.AIR_QUALITY_FILE)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        print(f"✗ ERROR: Unexpected error during air quality fetching: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _get_airly_installations(api_key):
    """
    Helper function to find nearby Airly sensor installations.
    
    Queries the Airly API to find all sensor installations within the 
    configured search radius of Bucharest's center coordinates.
    
    Args:
        api_key: The API key for authenticating with Airly API
        
    Returns:
        List of sensor installation data
    """
    url = config.AIRLY_BASE_URL + "/installations/nearest"
    params = {
        "lat": config.BUCHAREST_LAT,
        "lng": config.BUCHAREST_LON,
        "maxDistanceKM": config.SEARCH_RADIUS_KM,
        "maxResults": config.MAX_SENSORS
    }
    headers = {"Accept": "application/json", "apikey": api_key}
    
    print(f" → Searching for sensors within {config.SEARCH_RADIUS_KM}km...")
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    results = response.json()
    print(f"✓ Found {len(results)} sensors.")
    return results


def _get_airly_measurements(installations, api_key):
    """
    Helper function to fetch the latest measurements for installations.
    
    For each sensor installation, retrieves the current air quality measurements,
    including PM1, PM2.5, PM10 values, temperature, humidity and air quality index.
    
    Args:
        installations: List of sensor installation data
        api_key: The API key for authenticating with Airly API
        
    Returns:
        List of measurement records with location and pollutant data
    """
    all_data = []
    total = len(installations)
    print(f" → Fetching measurements from {total} sensors...")
    
    for i, installation in enumerate(installations):
        installation_id = installation['id']
        url = config.AIRLY_BASE_URL + "/measurements/installation"
        params = {"installationId": installation_id}
        headers = {"Accept": "application/json", "apikey": api_key}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            measurements = response.json().get('current', {})

            # Extract values from the API response
            current_values = {val['name']: val.get('value') for val in measurements.get('values', [])}
            current_indexes = {idx['name']: idx.get('value') for idx in measurements.get('indexes', [])}

            # Create a complete record with all available measurements
            record = {
                'id': installation_id,
                'latitude': installation['location']['latitude'],
                'longitude': installation['location']['longitude'],
                'street': installation.get('address', {}).get('street'),
                'city': installation.get('address', {}).get('city'),
                'country': installation.get('address', {}).get('country'),
                'aqi': current_indexes.get('AIRLY_CAQI'),
                'pm1': current_values.get('PM1'),
                'pm25': current_values.get('PM25'),
                'pm10': current_values.get('PM10'),
                'pressure': current_values.get('PRESSURE'),
                'humidity': current_values.get('HUMIDITY'),
                'temperature': current_values.get('TEMPERATURE'),
                'timestamp': datetime.now().isoformat()
            }
            all_data.append(record)
            
            # Brief pause to avoid hitting API rate limits
            if i < total - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"  Could not fetch data for sensor {installation_id}: {e}")
            
    print(f"✓ Collected measurements from {len(all_data)} sensors.")
    return all_data


# --- Real-time Data: Traffic and Noise ---

def _simulate_congestion(edges):
    """Add simulated congestion columns to a table of road segments."""
    time_of_day = datetime.now().hour
    
    # Simulate traffic patterns based on time of day
    if 7 <= time_of_day <= 9:  # Morning rush hour
        congestion_probs = [0.05, 0.15, 0.25, 0.35, 0.20]  # Higher congestion
    elif 16 <= time_of_day <= 19:  # Evening rush hour
        congestion_probs = [0.05, 0.10, 0.30, 0.40, 0.15]  # Higher congestion
    else:  # Normal hours
        congestion_probs = [0.25, 0.40, 0.20, 0.10, 0.05]  # Lower congestion
    
    # Assign congestion levels (1-5, where 5 is highest congestion)
    rng = np.random.default_rng()
    edges['congestion_level'] = rng.choice([1, 2, 3, 4, 5], size=len(edges), p=congestion_probs)
    
    # Map congestion levels to speed reduction factors
    edges['speed_factor'] = edges['congestion_level'].map({
        1: 1.0,    # Normal speed
        2: 0.8,    # Slightly reduced
        3: 0.6,    # Moderately reduced
        4: 0.4,    # Significantly reduced
        5: 0.2     # Heavy congestion
    })
    
    # Use length and speed factor to estimate travel time in seconds
    edges['travel_time_seconds'] = edges['length'] / edges['speed_factor'] / 10  # Simplified calculation
    return edges


def fetch_traffic_data(force_update=False):
    """
    Builds the current (simulated) traffic data for Bucharest.
    Congestion is re-simulated on every run.
    
    The road geometry comes from OpenStreetMap (via osmnx). It rarely changes and
    is large, so the previously saved file is reused when it exists and only the
    simulated congestion columns are regenerated. This also keeps the app working
    when the OpenStreetMap servers are slow or unreachable. Pass force_update=True
    to download the road network again.
    
    Returns:
        GeoDataFrame with traffic data for road network
    """
    print("Fetching traffic data...")
    cached = os.path.exists(config.TRAFFIC_FILE)
    
    if cached and not force_update:
        try:
            edges = gpd.read_file(config.TRAFFIC_FILE)
            edges = _simulate_congestion(edges)
            edges.to_file(config.TRAFFIC_FILE, driver="GeoJSON")
            print(f"✓ Reused saved road network; congestion re-simulated ({len(edges)} segments).")
            return edges
        except Exception as e:
            print(f"  Could not reuse saved road network ({e}); downloading it again.")
    
    try:
        # Get the road network from OSM using the osmnx library
        G = ox.graph_from_place("București, Romania", network_type="drive")
        
        # Convert to GeoDataFrame of edges (road segments)
        edges = ox.graph_to_gdfs(G, nodes=False)
        edges = _simulate_congestion(edges)
        
        # Save to GeoJSON file
        edges.to_file(config.TRAFFIC_FILE, driver="GeoJSON")
        print(f"✓ Traffic data saved to '{config.TRAFFIC_FILE}'")
        
        return edges
        
    except Exception as e:
        print(f"✗ ERROR: Failed to fetch traffic data: {e}")
        if cached:
            try:
                print("  Using the previously saved traffic file instead.")
                return gpd.read_file(config.TRAFFIC_FILE)
            except Exception:
                pass
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_noise_data():
    """
    Simulates noise level data for Bucharest.
    Updates on every program run.
    
    Creates a grid of points across the Bucharest area and assigns realistic
    noise levels based on:
    1. Distance from city center (higher noise in center, lower in outskirts)
    2. Time of day (higher during daytime, lower at night)
    3. Random variations to simulate real-world conditions
    
    Returns:
        GeoDataFrame with noise data points across the city
    """
    print("Generating noise level data...")
    
    try:
        # Create a 20x20 grid of points covering Bucharest's bounding box
        lat_range = np.linspace(config.BUCHAREST_BBOX["south"], config.BUCHAREST_BBOX["north"], 20)
        lon_range = np.linspace(config.BUCHAREST_BBOX["west"], config.BUCHAREST_BBOX["east"], 20)
        
        # Time-based factors (noise is higher during day)
        hour = datetime.now().hour
        time_factor = 1.0
        if 22 <= hour or hour < 6:  # Night
            time_factor = 0.6
        elif 6 <= hour < 8 or 18 <= hour < 22:  # Morning/Evening
            time_factor = 0.8
        
        # Generate points with noise levels
        points = []
        center_lat, center_lon = config.BUCHAREST_LAT, config.BUCHAREST_LON
        
        for lat in lat_range:
            for lon in lon_range:
                # Calculate distance from city center (for realistic noise distribution)
                # Higher noise in center, lower in outskirts
                dist_from_center = ((lat - center_lat)**2 + (lon - center_lon)**2)**0.5 * 111  # rough km conversion
                
                # Base noise level (higher in center, lower in outskirts)
                base_noise = max(30, 70 - (dist_from_center * 5))
                
                # Apply time factor and add random variation
                noise_level = base_noise * time_factor + np.random.normal(0, 3)
                noise_level = max(30, min(85, noise_level))  # Clamp between 30-85 dB (realistic urban noise range)
                
                points.append({
                    'geometry': Point(lon, lat),
                    'noise_level_db': round(noise_level, 1),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Create GeoDataFrame from the points
        gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
        
        # Save to GeoJSON file
        gdf.to_file(config.NOISE_FILE, driver="GeoJSON")
        print(f"✓ Noise data saved to '{config.NOISE_FILE}'")
        
        return gdf
        
    except Exception as e:
        print(f"✗ ERROR: Failed to generate noise data: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_weather_forecast():
    """
    Fetches weather and air quality forecast data from Open-Meteo API.
    Updates on every program run since forecasts are time-sensitive.
    
    This function retrieves two types of forecast data:
    1. Weather forecast (temperature, humidity, precipitation, etc.)
    2. Air quality forecast (PM10, PM2.5, European AQI)
    
    Both hourly (next 120 hours) and daily (next 5 days) forecasts are obtained.
    
    Returns:
        Dictionary with hourly and daily forecast data as GeoDataFrames
    """
    print("Fetching weather and air quality forecast data...")
    
    try:
        # Coordinates for Bucharest
        latitude = config.BUCHAREST_LAT
        longitude = config.BUCHAREST_LON
        
        # --- Weather Forecast ---
        # Configure parameters for the Open-Meteo Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloudcover",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 5
        }

        # --- Air Quality Forecast ---
        # Configure parameters for the Open-Meteo Air Quality API
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm10,pm2_5,european_aqi",
            "timezone": "auto",
            "forecast_days": 5
        }
        
        # Fetch data from both APIs
        print(f" → Requesting weather data from {weather_url}...")
        weather_resp = requests.get(weather_url, params=weather_params, timeout=30)
        
        print(f" → Requesting air quality data from {air_url}...")
        air_resp = requests.get(air_url, params=air_params, timeout=30)
        
        # Check for successful responses
        if weather_resp.status_code != 200 or air_resp.status_code != 200:
            print(f"❌ ERROR: Failed to fetch forecast data. Weather status: {weather_resp.status_code}, Air status: {air_resp.status_code}")
            return None
        
        weather_data = weather_resp.json()
        air_data = air_resp.json()
        
        # Verify data structure contains required elements
        if 'hourly' not in weather_data or 'daily' not in weather_data:
            print("❌ ERROR: Invalid weather data format received")
            return None
            
        if 'hourly' not in air_data:
            print("❌ ERROR: Invalid air quality data format received")
            return None
        
        # Process hourly data - combine weather and air quality
        hourly_records = []
        
        # Get minimum length to avoid index errors when data arrays have different lengths
        hourly_times = weather_data['hourly']['time']
        hourly_length = min(
            len(hourly_times), 
            len(weather_data['hourly']['temperature_2m']),
            len(air_data['hourly']['pm10'])
        )
        
        print(f" → Processing {hourly_length} hourly data points...")
        
        # Create records for each hourly forecast combining all data points
        for i in range(hourly_length):
            try:
                hourly_records.append({
                    'time': hourly_times[i],
                    'geometry': Point(longitude, latitude),  # Create point geometry
                    'temperature': weather_data['hourly']['temperature_2m'][i],
                    'humidity': weather_data['hourly'].get('relative_humidity_2m', [])[i] if i < len(weather_data['hourly'].get('relative_humidity_2m', [])) else None,
                    'wind_speed': weather_data['hourly']['wind_speed_10m'][i],
                    'precipitation': weather_data['hourly']['precipitation'][i],
                    'cloudcover': weather_data['hourly']['cloudcover'][i],
                    'pm10': air_data['hourly']['pm10'][i],
                    'pm25': air_data['hourly']['pm2_5'][i],
                    'aqi': air_data['hourly'].get('european_aqi', [])[i] if i < len(air_data['hourly'].get('european_aqi', [])) else None,
                    'forecast_type': 'hourly'
                })
            except IndexError as e:
                print(f"  Warning: Index error at hour {i}: {e}")
                break
        
        # Process daily data - summary data for each day
        daily_records = []
        daily_length = min(
            len(weather_data['daily']['time']),
            len(weather_data['daily']['temperature_2m_min']),
            len(weather_data['daily']['temperature_2m_max'])
        )
        
        print(f" → Processing {daily_length} daily data points...")
        
        # Create records for each daily forecast
        for i in range(daily_length):
            try:
                daily_records.append({
                    'time': weather_data['daily']['time'][i],
                    'geometry': Point(longitude, latitude),
                    'temp_min': weather_data['daily']['temperature_2m_min'][i],
                    'temp_max': weather_data['daily']['temperature_2m_max'][i],
                    'precipitation_sum': weather_data['daily']['precipitation_sum'][i],
                    'precipitation_probability': weather_data['daily']['precipitation_probability_max'][i],
                    'forecast_type': 'daily'
                })
            except IndexError as e:
                print(f"  Warning: Index error at day {i}: {e}")
                break
        
        # Create GeoDataFrames for both hourly and daily forecasts
        gdf_hourly = gpd.GeoDataFrame(hourly_records, crs="EPSG:4326")
        gdf_daily = gpd.GeoDataFrame(daily_records, crs="EPSG:4326")
        
        # Save to GeoJSON files
        os.makedirs(config.DYNAMIC_DATA_DIR, exist_ok=True)
        gdf_hourly.to_file(config.WEATHER_FORECAST_HOURLY_FILE, driver="GeoJSON")
        gdf_daily.to_file(config.WEATHER_FORECAST_DAILY_FILE, driver="GeoJSON")
        
        print(f"✓ Weather forecast data saved to '{config.WEATHER_FORECAST_HOURLY_FILE}' and '{config.WEATHER_FORECAST_DAILY_FILE}'")
        
        # Return combined data for processing
        return {
            'hourly': gdf_hourly,
            'daily': gdf_daily
        }
        
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch or process weather forecast data: {e}")
        import traceback
        print(traceback.format_exc())
        return None


# --- Main Data Collection Function ---
def collect_all_data(force_update=False):
    """
    Collects all necessary data with appropriate update frequencies.
    
    This is the main entry point for data collection that orchestrates
    all the individual data fetching functions. Each data type is collected
    with its appropriate frequency:
    
    1. Parks and sectors: Static data, rarely changes
    2. Air quality: Updated daily
    3. Weather forecast: Updated on every run
    4. Traffic and noise: Simulated real-time data, updated on every run
    
    Args:
        force_update: Force update for all data types regardless of cache status
        
    Returns:
        Dictionary of GeoDataFrames with all collected data
    """
    data = {}
    
    # Parks and sectors (static data - fetch once)
    data['parks_sectors'] = fetch_parks_and_sectors(force_update)
    
    # Air quality (daily update)
    data['air_quality'] = fetch_air_quality_data(force_update)
    
    # Weather forecast (update on every run)
    data['weather_forecast'] = fetch_weather_forecast()
    
    # Traffic and noise (real-time - update on every run)
    data['traffic'] = fetch_traffic_data(force_update)
    data['noise'] = fetch_noise_data()
    
    return data
