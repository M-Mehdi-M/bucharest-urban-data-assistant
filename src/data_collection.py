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
from . import source_health
from .storage import save_geojson
from . import air_budget

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
    from .park_data import fetch_park_geometries
    cached = None
    if os.path.exists(config.PARKS_SECTORS_FILE):
        try:
            cached = gpd.read_file(config.PARKS_SECTORS_FILE)
        except Exception:
            pass
        if not force_update and cached is not None and 'park_id' in cached.columns and os.path.exists(config.SECTORS_FILE):
            source_health.record('parks', 'ready', f'OpenStreetMap: {len(cached)} parcuri în cache cu geometrie și identitate OSM.', origin='cache', count=len(cached))
            return cached
    source_health.record('parks', 'loading', 'OpenStreetMap: actualizare geometrie și sectoare.')
    try:
        parks = fetch_park_geometries()
        save_geojson(parks, config.PARKS_SECTORS_FILE)
        source_health.record('parks', 'ready', f'OpenStreetMap: {len(parks)} parcuri descărcate cu geometrie.', origin='network', count=len(parks))
        return parks
    except Exception as exc:
        print(f"Park geometry refresh unavailable: {exc}; using last available cache.")
        if cached is not None:
            source_health.record('parks', 'cached', 'OpenStreetMap nu a putut fi actualizat. Se folosesc parcurile din cache.' + (' Geometria poligonală și suprafețele lipsesc din cache-ul vechi.' if 'park_id' not in cached else ''),
                                 origin='cache', count=len(cached), error=str(exc)[:350])
        else:
            source_health.record('parks', 'unavailable', 'OpenStreetMap este indisponibil și nu există un cache de parcuri.', count=0, error=str(exc)[:350])
        return cached


# --- Daily Update Data: Air Quality ---

def fetch_air_quality_data(force_update=False):
    from .air_source import collect_air_quality
    return collect_air_quality(force_update, _get_airly_installations, _get_airly_measurements)


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
    air_budget.observe(response, api_key)
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
    started = time.monotonic()
    consecutive_errors = 0
    print(f" → Fetching measurements from {total} sensors...")
    
    for i, installation in enumerate(installations):
        if air_budget.blocked(api_key):
            print('Airly quota cooldown active; preserving completed observations.')
            break
        if time.monotonic() - started >= config.AIRLY_COLLECTION_BUDGET_SECONDS:
            print('Airly collection time budget reached; keeping completed observations.')
            break
        installation_id = installation['id']
        url = config.AIRLY_BASE_URL + "/measurements/installation"
        params = {"installationId": installation_id}
        headers = {"Accept": "application/json", "apikey": api_key}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=(5, 15))
            air_budget.observe(response, api_key)
            response.raise_for_status()
            consecutive_errors = 0
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
                'air_observed_at': measurements.get('tillDateTime') or measurements.get('fromDateTime'),
                'air_fetched_at': datetime.now().astimezone().isoformat(),
                'air_source': 'Airly',
                'air_index': 'AIRLY_CAQI'
            }
            all_data.append(record)
            if (i + 1) % 10 == 0:
                source_health.record('air_quality', 'loading', f'Airly: {i + 1}/{total} senzori verificați.', count=len(all_data))
            
            # Brief pause to avoid hitting API rate limits
            if i < total - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"  Could not fetch data for sensor {installation_id}: {e}")
            consecutive_errors += 1
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status in (401, 403, 429) or consecutive_errors >= 3:
                print('Stopping Airly requests after an authentication/quota failure or repeated network errors.')
                break
            
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
    source_health.record('weather', 'loading', 'Open-Meteo: descărcare prognoză meteo și aer.')
    
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
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "Europe/Bucharest",
            "forecast_days": 5
        }

        # --- Air Quality Forecast ---
        # Configure parameters for the Open-Meteo Air Quality API
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm10,pm2_5,european_aqi",
            "timezone": "Europe/Bucharest",
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
            raise ValueError(f'Open-Meteo HTTP {weather_resp.status_code}/{air_resp.status_code}')
        
        weather_data = weather_resp.json()
        air_data = air_resp.json()
        
        # Verify data structure contains required elements
        if 'hourly' not in weather_data or 'daily' not in weather_data:
            print("❌ ERROR: Invalid weather data format received")
            return None
            
        if 'hourly' not in air_data:
            print("❌ ERROR: Invalid air quality data format received")
            return None
        
        from .forecast import combine_hourly_forecasts
        hourly_records = combine_hourly_forecasts(weather_data, air_data, longitude, latitude)

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
        save_geojson(gdf_hourly, config.WEATHER_FORECAST_HOURLY_FILE)
        save_geojson(gdf_daily, config.WEATHER_FORECAST_DAILY_FILE)
        source_health.record('weather', 'ready', f'Open-Meteo: {len(gdf_daily)} zile și {len(gdf_hourly)} ore de prognoză.', origin='network', count=len(gdf_hourly))
        
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
        try:
            from .forecast import load_forecast
            hourly = load_forecast(config.WEATHER_FORECAST_HOURLY_FILE)
            daily = load_forecast(config.WEATHER_FORECAST_DAILY_FILE)
            source_health.record('weather', 'cached', 'Open-Meteo nu a putut fi actualizat. Se folosește prognoza curentă din cache.', origin='cache', count=len(hourly))
            return {'hourly': hourly, 'daily': daily}
        except Exception:
            source_health.record('weather', 'unavailable', 'Open-Meteo nu este disponibil și nu există o prognoză curentă în cache.', count=0)
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
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Independent providers must not queue behind an overloaded Overpass server.
    tasks = {'parks_sectors': lambda: fetch_parks_and_sectors(force_update),
             'air_quality': lambda: fetch_air_quality_data(force_update),
             'weather_forecast': fetch_weather_forecast}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix='collection') as pool:
        futures = {pool.submit(loader): name for name, loader in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                data[name] = future.result()
            except Exception as exc:
                data[name] = None
                source = {'parks_sectors': 'parks', 'weather_forecast': 'weather'}.get(name, name)
                source_health.record(source, 'unavailable', f'{source}: colectarea a eșuat ({type(exc).__name__}).', count=0)
    
    # Traffic and noise (real-time - update on every run)
    data['traffic'] = fetch_traffic_data(force_update) if config.ENABLE_SIMULATED_DATA else None
    data['noise'] = fetch_noise_data() if config.ENABLE_SIMULATED_DATA else None
    
    return data
