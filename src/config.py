import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load variables from .env (e.g. GEMINI_API_KEY) before reading them below
load_dotenv()

# --- API Keys & URLs ---
# API key for Airly air quality data service with fallback to a default value
AIRLY_API_KEY = os.getenv("AIRLY_API_KEY", "")
# URL for the Overpass API which provides OpenStreetMap data for geographic queries
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
# Base URL for Airly API endpoints - used to fetch air quality measurements
AIRLY_BASE_URL = "https://airapi.airly.eu/v2"

# --- Gemini Configuration ---
# Google Gemini API settings used for natural language processing
# Put your key in a .env file in the project root: GEMINI_API_KEY=your_key_here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Maximum time in seconds to wait for a response from the language model
GEMINI_TIMEOUT = 120

# --- Geographic & Search Parameters ---
# Bounding box coordinates that define the area of Bucharest for data collection
BUCHAREST_BBOX = {
    "south": 44.38,
    "west": 25.96,
    "north": 44.52,
    "east": 26.26
}
# Central coordinates of Bucharest used as reference point for proximity calculations
BUCHAREST_LAT = 44.4268
BUCHAREST_LON = 26.1025
# Maximum radius in kilometers to search for data around Bucharest
SEARCH_RADIUS_KM = 20
# Maximum number of air quality sensors to query to avoid hitting API limits
MAX_SENSORS = 100

# --- File Paths ---
# Calculate absolute paths to ensure consistent file access regardless of working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create data directory if it doesn't exist to prevent file operation errors
os.makedirs(DATA_DIR, exist_ok=True)

# Define subdirectories for different types of data:
# - Static data: rarely changes (parks, sectors)
# - Dynamic data: updated regularly (air quality, weather, traffic)
STATIC_DATA_DIR = os.path.join(DATA_DIR, 'static')
DYNAMIC_DATA_DIR = os.path.join(DATA_DIR, 'dynamic')
os.makedirs(STATIC_DATA_DIR, exist_ok=True)
os.makedirs(DYNAMIC_DATA_DIR, exist_ok=True)

# Static data file paths - these files contain information that rarely changes
PARKS_SECTORS_FILE = os.path.join(STATIC_DATA_DIR, "bucharest_parks_sectors.geojson")

# Daily update data - air quality information that needs daily refreshing
AIR_QUALITY_FILE = os.path.join(DYNAMIC_DATA_DIR, "air_quality.geojson")

# Weather forecast data files - separate hourly and daily forecasts
WEATHER_FORECAST_HOURLY_FILE = os.path.join(DYNAMIC_DATA_DIR, "weather_forecast_hourly.geojson")
WEATHER_FORECAST_DAILY_FILE = os.path.join(DYNAMIC_DATA_DIR, "weather_forecast_daily.geojson")

# Real-time data files that need frequent updates
TRAFFIC_FILE = os.path.join(DYNAMIC_DATA_DIR, "traffic_current.geojson")
NOISE_FILE = os.path.join(DYNAMIC_DATA_DIR, "noise_current.geojson")

# Final processed data file that combines all sources for the application
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "bucharest_urban_data_processed.geojson")

# --- Update Frequencies ---
def should_update_file(filepath, update_frequency='daily'):
    """
    Determine if a file should be updated based on when it was last modified.

    This function checks if a data file needs refreshing based on configurable time periods.
    It helps implement smart caching to avoid unnecessary API calls and data processing.

    Args:
        filepath: Path to the file to check
        update_frequency: How often to update the file:
            - 'always': Update on every run
            - 'daily': Update if older than 24 hours (default)
            - 'weekly': Update if older than 7 days
            - 'monthly': Update if older than 30 days
            - 'never': Never update automatically

    Returns:
        Boolean indicating if the file should be updated
    """
    # If file doesn't exist, it should be updated
    if not os.path.exists(filepath):
        return True

    # Get file's last modification time
    last_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
    now = datetime.now()

    if update_frequency == 'always':
        return True
    elif update_frequency == 'daily':
        # Update if file is older than 24 hours
        return now - last_modified > timedelta(days=1)
    elif update_frequency == 'weekly':
        return now - last_modified > timedelta(weeks=1)
    elif update_frequency == 'monthly':
        return now - last_modified > timedelta(days=30)
    elif update_frequency == 'never':
        return False
    else:
        # Default: update daily
        return now - last_modified > timedelta(days=1)

# --- Agent Configuration ---
# Maximum number of conversation exchanges to maintain in history for context
MAX_HISTORY_LENGTH = 10  # Max number of exchanges to keep in conversation history

# --- Language Configuration ---
# Default language for the application with Romanian as default
LANGUAGE = os.getenv("LANGUAGE", "ro")  # Default language (ro=Romanian, en=English)

# --- Application Settings ---
# Debug mode flag for additional logging and diagnostic information
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
# Network port for the web server to listen on
PORT = int(os.getenv("PORT", 5000))
# Network interface to bind the web server to (127.0.0.1 = only this computer)
HOST = os.getenv("HOST", "127.0.0.1")  # Local only by default; use 0.0.0.0 to share on the network
