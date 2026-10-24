import os
import sys
import time
import argparse
import threading
import logging
import uuid
from collections import deque, OrderedDict
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add the project root to the Python path so we can import modules
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)
# Generated files (visualizations/, data/) use paths relative to the project folder
os.chdir(PROJECT_DIR)

# Import custom modules for data collection, processing and visualization
from src.data_collection import collect_all_data
from src.data_processing import process_and_integrate_data
from src.agent import UrbanDataAgent
from src.visualization import UrbanVisualizer
import src.config as config

# Configure logging with timestamp, level and message format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask web application instance with specified template and static file locations
app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables to store application state across requests
gdf_data = None  # Will hold the processed GeoDataFrame
agent = None     # Will hold the Urban Data Agent for query processing
visualizer = None  # Will hold the visualization component

# Thread management for background data updates
update_thread = None
is_updating = False
last_update_time = None
_load_lock = threading.Lock()  # Only one data load may run at a time

# --- Per-user conversations and simple rate limiting (in memory) ---
MAX_QUERY_LENGTH = 500          # Longest question accepted (characters)
MAX_SESSIONS = 500              # Conversations kept in memory; the oldest are dropped
RATE_LIMIT_QUERIES = 20         # Questions allowed per IP ...
RATE_LIMIT_WINDOW = 60          # ... in this many seconds
UPDATE_COOLDOWN = 300           # Minimum seconds between manual data updates
sessions = OrderedDict()        # session id -> conversation history
_sessions_lock = threading.Lock()
_request_log = {}               # IP -> timestamps of recent questions
_last_manual_update = 0.0


def rate_limited(ip):
    """Return True if this IP has asked too many questions recently."""
    now = time.time()
    with _sessions_lock:
        log = _request_log.setdefault(ip, deque())
        while log and now - log[0] > RATE_LIMIT_WINDOW:
            log.popleft()
        if len(log) >= RATE_LIMIT_QUERIES:
            return True
        log.append(now)
        # Forget IPs that have been quiet, so the dictionary cannot grow forever
        if len(_request_log) > 1000:
            for key in [k for k, v in _request_log.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW]:
                _request_log.pop(key, None)
    return False


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response

def debug_data(data, label="Data"):
    """Print debug information about the data structure for troubleshooting

    Args:
        data: The data object to inspect
        label: A descriptive label for the log entries
    """
    try:
        if data is None:
            logger.info(f"{label}: None")
            return

        if hasattr(data, "columns"):
            logger.info(f"{label} columns: {data.columns.tolist()}")
            if len(data) > 0:
                logger.info(f"{label} first row keys: {list(data.iloc[0].keys())}")
        elif isinstance(data, dict):
            logger.info(f"{label} keys: {list(data.keys())}")
        else:
            logger.info(f"{label} type: {type(data)}")
    except Exception as e:
        logger.error(f"Error debugging {label}: {e}")

def load_data(force_update=False):
    """Load and process urban data from various sources

    Args:
        force_update: If True, force refresh of all data sources ignoring cache

    Returns:
        bool: True if data loading succeeded, False otherwise
    """
    global gdf_data, agent, visualizer, is_updating, last_update_time

    if not _load_lock.acquire(blocking=False):
        logger.info("O încărcare de date este deja în curs; se ignoră cererea.")  # A load is already running
        return False

    try:
        is_updating = True
        logger.info("Începe colectarea și procesarea datelor...")  # Starting data collection and processing

        # 1. Collect all necessary data from various sources
        data_dict = collect_all_data(force_update=force_update)
        debug_data(data_dict, "Dicționar date colectate")

        # Check if we have the essential data about parks and sectors
        if not data_dict or 'parks_sectors' not in data_dict or data_dict['parks_sectors'].empty:
            logger.error("Eșec la colectarea datelor esențiale")  # Failed to collect essential data
            is_updating = False
            return False

        # 2. Process and integrate data from different sources
        logger.info("Procesare și integrare date...")  # Processing and integrating data
        processed_data = process_and_integrate_data(data_dict)
        debug_data(processed_data, "Date procesate")

        if processed_data is None or processed_data.empty:
            logger.error("Procesarea datelor a eșuat sau a produs rezultate goale")  # Data processing failed or produced empty result
            is_updating = False
            return False

        # Successfully processed - update global variables for app use
        gdf_data = processed_data

        # 3. Initialize the agent and visualizer with the processed data
        logger.info("Inițializare agent și vizualizator...")  # Initializing agent and visualizer
        agent = UrbanDataAgent(gdf_data)
        visualizer = UrbanVisualizer(gdf_data)

        last_update_time = datetime.now()
        is_updating = False
        logger.info("Încărcare date finalizată cu succes")  # Data loading completed successfully

        return True
    except Exception as e:
        logger.error(f"Eroare la încărcarea datelor: {e}")  # Error loading data
        is_updating = False
        return False
    finally:
        _load_lock.release()

def update_data_periodically():
    """Background thread function that updates data every 6 hours

    This runs continuously in the background and refreshes data
    to ensure the application has up-to-date information.
    """
    global is_updating

    while True:
        # Sleep for 6 hours between updates
        time.sleep(6 * 60 * 60)

        # Only update if not already updating
        if not is_updating:
            logger.info("Începe actualizarea programată a datelor...")  # Scheduled data update starting
            load_data(force_update=True)
            logger.info("Actualizarea programată a datelor s-a terminat.")  # Scheduled data update completed

@app.route('/')
def home():
    """Flask route handler for the home page

    Prepares all necessary data for rendering the main web interface,
    including maps, statistics and status information.
    """
    global gdf_data, visualizer, last_update_time

    # Create default park map
    map_html = None  # No map until data is loaded
    parks_count = 0
    sectors_data = []
    update_info = "Datele nu au fost încă încărcate"  # Data not yet loaded
    error_message = None

    try:
        if gdf_data is not None and visualizer is not None:
            logger.info("Creare hartă pentru pagina principală")  # Creating map for home page
            debug_data(gdf_data, "GeoDataFrame pentru vizualizare")

            # Check which columns are available for visualization
            available_cols = gdf_data.columns.tolist()
            column = 'aqi' if 'aqi' in available_cols else None

            # Create park map with available columns
            park_map = visualizer.create_simplified_map()
            map_path = visualizer.save_map(park_map, "current_view.html")
            map_html = os.path.basename(map_path)

            # Get park count for statistics
            parks_count = len(gdf_data)

            # Get sectors data if available for the sector breakdown
            if 'sector' in available_cols:
                sector_counts = gdf_data.groupby('sector').size().sort_values(ascending=False)
                sectors_data = [{"name": k, "count": v} for k, v in sector_counts.items()]
            else:
                sectors_data = [{"name": "Toate Sectoarele", "count": parks_count}]  # All sectors

            # Format last update time for display
            if last_update_time:
                update_info = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Eroare la crearea conținutului paginii principale: {e}")  # Error creating home page content
        error_message = f"Eroare la pregătirea vizualizării: {str(e)}"  # Error preparing visualization

    return render_template(
        'index.html',
        map_file=map_html,
        parks_count=parks_count,
        sectors=sectors_data,
        is_updating=is_updating,
        last_update=update_info,
        error_message=error_message
    )

# Add a route to serve visualization files
@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve generated visualization images from the visualizations directory

    This route allows the web interface to access generated visualizations
    like charts, maps, and other graphical data representations.
    """
    try:
        # Create visualization directory if it doesn't exist
        visualization_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)

        return send_from_directory(visualization_dir, filename)
    except Exception as e:
        logger.error(f"Eroare la servirea vizualizării {filename}: {e}")  # Error serving visualization
        return "Fișierul de vizualizare nu a fost găsit", 404  # Visualization file not found

# Update the process_query route to include visualization path
@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint for processing natural language queries about urban data

    Receives user queries via POST requests and passes them to the agent
    for processing, returning the response and any generated visualizations as JSON.
    """
    global agent

    if agent is None:
        return jsonify({'error': 'Agentul nu este inițializat. Vă rugăm să așteptați finalizarea încărcării datelor.'}), 503  # Agent not initialized

    if is_updating:
        return jsonify({'error': 'Datele sunt în curs de actualizare. Vă rugăm să încercați din nou în câteva momente.'}), 503  # Data is being updated

    if rate_limited(request.remote_addr):
        return jsonify({'error': 'Prea multe întrebări într-un timp scurt. Te rog așteaptă un minut.'}), 429  # Too many requests

    data = request.get_json(silent=True) or {}
    user_query = str(data.get('query', '')).strip()

    if not user_query:
        return jsonify({'error': 'Nu a fost furnizată nicio întrebare'}), 400  # No query provided
    if len(user_query) > MAX_QUERY_LENGTH:
        return jsonify({'error': f'Întrebarea este prea lungă (maximum {MAX_QUERY_LENGTH} caractere).'}), 400  # Query too long

    # Each browser gets its own conversation, identified by a random cookie
    sid = request.cookies.get('sid')
    new_cookie = False
    with _sessions_lock:
        if not sid or sid not in sessions:
            sid = uuid.uuid4().hex
            new_cookie = True
            sessions[sid] = []
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        sessions.move_to_end(sid)
        history = list(sessions[sid])

    try:
        logger.info(f"Procesare întrebare: {user_query}")  # Processing query
        response_data = agent.process_query(user_query, history=history)

        with _sessions_lock:
            sessions[sid] = response_data.get('history', history)

        reply = jsonify({
            'response': response_data.get('response', ''),
            'visualization': response_data.get('visualization'),
            'timestamp': datetime.now().isoformat()
        })
        if new_cookie:
            reply.set_cookie('sid', sid, httponly=True, samesite='Lax')
        return reply

    except Exception as e:
        logger.error(f"Eroare la procesarea întrebării: {e}")  # Error processing query
        return jsonify({'error': 'A apărut o eroare internă. Te rog încearcă din nou.'}), 500

@app.route('/maps/<path:filename>')
def serve_map(filename):
    """Serve generated map files from the data directory

    This route allows the web interface to access the generated
    map HTML files that are stored in the data directory.
    """
    try:
        return send_from_directory(config.DATA_DIR, filename)
    except Exception as e:
        logger.error(f"Eroare la servirea hărții {filename}: {e}")  # Error serving map
        return "Fișierul hărții nu a fost găsit", 404  # Map file not found

@app.route('/update-data', methods=['POST'])
def update_data():
    """API endpoint to manually trigger a data update

    This allows administrators to force a refresh of all data
    sources without waiting for the scheduled update.
    """
    global is_updating, _last_manual_update

    if is_updating:
        return jsonify({'message': 'Actualizarea datelor este deja în curs'}), 409  # Data update already in progress

    wait = int(UPDATE_COOLDOWN - (time.time() - _last_manual_update))
    if wait > 0:
        return jsonify({'message': f'Datele au fost actualizate recent. Încearcă din nou peste {wait} secunde.'}), 429
    _last_manual_update = time.time()

    # Start update in a thread to not block the response
    threading.Thread(target=load_data, args=(True,)).start()

    return jsonify({'message': 'Actualizarea datelor a început'})  # Data update started

@app.route('/api/map', methods=['GET'])
def get_map():
    """API endpoint to generate different types of maps on demand

    Allows the web interface to request different visualizations
    based on the map type and data column parameters.
    """
    global visualizer, gdf_data

    if visualizer is None or gdf_data is None:
        return jsonify({'error': 'Datele nu au fost încărcate'}), 503  # Data not loaded

    map_type = request.args.get('type', 'parks')
    column = request.args.get('column')

    try:
        if map_type == 'heatmap' and column in gdf_data.columns:
            map_obj = visualizer.create_heatmap(column)
        else:  # default to simplified parks map
            map_obj = visualizer.create_simplified_map()

        map_path = visualizer.save_map(map_obj, f"{map_type}_{int(time.time())}.html")
        map_filename = os.path.basename(map_path)

        return jsonify({
            'map_url': f'/maps/{map_filename}',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Eroare la crearea hărții: {e}")  # Error creating map
        return jsonify({'error': 'Harta nu a putut fi creată.'}), 500

@app.route('/debug-data')
def debug_data_endpoint():
    """Debugging endpoint to inspect the loaded data structure

    This endpoint is useful during development and troubleshooting
    to view the structure and contents of the loaded data.
    """
    if gdf_data is None:
        return jsonify({
            "status": "Nicio dată încărcată",  # No data loaded
            "is_updating": is_updating
        })

    # Without debug mode only a harmless status summary is shown
    if not config.DEBUG_MODE:
        return jsonify({
            "status": "Date încărcate",  # Data loaded
            "is_updating": is_updating,
            "rows": len(gdf_data),
            "last_update": last_update_time.isoformat() if last_update_time else None
        })

    try:
        columns = gdf_data.columns.tolist()
        sample = {}
        if len(gdf_data) > 0:
            for col in columns:
                try:
                    sample[col] = str(gdf_data[col].iloc[0])
                except Exception:
                    sample[col] = "Eroare la extragerea valorii"  # Error extracting value

        return jsonify({
            "status": "Date încărcate",  # Data loaded
            "is_updating": is_updating,
            "rows": len(gdf_data),
            "columns": columns,
            "sample": sample,
            "last_update": last_update_time.isoformat() if last_update_time else None
        })
    except Exception as e:
        logger.error(f"Eroare debug-data: {e}")
        return jsonify({"status": "Eroare"}), 500  # Error

def parse_arguments():
    """Parse command line arguments for application configuration

    Defines and processes all command-line options that control
    how the application starts and operates.
    """
    parser = argparse.ArgumentParser(description="Asistent Date Urbane București - Server Web")  # Bucharest Urban Data Assistant Web Server
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=int(os.getenv("PORT", 5000)),
        help="Portul pe care va rula serverul web (implicit: 5000)"
    )
    parser.add_argument(
        "--host",
        default=config.HOST,
        help="Adresa pe care va rula serverul web (implicit: 127.0.0.1, doar local)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Rulează aplicația Flask în modul debug"  # Run the Flask app in debug mode
    )
    parser.add_argument(
        "-f", "--force-update",
        action="store_true",
        help="Forțează actualizarea tuturor surselor de date la pornire, ignorând datele din cache"  # Force update all data sources
    )
    return parser.parse_args()

def main():
    """Main entry point for the web application

    Handles initialization, data loading, thread setup, and
    starts the Flask web server for the application.
    """
    args = parse_arguments()

    print("Inițializare Asistent Date Urbane București - Server Web...")  # Initializing Bucharest Urban Data Assistant Web Server

    # Load data in a separate thread to not block app startup
    threading.Thread(target=load_data, args=(args.force_update,)).start()

    # Start the periodic data update thread
    global update_thread
    update_thread = threading.Thread(target=update_data_periodically, daemon=True)
    update_thread.start()

    # The Flask debugger lets visitors run code, so never combine it with a public address
    host = args.host
    if args.debug and host not in ("127.0.0.1", "localhost"):
        logger.warning("Modul debug este permis doar local; se folosește 127.0.0.1.")
        host = "127.0.0.1"

    # Run the Flask app (no reloader: it would start the data loader twice)
    app.run(host=host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()
