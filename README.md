# Verde București · Bucharest Urban Data Assistant

A Romanian conversational assistant for Bucharest parks, with interactive maps, nearby air measurements, weather forecasts, and routes over real OpenStreetMap networks.

## Run locally

Use Python 3.11–3.13. Create a virtual environment, install dependencies, and copy `.env.example` to `.env`:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv/Scripts/python.exe -X utf8 app.py
```

Set `GEMINI_API_KEY` for conversations. `AIRLY_API_KEY` enables sensor downloads. The map can load without a model key. The server defaults to `http://127.0.0.1:5000`.

This workspace also has an ignored, project-local Python runtime prepared during development. If the old `env` references a missing Python installation, run:

```powershell
.runtime/python/python.exe -s -X utf8 app.py
```

`python main.py` starts the terminal interface. `python app.py -f` forces source refresh. `--port` changes the port; `--host` changes the bind address. Debug mode always binds locally.

## What changed

- A responsive map-and-chat interface, live readiness status, examples, and expandable evidence details.
- Data refresh builds a complete replacement snapshot before publishing it. Queries continue using the previous snapshot during refresh or provider failure.
- Startup can publish reprocessed source caches before attempting network updates. A new installation still requires an initial park download.
- Source cards report collection progress and distinguish live downloads, usable caches, provider quotas, and unavailable data. `/api/status` includes the same diagnostics and the number of parks with usable air readings.
- Park downloads use bounded requests with a second public Overpass provider. An overloaded OpenStreetMap server does not prevent independent air and weather collection.
- An air cache without provider observation timestamps is refreshed even when its file is recent. Offline sensors are excluded before finding the nearest functioning sensor.
- Airly refreshes default to six hours and at most 20 measurement requests per refresh. Sensor selection spreads requests across park coverage; other recent cached observations are retained. Provider quota headers and `Retry-After` are persisted and respected even by manual refresh and restarts. Configure `AIR_REFRESH_MINUTES` and `AIRLY_MAX_MEASUREMENTS_PER_REFRESH` for your account's allowance.
- Traffic and noise simulations are disabled by default. Enable `ENABLE_SIMULATED_DATA=true` only for demonstrations; answers and the interface label this mode.
- Park downloads preserve OSM IDs, polygon geometry, and calculated hectares. Same-name parks are not automatically merged. Older point caches remain usable for basic exploration when migration fails.
- Missing air observations stay missing. Unknown, future, over-24-hour-old, and over-3-km-away readings cannot support normal air recommendations. Sensor distance uses local UTM metres. Observations are checked again when a question runs.
- Airly CAQI is identified explicitly. It is not interpreted using US AQI categories.
- Walking, cycling, and driving use separate cached OSM networks. Missing networks produce an unavailable result; no synthetic streets are generated. Tours use real walking legs.
- Largest-park answers use polygon area. Sector density uses verified boundary geometry. Population-based comparisons require a future verified population dataset.
- Model selection requests JSON constrained by a tool-selection schema. Server-side checks reject unknown arguments, wrong types, and excessive analysis sizes. Invalid model output prompts clarification.
- Statistical failures cannot become predetermined claims about pollution reduction.
- Generated maps have a dedicated directory. Dataset caches cannot be fetched through the map-serving route.

## Architecture

| Module | Responsibility |
| --- | --- |
| `app.py` | HTTP API, sessions, atomic snapshots, background refresh, WSGI factory |
| `src/agent.py` | Tool registry, LangGraph workflow, remaining descriptive/forecast tools |
| `src/contracts.py` | Query/argument validation and model selection schema |
| `src/recommendations.py` | Evidence-gated, relative park rankings |
| `src/routing.py` | Mode-specific real-network routing and route artifacts |
| `src/park_data.py` | Park polygons, OSM identity, sector assignment and boundaries |
| `src/sectors.py` | Counts, verified density, sector comparisons and charts |
| `src/forecast.py` | Current-date forecast loading and weather/air alignment by timestamp |
| `src/data_collection.py` | Source downloads, caches and optional simulations |
| `src/air_source.py`, `src/air_budget.py` | Observation-aware air cache, sensor selection and persistent quota handling |
| `src/osm_source.py` | Bounded OSM downloads with fallback providers and XML geometry parsing |
| `src/source_health.py`, `src/storage.py` | Collection diagnostics and atomic cache publication |
| `src/data_processing.py` | Local-projection integration, provenance, missing-data handling |
| `src/spatial_analysis.py` | Geometric coverage and exploratory candidate zones |
| `src/visualization.py` | Map and chart rendering |
| `static/`, `templates/` | Browser interface; local CSS and JavaScript |

Air observations preserve provider observation time separately from fetch time. Processing preserves sensor identity, source, index, and distance. Scores compare the available parks; they are not absolute environmental grades. Rankings require sufficient evidence and can legitimately be unavailable.

## API

| Route | Behavior |
| --- | --- |
| `GET /` | Map, statistics, conversation interface |
| `GET /api/status` | Readiness, refresh state, snapshot time, counts and configuration status; no credentials |
| `POST /api/query` | JSON `{ "query": "..." }`; returns response, optional visualization, evidence and snapshot metadata |
| `POST /api/query` | Add `"reset_session": true` to start a fresh conversation |
| `GET /api/map?type=parks` | Park map |
| `GET /api/map?type=heatmap&column=aqi` | Heatmap when usable data exists |
| `POST /update-data` | Start a background refresh; 202 accepted, 409 already running, 429 cooldown |
| `GET /maps/<filename>.html` | Generated map artifacts only |
| `GET /visualizations/<filename>.png` | Generated image artifacts |
| `GET /debug-data` | Compatibility alias for status |

Queries are limited to 500 characters and 20 requests/minute per IP. Request bodies are limited to 16 KiB. Per-browser conversations retain ten exchanges. Limits and sessions are bounded and held in memory. Query and status responses are not cacheable.

`/api/status.sources` reports `parks`, `air_quality`, and `weather`. Each entry has a state and explanation; air quota delays include `retry_at`. `ready: true` means that parks can be explored, while `degraded: true` explains limits on individual sources. Refreshing a source successfully does not imply every provider is available.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src app.py tests --select F821,F822,F823,E9
node --check static/app.js
```

Tests use synthetic geospatial fixtures and mock provider/model access. They do not consume API quota. Coverage includes malformed JSON, tool argument bounds, conversation isolation/reset, snapshot failure, stale/missing sensor data, nearest-join ties, polygon retention, metric coverage, travel-mode selection, parallel routing edges, and unavailable networks.

`requirements.txt` declares bounded application dependencies. `requirements.lock.txt` records the resolved application **and development** environment verified on Windows/Python 3.12; other Python versions may need the bounded requirements to resolve compatible wheels. CI checks Python 3.11, 3.12, and 3.13 without external service calls.

## Serving beyond development

Use a production WSGI server, with the factory so initialization actually runs:

```powershell
waitress-serve --call --host=127.0.0.1 --port=5000 app:create_app
```

Or, using the project-local runtime:

```powershell
.runtime/python/python.exe -s -X utf8 -m waitress --call --host=127.0.0.1 --port=5000 app:create_app
```

Use one application process: snapshots, plotting coordination, refresh scheduling, sessions, and quotas are process-local. Multiple processes require a shared task/state design. Plotting remains serialized to protect Matplotlib state; this is a deliberate remaining concurrency limit.

The ngrok workflow is **manual only** and uses Waitress. Normal pushes run checks rather than opening public tunnels. The application has no account system; keep it local or use authenticated hosting for private/public deployments. Generated artifacts are not private per-user documents.

## Current limits

- Coverage is a clearly labeled geometric buffer, not a walking isochrone or population coverage. Local projection fixes measurement distortion, but barriers and park entrances remain unmodeled.
- Routing snaps representative park points to street nodes, with the gap disclosed. Exact entrances and accessibility conditions are not verified. Driving duration is unavailable without verified speed data. Walking/cycling durations assume 5/15 km/h; they are estimates.
- Candidate park zones are exploratory. Ownership, land use, population, and construction feasibility are not checked.
- A sensor near a park does not establish conditions inside it. Several parks can share a sensor. The 3-km and 24-hour gates are application eligibility rules, not scientific guarantees.
- Forecasts are city-level predictions. Airly CAQI and European AQI forecast values are different indices.
- Existing caches without observation timestamps cannot be treated as fresh measurements. Refresh them with a valid Airly key. Static geometry refresh requires OpenStreetMap availability.
- Some legacy descriptive and weather tools remain in `agent.py`; full localization and further extraction are future work. External map tiles and Folium libraries need network access.
- The language model can still misclassify a request or phrase a summary incorrectly. Evidence-rich recommendation, routing, tour, details, coverage, and exploratory-location answers preserve deterministic tool text.

Data © OpenStreetMap contributors · Air quality by Airly · Forecasts by Open-Meteo · Language understanding by Google Gemini.
