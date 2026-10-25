"""Evidence-based rankings: missing measurements never become good scores."""
import pandas as pd


def rank_parks(data, criteria=None, simulation=False):
    criteria = ["air_quality"] if criteria is None else criteria
    definitions = {
        "air_quality": ("aqi", 0.4),
        "noise": ("noise_level_db", 0.3),
        "congestion": ("congestion_level", 0.3),
    }
    if not criteria or any(item not in definitions for item in criteria):
        raise ValueError("Criterii disponibile: air_quality, noise, congestion.")
    if not simulation and any(item in ("noise", "congestion") for item in criteria):
        raise ValueError("Zgomotul și traficul sunt simulate și nu pot fundamenta o recomandare reală.")
    frame = data.copy()
    columns = [definitions[item][0] for item in set(criteria)]
    if any(column not in frame for column in columns):
        raise ValueError("Nu există măsurători pentru toate criteriile solicitate.")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    frame = frame.dropna(subset=columns)
    if "air_quality" in criteria and 'sensor_distance_meters' not in frame:
        raise ValueError('Distanța senzorului lipsește; nu pot verifica apropierea măsurătorii.')
    if "sensor_distance_meters" in frame and "air_quality" in criteria:
        from . import config
        frame = frame[frame.sensor_distance_meters.notna() & (frame.sensor_distance_meters <= config.MAX_SENSOR_DISTANCE_METERS)]
    if "air_observed_at" in frame and "air_quality" in criteria:
        from . import config
        observations = pd.to_datetime(frame.air_observed_at, utc=True, errors="coerce", format='mixed')
        age_seconds = (pd.Timestamp.now(tz="UTC") - observations).dt.total_seconds()
        frame = frame[(age_seconds >= 0) & (age_seconds <= config.MAX_AIR_AGE_HOURS * 3600)]
    elif "air_quality" in criteria:
        raise ValueError("Lipsește ora observațiilor de aer; actualizează datele înainte de recomandare.")
    if frame.empty:
        raise ValueError("Nu există suficiente măsurători recente și apropiate pentru o recomandare.")
    frame["total_score"] = 0.0
    weight_sum = sum(definitions[item][1] for item in set(criteria))
    for item in set(criteria):
        column, weight = definitions[item]
        span = frame[column].max() - frame[column].min()
        score = 1 - (frame[column] - frame[column].min()) / span if span > 0 else 0.5
        frame["total_score"] += score * weight / weight_sum
    return frame.sort_values(["total_score", "name"], ascending=[False, True], kind="stable")
