"""Freshness checks shared by collection, integration and recommendations."""
import pandas as pd

from . import config


def fresh_air_mask(data):
    if 'air_observed_at' not in data or 'aqi' not in data:
        return pd.Series(False, index=data.index)
    observations = pd.to_datetime(data.air_observed_at, utc=True, errors='coerce', format='mixed')
    age = (pd.Timestamp.now(tz='UTC') - observations).dt.total_seconds()
    values = pd.to_numeric(data.aqi, errors='coerce')
    return (age >= 0) & (age <= config.MAX_AIR_AGE_HOURS * 3600) & values.notna() & (values >= 0)
