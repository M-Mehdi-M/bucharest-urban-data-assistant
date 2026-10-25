"""Sector comparisons from dataset counts and verified boundary geometry."""
from pathlib import Path
from uuid import uuid4

import geopandas as gpd

from . import config


def sector_metrics(data):
    table = data.groupby('sector').size().to_frame('park_count')
    for column, target in [('aqi', 'avg_aqi'), ('noise_level_db', 'avg_noise')]:
        if column in data:
            table[target] = data.groupby('sector')[column].mean()
    if Path(config.SECTORS_FILE).exists():
        boundaries = gpd.read_file(config.SECTORS_FILE).dissolve(by='sector').to_crs(config.METRIC_CRS)
        table['area_sqkm'] = boundaries.geometry.area / 1_000_000
        table['parks_per_sqkm'] = table.park_count / table.area_sqkm
    return table.reset_index()


def column_for(metric):
    mapping = {'parks': 'park_count', 'park_count': 'park_count', 'air': 'avg_aqi',
               'air_quality': 'avg_aqi', 'avg_aqi': 'avg_aqi', 'noise': 'avg_noise',
               'density': 'parks_per_sqkm', 'parks_per_sqkm': 'parks_per_sqkm'}
    if metric not in mapping:
        raise ValueError('Alege parks, air, noise sau density. Nu există date de populație verificate.')
    return mapping[metric]


def select_metrics(data, metric='parks', sectors=None):
    table = sector_metrics(data)
    column = column_for(metric)
    if sectors:
        import re
        wanted = []
        for sector in sectors:
            match = re.search(r'[1-6]', str(sector))
            if not match:
                raise ValueError('Precizează un sector de la 1 la 6.')
            wanted.append(f'Sector {match.group()}')
        table = table[table.sector.isin(wanted)]
    if column not in table or not table[column].notna().any():
        raise ValueError('Nu există date verificate pentru acest indicator. Densitatea necesită limitele administrative, iar aerul necesită observații recente.')
    return table.dropna(subset=[column]).sort_values(column, ascending=column in ('avg_aqi', 'avg_noise')), column


def report(data, metric='parks', sectors=None):
    table, column = select_metrics(data, metric, sectors)
    units = {'park_count': 'înregistrări de parc', 'avg_aqi': 'Airly CAQI mediu',
             'avg_noise': 'dB simulați', 'parks_per_sqkm': 'înregistrări de parc/km²'}
    lines = [f"Comparație: {units[column]}."]
    for _, row in table.iterrows():
        lines.append(f"{row['sector']}: {row[column]:.2f}")
    lines.append('Numărul de înregistrări nu măsoară suprafața verde. Mediile de aer se bazează pe senzorii din apropiere; același senzor poate deservi mai multe parcuri.')
    return '\n'.join(lines)


def chart(data, metric='parks'):
    import matplotlib.pyplot as plt
    table, column = select_metrics(data, metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        ax.bar(table.sector, table[column], color='#176b50')
        ax.set_ylabel(column)
        ax.set_title('București · comparație din datele disponibile')
        fig.tight_layout()
        directory = Path(config.BASE_DIR) / 'visualizations'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'sectors_{uuid4().hex}.png'
        fig.savefig(path, dpi=150)
        return str(path)
    finally:
        plt.close(fig)
