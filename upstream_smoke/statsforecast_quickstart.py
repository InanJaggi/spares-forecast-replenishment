# Minimal example from the statsforecast README ("Quick Start" section),
# https://github.com/Nixtla/statsforecast (Apache-2.0). Run unchanged except for
# the print() calls and timing, to prove the pinned install works.
import time

import statsforecast
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

print(f"statsforecast {statsforecast.__version__}")
t0 = time.perf_counter()

df = AirPassengersDF
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq='ME',
)
sf.fit(df)
print(sf.predict(h=12, level=[95]))

print(f"elapsed {time.perf_counter() - t0:.2f}s")
