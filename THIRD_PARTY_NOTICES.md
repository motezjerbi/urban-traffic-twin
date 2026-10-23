# Third-party notices

Urban Traffic Twin uses, bundles or derives data from the works below. This file is an
attribution notice. It does not license the repository's own code: the repository has no
LICENSE yet (Q47), and its owner chooses one.

## Bundled code

### three.js (r128), MIT License

`src/traffic_twin/app/three.min.js` is an unmodified copy of three.js r128, used by the 2D/3D
map of the user app (`mock.html`). Its header keeps the original notice:
"Copyright 2010-2021 Three.js Authors", SPDX-License-Identifier: MIT.

> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
> to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

## Dependencies (not redistributed)

### Eclipse SUMO, Eclipse Public License 2.0

The simulator is Eclipse SUMO (Simulation of Urban MObility), installed as a dependency
through the `eclipse-sumo` wheel (`pip install -e ".[sim]"`). It is licensed under EPL-2.0
(with GPL-2.0-or-later as a secondary license). The repository does not redistribute SUMO.
The network files under `sumo/` are generated with SUMO's tools from the data below.

Python packages (FastAPI, pandas, Streamlit, Plotly, stable-baselines3, XGBoost, reportlab,
python-pptx and others; see `pyproject.toml`) are installed from PyPI under their own
licenses and are not redistributed.

## Data

### OpenStreetMap, Open Database License (ODbL) 1.0

The road network of the Pitié-Salpêtrière junction (`sumo/network/osm/pitie.osm` and the
curated plain XML under `sumo/network/pitie/` derived from it) comes from OpenStreetMap.

**© OpenStreetMap contributors**, available under the Open Database License:
<https://www.openstreetmap.org/copyright>. The derived network files are a derivative
database under the same license.

### City of Paris open data: "Comptage routier"

The demand calibration uses the City of Paris permanent road sensors, dataset
"Comptage routier - Données trafic issues des capteurs permanents"
(`comptages-routiers-permanents` on <https://opendata.paris.fr>, Ville de Paris). The extracts
are versioned under `data/reference/`. The dataset is published under the license shown on
its page (the Open Database License is expected; to be confirmed there before any
redistribution beyond this repository). Source: Ville de Paris.

## Authorship

- **Motez Jerbi**: repository owner; the user app (`src/traffic_twin/app/mock.html`), the
  Streamlit dashboard (`app.py`) and `run_mock.py`.
- **mrvolcon1**: the digital twin, calibration and validation, controllers, RL, ETA, API,
  evaluation, report and demo tooling.

Parts of the code were written with the help of an AI assistant (Claude); see the commit
trailers.
