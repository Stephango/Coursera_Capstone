# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Coursera IBM Data Science Professional Certificate capstone project. The repository contains Jupyter notebooks for geospatial neighborhood analysis and machine learning clustering, focused on Toronto, Canada.

There are two distinct analysis tracks:

1. **Toronto Neighborhood Segmentation & Clustering** — Scrapes postal code data from Wikipedia, merges with geospatial coordinates, queries the Foursquare API for nearby venues, performs one-hot encoding on venue categories, then applies K-Means clustering to group neighborhoods. Visualized with Folium interactive maps.

2. **Best Location for an African Restaurant in Toronto** — Extends the above with Toronto census/wellbeing data (`wellbeing_toronto.csv`) to incorporate African population percentage and household income as clustering features alongside Foursquare venue density. Uses the elbow method to select the optimal number of K-Means clusters (k=6), then examines each cluster to recommend a target neighborhood (Weston).

## Running Notebooks

These notebooks were developed in IBM Watson Studio / JupyterLab with a conda environment. Key environment setup commands that appear inline in the notebooks:

```bash
conda install -c conda-forge folium=0.5.0 --yes
conda install -c conda-forge geopy --yes
```

To launch Jupyter locally:

```bash
jupyter notebook
# or
jupyter lab
```

To run a single notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute "The Battle of Neighborhood Project Capstone.ipynb"
```

## Key Libraries

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` (`KMeans`) | Clustering |
| `folium` | Interactive map rendering |
| `geopy` (`Nominatim`) | Address → lat/lng geocoding |
| `matplotlib` (`cm`, `colors`) | Cluster color mapping |
| `requests` | Foursquare REST API calls |

## Data Sources and Files

- **Wikipedia** — Scraped via `pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')` to get Toronto postal codes, boroughs, and neighborhoods.
- **`Geospatial_Coordinates.csv` / `geospatial_coordinates.csv`** — Local CSV mapping postal codes to lat/lng (note: two filenames exist across notebooks with different capitalizations; ensure the correct file is present).
- **`wellbeing_toronto.csv`** — Toronto census data including after-tax household income and African population percentage by neighborhood.
- **Foursquare API** — Venue category data fetched via the v2 `venues/explore` endpoint. `CLIENT_ID` and `CLIENT_SECRET` are hardcoded directly in the notebooks (not in environment variables).

## Core Analysis Pattern

Every notebook follows the same pipeline:

1. **Collect** — Web-scrape Wikipedia postal data + load local CSVs + call Foursquare API (`getNearbyVenues()`)
2. **Clean** — Drop `Not assigned` boroughs, groupby postal code to join multi-neighborhood rows
3. **Merge** — Inner join postal/neighborhood data with geospatial coordinates on `Postal Code`
4. **Engineer features** — `pd.get_dummies` on Foursquare venue categories; compute `mean()` per neighborhood for top venues
5. **Cluster** — `KMeans(n_clusters=k, random_state=0).fit(...)` with optional elbow-method loop over k=3..10
6. **Visualize** — `folium.Map` + `folium.CircleMarker` with rainbow color scheme per cluster
7. **Interpret** — Examine each cluster's characteristics to form a business recommendation

## Foursquare API

The Foursquare API version used is `20180604` (v2 legacy). Credentials are embedded as plain strings in the notebooks. If the credentials are expired or rate-limited, the `getNearbyVenues()` function will return empty results silently — check the raw `requests.get(url).json()` response before assuming clustering output is correct.

---

## BRVM Portfolio Application (`app.py`)

A Streamlit web app for building and managing a profitable BRVM (West African stock exchange) portfolio.

### Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Architecture

```
app.py                        # Streamlit entry point — 4 tabs
src/
  data/
    fundamentals_data.py      # Static dict of ~41 BRVM stocks (PER, PBR, ROE, div yield)
    brvm_scraper.py           # Live scraping + 1h JSON cache + GBM price history
  analysis/
    fundamentals.py           # Min-max scoring (0–100 composite value score)
    optimizer.py              # PyPortfolioOpt Markowitz + Monte Carlo frontier
data/cache/                   # Runtime JSON cache (gitignored)
```

### Key Design Decisions

- **Data flow**: `get_quotes()` tries: fresh cache → live scrape → static fallback. Always returns a DataFrame, never raises.
- **Price history**: `get_historical_prices()` generates 504 business days via GBM with a deterministic seed per ticker (`hashlib.md5(ticker)`), anchored so the last price equals `prix_ref`. Call is `@st.cache_data`-wrapped in `app.py`.
- **Covariance**: Always uses Ledoit-Wolf shrinkage (`CovarianceShrinkage.ledoit_wolf()`) — prevents singular matrix with correlated or few assets.
- **EfficientFrontier re-instantiation**: `optimize_portfolio()` creates a fresh `EfficientFrontier(mu, S)` on every call — pypfopt EF objects are stateful and cannot be reused after solving.
- **Session state**: Portfolio held as `st.session_state.portfolio` (list of dicts with `ticker`, `quantite`, `prix_achat`). Persists across tab switches within a session.
- **Pandas 3.x**: `style.map()` is used everywhere (not the deprecated `applymap`).

### Adding a New BRVM Stock

Add an entry to `BRVM_STOCKS` in `src/data/fundamentals_data.py` with keys: `nom`, `secteur`, `pays`, `prix_ref`, `per`, `pbr`, `roe`, `rendement_dividende`, `vol_annuelle`, `rendement_annuel`. It will automatically appear in all tabs.
