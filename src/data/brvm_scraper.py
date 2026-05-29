"""
Collecte de données BRVM : scraping live + cache JSON + historique synthétique.
"""
import json
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.fundamentals_data import BRVM_STOCKS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "brvm_quotes.json")
CACHE_TTL_SECONDS = 3600  # 1 heure

BRVM_URL = "https://www.brvm.org/en/cours-de-bourse/0/title"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> Optional[dict]:
    """Charge le cache JSON si valide (< TTL)."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", 0)
        if time.time() - ts < CACHE_TTL_SECONDS:
            return data
    except Exception:
        pass
    return None


def _save_cache(quotes: list[dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {"timestamp": time.time(), "quotes": quotes}
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Live scraping
# ---------------------------------------------------------------------------

def _scrape_brvm() -> Optional[list[dict]]:
    """
    Tente de scraper le tableau de cours de la BRVM.
    Retourne None si le scraping échoue.
    """
    try:
        resp = requests.get(BRVM_URL, headers=REQUEST_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table")
        if table is None:
            return None

        rows = table.find_all("tr")
        quotes = []
        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 5:
                continue
            try:
                ticker = cols[0].strip()
                last_price_str = cols[2].replace("\xa0", "").replace(",", ".").replace(" ", "")
                change_str = cols[3].replace(",", ".").replace("%", "").strip()
                volume_str = cols[4].replace("\xa0", "").replace(" ", "").replace(",", "")

                last_price = float(last_price_str) if last_price_str else None
                change_pct = float(change_str) if change_str else 0.0
                volume = int(volume_str) if volume_str.isdigit() else 0

                if last_price is None or last_price <= 0:
                    continue

                quotes.append({
                    "ticker": ticker,
                    "nom": cols[1] if len(cols) > 1 else ticker,
                    "cours": last_price,
                    "variation": change_pct,
                    "volume": volume,
                })
            except (ValueError, IndexError):
                continue
        return quotes if quotes else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback data from fundamentals_data.py
# ---------------------------------------------------------------------------

def _build_fallback_quotes() -> list[dict]:
    """Construit les cotations de secours à partir des données statiques."""
    rng = np.random.default_rng(int(datetime.now().strftime("%Y%m%d%H")))
    quotes = []
    for ticker, info in BRVM_STOCKS.items():
        daily_vol = info["vol_annuelle"] / np.sqrt(252)
        variation = float(rng.normal(0, daily_vol) * 100)
        cours = info["prix_ref"] * (1 + variation / 100)
        quotes.append({
            "ticker": ticker,
            "nom": info["nom"],
            "secteur": info["secteur"],
            "pays": info["pays"],
            "cours": round(cours, 0),
            "variation": round(variation, 2),
            "volume": int(rng.integers(100, 5000)),
        })
    return quotes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_quotes() -> pd.DataFrame:
    """
    Retourne les cotations BRVM sous forme de DataFrame.
    Ordre de priorité : cache valide → scraping live → données de secours.
    """
    cached = _load_cache()
    if cached is not None:
        quotes = cached["quotes"]
    else:
        scraped = _scrape_brvm()
        if scraped:
            # Enrichir avec les métadonnées statiques
            for q in scraped:
                ticker = q.get("ticker", "")
                meta = BRVM_STOCKS.get(ticker, {})
                q.setdefault("secteur", meta.get("secteur", "Autre"))
                q.setdefault("pays", meta.get("pays", "–"))
            quotes = scraped
            _save_cache(quotes)
        else:
            quotes = _build_fallback_quotes()
            _save_cache(quotes)

    df = pd.DataFrame(quotes)

    # Enrichir les colonnes manquantes depuis BRVM_STOCKS
    if "secteur" not in df.columns:
        df["secteur"] = df["ticker"].map(lambda t: BRVM_STOCKS.get(t, {}).get("secteur", "Autre"))
    if "pays" not in df.columns:
        df["pays"] = df["ticker"].map(lambda t: BRVM_STOCKS.get(t, {}).get("pays", "–"))

    # Assurer les types
    df["cours"] = pd.to_numeric(df["cours"], errors="coerce")
    df["variation"] = pd.to_numeric(df["variation"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


def get_historical_prices(tickers: list[str], n_days: int = 504) -> pd.DataFrame:
    """
    Génère un historique synthétique de prix (GBM) pour la liste de tickers.
    Le dernier prix correspond au prix de référence dans fundamentals_data.

    Parameters
    ----------
    tickers : liste de codes ticker BRVM
    n_days  : nombre de jours boursiers (252 ≈ 1 an, 504 ≈ 2 ans)

    Returns
    -------
    DataFrame avec les dates en index et les tickers en colonnes.
    """
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    price_data: dict[str, np.ndarray] = {}

    for ticker in tickers:
        info = BRVM_STOCKS.get(ticker, {})
        price_ref = info.get("prix_ref", 1000)
        mu_annual = info.get("rendement_annuel", 0.08)
        sigma_annual = info.get("vol_annuelle", 0.20)

        mu_daily = mu_annual / 252
        sigma_daily = sigma_annual / np.sqrt(252)

        # Seed déterministe par ticker → même historique entre chargements
        seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        # Mouvement brownien géométrique (log-rendements)
        log_returns = rng.normal(
            mu_daily - 0.5 * sigma_daily**2,
            sigma_daily,
            size=n_days,
        )

        # Normaliser pour que le dernier prix = prix_ref
        cum_log = np.cumsum(log_returns)
        cum_log = cum_log - cum_log[-1]  # ancrer à 0 à la fin
        prices = price_ref * np.exp(cum_log)

        price_data[ticker] = prices

    return pd.DataFrame(price_data, index=dates)
