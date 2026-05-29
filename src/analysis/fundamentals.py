"""
Analyse fondamentale : scoring et filtrage des valeurs BRVM.
"""
import pandas as pd
import numpy as np

from src.data.fundamentals_data import BRVM_STOCKS


# Pondération du score composite (somme = 1)
WEIGHTS = {
    "per": 0.25,       # Price/Earnings  → faible = bien
    "pbr": 0.20,       # Price/Book      → faible = bien
    "roe": 0.30,       # Return on Equity → élevé = bien
    "rendement_dividende": 0.25,  # Div. yield → élevé = bien
}

SEUIL_SOUS_EVALUE = 70  # score > 70 → valeur décotée intéressante


def build_fundamentals_df() -> pd.DataFrame:
    """
    Construit un DataFrame avec tous les indicateurs fondamentaux
    et le score composite pour chaque valeur BRVM.
    """
    rows = []
    for ticker, info in BRVM_STOCKS.items():
        rows.append({
            "Ticker": ticker,
            "Nom": info["nom"],
            "Secteur": info["secteur"],
            "Pays": info["pays"],
            "PER": info.get("per"),
            "PBR": info.get("pbr"),
            "ROE (%)": info.get("roe"),
            "Rend. Div. (%)": info.get("rendement_dividende"),
            "Prix Réf. (XOF)": info.get("prix_ref"),
        })
    df = pd.DataFrame(rows)
    df = _add_scores(df)
    return df


def _min_max_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise en [0, 1]. Si invert=True, une valeur faible donne un score élevé."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    normalized = (series - mn) / (mx - mn)
    return 1 - normalized if invert else normalized


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    score_per = _min_max_normalize(df["PER"], invert=True)     # faible PER = bien
    score_pbr = _min_max_normalize(df["PBR"], invert=True)     # faible PBR = bien
    score_roe = _min_max_normalize(df["ROE (%)"])              # élevé ROE = bien
    score_div = _min_max_normalize(df["Rend. Div. (%)"])       # élevé yield = bien

    df["Score Valeur"] = (
        WEIGHTS["per"] * score_per
        + WEIGHTS["pbr"] * score_pbr
        + WEIGHTS["roe"] * score_roe
        + WEIGHTS["rendement_dividende"] * score_div
    ) * 100

    df["Score Valeur"] = df["Score Valeur"].round(1)
    df["Sous-évalué"] = df["Score Valeur"] >= SEUIL_SOUS_EVALUE

    return df.sort_values("Score Valeur", ascending=False).reset_index(drop=True)


def filter_by_sector(df: pd.DataFrame, secteur: str) -> pd.DataFrame:
    if secteur == "Tous":
        return df
    return df[df["Secteur"] == secteur].reset_index(drop=True)


def get_top_picks(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df[df["Sous-évalué"]].head(n)
