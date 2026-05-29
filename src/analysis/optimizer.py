"""
Optimisation de portefeuille : Markowitz (PyPortfolioOpt) + Monte Carlo.
"""
from typing import Optional
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


def _get_ef(prices_df: pd.DataFrame, risk_free_rate: float = 0.03):
    """Construit mu et S, retourne un objet EfficientFrontier."""
    from pypfopt import expected_returns, risk_models, EfficientFrontier

    mu = expected_returns.mean_historical_return(prices_df, frequency=252)
    S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
    return EfficientFrontier(mu, S), mu, S


def optimize_portfolio(
    prices_df: pd.DataFrame,
    method: str = "max_sharpe",
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.03,
) -> tuple[dict, tuple]:
    """
    Optimise le portefeuille selon la méthode choisie.

    Parameters
    ----------
    prices_df      : DataFrame avec les prix historiques (dates × tickers)
    method         : 'max_sharpe' | 'min_volatility' | 'efficient_return'
    target_return  : rendement cible annuel (uniquement pour 'efficient_return')
    risk_free_rate : taux sans risque annuel (défaut 3 %)

    Returns
    -------
    (weights_dict, (expected_return, annual_vol, sharpe_ratio))
    """
    if prices_df.shape[1] < 2:
        raise ValueError("Au moins 2 actifs requis pour l'optimisation.")

    ef, mu, S = _get_ef(prices_df, risk_free_rate)

    try:
        if method == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == "min_volatility":
            ef.min_volatility()
        elif method == "efficient_return":
            if target_return is None:
                target_return = float(mu.mean())
            # Clamp target_return entre le min et max rendements attendus
            target_return = float(np.clip(target_return, mu.min() + 1e-4, mu.max() - 1e-4))
            ef.efficient_return(target_return)
        else:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
    except Exception as exc:
        raise RuntimeError(f"Optimisation échouée ({method}): {exc}") from exc

    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return weights, perf


def monte_carlo_frontier(
    prices_df: pd.DataFrame,
    n_simulations: int = 5000,
    risk_free_rate: float = 0.03,
) -> pd.DataFrame:
    """
    Simule n_simulations portefeuilles aléatoires et retourne un DataFrame
    avec les colonnes : Rendement, Volatilité, Sharpe, Poids (liste).

    Parameters
    ----------
    prices_df     : DataFrame avec les prix historiques
    n_simulations : nombre de simulations (défaut 5 000)
    risk_free_rate: taux sans risque annuel

    Returns
    -------
    DataFrame trié par ratio de Sharpe décroissant.
    """
    from pypfopt import expected_returns, risk_models

    mu = expected_returns.mean_historical_return(prices_df, frequency=252).values
    cov = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf().values
    n = len(mu)

    rng = np.random.default_rng(42)
    results = []

    for _ in range(n_simulations):
        w = rng.dirichlet(np.ones(n))
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        results.append({"Rendement": ret, "Volatilité": vol, "Sharpe": sharpe})

    return pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
