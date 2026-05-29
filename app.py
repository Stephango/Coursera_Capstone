"""
Application BRVM — Portefeuille d'Actions Rentable
Lancez avec : streamlit run app.py
"""
import sys
import os

# Assurer que le répertoire racine est dans le path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.brvm_scraper import get_quotes, get_historical_prices
from src.data.fundamentals_data import BRVM_STOCKS, SECTEURS
from src.analysis.fundamentals import (
    build_fundamentals_df,
    filter_by_sector,
    get_top_picks,
    SEUIL_SOUS_EVALUE,
)
from src.analysis.optimizer import optimize_portfolio, monte_carlo_frontier

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BRVM — Portefeuille Intelligent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Couleurs BRVM (vert Afrique de l'Ouest)
PRIMARY = "#00843D"
SECONDARY = "#FFD100"

st.markdown(
    f"""
    <style>
    .stTabs [data-baseweb="tab-list"] {{ gap: 16px; }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600; font-size: 15px; padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }}
    .metric-card {{
        background: #f0f7f4; border-left: 4px solid {PRIMARY};
        padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 BRVM — Portefeuille d'Actions Rentable")
st.caption(
    "Bourse Régionale des Valeurs Mobilières · Données XOF (Franc CFA) · "
    "Cours en temps quasi-réel ou données de référence"
)

# ---------------------------------------------------------------------------
# Cache Streamlit
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Chargement des cotations BRVM…")
def cached_quotes() -> pd.DataFrame:
    return get_quotes()


@st.cache_data(ttl=86400, show_spinner="Génération de l'historique des prix…")
def cached_history(tickers: tuple[str, ...], n_days: int = 504) -> pd.DataFrame:
    return get_historical_prices(list(tickers), n_days)


@st.cache_data(show_spinner="Calcul des scores fondamentaux…")
def cached_fundamentals() -> pd.DataFrame:
    return build_fundamentals_df()


# ---------------------------------------------------------------------------
# Navigation — 4 onglets
# ---------------------------------------------------------------------------
tab_marche, tab_fondamentaux, tab_optim, tab_portfolio = st.tabs([
    "📊  Marché",
    "🔍  Analyse Fondamentale",
    "⚡  Optimisation",
    "📂  Portefeuille",
])


# ===========================================================================
# ONGLET 1 — MARCHÉ
# ===========================================================================
with tab_marche:
    df_quotes = cached_quotes()

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    n_total = len(df_quotes)
    n_hausse = int((df_quotes["variation"] > 0).sum())
    n_baisse = int((df_quotes["variation"] < 0).sum())
    variation_moy = float(df_quotes["variation"].mean())

    col_stat1.metric("Valeurs cotées", n_total)
    col_stat2.metric("En hausse", n_hausse, delta=f"+{n_hausse}")
    col_stat3.metric("En baisse", n_baisse, delta=f"-{n_baisse}", delta_color="inverse")
    col_stat4.metric(
        "Variation moyenne",
        f"{variation_moy:+.2f}%",
        delta=f"{variation_moy:+.2f}%",
        delta_color="normal",
    )

    st.divider()

    # Tableau principal des cotations
    st.subheader("Cours de Bourse — BRVM")
    display_cols = {
        "ticker": "Ticker",
        "nom": "Société",
        "secteur": "Secteur",
        "cours": "Cours (XOF)",
        "variation": "Variation (%)",
        "volume": "Volume",
    }
    df_display = df_quotes[[c for c in display_cols if c in df_quotes.columns]].rename(
        columns=display_cols
    )

    def color_variation(val):
        color = PRIMARY if val > 0 else ("#c0392b" if val < 0 else "gray")
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        df_display.style.map(color_variation, subset=["Variation (%)"]).format(
            {"Cours (XOF)": "{:,.0f}", "Variation (%)": "{:+.2f}%", "Volume": "{:,}"}
        ),
        use_container_width=True,
        height=420,
    )

    st.divider()

    col_gainers, col_losers = st.columns(2)

    with col_gainers:
        st.subheader("🏆 Top 5 Hausses")
        top5_up = df_quotes.nlargest(5, "variation")[["ticker", "cours", "variation"]]
        for _, row in top5_up.iterrows():
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>{row['ticker']}</strong> — {row['cours']:,.0f} XOF "
                f"<span style='color:{PRIMARY};font-weight:bold'>{row['variation']:+.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_losers:
        st.subheader("📉 Top 5 Baisses")
        top5_down = df_quotes.nsmallest(5, "variation")[["ticker", "cours", "variation"]]
        for _, row in top5_down.iterrows():
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>{row['ticker']}</strong> — {row['cours']:,.0f} XOF "
                f"<span style='color:#c0392b;font-weight:bold'>{row['variation']:+.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Répartition par secteur
    st.subheader("Répartition par Secteur")
    if "secteur" in df_quotes.columns:
        sect_counts = df_quotes.groupby("secteur").size().reset_index(name="Nombre")
        fig_pie = px.pie(
            sect_counts,
            names="secteur",
            values="Nombre",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)


# ===========================================================================
# ONGLET 2 — ANALYSE FONDAMENTALE
# ===========================================================================
with tab_fondamentaux:
    df_fund = cached_fundamentals()

    st.subheader("Filtres")
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        secteur_filtre = st.selectbox(
            "Secteur", ["Tous"] + SECTEURS, key="secteur_fondamentaux"
        )
    with col_f2:
        score_min = st.slider(
            "Score Valeur minimum", 0, 100, 0, key="score_min_slider"
        )

    df_filtered = filter_by_sector(df_fund, secteur_filtre)
    df_filtered = df_filtered[df_filtered["Score Valeur"] >= score_min]

    # KPI
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Valeurs analysées", len(df_filtered))
    col_k2.metric(
        "Sous-évaluées (score > 70)",
        int(df_filtered["Sous-évalué"].sum()),
    )
    col_k3.metric(
        "PER médian",
        f"{df_filtered['PER'].median():.1f}x" if not df_filtered.empty else "–",
    )
    col_k4.metric(
        "Rend. Div. moyen",
        f"{df_filtered['Rend. Div. (%)'].mean():.1f}%" if not df_filtered.empty else "–",
    )

    st.divider()

    # Tableau fondamentaux
    st.subheader("Tableau des Indicateurs Fondamentaux")

    def highlight_undervalued(row):
        color = "#e8f5e9" if row.get("Sous-évalué", False) else ""
        return [f"background-color: {color}"] * len(row)

    table_cols = [
        "Ticker", "Nom", "Secteur", "Pays",
        "PER", "PBR", "ROE (%)", "Rend. Div. (%)",
        "Prix Réf. (XOF)", "Score Valeur", "Sous-évalué",
    ]
    show_cols = [c for c in table_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[show_cols].style.apply(highlight_undervalued, axis=1).format(
            {
                "PER": "{:.1f}x",
                "PBR": "{:.2f}x",
                "ROE (%)": "{:.1f}%",
                "Rend. Div. (%)": "{:.1f}%",
                "Prix Réf. (XOF)": "{:,.0f}",
                "Score Valeur": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=420,
    )

    st.divider()

    # Graphique en barres du score valeur
    st.subheader("Scores de Valeur")
    if not df_filtered.empty:
        df_chart = df_filtered.copy().sort_values("Score Valeur")
        fig_bar = px.bar(
            df_chart,
            x="Score Valeur",
            y="Ticker",
            orientation="h",
            color="Score Valeur",
            color_continuous_scale=["#c0392b", "#f39c12", PRIMARY],
            range_color=[0, 100],
            hover_data=["Nom", "Secteur", "PER", "ROE (%)"],
            labels={"Score Valeur": "Score (0–100)", "Ticker": ""},
        )
        fig_bar.add_vline(
            x=SEUIL_SOUS_EVALUE, line_dash="dash", line_color="gray",
            annotation_text="Seuil 70", annotation_position="top right"
        )
        fig_bar.update_layout(
            height=max(350, len(df_chart) * 22),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top picks
    top_picks = get_top_picks(df_filtered)
    if not top_picks.empty:
        st.subheader("⭐ Valeurs Décotées Recommandées")
        st.dataframe(
            top_picks[["Ticker", "Nom", "Secteur", "Score Valeur", "PER", "Rend. Div. (%)"]].style.format(
                {"PER": "{:.1f}x", "Rend. Div. (%)": "{:.1f}%", "Score Valeur": "{:.1f}"}
            ),
            use_container_width=True,
        )


# ===========================================================================
# ONGLET 3 — OPTIMISATION
# ===========================================================================
with tab_optim:
    st.subheader("Optimisation de Portefeuille — Markowitz")

    all_tickers = list(BRVM_STOCKS.keys())

    col_o1, col_o2 = st.columns([2, 1])
    with col_o1:
        selected_tickers = st.multiselect(
            "Sélectionnez les valeurs à optimiser (min. 3)",
            options=all_tickers,
            default=["SGBCI", "PALMCI", "TOTALCI", "ONATEL", "CIE"],
            format_func=lambda t: f"{t} — {BRVM_STOCKS[t]['nom'][:40]}",
            key="optim_tickers",
        )
    with col_o2:
        method_labels = {
            "max_sharpe": "Maximiser le ratio de Sharpe",
            "min_volatility": "Minimiser la volatilité",
            "efficient_return": "Rendement cible",
        }
        method_key = st.selectbox(
            "Méthode d'optimisation",
            options=list(method_labels.keys()),
            format_func=lambda k: method_labels[k],
            key="optim_method",
        )

    target_return_val = None
    if method_key == "efficient_return":
        target_return_val = st.slider(
            "Rendement annuel cible (%)", 5, 30, 10, key="target_ret_slider"
        ) / 100

    col_days, col_rf = st.columns(2)
    n_days_optim = col_days.select_slider(
        "Historique utilisé",
        options=[126, 252, 378, 504],
        value=252,
        format_func=lambda d: {126: "6 mois", 252: "1 an", 378: "18 mois", 504: "2 ans"}.get(d, f"{d} j"),
        key="n_days_optim",
    )
    rf_rate = col_rf.number_input(
        "Taux sans risque (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5,
        key="rf_rate_input"
    ) / 100

    run_optim = st.button("🚀 Lancer l'optimisation", type="primary", key="run_optim_btn")

    if run_optim:
        if len(selected_tickers) < 2:
            st.error("Veuillez sélectionner au moins 2 valeurs.")
        else:
            with st.spinner("Optimisation en cours…"):
                prices_df = cached_history(tuple(selected_tickers), n_days_optim)
                try:
                    weights, (exp_ret, ann_vol, sharpe) = optimize_portfolio(
                        prices_df,
                        method=method_key,
                        target_return=target_return_val,
                        risk_free_rate=rf_rate,
                    )

                    # --- Métriques ---
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rendement Attendu", f"{exp_ret:.1%}")
                    c2.metric("Volatilité Annuelle", f"{ann_vol:.1%}")
                    c3.metric("Ratio de Sharpe", f"{sharpe:.2f}")

                    # --- Poids optimaux ---
                    weights_clean = {k: v for k, v in weights.items() if v > 0.005}
                    df_weights = pd.DataFrame(
                        [{"Ticker": k, "Poids (%)": v * 100} for k, v in weights_clean.items()]
                    )

                    col_pie, col_table = st.columns([1, 1])
                    with col_pie:
                        st.subheader("Allocation Optimale")
                        fig_pie_w = px.pie(
                            df_weights, names="Ticker", values="Poids (%)",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hole=0.35,
                        )
                        fig_pie_w.update_layout(height=320, margin=dict(t=20, b=20))
                        st.plotly_chart(fig_pie_w, use_container_width=True)

                    with col_table:
                        st.subheader("Poids Détaillés")
                        df_weights["Société"] = df_weights["Ticker"].map(
                            lambda t: BRVM_STOCKS.get(t, {}).get("nom", t)[:35]
                        )
                        st.dataframe(
                            df_weights[["Ticker", "Société", "Poids (%)"]].style.format(
                                {"Poids (%)": "{:.1f}%"}
                            ),
                            use_container_width=True,
                            height=280,
                        )

                    # --- Frontière efficiente (Monte Carlo) ---
                    st.subheader("Frontière Efficiente (5 000 simulations)")
                    with st.spinner("Simulation Monte Carlo…"):
                        mc_df = monte_carlo_frontier(
                            prices_df, n_simulations=5000, risk_free_rate=rf_rate
                        )

                    fig_frontier = go.Figure()
                    fig_frontier.add_trace(go.Scatter(
                        x=mc_df["Volatilité"] * 100,
                        y=mc_df["Rendement"] * 100,
                        mode="markers",
                        marker=dict(
                            color=mc_df["Sharpe"],
                            colorscale="Viridis",
                            size=4,
                            opacity=0.6,
                            colorbar=dict(title="Sharpe"),
                        ),
                        name="Portefeuilles simulés",
                        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[ann_vol * 100],
                        y=[exp_ret * 100],
                        mode="markers",
                        marker=dict(color="red", size=14, symbol="star"),
                        name="Portefeuille optimal",
                        hovertemplate=f"Optimal<br>Vol: {ann_vol:.1%}<br>Ret: {exp_ret:.1%}<extra></extra>",
                    ))
                    fig_frontier.update_layout(
                        xaxis_title="Volatilité annuelle (%)",
                        yaxis_title="Rendement attendu (%)",
                        height=420,
                        margin=dict(t=20, b=40),
                        legend=dict(x=0.02, y=0.98),
                    )
                    st.plotly_chart(fig_frontier, use_container_width=True)

                except RuntimeError as e:
                    st.error(f"Erreur d'optimisation : {e}")
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")
    else:
        st.info(
            "👆 Sélectionnez vos valeurs, choisissez une méthode, puis cliquez sur "
            "**Lancer l'optimisation**."
        )


# ===========================================================================
# ONGLET 4 — PORTEFEUILLE
# ===========================================================================
with tab_portfolio:
    st.subheader("Mon Portefeuille")

    # Initialisation du portefeuille en session_state
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []  # list[dict]

    # --- Ajout de position ---
    with st.expander("➕ Ajouter / Modifier une position", expanded=len(st.session_state.portfolio) == 0):
        col_a1, col_a2, col_a3, col_a4 = st.columns([2, 1, 1, 1])
        ticker_add = col_a1.selectbox(
            "Valeur",
            options=list(BRVM_STOCKS.keys()),
            format_func=lambda t: f"{t} — {BRVM_STOCKS[t]['nom'][:35]}",
            key="port_ticker_add",
        )
        qty_add = col_a2.number_input("Quantité", min_value=1, value=10, step=1, key="port_qty_add")
        price_add = col_a3.number_input(
            "Prix d'achat (XOF)",
            min_value=1,
            value=int(BRVM_STOCKS[ticker_add]["prix_ref"]),
            step=50,
            key="port_price_add",
        )

        col_a4.markdown("&nbsp;")
        if col_a4.button("Ajouter", type="primary", key="port_add_btn"):
            # Mise à jour si ticker existant, sinon ajout
            existing = next(
                (i for i, p in enumerate(st.session_state.portfolio) if p["ticker"] == ticker_add),
                None,
            )
            entry = {"ticker": ticker_add, "quantite": int(qty_add), "prix_achat": float(price_add)}
            if existing is not None:
                st.session_state.portfolio[existing] = entry
                st.success(f"Position {ticker_add} mise à jour.")
            else:
                st.session_state.portfolio.append(entry)
                st.success(f"{ticker_add} ajouté au portefeuille.")
            st.rerun()

    # --- Suppression ---
    if st.session_state.portfolio:
        tickers_port = [p["ticker"] for p in st.session_state.portfolio]
        ticker_del = st.selectbox(
            "Supprimer une position", ["–"] + tickers_port, key="port_del_select"
        )
        if ticker_del != "–" and st.button("🗑 Supprimer", key="port_del_btn"):
            st.session_state.portfolio = [
                p for p in st.session_state.portfolio if p["ticker"] != ticker_del
            ]
            st.rerun()

    # --- Affichage du portefeuille ---
    if not st.session_state.portfolio:
        st.info("Votre portefeuille est vide. Ajoutez des positions ci-dessus.")
    else:
        df_quotes_port = cached_quotes()
        quotes_map = df_quotes_port.set_index("ticker")["cours"].to_dict()

        rows_port = []
        for pos in st.session_state.portfolio:
            t = pos["ticker"]
            cours_actuel = quotes_map.get(t, BRVM_STOCKS[t]["prix_ref"])
            valeur_achat = pos["prix_achat"] * pos["quantite"]
            valeur_actuelle = cours_actuel * pos["quantite"]
            pl = valeur_actuelle - valeur_achat
            pl_pct = (pl / valeur_achat) * 100 if valeur_achat > 0 else 0.0
            rows_port.append({
                "Ticker": t,
                "Société": BRVM_STOCKS[t]["nom"][:35],
                "Secteur": BRVM_STOCKS[t]["secteur"],
                "Qté": pos["quantite"],
                "Prix achat (XOF)": pos["prix_achat"],
                "Cours actuel (XOF)": cours_actuel,
                "Valeur actuelle (XOF)": valeur_actuelle,
                "P&L (XOF)": pl,
                "P&L (%)": pl_pct,
            })

        df_port = pd.DataFrame(rows_port)
        total_invested = sum(p["prix_achat"] * p["quantite"] for p in st.session_state.portfolio)
        total_current = df_port["Valeur actuelle (XOF)"].sum()
        total_pl = total_current - total_invested
        total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0

        # KPI portefeuille
        kp1, kp2, kp3 = st.columns(3)
        kp1.metric("Valeur Actuelle", f"{total_current:,.0f} XOF")
        kp2.metric("Investi", f"{total_invested:,.0f} XOF")
        kp3.metric(
            "P&L Total",
            f"{total_pl:+,.0f} XOF",
            delta=f"{total_pl_pct:+.2f}%",
            delta_color="normal",
        )

        st.divider()

        # Tableau P&L
        def color_pl(val):
            color = PRIMARY if val > 0 else ("#c0392b" if val < 0 else "gray")
            return f"color: {color}; font-weight: bold"

        st.dataframe(
            df_port.style.map(color_pl, subset=["P&L (XOF)", "P&L (%)"]).format(
                {
                    "Prix achat (XOF)": "{:,.0f}",
                    "Cours actuel (XOF)": "{:,.0f}",
                    "Valeur actuelle (XOF)": "{:,.0f}",
                    "P&L (XOF)": "{:+,.0f}",
                    "P&L (%)": "{:+.2f}%",
                }
            ),
            use_container_width=True,
            height=300,
        )

        st.divider()

        # Répartition du portefeuille + courbe de performance
        col_p1, col_p2 = st.columns([1, 2])

        with col_p1:
            st.subheader("Répartition")
            fig_alloc = px.pie(
                df_port,
                names="Ticker",
                values="Valeur actuelle (XOF)",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            fig_alloc.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_alloc, use_container_width=True)

        with col_p2:
            st.subheader("Performance — 6 mois")
            port_tickers = [p["ticker"] for p in st.session_state.portfolio]
            hist_df = cached_history(tuple(port_tickers), n_days=126)

            # Pondération proportionnelle à la valeur actuelle
            alloc_weights = (
                df_port.set_index("Ticker")["Valeur actuelle (XOF)"] / total_current
            )

            port_returns = (hist_df.pct_change().dropna() * alloc_weights).sum(axis=1)
            port_cumul = (1 + port_returns).cumprod()

            # Indice BRVM synthétique (équipondéré sur toutes les valeurs disponibles)
            all_hist = cached_history(tuple(BRVM_STOCKS.keys()), n_days=126)
            brvm_idx = (1 + all_hist.pct_change().dropna().mean(axis=1)).cumprod()

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=port_cumul.index, y=(port_cumul - 1) * 100,
                name="Mon Portefeuille", line=dict(color=PRIMARY, width=2.5)
            ))
            fig_perf.add_trace(go.Scatter(
                x=brvm_idx.index, y=(brvm_idx - 1) * 100,
                name="BRVM Composite (synthétique)",
                line=dict(color="gray", width=1.5, dash="dash")
            ))
            fig_perf.add_hline(y=0, line_color="black", line_width=0.8, opacity=0.5)
            fig_perf.update_layout(
                yaxis_title="Performance (%)",
                xaxis_title="",
                height=300,
                margin=dict(t=20, b=20),
                legend=dict(x=0.02, y=0.98),
                hovermode="x unified",
            )
            st.plotly_chart(fig_perf, use_container_width=True)

# Footer
st.divider()
st.caption(
    "⚠️ Cette application est fournie à titre éducatif uniquement. "
    "Les données et analyses présentées ne constituent pas un conseil en investissement. "
    "Investir en bourse comporte des risques de perte en capital."
)
