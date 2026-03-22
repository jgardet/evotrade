"""
Streamlit Dashboard — real-time view of the agent's progress.
Shows strategy performance, component leaderboard, and insight feed.
"""
import asyncio
import json
from datetime import datetime

import asyncpg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import settings

st.set_page_config(
    page_title="Freqtrade Strategy Agent",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()
    return loop


def run_async(coro):
    loop = get_event_loop()
    return loop.run_until_complete(coro)


async def fetch(query: str, *args) -> list[dict]:
    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        rows = await conn.fetch(query, *args)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def q(query, *args):
    return run_async(fetch(query, *args))


# ============================================================
# Layout
# ============================================================

st.title("📈 Freqtrade Strategy Research Agent")
st.caption(f"Live knowledge base — refreshed at {datetime.now().strftime('%H:%M:%S')}")

# Top metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_runs = q("SELECT COUNT(*) as n FROM backtest_runs")[0]["n"]
total_strategies = q("SELECT COUNT(*) as n FROM strategies")[0]["n"]
total_insights = q("SELECT COUNT(*) as n FROM insights WHERE active=TRUE")[0]["n"]
hypotheses_complete = q("SELECT COUNT(*) as n FROM hypotheses WHERE status='complete'")[0]["n"]
best_score_row = q("SELECT MAX(composite_score) as s FROM backtest_runs WHERE disqualified=FALSE")
best_score = best_score_row[0]["s"] if best_score_row else None

col1.metric("Total Backtest Runs", total_runs)
col2.metric("Strategies Generated", total_strategies)
col3.metric("Active Insights", total_insights)
col4.metric("Experiments Complete", hypotheses_complete)
col5.metric("Best Composite Score", f"{best_score:.3f}" if best_score else "—")

st.divider()

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Best Strategies", "🧩 Component Library",
    "💡 Insights", "🔬 Experiments", "📊 Score History", "⚙️ Config"
])

# ---- Tab 1: Best Strategies ----
with tab1:
    st.subheader("Top Strategies by Composite Score")
    rows = q("""
        SELECT s.name, b.pair, b.timeframe, b.composite_score, b.sharpe,
               b.max_drawdown, b.win_rate, b.trade_count, b.regime_trend, b.regime_vol,
               b.created_at
        FROM backtest_runs b
        JOIN strategies s ON s.id = b.strategy_id
        WHERE b.disqualified = FALSE AND b.is_holdout = FALSE
        ORDER BY b.composite_score DESC NULLS LAST
        LIMIT 20
    """)
    if rows:
        df = pd.DataFrame(rows)
        df["max_drawdown"] = df["max_drawdown"].apply(lambda x: f"{x:.1%}" if x else "—")
        df["win_rate"] = df["win_rate"].apply(lambda x: f"{x:.1%}" if x else "—")
        df["composite_score"] = df["composite_score"].apply(lambda x: f"{x:.3f}" if x else "—")
        df["sharpe"] = df["sharpe"].apply(lambda x: f"{x:.2f}" if x else "—")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No qualified strategies yet. Run some experiments first.")

# ---- Tab 2: Component Library ----
with tab2:
    st.subheader("Component Leaderboard")
    rows = q("SELECT * FROM v_component_leaderboard LIMIT 30")
    if rows:
        df = pd.DataFrame(rows)
        if "avg_sharpe_contrib" in df.columns:
            fig = px.bar(
                df.dropna(subset=["avg_sharpe_contrib"]).head(20),
                x="name",
                y="avg_sharpe_contrib",
                color="category",
                title="Average Sharpe Contribution by Component",
                labels={"avg_sharpe_contrib": "Avg Sharpe Contribution", "name": "Component"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No component scores yet.")

    st.subheader("All Components")
    comps = q("SELECT name, category, origin, description, created_at FROM components ORDER BY created_at DESC")
    if comps:
        st.dataframe(pd.DataFrame(comps), use_container_width=True)

# ---- Tab 3: Insights ----
with tab3:
    st.subheader("Active Knowledge Base Insights")
    insights = q("""
        SELECT statement, insight_type, confidence, created_at, last_updated
        FROM insights
        WHERE active=TRUE
        ORDER BY confidence DESC
        LIMIT 50
    """)
    if insights:
        for ins in insights:
            conf = ins["confidence"]
            color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
            with st.expander(f"{color} [{ins['insight_type']}] {ins['statement'][:80]}..."):
                st.write(f"**Statement:** {ins['statement']}")
                st.write(f"**Confidence:** {conf:.0%}")
                st.write(f"**Type:** {ins['insight_type']}")
                st.write(f"**Created:** {ins['created_at']}")
                st.write(f"**Last updated:** {ins['last_updated']}")
    else:
        st.info("No insights synthesized yet.")

# ---- Tab 4: Experiments / Hypotheses ----
with tab4:
    st.subheader("Experiment History")
    hyps = q("""
        SELECT type, rationale, status, outcome, priority_score,
               outcome_summary, created_at, completed_at
        FROM hypotheses
        ORDER BY created_at DESC
        LIMIT 50
    """)
    if hyps:
        df = pd.DataFrame(hyps)
        # Color-code by outcome
        st.dataframe(df, use_container_width=True)

        # Experiment type distribution
        type_counts = q("SELECT type, COUNT(*) as n FROM hypotheses GROUP BY type")
        if type_counts:
            fig = px.pie(
                pd.DataFrame(type_counts),
                names="type", values="n",
                title="Experiment Type Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No experiments yet.")

# ---- Tab 5: Score History ----
with tab5:
    st.subheader("Composite Score Over Time")
    history = q("""
        SELECT br.created_at, br.composite_score, br.pair, br.timeframe, s.name,
               (br.raw_output->>'walk_forward')::boolean AS is_walk_forward
        FROM backtest_runs br
        JOIN strategies s ON s.id = br.strategy_id
        WHERE br.disqualified = FALSE AND br.is_holdout = FALSE
        ORDER BY br.created_at
    """)
    if history:
        df = pd.DataFrame(history)
        main_runs = df[~df["is_walk_forward"].fillna(False)]
        wf_runs = df[df["is_walk_forward"].fillna(False)]

        fig = px.scatter(
            main_runs, x="created_at", y="composite_score",
            color="pair", hover_data=["name", "timeframe"],
            title="Strategy Score Progression (in-sample runs)",
            trendline="lowess",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not wf_runs.empty:
            st.subheader("Walk-Forward Window Scores")
            fig2 = px.box(
                wf_runs, x="pair", y="composite_score", color="timeframe",
                title="Walk-Forward Score Distribution by Pair",
            )
            st.plotly_chart(fig2, use_container_width=True)

        sharpe_data = q("SELECT sharpe, pair FROM backtest_runs WHERE disqualified=FALSE AND sharpe IS NOT NULL")
        if sharpe_data:
            fig3 = px.histogram(
                pd.DataFrame(sharpe_data), x="sharpe", color="pair",
                title="Sharpe Ratio Distribution", nbins=30,
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No scored runs yet.")

# ---- Tab 6: Config ----
with tab6:
    st.subheader("Active Configuration")
    stored = q("SELECT key, value, created_at FROM agent_config ORDER BY key")
    if stored:
        st.caption("These values were locked on first run. Holdout dates cannot be changed.")
        st.dataframe(pd.DataFrame(stored), use_container_width=True)
    else:
        st.info("No configuration stored yet — agent hasn't run its first experiment.")

    st.divider()
    st.subheader("Walk-Forward Windows")
    from src.config import settings as cfg
    if cfg.walk_forward_enabled:
        windows = cfg.walk_forward_windows
        wf_df = pd.DataFrame([
            {"window": i + 1, "start": str(s), "end": str(e),
             "days": (e - s).days}
            for i, (s, e) in enumerate(windows)
        ])
        st.write(f"**{len(windows)} windows** ({cfg.walk_forward_window_days}d wide, "
                 f"{cfg.walk_forward_step_days}d step) | "
                 f"Pass threshold: {cfg.walk_forward_min_pass_ratio:.0%}")
        st.dataframe(wf_df, use_container_width=True)

        # Show coverage: what % of each window has data
        wf_coverage = q("""
            SELECT
                raw_output->>'wf_window' AS window,
                COUNT(*) AS runs,
                AVG(composite_score) AS avg_score,
                SUM(CASE WHEN NOT disqualified AND composite_score > 0 THEN 1 ELSE 0 END) AS passed
            FROM backtest_runs
            WHERE raw_output->>'walk_forward' = 'true'
            GROUP BY raw_output->>'wf_window'
            ORDER BY window
        """)
        if wf_coverage:
            st.subheader("Walk-Forward Results by Window")
            st.dataframe(pd.DataFrame(wf_coverage), use_container_width=True)
    else:
        st.info("Walk-forward validation is disabled (WALK_FORWARD_ENABLED=false)")

# Auto-refresh every 60 seconds
st.markdown(
    """
    <script>
    setTimeout(() => window.location.reload(), 60000);
    </script>
    """,
    unsafe_allow_html=True,
)
