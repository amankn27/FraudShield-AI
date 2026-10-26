"""FraudShield AI analyst dashboard.

Run from the repository root:
    streamlit run dashboard/app.py

The dashboard talks to FastAPI only; it never connects directly to PostgreSQL.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.api_client import APIClientError, FraudShieldAPIClient  # noqa: E402
from dashboard.utils import daily_frame, records_to_frame, shap_contributions_to_frame  # noqa: E402

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
INVESTIGATION_STATUSES = ["NEW", "IN_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"]
DEFAULT_API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FraudShield AI | Analyst Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: rgba(110, 125, 145, 0.08); padding: 14px 16px; border-radius: 12px;}
      .hero {padding: 1.1rem 1.35rem; border: 1px solid rgba(125,135,150,.28); border-radius: 16px;
             background: linear-gradient(110deg, rgba(35,75,130,.11), rgba(22,140,110,.07)); margin-bottom: 1rem;}
      .muted {color: #7b8494; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except APIClientError as exc:
        st.error(f"{label}: {exc}")
    except Exception as exc:  # defensive UI boundary
        st.error(f"{label}: unexpected dashboard error ({type(exc).__name__}).")
    return None


def _format_score(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _render_shap(result: dict[str, Any], *, key_prefix: str) -> None:
    if not result.get("shap_available"):
        st.info("SHAP explanation was not returned for this prediction.")
        return
    st.caption(result.get("interpretation_note") or "SHAP explains model behavior; it does not establish causation.")
    shap_frame = shap_contributions_to_frame(result)
    if shap_frame.empty:
        st.info("The API did not return SHAP feature contributions.")
        return
    fig = px.bar(
        shap_frame,
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale=["#2563eb", "#94a3b8", "#e34a56"],
        labels={"shap_value": "SHAP contribution", "feature": "Feature"},
        title="Local feature contributions",
        hover_data={"feature_value": ":.5g", "direction": True, "shap_value": ":.5g"},
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(height=max(300, 32 * len(shap_frame)), coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_shap")
    st.dataframe(shap_frame.sort_values("shap_value", key=lambda x: x.abs(), ascending=False), use_container_width=True, hide_index=True)
    if result.get("shap_additivity_error") is not None:
        st.caption(f"SHAP additivity check error: {float(result['shap_additivity_error']):.3e}")


def _render_metrics_cards(summary: dict[str, Any]) -> None:
    total = int(summary.get("total_predictions") or 0)
    flagged = int(summary.get("flagged_predictions") or 0)
    high = int(summary.get("high_risk_predictions") or 0)
    mean_score = summary.get("mean_model_score")
    p95 = summary.get("p95_latency_ms")
    columns = st.columns(5)
    columns[0].metric("Predictions logged", f"{total:,}")
    columns[1].metric("Flagged by model", f"{flagged:,}", help="Predicted positive; not confirmed fraud labels.")
    columns[2].metric("High / critical risk", f"{high:,}")
    columns[3].metric("Mean model score", _format_score(mean_score))
    columns[4].metric("P95 latency", f"{float(p95):.1f} ms" if p95 is not None else "—")


def _fetch_high_risk(client: FraudShieldAPIClient, risk_level: str, limit: int = 200) -> dict[str, Any] | None:
    return _safe_call(f"Could not load {risk_level.lower()} risk transactions", client.list_predictions,
                      limit=limit, offset=0, risk_level=risk_level)


def _render_detail_and_update(client: FraudShieldAPIClient, prediction_id: str, key_prefix: str) -> None:
    detail = _safe_call("Could not load prediction detail", client.prediction_detail, prediction_id)
    if not detail:
        return
    st.markdown("#### Prediction detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model score", _format_score(detail.get("model_score")))
    c2.metric("Risk score", f"{float(detail.get('risk_score_0_100') or 0):.2f}/100")
    c3.metric("Risk level", str(detail.get("risk_level", "—")))
    c4.metric("Investigation", str(detail.get("investigation_status", "—")))
    st.write({
        "prediction_id": detail.get("prediction_id"),
        "created_at": detail.get("created_at"),
        "model_version": detail.get("model_version"),
        "policy_version": detail.get("policy_version"),
        "decision": detail.get("decision"),
        "prediction_latency_ms": detail.get("prediction_latency_ms"),
    })
    shap_summary = detail.get("shap_summary") or {}
    if shap_summary:
        st.markdown("##### Stored SHAP summary")
        contrib = []
        for direction_key in ("top_positive_contributions", "top_negative_contributions"):
            for item in shap_summary.get(direction_key) or []:
                contrib.append({
                    "feature": item.get("feature"), "shap_value": item.get("shap_value"),
                    "direction": item.get("direction", direction_key),
                })
        if contrib:
            df = pd.DataFrame(contrib)
            df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
            df = df.sort_values("shap_value", key=lambda x: x.abs(), ascending=True)
            fig = px.bar(df, x="shap_value", y="feature", orientation="h", color="shap_value",
                         color_continuous_scale=["#2563eb", "#94a3b8", "#e34a56"],
                         labels={"shap_value": "SHAP contribution", "feature": "Feature"})
            fig.add_vline(x=0, line_color="gray")
            fig.update_layout(height=max(260, 30 * len(df)), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_stored_shap")
    events = detail.get("investigation_events") or []
    if events:
        st.markdown("##### Investigation audit history")
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

    st.markdown("##### Update investigation")
    with st.form(f"{key_prefix}_update_form"):
        current_status = detail.get("investigation_status", "NEW")
        default_index = INVESTIGATION_STATUSES.index(current_status) if current_status in INVESTIGATION_STATUSES else 0
        new_status = st.selectbox("New status", INVESTIGATION_STATUSES, index=default_index, key=f"{key_prefix}_status")
        note = st.text_input("Analyst note (optional)", max_chars=2000, key=f"{key_prefix}_note")
        submitted = st.form_submit_button("Save investigation update", type="primary")
    if submitted:
        updated = _safe_call("Could not update investigation", client.update_investigation, prediction_id, new_status, note)
        if updated:
            st.success(f"Investigation status updated to {updated.get('investigation_status')}.")
            st.rerun()


def page_overview(client: FraudShieldAPIClient, health: dict[str, Any] | None) -> None:
    st.markdown('<div class="hero"><h2>Fraud overview</h2><p>Operational snapshot of prediction records, risk decisions, and analyst review outcomes.</p></div>', unsafe_allow_html=True)
    summary = _safe_call("Could not load overview metrics", client.monitoring_summary)
    if summary is None:
        st.info("The overview requires a configured, reachable database. Start PostgreSQL, set DATABASE_URL in .env, apply migrations, and restart FastAPI.")
        return
    _render_metrics_cards(summary)
    confirmed = int((summary.get("investigation_status_counts") or {}).get("CONFIRMED_FRAUD", 0))
    false_positive = int((summary.get("investigation_status_counts") or {}).get("FALSE_POSITIVE", 0))
    reviewed = confirmed + false_positive
    columns = st.columns(3)
    columns[0].metric("Confirmed fraud", f"{confirmed:,}")
    columns[1].metric("Reviewed false positives", f"{false_positive:,}")
    columns[2].metric("Reviewed outcomes", f"{reviewed:,}")
    st.caption("Predicted-positive counts are model decisions. Confirmed fraud and false-positive counts come from analyst review status.")

    daily = daily_frame(summary.get("daily_predictions") or [])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Prediction volume")
        if not daily.empty:
            fig = px.line(daily, x="date", y="total_predictions", markers=True, labels={"date": "Date", "total_predictions": "Predictions"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction history has been recorded yet.")
    with right:
        st.markdown("#### Risk distribution")
        risk_counts = summary.get("risk_level_counts") or {}
        if risk_counts:
            risk_df = pd.DataFrame([{"Risk level": key, "Count": value} for key, value in risk_counts.items()])
            fig = px.bar(risk_df, x="Risk level", y="Count", color="Risk level", category_orders={"Risk level": RISK_LEVELS})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk decisions are available yet.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Investigation queue")
        status_counts = summary.get("investigation_status_counts") or {}
        if status_counts:
            frame = pd.DataFrame([{"Status": key, "Count": value} for key, value in status_counts.items()])
            st.bar_chart(frame.set_index("Status"))
        else:
            st.info("No investigation records are available yet.")
    with right:
        st.markdown("#### Models in use")
        versions = summary.get("model_version_counts") or {}
        if versions:
            st.dataframe(pd.DataFrame([{"Model version": k, "Predictions": v} for k, v in versions.items()]), use_container_width=True, hide_index=True)
        else:
            st.info("No model versions are recorded yet.")
    for note in summary.get("notes", []):
        st.caption(f"Note: {note}")


def page_transaction_analyzer(client: FraudShieldAPIClient) -> None:
    st.markdown('<div class="hero"><h2>Transaction analyzer</h2><p>Submit one 30-feature transaction to the FastAPI model and review its SHAP contribution summary.</p></div>', unsafe_allow_html=True)
    example_path = PROJECT_ROOT / "reports" / "api" / "example_request.json"
    try:
        example = json.loads(example_path.read_text(encoding="utf-8"))
    except Exception:
        example = {"Time": 0.0, "Amount": 100.0, **{f"V{i}": 0.0 for i in range(1, 29)}}

    with st.form("transaction_analyzer_form"):
        st.markdown("#### Transaction fields")
        col1, col2 = st.columns(2)
        payload: dict[str, float] = {
            "Time": col1.number_input("Time (seconds from dataset start)", min_value=0.0, value=float(example.get("Time", 0.0)), format="%.3f"),
            "Amount": col2.number_input("Amount", min_value=0.0, value=float(example.get("Amount", 100.0)), format="%.4f"),
        }
        with st.expander("Advanced anonymized PCA features (V1–V28)", expanded=False):
            cols = st.columns(4)
            for idx in range(1, 29):
                name = f"V{idx}"
                payload[name] = cols[(idx - 1) % 4].number_input(
                    name, value=float(example.get(name, 0.0)), format="%.6f", key=f"analyzer_{name}"
                )
        include_shap = st.checkbox("Calculate SHAP explanation", value=True)
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)

    if submitted:
        result = _safe_call("Prediction failed", client.predict, payload, include_shap=include_shap, top_k=8)
        if result:
            st.session_state["last_prediction_result"] = result
            st.session_state["last_prediction_payload_summary"] = {"Time": payload["Time"], "Amount": payload["Amount"]}
    result = st.session_state.get("last_prediction_result")
    if not result:
        st.info("Enter transaction values and select Analyze transaction. The default values are from the project's example request.")
        return

    st.markdown("#### Prediction result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model score", _format_score(result.get("model_score")))
    c2.metric("Risk score", f"{float(result.get('risk_score_0_100') or 0):.2f}/100")
    c3.metric("Risk level", str(result.get("risk_level", "—")))
    c4.metric("Decision", str(result.get("decision", "—")))
    st.write({
        "model_version": result.get("model_version"), "policy_version": result.get("policy_version"),
        "latency_ms": result.get("latency_ms"), "prediction_id": result.get("prediction_id"),
        "persistence_status": result.get("persistence_status"), "request_id": result.get("request_id"),
    })
    st.warning(result.get("score_interpretation", "Model score is not a calibrated real-world fraud probability."))
    if result.get("persistence_status") != "saved":
        st.info("This prediction was not saved to the history database. Configure PostgreSQL to enable historical review and monitoring.")
    if include_shap and result.get("shap_available"):
        st.markdown("#### SHAP explanation")
        _render_shap(result, key_prefix="analyzer")


def page_high_risk(client: FraudShieldAPIClient) -> None:
    st.markdown('<div class="hero"><h2>High-risk transactions</h2><p>Prioritize HIGH and CRITICAL model decisions for human review.</p></div>', unsafe_allow_html=True)
    status_filter = st.selectbox("Investigation status", ["ALL", *INVESTIGATION_STATUSES], key="highrisk_status_filter")
    with st.spinner("Loading flagged transactions..."):
        high = _fetch_high_risk(client, "HIGH", 200)
        critical = _fetch_high_risk(client, "CRITICAL", 200)
    if high is None or critical is None:
        return
    records = (high.get("items") or []) + (critical.get("items") or [])
    if status_filter != "ALL":
        records = [row for row in records if row.get("investigation_status") == status_filter]
    frame = records_to_frame(records)
    st.caption(f"Loaded {len(frame):,} recent high/critical records. API totals for the unfiltered bands: HIGH={high.get('total', 0):,}; CRITICAL={critical.get('total', 0):,}. Each band is capped at 200 rows in this view.")
    if frame.empty:
        st.info("No transactions match this filter.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config={
        "model_score": st.column_config.NumberColumn("Model score", format="%.4f"),
        "risk_score_0_100": st.column_config.NumberColumn("Risk score", format="%.2f"),
        "prediction_latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.2f"),
        "created_at": st.column_config.DatetimeColumn("Created at", format="YYYY-MM-DD HH:mm:ss"),
    })
    chosen = st.selectbox("Select a prediction to inspect", frame["prediction_id"].astype(str).tolist(), key="highrisk_selected_id")
    if chosen:
        _render_detail_and_update(client, chosen, "highrisk")


def page_history(client: FraudShieldAPIClient) -> None:
    st.markdown('<div class="hero"><h2>Historical predictions</h2><p>Search the persisted prediction log and follow the investigation audit history.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    risk = c1.selectbox("Risk level", ["ALL", *RISK_LEVELS], key="history_risk")
    status = c2.selectbox("Investigation status", ["ALL", *INVESTIGATION_STATUSES], key="history_status")
    page_num = c3.number_input("Page", min_value=1, value=1, step=1, key="history_page")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1, key="history_page_size")
    offset = (int(page_num) - 1) * int(page_size)
    response = _safe_call("Could not load history", client.list_predictions, limit=int(page_size), offset=offset,
                          risk_level=risk, investigation_status=status)
    if response is None:
        st.info("Configure PostgreSQL and start the API to browse historical records.")
        return
    records = response.get("items") or []
    total = int(response.get("total", 0))
    st.caption(f"Showing {offset + 1 if total else 0}–{min(offset + len(records), total)} of {total:,} matching predictions.")
    frame = records_to_frame(records)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if total == 0:
        st.info("No matching prediction records. Analyze a transaction to create the first record.")
        return
    if offset >= total:
        st.warning("This page is beyond the last result. Choose an earlier page.")
        return
    chosen = st.selectbox("Prediction ID", frame["prediction_id"].astype(str).tolist(), key="history_selected_id")
    if chosen:
        _render_detail_and_update(client, chosen, "history")


def page_monitoring(client: FraudShieldAPIClient, health: dict[str, Any] | None, model_info: dict[str, Any] | None) -> None:
    st.markdown('<div class="hero"><h2>Monitoring dashboard</h2><p>Monitor service readiness, inference latency, model versions, score volume, and analyst workload.</p></div>', unsafe_allow_html=True)
    if health:
        c1, c2, c3 = st.columns(3)
        c1.metric("API readiness", "READY" if health.get("ready") else "DEGRADED")
        c2.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
        c3.metric("Database", "Connected" if health.get("database_ready") else "Not ready")
        st.write({k: health.get(k) for k in ("service", "database_enabled", "database_required", "missing_artifacts")})
    if model_info:
        with st.expander("Active model metadata"):
            st.json(model_info)
    summary = _safe_call("Could not load operational metrics", client.monitoring_summary)
    if summary is None:
        st.info("Operational metrics require a reachable configured database. API health and model metadata can still be inspected above.")
        return
    _render_metrics_cards(summary)
    daily = daily_frame(summary.get("daily_predictions") or [])
    if not daily.empty:
        st.markdown("#### Throughput and latency by day")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.line(daily, x="date", y=["total_predictions", "flagged_predictions", "high_risk_predictions"], markers=True,
                                    labels={"value": "Count", "date": "Date", "variable": "Series"}), use_container_width=True)
        with right:
            st.plotly_chart(px.line(daily, x="date", y="average_latency_ms", markers=True,
                                    labels={"average_latency_ms": "Average latency (ms)", "date": "Date"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Risk level counts")
        risks = summary.get("risk_level_counts") or {}
        if risks:
            st.plotly_chart(px.bar(pd.DataFrame([{"risk_level": k, "count": v} for k, v in risks.items()]),
                                   x="risk_level", y="count", color="risk_level", category_orders={"risk_level": RISK_LEVELS}), use_container_width=True)
    with right:
        st.markdown("#### Investigation workload")
        statuses = summary.get("investigation_status_counts") or {}
        if statuses:
            st.plotly_chart(px.bar(pd.DataFrame([{"status": k, "count": v} for k, v in statuses.items()]),
                                   x="status", y="count", color="status"), use_container_width=True)
    st.caption("Input-feature drift is not measured from these logs because raw transaction features are intentionally not persisted. Model-quality metrics require trustworthy outcome labels after investigation.")


def main() -> None:
    st.sidebar.markdown("# 🛡️ FraudShield AI")
    st.sidebar.caption("Analyst console")
    api_url = st.sidebar.text_input("FastAPI base URL", value=st.session_state.get("api_url", DEFAULT_API_URL))
    st.session_state["api_url"] = api_url
    client = FraudShieldAPIClient(api_url)
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Workspace",
        ["Fraud overview", "Transaction analyzer", "High-risk transactions", "Historical predictions", "Monitoring dashboard"],
        key="dashboard_page",
    )
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.rerun()

    health = _safe_call("API health check failed", client.health)
    model_info = None
    if page in {"Monitoring dashboard", "Fraud overview"}:
        model_info = _safe_call("Could not load model metadata", client.model_info)

    st.sidebar.markdown("---")
    if health:
        if health.get("ready"):
            st.sidebar.success("API ready")
        else:
            st.sidebar.warning("API degraded")
        st.sidebar.caption(f"Database: {'ready' if health.get('database_ready') else 'not ready'}")
    else:
        st.sidebar.error("API unreachable")
        st.sidebar.caption("Start FastAPI in another terminal, then refresh.")

    if page == "Fraud overview":
        page_overview(client, health)
    elif page == "Transaction analyzer":
        page_transaction_analyzer(client)
    elif page == "High-risk transactions":
        page_high_risk(client)
    elif page == "Historical predictions":
        page_history(client)
    else:
        page_monitoring(client, health, model_info)

    st.sidebar.caption("Development dashboard · model scores are not calibrated fraud probabilities.")


if __name__ == "__main__":
    main()
