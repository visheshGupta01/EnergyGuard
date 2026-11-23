"""Streamlit dashboard for EnergyGuard AI."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.config import MODEL_DIR, REPORT_DIR, PROCESSED_DIR

st.set_page_config(page_title="EnergyGuard AI", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#145374); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0;}
.hero p {color: #d9eaf4; margin: .4rem 0 0;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚡ EnergyGuard AI</h1><p>Understand household electricity patterns with machine learning.</p></div>', unsafe_allow_html=True)

pages = ["Overview", "Consumption Forecast", "Anomaly Review", "Theft-Risk Screening", "Model Evaluation"]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.caption("Semester 3 Machine Learning Project")
st.sidebar.info("Risk predictions are screening signals, not proof of theft.")

def load_json(name):
    path = REPORT_DIR / name
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except Exception: return None

def load_model(name):
    path = MODEL_DIR / name
    if not path.exists(): return None
    try: return joblib.load(path)
    except Exception as exc:
        st.error(f"Could not load {name}: {exc}")
        return None

if page == "Overview":
    st.subheader("Project overview")
    st.write("EnergyGuard AI demonstrates three independent ML tasks using two public electricity datasets.")
    a, b, c = st.columns(3)
    a.metric("Forecasting", "2 models", "Linear + Random Forest")
    b.metric("Anomaly detection", "Isolation Forest", "Unsupervised")
    c.metric("Theft-risk", "Random Forest", "Binary classification")
    st.divider()
    st.markdown("#### Dataset readiness")
    for label, filename in [("UCI household data", "household_power_consumption.txt"), ("SGCC consumer data", "sgcc.csv")]:
        exists = (Path("data/raw") / filename).exists()
        st.write(("✅" if exists else "⬜") + f" **{label}** — " + ("found in data/raw" if exists else f"not found; add data/raw/{filename}"))
    audit = load_json("dataset_audit.json")
    if audit:
        with st.expander("Latest dataset audit"): st.json(audit)
    st.markdown("#### Start here")
    st.write("1. Add both datasets to data/raw. 2. Run preprocessing and training commands from the README. 3. Return here to explore trained models.")
    st.caption("UCI describes one household in France. SGCC is Chinese utility data. Neither dataset establishes performance for Indian consumers.")

elif page == "Consumption Forecast":
    st.subheader("Next-hour consumption forecast")
    model = load_model("forecast_model.joblib")
    data_path = PROCESSED_DIR / "uci_hourly_features.csv"
    if model is None or not data_path.exists():
        st.warning("Forecast model/data not found. Add the UCI dataset and run: python -m src.train --forecast")
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        latest = df.iloc[-1]
        st.metric("Latest observed hourly energy", f"{latest['energy_kwh']:.3f} kWh")
        st.caption(f"Latest feature timestamp: {latest['timestamp']}")
        features = ["energy_kwh", "hour", "weekday", "lag_1", "lag_2", "lag_24", "rolling_mean_3", "rolling_mean_24"]
        prediction = float(model.predict(df.iloc[[-1]][features])[0])
        st.metric("Predicted next-hour energy", f"{prediction:.3f} kWh")
        st.line_chart(df.set_index("timestamp")["energy_kwh"].tail(168))
        st.caption("The displayed forecast uses the most recent complete feature row. It is a model estimate, not a utility bill.")

elif page == "Anomaly Review":
    st.subheader("Unusual consumption observations")
    metrics = load_json("anomaly_metrics.json")
    path = REPORT_DIR / "anomaly_review.csv"
    if not path.exists():
        st.warning("Anomaly results not found. Run: python -m src.train --anomaly after adding UCI data.")
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if metrics:
            a, b, c = st.columns(3)
            a.metric("Reviewed observations", f"{metrics.get('review_rows', 0):,}")
            b.metric("Flagged as unusual", f"{metrics.get('flagged_count', 0):,}")
            c.metric("Flagged share", f"{metrics.get('flagged_percentage', 0):.2f}%")
        flagged = df[df["is_anomaly"].astype(str).str.lower().isin(["true", "1"])].sort_values("anomaly_score", ascending=False)
        st.dataframe(flagged.head(100), use_container_width=True)
        st.download_button("Download anomaly review CSV", df.to_csv(index=False), "anomaly_review.csv", "text/csv")
        st.warning("Isolation Forest identifies observations that differ from learned patterns. These are not theft labels.")

elif page == "Theft-Risk Screening":
    st.subheader("Consumer profile screening")
    artifact = load_model("theft_model.joblib")
    if artifact is None:
        st.warning("Theft-risk model not found. Add SGCC data and run: python -m src.train --theft")
    else:
        pipeline, features = artifact["pipeline"], artifact["features"]
        threshold = float(artifact["threshold"])
        st.write("Enter consumer-level summary features. The trained pipeline imputes remaining missing values.")
        defaults = {"mean_daily":1.0,"std_daily":.8,"median_daily":1.0,"min_daily":0.0,"max_daily":3.0,"zero_day_ratio":.05,"missing_day_ratio":0.0,"first_quarter_mean":1.0,"last_quarter_mean":1.0,"consumption_cv":.8}
        with st.form("risk_form"):
            values = {}
            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    values[feature] = st.number_input(feature.replace("_"," ").title(), min_value=0.0, value=float(defaults.get(feature,0.0)), step=.05, format="%.4f")
            submitted = st.form_submit_button("Estimate screening risk", type="primary")
        if submitted:
            probability = float(pipeline.predict_proba(pd.DataFrame([values]))[0,1])
            predicted = int(probability >= threshold)
            left, right = st.columns(2)
            left.metric("Estimated class-1 probability", f"{probability:.1%}")
            right.metric("Screening category", "Higher-risk pattern" if predicted else "Lower-risk pattern")
            st.progress(min(max(probability,0.0),1.0))
            st.caption(f"Decision threshold selected using validation consumers: {threshold:.2f}")
            st.error("This result is not evidence that theft occurred. Do not use it for disconnection, penalties or enforcement. Independent investigation is essential.")
        st.info("These are consumer-level statistical summaries. For a real SGCC consumer, generate features using the same preprocessing pipeline rather than guessing values.")

elif page == "Model Evaluation":
    st.subheader("Held-out model evaluation")
    forecast, anomaly, theft = load_json("forecast_metrics.json"), load_json("anomaly_metrics.json"), load_json("theft_metrics.json")
    if forecast:
        st.markdown("#### Forecasting")
        st.write(f"Selected model: **{forecast.get('selected_model','N/A')}**")
        st.json(forecast.get("test", {}))
        image = REPORT_DIR / "forecast_test.png"
        if image.exists(): st.image(str(image), caption="Actual vs predicted next-hour energy")
    else: st.info("Forecast metrics will appear after training.")
    if theft:
        st.markdown("#### Theft-risk classification")
        st.json(theft.get("test", {}))
        image = REPORT_DIR / "theft_confusion_matrix.png"
        if image.exists(): st.image(str(image), caption="Confusion matrix on held-out consumers")
    else: st.info("Theft-risk metrics will appear after training.")
    if anomaly:
        st.markdown("#### Anomaly detection")
        st.json(anomaly)
        st.caption("No precision/recall is reported because UCI does not provide anomaly ground-truth labels.")
    else: st.info("Anomaly metrics will appear after training.")

st.divider()
st.caption("EnergyGuard AI • Academic demonstration • Model outputs require human interpretation.")
