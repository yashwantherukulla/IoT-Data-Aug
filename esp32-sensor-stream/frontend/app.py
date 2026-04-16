import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("ESP32 Sensor Stream Dashboard")

# Sidebar configuration
st.sidebar.header("Settings")
k_samples = st.sidebar.slider("Display Window (Samples)", 50, 1000, 200)

# Metric human-readable labels and description tooltips
METRIC_META = {
    "temporal_range": ("Temporal Range",   "Max − min of mean amplitude across time"),
    "f0_amplitude":   ("F0 Amplitude",     "HPS-based fundamental frequency amplitude"),
    "contrast":       ("Contrast",         "Mean(top 5%) − mean(bottom 5%) of bins"),
    "flatness":       ("Flatness",         "Geometric mean / arithmetic mean  (0→tonal, 1→noise-like)"),
    "entropy":        ("Entropy (bits)",   "Shannon entropy over normalised spectrogram bins"),
}

# Create placeholders for dynamic charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Accelerometer Data (g)")
    acc_placeholder = st.empty()
with col2:
    st.subheader("Gyroscope Data (°/s)")
    gyro_placeholder = st.empty()

# Single placeholder for metrics panel — replaced each loop
metrics_placeholder = st.empty()

# Single placeholder for the entire spectrogram section — cleared & redrawn each loop
spec_placeholder = st.empty()

# Fetch loop
while True:
    try:
        response = requests.get("http://localhost:8000/api/data")
        data = response.json()

        if data["acc_x"]:  # If we have received data
            # Slice to last K samples
            acc_df = pd.DataFrame({
                "X": data["acc_x"][-k_samples:],
                "Y": data["acc_y"][-k_samples:],
                "Z": data["acc_z"][-k_samples:]
            })

            gyro_df = pd.DataFrame({
                "X": data["gyro_x"][-k_samples:],
                "Y": data["gyro_y"][-k_samples:],
                "Z": data["gyro_z"][-k_samples:]
            })

            # Update the charts in real-time (replaces, doesn't append)
            acc_placeholder.line_chart(acc_df)
            gyro_placeholder.line_chart(gyro_df)

        # ------------------------------------------------------------------ #
        # Metrics panel
        # ------------------------------------------------------------------ #
        metrics_response = requests.get("http://localhost:8000/api/metrics")
        metrics = metrics_response.json()

        with metrics_placeholder.container():
            acc_m = metrics.get("acc")
            gyr_m = metrics.get("gyr")
            if acc_m or gyr_m:
                st.divider()
                st.header("Signal Metrics  (latest window)")
                m_col1, m_col2 = st.columns(2)

                def render_metrics(m: dict, label: str, col):
                    with col:
                        st.subheader(label)
                        if not m:
                            st.info("Waiting for first window…")
                            return
                        # Five metric tiles in a row
                        tiles = st.columns(len(METRIC_META))
                        for tile, (key, (human_name, tooltip)) in zip(tiles, METRIC_META.items()):
                            val = m.get(key)
                            with tile:
                                st.metric(
                                    label=human_name,
                                    value=f"{val:.4f}" if val is not None else "—",
                                    help=tooltip,
                                )

                render_metrics(acc_m, "Accelerometer", m_col1)
                render_metrics(gyr_m, "Gyroscope", m_col2)

        # ------------------------------------------------------------------ #
        # Spectrograms
        # ------------------------------------------------------------------ #
        spec_response = requests.get("http://localhost:8000/api/spectrograms")
        specs = spec_response.json()

        # Rebuild the spectrogram section inside a container so it replaces every refresh
        with spec_placeholder.container():
            if specs["acc"] or specs["gyr"]:
                st.divider()
                st.header("Top 3 Recent Spectrograms")
                s_col1, s_col2 = st.columns(2)

                with s_col1:
                    st.subheader("Accelerometer")
                    if specs["acc"]:
                        cols = st.columns(len(specs["acc"]))
                        for idx, img_b64 in enumerate(specs["acc"]):
                            with cols[idx]:
                                st.image(
                                    f"data:image/png;base64,{img_b64}",
                                    width="stretch",
                                    caption=f"Newest {idx+1}",
                                )
                    else:
                        st.info("Waiting for enough data to generate Acc spectrogram...")

                with s_col2:
                    st.subheader("Gyroscope")
                    if specs["gyr"]:
                        cols = st.columns(len(specs["gyr"]))
                        for idx, img_b64 in enumerate(specs["gyr"]):
                            with cols[idx]:
                                st.image(
                                    f"data:image/png;base64,{img_b64}",
                                    width="stretch",
                                    caption=f"Newest {idx+1}",
                                )
                    else:
                        st.info("Waiting for enough data to generate Gyr spectrogram...")

    except Exception as e:
        st.warning(f"Waiting for backend server... {e}")

    time.sleep(0.5)