# app.py

import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Internal Module Imports
from src.pipeline import run_pipeline
from src.classifier import CLASS_NAMES

# Page Tab Configuration Layout
st.set_page_config(
    page_title="ExosPlore — TESS Exoplanet Detection Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌌 ExosPlore — TESS Exoplanet Detection Pipeline")
st.markdown("""
**Bharatiya Antariksh Hackathon 2026** — *Problem Statement 7 (ISRO)*
An end-to-end AI pipeline that ingests raw TESS space telescope light curves to automatically detect, classify, and characterize exoplanetary transit signals.
""")
st.write("---")

# ==========================================
# SIDEBAR CONFIGURATIONS
# ==========================================
st.sidebar.header("📁 Data Ingestion Control")
uploaded_file = st.sidebar.file_uploader("Upload Raw TESS .fits File", type=["fits"])

# --- NEW: BIG DATA PRE-CACHING AUTOMATION ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Big Data Processing")
st.sidebar.markdown("Prepare Sector 20k+ datasets for rapid network training.")

if st.sidebar.button("⚡ Execute Pre-Caching Pipeline"):
    with st.sidebar.spinner(
        "Compressing raw FITS telemetry into binary matrices... Please wait."
    ):
        import subprocess

        try:
            # Automates the terminal command directly from the UI
            subprocess.run(["python", "-m", "src.cache_dataset"], check=True)
            st.sidebar.success(
                "✅ Pre-Caching Complete! Arrays saved to data/processed/"
            )
        except Exception as e:
            st.sidebar.error(f"❌ Caching Failed: {str(e)}")

st.sidebar.markdown("---")

# Status of model weights check
MODEL_PATH = "models/transit_cnn.pth"
if os.path.exists(MODEL_PATH):
    st.sidebar.success("✅ Neural Network Weights Loaded")
else:
    st.sidebar.warning("⚠️ Model Running in Emulation Mode")


# ==========================================
# MAIN EXECUTION ROUTING LOOP
# ==========================================
if uploaded_file is not None:
    # Save the uploaded buffer stream to a temp file safely
    temp_path = "temp_uploaded_target.fits"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(
        f"Processing target: **{uploaded_file.name}**... Running Physics-Aware Preprocessing & BLS Engines..."
    )

    # Run the entire pipeline execution block
    with st.spinner("Analyzing light curve telemetry..."):
        results = run_pipeline(temp_path, model_path=MODEL_PATH)

    if "Success" in results["status"]:
        st.success("Pipeline Analysis Complete!")

        # --- TOP LEVEL SUMMARY SUMMARY BADGES ---
        col1, col2, col3, col4 = st.columns(4)

        # Map structural colors for the badge presentation
        color_map = {
            "Planetary Transit": "background-color: #2ecc71; color: white;",  # Green
            "Eclipsing Binary": "background-color: #e74c3c; color: white;",  # Red
            "Stellar Blend": "background-color: #e67e22; color: white;",  # Orange
            "Stellar Variability": "background-color: #f1c40f; color: black;",  # Yellow
            "Noise/Unknown": "background-color: #7f8c8d; color: white;",  # Grey
        }

        with col1:
            st.markdown(
                f"""
            <div style="padding: 15px; border-radius: 5px; {color_map.get(results['predicted_class'], color_map['Noise/Unknown'])} text-align: center;">
                <h4 style="margin: 0;">Predicted Class</h4>
                <h2 style="margin: 5px 0 0 0; font-size: 22px;">{results['predicted_class']}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.metric(
                "Classification Confidence", f"{results['confidence'] * 100:.2f}%"
            )
        with col3:
            st.metric(
                "BLS Candidate Period", f"{results['top_candidate']['period']:.5f} Days"
            )
        with col4:
            st.metric(
                "Calculated Signal SNR", f"{results['top_candidate']['snr_bls']:.2f}"
            )

        st.write("---")

        # --- TAB LAYOUT SYSTEM ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 Light Curve & Preprocessing",
                "🎯 Detection & Classification",
                "📋 Characterization Parameters",
                "🚫 Rejected Detections Dashboard",
                "📊 Model Evaluation Metrics",
            ]
        )

        # --- TAB 1: LIGHT CURVE PLOTS ---
        with tab1:
            st.subheader("Physics-Aware Preprocessing Sequence Visualization")

            # Interactive Plotly Raw Light Curve Layout
            fig_raw = go.Figure()
            fig_raw.add_trace(
                go.Scatter(
                    x=results["plots"]["raw_time"],
                    y=results["plots"]["raw_flux"],
                    mode="markers",
                    marker=dict(size=2, color="#34495e"),
                    name="Raw PDCSAP Flux",
                )
            )
            fig_raw.update_layout(
                title="Normalized Flux Over Time (Outliers Clipped via Sigma-5)",
                xaxis_title="Time (BJD - 2457000)",
                yaxis_title="Normalized Flux",
                height=350,
            )
            st.plotly_chart(fig_raw, use_container_width=True)

            # Detrended Flattened View Output
            fig_flat = go.Figure()
            fig_flat.add_trace(
                go.Scatter(
                    x=results["plots"]["flat_time"],
                    y=results["plots"]["flat_flux"],
                    mode="markers",
                    marker=dict(size=2, color="#2980b9"),
                    name="Flattened Flux",
                )
            )
            fig_flat.update_layout(
                title="Flattened & Detrended Flux (Stellar Variability Removed via Savitzky-Golay)",
                xaxis_title="Time (BJD - 2457000)",
                yaxis_title="Flattened Flux",
                height=350,
            )
            st.plotly_chart(fig_flat, use_container_width=True)

            # BLS Periodogram
            st.subheader("Box Least Squares (BLS) Power Spectrum")
            fig_bls = go.Figure()
            fig_bls.add_trace(
                go.Scatter(
                    x=results["top_candidate"]["period_grid"],
                    y=results["top_candidate"]["bls_power_array"],
                    mode="lines",
                    line=dict(color="#8e44ad"),
                    name="BLS Power",
                )
            )
            # Mark the maximized target peak point
            fig_bls.add_vline(
                x=results["top_candidate"]["period"],
                line_width=2,
                line_dash="dash",
                line_color="red",
            )
            fig_bls.update_layout(
                title=f"BLS Periodogram Spectrum (Peak Power Period Identified at {results['top_candidate']['period']:.5f} Days)",
                xaxis_title="Period (Days)",
                yaxis_title="BLS Power Metric",
                height=350,
            )
            st.plotly_chart(fig_bls, use_container_width=True)

        # --- TAB 2: DETECTION & CLASSIFICATION ---
        with tab2:
            c_left, c_right = st.columns([3, 2])

            with c_left:
                st.subheader("Phase-Folded Candidate Profiles")

                # Plot folded data profile
                fig_fold = go.Figure()

                if results["is_planet"] and results["characterization"] is not None:
                    char = results["characterization"]
                    fig_fold.add_trace(
                        go.Scatter(
                            x=char["phase_sorted"],
                            y=char["flux_sorted"],
                            mode="markers",
                            marker=dict(size=3, color="#bdc3c7"),
                            name="Phase-Folded Cadences",
                        )
                    )
                    fig_fold.add_trace(
                        go.Scatter(
                            x=char["phase_sorted"],
                            y=char["model_flux"],
                            mode="lines",
                            line=dict(color="#e74c3c", width=3),
                            name="Fitted Model Optimization Profile",
                        )
                    )
                else:
                    # Generic curve profile fallback for alternative classes
                    cand = results["top_candidate"]
                    phases = np.linspace(-0.5, 0.5, 200)
                    fig_fold.add_trace(
                        go.Scatter(
                            x=phases,
                            y=cand["phase_folded_flux_200"],
                            mode="markers+lines",
                            marker=dict(size=4),
                            name="Folded Candidate Array Segment",
                        )
                    )

                fig_fold.update_layout(
                    title="Phase-Folded Signal Window Centered at Phase 0.0",
                    xaxis_title="Phase Window Coordinate",
                    yaxis_title="Normalized Flux Signature",
                    height=450,
                )
                st.plotly_chart(fig_fold, use_container_width=True)

            with c_right:
                st.subheader("1D-CNN + Attention Softmax Probability Distribution")

                # Probabilities Horizontal Bar Chart View
                prob_data = pd.DataFrame(
                    {
                        "Class Label Categories": list(
                            results["all_probabilities"].keys()
                        ),
                        "Model Confidence Score": list(
                            results["all_probabilities"].values()
                        ),
                    }
                )

                fig_prob = go.Figure(
                    go.Bar(
                        x=prob_data["Model Confidence Score"],
                        y=prob_data["Class Label Categories"],
                        orientation="h",
                        marker_color=[
                            "#2ecc71",
                            "#e74c3c",
                            "#e67e22",
                            "#f1c40f",
                            "#7f8c8d",
                        ],
                    )
                )
                fig_prob.update_layout(
                    title="Softmax Evaluation Matrix",
                    xaxis=dict(range=[0, 1]),
                    height=400,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        # --- TAB 3: PHYSICAL PARAMETERS ---
        with tab3:
            st.subheader("Physical Extraction Metrics & System Parameters Table")

            if results["is_planet"] and results["characterization"] is not None:
                char = results["characterization"]

                param_matrix = {
                    "System Metric Target Parameter": [
                        "Orbital System Period Base",
                        "Optimized Transit Duration Profile",
                        "Calculated Geometry Transit Depth",
                        "Operational Signal-to-Noise Ratio (SNR)",
                        "Pipeline Pipeline Confidence Rating",
                        "Physical Model Engine Utilized",
                    ],
                    "Extracted Value": [
                        f"{char['period_days']:.5f} Days",
                        f"{char['duration_hours']:.3f} Hours",
                        f"{char['depth_ppm']:.2f} ppm (Parts Per Million)",
                        f"{char['snr']:.2f}",
                        f"{results['confidence'] * 100:.2f}%",
                        (
                            "BATMAN Physical Simulator Engine"
                            if char["using_batman"]
                            else "Scipy Geometric Trapezoid Model Optimization Head"
                        ),
                    ],
                }
                st.table(pd.DataFrame(param_matrix))
            else:
                st.error("❌ Physical Characterization Skipped.")
                st.write(
                    f"The structural classifier engine designated this object as a **{results['predicted_class']}** rather than a Planetary Transit system."
                )
                st.info(
                    "System Rejection Parameters Diagnostics: Only candidate detections classified as Class 0 ('Planetary Transit') route to the optimization engine to ensure processing compute constraints are preserved."
                )

        # --- TAB 4: REJECTED DETECTIONS DEMO ---
        with tab4:
            st.subheader("System False Positive Rejection Matrix Framework")
            st.markdown("""
            This pane displays the operational framework metrics for understanding how the 1D-CNN + Attention layer segregates planetary signals from common false positive configurations.
            """)

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown("#### 🔴 Eclipsing Binaries (Class 1)")
                st.caption(
                    "Identified by alternating primary and secondary V-shaped eclipse geometries, deep configurations, or missing physical flat bottoms."
                )
            with rc2:
                st.markdown("#### 🟠 Stellar Blends (Class 2)")
                st.caption(
                    "Identified by shallow, heavily integrated environmental background structures mixed with noise profiles from adjacent companion stars."
                )
            with rc3:
                st.markdown("#### 🟡 Stellar Variability (Class 3)")
                st.caption(
                    "Identified by long-range sinusoidal variations, spot rotation cycles, or flare events."
                )

            st.image(
                "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=600&auto=format&fit=crop&q=60",
                caption="TESS Space Telescope Science Verification Field Context",
                width=400,
            )

        # --- TAB 5: EVALUATION METRICS ---
        with tab5:
            st.markdown("### 📊 Pipeline Validation Performance Suite")
            st.markdown(
                "This panel displays system-wide verification metrics calculated against the curated testing catalog."
            )

            import os
            import json
            import pandas as pd
            import subprocess

            # --- AUTOMATION UPGRADE: Run evaluation directly from UI ---
            if st.button("🚀 Execute Live Evaluation"):
                with st.spinner(
                    "Evaluating network weights against testing catalog... Please wait."
                ):
                    subprocess.run(["python", "evaluate.py"])
                    st.rerun()

            st.markdown("---")

            metrics_path = "models/metrics.json"

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        real_metrics = json.load(f)

                    report_data = real_metrics["classification_report"]

                    classes = [
                        "Planetary Transit (0)",
                        "Eclipsing Binary (1)",
                        "Stellar Blend (2)",
                        "Stellar Variability (3)",
                        "Noise/Unknown (4)",
                    ]
                    parsed_report = {
                        "Stellar Category Class": classes,
                        "Precision Rating": [
                            round(report_data[c]["precision"], 2) for c in classes
                        ],
                        "Recall Rate": [
                            round(report_data[c]["recall"], 2) for c in classes
                        ],
                        "Calculated F1-Score": [
                            round(report_data[c]["f1-score"], 2) for c in classes
                        ],
                    }
                    df_report = pd.DataFrame(parsed_report)

                    matrix_data = real_metrics["confusion_matrix"]
                    columns_labels = [
                        "Pred (0)",
                        "Pred (1)",
                        "Pred (2)",
                        "Pred (3)",
                        "Pred (4)",
                    ]
                    index_labels = [
                        "True Transit (0)",
                        "True Eclipse (1)",
                        "True Blend (2)",
                        "True Variab. (3)",
                        "True Noise (4)",
                    ]
                    df_matrix = pd.DataFrame(
                        matrix_data, columns=columns_labels, index=index_labels
                    )

                    ui_col1, ui_col2 = st.columns(2)
                    with ui_col1:
                        st.markdown(
                            "#### 📑 Scientific Classification Performance Metrics"
                        )
                        st.table(df_report)

                    with ui_col2:
                        st.markdown("#### 🧩 Confusion Resolution Matrix")
                        st.table(df_matrix)

                    st.success(
                        "🎯 Live pipeline evaluation metrics successfully loaded from production checkpoint assets."
                    )

                except Exception as e:
                    st.error(f"⚠️ Error parsing the metrics file: {str(e)}")
            else:
                st.warning(
                    "⚠️ Metrics data not found. Click the button above to generate the live performance report."
                )

    else:
        st.error(f"Error Processing File: {results['status']}")

    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info(
        "👋 Welcome to ExosPlore! Please upload a raw TESS .fits science file on the sidebar panel to launch automated processing."
    )
