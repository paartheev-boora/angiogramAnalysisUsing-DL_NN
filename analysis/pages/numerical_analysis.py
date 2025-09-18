import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

def show_numerical_analysis():
    st.title(" Numerical Feature Analysis")

    # ---- INIT SESSION STATE ----
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    # Guard clause
    if not st.session_state["uploaded_files"]:
        st.warning("No files uploaded yet. Please go to the Input Uploader page and upload angiogram data.")
        st.stop()

    if not st.session_state["results"]:
        st.warning(" Please upload and run analysis first from Input page.")
        return

    # Collect all reports into a single dataframe
    all_data = []
    for video, report in st.session_state["results"].items():
        row = {"Video": video}
        # If numerical_data missing, ensure it's present as empty dict
        numerical = report.get("numerical_data", {}) or {}
        row.update(numerical)
        all_data.append(row)

    if not all_data:
        st.warning("No numerical data found in results.")
        return

    df = pd.DataFrame(all_data)
    st.subheader(" Extracted Numerical Features")
    st.dataframe(df, use_container_width=True)

    # ---------------------------
    # Build threshold comparison table
    # ---------------------------
    # Define thresholds (tunable)
    thresholds = {
        "avg_flow": 0.3,
        "avg_contrast": 40,
        "avg_area": 20000,
    }

    # For features not in thresholds, we'll show "N/A"
    rows = []
    for _, r in df.iterrows():
        video = r["Video"]
        for feature in df.columns:
            if feature == "Video":
                continue
            val = r.get(feature, "N/A")
            thr = thresholds.get(feature, "N/A")

            # Determine feedback. Assumption: higher is better for these features.
            if val == "N/A" or pd.isna(val):
                feedback = "N/A"
            elif thr == "N/A":
                feedback = "No threshold defined"
            else:
                try:
                    v = float(val)
                    t = float(thr)
                    if v >= t:
                        feedback = "✅ Normal"
                    else:
                        feedback = "⚠️ Below Threshold"
                except Exception:
                    feedback = "N/A"

            rows.append({
                "Video": video,
                "Feature": feature,
                "Value": val if (val == "N/A") else round(float(val), 4),
                "Threshold": thr,
                "Feedback": feedback
            })

    df_thresh = pd.DataFrame(rows)

    # Summarize per-video numerical decision based on number of below-threshold flags
    decisions = {}
    for video, group in df_thresh.groupby("Video"):
        below_count = sum(1 for x in group["Feedback"] if isinstance(x, str) and "Below" in x)
        if below_count >= 2:
            decision = "❌ PCI Recommended (Numerical Path)"
        elif below_count == 1:
            decision = "⚠️ Further Evaluation Needed"
        else:
            decision = "✅ No PCI Needed (Numerical Path)"
        decisions[video] = {"below_count": below_count, "decision": decision}

    # Display threshold table
    st.subheader(" Comparison Table: Clinical Thresholds vs Patient Values")
    # Color styling for feedback column using pandas Styler
    def color_feedback(val):
        if isinstance(val, str):
            if "Below" in val:
                return "background-color: rgba(255,0,0,0.15)"
            if "Normal" in val or "✅" in val:
                return "background-color: rgba(0,255,0,0.10)"
            if "No threshold" in val or "No threshold" in val:
                return "background-color: rgba(200,200,200,0.08)"
        return ""

    try:
        styled = df_thresh.style.applymap(color_feedback, subset=["Feedback"])
        st.dataframe(styled, width="stretch")
    except Exception:
        # Fallback to plain table if styled table not supported
        st.dataframe(df_thresh, width="stretch")

    # CSV download
    csv_buf = io.StringIO()
    df_thresh.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    st.download_button(
        label="📥 Download threshold comparison (CSV)",
        data=csv_bytes,
        file_name="numerical_thresholds.csv",
        mime="text/csv"
    )

    # Show per-video decision summary
    st.subheader(" Numerical Path Summary per Video")
    for v, meta in decisions.items():
        st.markdown(f"**{v}** — {meta['decision']} _(flags: {meta['below_count']})_")

    # ---------------------------
    # Convert wide → long format for plotting
    # ---------------------------
    df_long = df.melt(id_vars="Video", var_name="Feature", value_name="Value")

    st.subheader("Numerical Feature Comparison Across Angiographies")

    # ---- Plot each feature separately ----
    for feature in df.columns:
        if feature == "Video":
            continue

        # Filter for one feature
        df_feature = df[["Video", feature]].copy()
        # Ensure numeric values for plotting; replace missing with NaN
        df_feature[feature] = pd.to_numeric(df_feature[feature], errors="coerce")

        fig = px.bar(
            df_feature,
            x="Video",
            y=feature,
            color="Video",
            text=feature,
            title=f"{feature} Comparison"
        )

        # Add threshold if defined
        if feature in thresholds:
            fig.add_hline(
                y=thresholds[feature],
                line_dash="dot",
                line_color="red",
                annotation_text=f"Threshold: {thresholds[feature]}",
                annotation_position="top left"
            )

        st.plotly_chart(fig, use_container_width=True)
        
if __name__ == "__main__":
    show_numerical_analysis()