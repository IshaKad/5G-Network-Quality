import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from prediction.predict_network_state import predict_network_state

VALIDATION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "validation",
    "features_validation.csv"
)

st.set_page_config(
    page_title="5G Network Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #111827, #263449);
        padding: 30px 34px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 16px;
    }

    .active-status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(34, 197, 94, 0.15);
        color: #86efac;
        font-size: 13px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_validation_data():
    return pd.read_csv(VALIDATION_PATH)


def get_float(row, column, default=np.nan):

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_value(row, column, default="Unknown"):

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def signal_status(value):

    if pd.isna(value):
        return "Unavailable"

    if value >= 80:
        return "Excellent"
    elif value >= 65:
        return "Good"
    elif value >= 50:
        return "Fair"
    else:
        return "Weak"


def connection_score(rsrp, rsrq, sinr):

    if any(pd.isna(value) for value in [rsrp, rsrq, sinr]):
        return None

    rsrp_score = np.clip((rsrp - 30) / 60 * 100, 0, 100)
    rsrq_score = np.clip((rsrq - 20) / 50 * 100, 0, 100)
    sinr_score = np.clip((sinr - 20) / 75 * 100, 0, 100)

    score = (
        0.45 * rsrp_score
        + 0.25 * rsrq_score
        + 0.30 * sinr_score
    )

    return int(round(np.clip(score, 0, 100)))


try:
    data = load_validation_data()
except Exception as e:
    st.error("The dashboard could not load the validation data.")
    st.exception(e)
    st.stop()


st.sidebar.title("Dashboard Controls")

st.sidebar.write(
    "Select a measurement to see the current connection, "
    "future prediction and AI handover decision."
)

sample_index = st.sidebar.slider(
    "Measurement",
    min_value=0,
    max_value=len(data) - 1,
    value=min(100, len(data) - 1),
    step=1
)

st.sidebar.caption(
    f"Showing measurement {sample_index + 1:,} of {len(data):,}"
)

selected_row = data.iloc[[sample_index]].copy()

try:
    prediction = predict_network_state(selected_row.copy())
except Exception as e:
    st.error("Prediction failed for the selected measurement.")
    st.exception(e)
    st.stop()

current = selected_row.iloc[0]
result = prediction.iloc[0]


current_rsrp = get_float(
    current,
    "serving_rsrp"
)

current_rsrq = get_float(
    current,
    "serving_rsrq"
)

current_sinr = get_float(
    current,
    "serving_sinr"
)


pred_rsrp = get_float(
    result,
    "predicted_rsrp"
)

pred_rsrq = get_float(
    result,
    "predicted_rsrq"
)

pred_sinr = get_float(
    result,
    "predicted_sinr"
)


handover_probability = get_float(
    result,
    "handover_probability",
    0.0
)


decision = str(
    get_value(
        result,
        "decision",
        "NO_HANDOVER"
    )
)


recommended_pci = get_value(
    result,
    "recommended_target_pci",
    None
)


decision_reason = str(
    get_value(
        result,
        "decision_reason",
        "No immediate handover is recommended."
    )
)


serving_cell = get_value(
    current,
    "serving_cell",
    "Unknown"
)


score = connection_score(
    current_rsrp,
    current_rsrq,
    current_sinr
)


if score is None:
    connection_status = "DATA UNAVAILABLE"
elif score >= 80:
    connection_status = "EXCELLENT"
elif score >= 65:
    connection_status = "GOOD"
elif score >= 50:
    connection_status = "FAIR"
else:
    connection_status = "NEEDS ATTENTION"


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">5G Network Intelligence</div>
        <div class="hero-subtitle">
            Predicting connection quality before it changes
            and making intelligent handover decisions.
        </div>
        <span class="active-status">● SYSTEM ACTIVE</span>
    </div>
    """,
    unsafe_allow_html=True
)


st.subheader("Why this matters")

st.info(
    "Your device can detect several nearby 5G cells while staying "
    "connected to one of them. As you move, the current connection "
    "may become weaker while another cell becomes a better option.\n\n"
    "This system looks ahead. It predicts the expected connection "
    "quality for the next 3 seconds and decides whether switching "
    "cells would actually improve the connection."
)


st.subheader("Your connection")

st.caption(
    "What your device is experiencing right now."
)


health_col, metrics_col = st.columns([1, 2])


with health_col:

    with st.container(border=True):

        st.caption("CONNECTION HEALTH")

        if score is not None:

            st.markdown(
                f"# {score}/100"
            )

            st.write(
                f"**{connection_status} CONNECTION**"
            )

        else:

            st.markdown("# —")

            st.write(
                "**DATA UNAVAILABLE**"
            )

        st.caption(
            "Current radio conditions are being monitored "
            "by the AI prediction system."
        )

        st.info(
            f"Connected to Cell {serving_cell}"
        )


with metrics_col:

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Signal strength",
            "—" if pd.isna(current_rsrp) else f"{current_rsrp:.0f}",
            help=(
                "RSRP (Reference Signal Received Power) "
                "measures the strength of the received 5G "
                "radio signal."
            )
        )

        st.caption(
            signal_status(current_rsrp)
        )

    with metric2:

        st.metric(
            "Signal quality",
            "—" if pd.isna(current_rsrq) else f"{current_rsrq:.0f}",
            help=(
                "RSRQ (Reference Signal Received Quality) "
                "describes the quality of the received radio "
                "signal and the effect of interference."
            )
        )

        st.caption(
            signal_status(current_rsrq)
        )

    with metric3:

        st.metric(
            "Signal clarity",
            "—" if pd.isna(current_sinr) else f"{current_sinr:.0f}",
            help=(
                "SINR (Signal-to-Interference-plus-Noise Ratio) "
                "indicates how clearly the desired signal can "
                "be received compared with interference and noise."
            )
        )

        st.caption(
            signal_status(current_sinr)
        )


st.subheader("What is likely to happen next?")

st.caption(
    "The AI predicts the expected connection state approximately "
    "3 seconds into the future."
)


pred1, pred2, pred3 = st.columns(3)


future_values = [
    ("Signal strength", pred_rsrp, current_rsrp),
    ("Signal quality", pred_rsrq, current_rsrq),
    ("Signal clarity", pred_sinr, current_sinr)
]


for column, (name, predicted, current_value) in zip(
    [pred1, pred2, pred3],
    future_values
):

    with column:

        with st.container(border=True):

            if pd.isna(predicted):

                st.metric(
                    name,
                    "—",
                    help=(
                        "AI prediction for this connection "
                        "metric approximately 3 seconds ahead."
                    )
                )

            else:

                if pd.isna(current_value):
                    delta = None
                else:
                    delta = predicted - current_value

                st.metric(
                    name,
                    f"{predicted:.1f}",
                    delta=(
                        f"{delta:+.1f}"
                        if delta is not None
                        else None
                    ),
                    help=(
                        "AI prediction for this connection "
                        "metric approximately 3 seconds ahead."
                    )
                )


st.subheader("AI handover decision")

st.caption(
    "Should the device remain connected to this cell, "
    "or would another nearby cell be a better choice?"
)


if decision == "HANDOVER":

    if pd.notna(recommended_pci):

        target_text = f"Cell {recommended_pci}"

    else:

        target_text = "a nearby cell"


    st.warning(
        f"### Switch connection\n\n"
        f"The AI recommends moving to **{target_text}**.\n\n"
        f"The nearby cell provides a stronger connection option."
    )

    st.success(
        "The model detected a high likelihood that a handover is needed."
    )

    st.success(
        "A nearby cell provides a meaningful signal improvement."
    )

else:

    if handover_probability < 0.90:

        st.success(
            "### ✓ Stay connected\n\n"
            "The AI does not recommend switching to another "
            "cell right now."
        )

        st.write(
            "✓ The current connection does not show a strong need to switch."
        )

        st.write(
            "✓ The AI expects the connection to remain sufficiently stable."
        )

        st.write(
            "✓ Staying connected avoids an unnecessary cell change."
        )

    else:

        st.warning(
            "### Stay connected for now\n\n"
            "The AI detected a possible need for a handover, "
            "but no nearby cell provides enough improvement."
        )

        st.write(
            "✓ A possible handover was detected."
        )

        st.write(
            "✓ No neighboring cell provides enough improvement."
        )

        st.write(
            "✓ Staying connected is safer than making an unnecessary switch."
        )


st.metric(
    "AI-estimated switch likelihood",
    f"{handover_probability * 100:.1f}%",
    help=(
        "The probability estimated by the AI model that "
        "a handover may be required within the prediction horizon."
    )
)


st.subheader("Nearby connection options")

st.caption(
    "Other 5G cells detected around the device. "
    "The AI compares these options when a handover may be needed."
)


neighbor_rows = []


for i in range(1, 4):

    pci_column = f"neighbor_{i}_pci"
    rsrp_column = f"neighbor_{i}_rsrp"

    if pci_column not in current.index:
        continue

    pci = current.get(pci_column)
    rsrp = current.get(rsrp_column)

    if pd.isna(pci):
        continue

    try:
        rsrp = float(rsrp)
    except (ValueError, TypeError):
        rsrp = np.nan

    neighbor_rows.append(
        {
            "pci": pci,
            "rsrp": rsrp
        }
    )


if neighbor_rows:

    for neighbor in neighbor_rows:

        pci = neighbor["pci"]
        rsrp = neighbor["rsrp"]

        left, middle, right = st.columns([2.5, 2.5, 1])

        with left:

            st.write(
                f"**Cell {pci}**"
            )

            st.caption(
                "PCI (Physical Cell Identity) is a technical identifier "
                "used by the 5G radio network to distinguish cells."
            )

        with middle:

            if pd.isna(rsrp):
                st.metric(
                    "Signal",
                    "—"
                )
            else:
                st.metric(
                    "Signal",
                    f"{rsrp:.0f}"
                )

        with right:

            if pd.isna(current_rsrp) or pd.isna(rsrp):
                relation = "Unknown"

            elif rsrp > current_rsrp:
                relation = "Stronger"

            elif rsrp < current_rsrp:
                relation = "Weaker"

            else:
                relation = "Similar"

            st.write("")
            st.caption(relation)

        st.divider()

else:

    st.info(
        "No nearby cells were detected in this measurement."
    )


with st.expander("Technical details"):

    st.write(
        "These values are intended for technical users "
        "and project evaluation."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Current network state")

        st.write(
            f"Serving cell: {serving_cell}"
        )

        st.write(
            f"RSRP: {current_rsrp:.0f}"
            if not pd.isna(current_rsrp)
            else "RSRP: unavailable"
        )

        st.write(
            f"RSRQ: {current_rsrq:.0f}"
            if not pd.isna(current_rsrq)
            else "RSRQ: unavailable"
        )

        st.write(
            f"SINR: {current_sinr:.0f}"
            if not pd.isna(current_sinr)
            else "SINR: unavailable"
        )

        st.caption(
            "RSRP = Reference Signal Received Power"
        )

        st.caption(
            "RSRQ = Reference Signal Received Quality"
        )

        st.caption(
            "SINR = Signal-to-Interference-plus-Noise Ratio"
        )

        st.caption(
            "PCI = Physical Cell Identity"
        )

    with right:

        st.markdown("#### AI output")

        st.write(
            f"Predicted RSRP: {pred_rsrp:.4f}"
            if not pd.isna(pred_rsrp)
            else "Predicted RSRP: unavailable"
        )

        st.write(
            f"Predicted RSRQ: {pred_rsrq:.4f}"
            if not pd.isna(pred_rsrq)
            else "Predicted RSRQ: unavailable"
        )

        st.write(
            f"Predicted SINR: {pred_sinr:.4f}"
            if not pd.isna(pred_sinr)
            else "Predicted SINR: unavailable"
        )

        st.write(
            f"Handover probability: {handover_probability:.4f}"
        )

        st.write(
            "Decision threshold: 0.90"
        )

        st.write(
            "Required RSRP improvement: 2"
        )

        st.write(
            "Prediction horizon: 3 seconds"
        )

        st.write(
            "Model: XGBoost"
        )

    st.divider()

    st.markdown("#### Model decision")

    st.write(decision)

    st.write(decision_reason)

    st.caption(
        "The radio values displayed by this project are the "
        "encoded/raw measurements supplied by the dataset. "
        "They should not be interpreted directly as dBm or dB."
    )


st.markdown(
    """
    <div class="footer">
        5G Network Quality Prediction & Intelligent Handover System
        · AI-driven network intelligence
    </div>
    """,
    unsafe_allow_html=True
)