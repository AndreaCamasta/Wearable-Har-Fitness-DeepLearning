import os
# --- DISATTIVAZIONE XLA PER EVITARE SEGMENTATION FAULT IN CLOUD ---
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import time
import graphviz
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hierarchical HAR Production System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — DARK, PROFESSIONAL, MLOPS-STYLE THEME
# ============================================================
st.markdown("""
<style>
    /* Global tweaks */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        letter-spacing: -0.5px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(139, 92, 246, 0.4);
    }

    /* Card container */
    .metric-card {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f0f6fc;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px 4px;
    }
    .badge-idle {
        background-color: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .badge-active {
        background-color: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        border: 1px solid #6366F1;
        animation: pulse 1.4s infinite;
    }
    .badge-success {
        background-color: rgba(46, 204, 113, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.5); }
        70% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    }

    /* Result banner */
    .result-banner {
        background: linear-gradient(90deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
        border-left: 4px solid #8B5CF6;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 1.1rem;
        color: #f0f6fc;
        margin-bottom: 10px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. ASYNC ASSET LOADING (CACHED)
# ============================================================
@st.cache_resource
def load_production_assets():
    with open('pipeline_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)

    with open('samples_demo.pkl', 'rb') as f:
        samples_demo = pickle.load(f)

    models = {
        'L0': tf.keras.models.load_model('model_level0.keras'),
        'static': tf.keras.models.load_model('model_static.keras'),
        'dynamic': tf.keras.models.load_model('model_dynamic.keras'),
        'complex': tf.keras.models.load_model('model_complex.keras')
    }
    return models, meta, samples_demo

assets_loaded = False
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    try:
        models, meta, samples_demo = load_production_assets()
        assets_loaded = True
        st.success("🚀 Ecosistema e Dati Reali caricati in memoria!")
        st.markdown("**Modelli attivi:**")
        for name in ["L0 (Gatekeeper)", "Static Specialist", "Dynamic Specialist", "Complex Specialist"]:
            st.markdown(f"<span class='badge badge-success'>✔ {name}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Errore nel caricamento delle risorse: {e}")

    st.markdown("---")
    st.markdown("### 📦 Dataset")
    st.caption("PAMAP2 — Physical Activity Monitoring (Accelerometer, Gyroscope, Magnetometer)")
    st.markdown("### 🏗️ Architettura")
    st.caption("4× CNN 1D coordinate in cascata gerarchica (Level 0 → Level 1 Specialists)")
    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow/Keras · Plotly")

# ============================================================
# HEADER
# ============================================================
st.title("🧠 Hierarchical Human Activity Recognition System")
st.subheader("Simulatore di Inferenza Edge in tempo reale a 4 Modelli Neurali Coordinati")
st.markdown(
    "<span class='badge badge-success'>PAMAP2 Dataset</span>"
    "<span class='badge badge-success'>Cascade Architecture</span>"
    "<span class='badge badge-success'>Real-Time Edge Inference</span>"
    "<span class='badge badge-success'>MLOps Dashboard</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================
# 2. ARCHITECTURE DIAGRAM (GRAPHVIZ)
# ============================================================
def build_cascade_diagram(active_branch=None):
    """
    Builds a Graphviz diagram of the hierarchical cascade.
    active_branch: None | 'static' | 'dynamic' | 'complex'
    Highlights the active routing path when a branch is provided.
    """
    dot = graphviz.Digraph()
    dot.attr(
        rankdir="LR",
        bgcolor="transparent",
        fontname="Helvetica",
        fontcolor="#f0f6fc"
    )
    dot.attr("node", style="filled", fontname="Helvetica", fontcolor="white",
              color="#30363d", fontsize="12")
    dot.attr("edge", color="#30363d", fontcolor="#8b949e", fontsize="10")

    default_fill = "#161b22"
    default_stroke = "#30363d"
    active_fill = "#6366F1"

    dot.node("data", "📡 Sensor Window\n(500 × 4)", shape="box", fillcolor=default_fill)
    dot.node("L0", "🔍 Level 0\nGatekeeper CNN", shape="box",
              fillcolor=active_fill if active_branch else default_fill)

    dot.node("static", "🧍 Static\nSpecialist", shape="box",
              fillcolor=active_fill if active_branch == "static" else default_fill)
    dot.node("dynamic", "🏃 Dynamic\nSpecialist", shape="box",
              fillcolor=active_fill if active_branch == "dynamic" else default_fill)
    dot.node("complex", "🎛️ Complex\nSpecialist", shape="box",
              fillcolor=active_fill if active_branch == "complex" else default_fill)

    dot.node("out", "🏆 Final\nActivity", shape="ellipse", fillcolor="#22c55e" if active_branch else default_fill)

    dot.edge("data", "L0")
    dot.edge("L0", "static", label="Statico",
              color=active_fill if active_branch == "static" else "#30363d",
              penwidth="2.5" if active_branch == "static" else "1")
    dot.edge("L0", "dynamic", label="Dinamico",
              color=active_fill if active_branch == "dynamic" else "#30363d",
              penwidth="2.5" if active_branch == "dynamic" else "1")
    dot.edge("L0", "complex", label="Complesso",
              color=active_fill if active_branch == "complex" else "#30363d",
              penwidth="2.5" if active_branch == "complex" else "1")

    for b in ["static", "dynamic", "complex"]:
        dot.edge(b, "out", color="#22c55e" if active_branch == b else "#30363d",
                  penwidth="2.5" if active_branch == b else "1")

    return dot

with st.expander("🗺️ Architettura del Sistema — Hierarchical Cascade Flow", expanded=True):
    diagram_placeholder = st.empty()
    diagram_placeholder.graphviz_chart(build_cascade_diagram(None), use_container_width=True)
    st.caption(
        "Ogni finestra sensoriale attraversa prima il **Level 0 (Gatekeeper)**, che instrada il campione "
        "verso uno dei tre specialisti dedicati. Questo design elimina il cross-talk tra classi eterogenee "
        "e migliora sensibilmente la precisione rispetto a un classificatore piatto a 9 classi."
    )

st.markdown("---")

# ============================================================
# CONTROL PANEL + LIVE DASHBOARD LAYOUT
# ============================================================
col_control, col_monitor = st.columns([1, 2])

with col_control:
    st.header("🎛️ Pannello di Controllo Stream")

    category_mapping = {
        "Attività Statiche (Lying/Sitting/Standing)": "static",
        "Attività Dinamiche (Walking/Running/Stairs)": "dynamic",
        "Attività Complesse (Cycling/Vacuum/Ironing)": "complex"
    }

    selected_label = st.selectbox(
        "Seleziona la macro-categoria reale inviata dallo Smartwatch:",
        list(category_mapping.keys())
    )
    selected_key = category_mapping[selected_label]

    stream_speed = st.slider("Velocità di campionamento (secondi per finestra):", 0.2, 2.0, 0.5)
    start_stream = st.button("▶️ Avvia Stream Dati Sensore", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏷️ Legenda Livelli")
    st.markdown(
        "<span class='badge badge-idle'>⚪ Idle</span>"
        "<span class='badge badge-active'>🔵 Elaborazione</span>"
        "<span class='badge badge-success'>🟢 Completato</span>",
        unsafe_allow_html=True
    )

with col_monitor:
    st.header("📊 Monitoraggio della Cascata Decisionale")
    routing_placeholder = st.empty()
    result_placeholder = st.empty()

    st.markdown("### 📈 Segnale Sensoriale Live (500 timesteps × 4 canali)")
    chart_placeholder = st.empty()

    st.markdown("### ⚡ Metriche di Produzione (MLOps)")
    m1, m2, m3, m4 = st.columns(4)
    val_window = m1.empty()
    val_latency = m2.empty()
    val_confidence = m3.empty()
    val_throughput = m4.empty()

    st.markdown("### 🧾 Log Sessione — Storico Predizioni")
    history_placeholder = st.empty()


def render_metric_card(placeholder, label, value):
    placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_routing_status(stage, macro_label=None, specialist_label=None):
    """
    stage: 'l0_running' | 'l0_done' | 'l1_running' | 'l1_done'
    """
    l0_badge = "badge-idle"
    l1_badge = "badge-idle"
    l0_text = "Level 0 — Gatekeeper"
    l1_text = "Level 1 — Specialista"

    if stage == "l0_running":
        l0_badge = "badge-active"
        l0_text = "Level 0 — Analisi pattern geometrico..."
    elif stage in ("l0_done", "l1_running", "l1_done"):
        l0_badge = "badge-success"
        l0_text = f"Level 0 — Stato rilevato: {macro_label}"

    if stage == "l1_running":
        l1_badge = "badge-active"
        l1_text = f"Level 1 — Attivazione {specialist_label}..."
    elif stage == "l1_done":
        l1_badge = "badge-success"
        l1_text = f"Level 1 — {specialist_label} ha concluso l'inferenza"

    routing_placeholder.markdown(
        f"<span class='badge {l0_badge}'>{l0_text}</span> &nbsp;➡️&nbsp; "
        f"<span class='badge {l1_badge}'>{l1_text}</span>",
        unsafe_allow_html=True
    )


def render_signal_chart(window, window_id):
    fig = go.Figure()
    channel_names = ["Canale 1 (Acc X)", "Canale 2 (Acc Y)", "Canale 3 (Acc Z)", "Canale 4 (Gyro/Mag)"]
    colors = ["#6366F1", "#22c55e", "#f59e0b", "#ef4444"]
    timesteps = np.arange(window.shape[0])

    for ch in range(window.shape[1]):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=window[:, ch],
            mode="lines",
            name=channel_names[ch] if ch < len(channel_names) else f"Canale {ch+1}",
            line=dict(color=colors[ch % len(colors)], width=1.6)
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Finestra sensoriale #{window_id} — 500 timesteps",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timestep",
        yaxis_title="Valore normalizzato"
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{window_id}_{time.time()}")


# Session state for the running history log across the stream session
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# ============================================================
# 3. REAL-TIME STREAMING INFERENCE LOGIC
# ============================================================
if start_stream and assets_loaded:
    X_demo = samples_demo[selected_key]['x']  # Forma originale (5, 500, 4)
    st.session_state.history_log = []  # reset log for a new session run

    latencies = []
    total_start = time.time()

    for i, window in enumerate(X_demo):
        window_start_time = time.time()
        real_window = np.expand_dims(window, axis=0)

        # --- LEVEL 0: IL PORTINAIO ---
        render_routing_status("l0_running")
        render_signal_chart(window, i + 1)

        pred_L0_probs = models['L0'].predict(real_window, verbose=0)
        macro_pred = np.argmax(pred_L0_probs, axis=1)[0]
        macro_labels = {0: "Statico", 1: "Dinamico Lineare", 2: "Complesso/Casalingo"}
        macro_label_text = macro_labels[macro_pred]

        render_routing_status("l0_done", macro_label=macro_label_text)
        time.sleep(0.15)  # Delay grafico per rendere scannabile il processo

        # --- LEVEL 1: GLI SPECIALISTI ---
        if macro_pred == 0:
            branch_key, specialist_name = "static", "Specialista Statico"
        elif macro_pred == 1:
            branch_key, specialist_name = "dynamic", "Specialista Dinamico (Locomozione)"
        else:
            branch_key, specialist_name = "complex", "Specialista Attività Complesse"

        render_routing_status("l1_running", macro_label=macro_label_text, specialist_label=specialist_name)
        diagram_placeholder.graphviz_chart(build_cascade_diagram(branch_key), use_container_width=True)

        probs = models[branch_key].predict(real_window, verbose=0)
        final_idx = np.argmax(probs, axis=1)[0]

        if branch_key == "static":
            final_activity = meta['le_static'].inverse_transform([final_idx])[0]
        elif branch_key == "dynamic":
            final_activity = meta['le_dynamic'].inverse_transform([final_idx])[0]
        else:
            final_activity = meta['le_complex'].inverse_transform([final_idx])[0]

        latency = (time.time() - window_start_time) * 1000  # ms
        conf = float(np.max(probs) * 100)
        activity_name = meta['activity_mapping'].get(final_activity, f"Attività {final_activity}")

        render_routing_status("l1_done", macro_label=macro_label_text, specialist_label=specialist_name)

        result_placeholder.markdown(f"""
            <div class="result-banner">
                🏆 <b>Predizione Finale — Finestra {i+1}/5:</b> l'utente sta facendo
                <b>{activity_name}</b> &nbsp; | &nbsp; Confidenza: <b>{conf:.1f}%</b>
            </div>
        """, unsafe_allow_html=True)

        # --- MLOps METRICS ---
        latencies.append(latency)
        elapsed = time.time() - total_start
        throughput = (i + 1) / elapsed if elapsed > 0 else 0.0
        stability = np.std(latencies) if len(latencies) > 1 else 0.0

        render_metric_card(val_window, "Window ID", f"{i+1} / {len(X_demo)}")
        render_metric_card(val_latency, "Latenza Inferenza", f"{latency:.2f} ms")
        render_metric_card(val_confidence, "Confidenza Rete", f"{conf:.1f} %")
        render_metric_card(val_throughput, "Throughput", f"{throughput:.2f} win/s")

        # --- HISTORY LOG ---
        st.session_state.history_log.append({
            "Window": i + 1,
            "Level 0": macro_label_text,
            "Specialista": specialist_name,
            "Attività Predetta": activity_name,
            "Confidenza (%)": round(conf, 1),
            "Latenza (ms)": round(latency, 2),
            "Jitter Latenza (ms)": round(stability, 2)
        })
        history_df = pd.DataFrame(st.session_state.history_log)
        history_placeholder.dataframe(history_df, use_container_width=True, hide_index=True)

        time.sleep(stream_speed)

    routing_placeholder.markdown(
        "<span class='badge badge-success'>🏁 Streaming della sequenza completato</span>",
        unsafe_allow_html=True
    )
    diagram_placeholder.graphviz_chart(build_cascade_diagram(None), use_container_width=True)

elif start_stream and not assets_loaded:
    st.error("Impossibile avviare lo stream: le risorse del modello non sono state caricate correttamente.")
