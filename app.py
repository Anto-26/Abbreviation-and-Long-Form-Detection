#!/usr/bin/env python3
"""Streamlit web app for abbreviation and long-form detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.predictor import AbbreviationDetector, DEFAULT_MODEL_PATH

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Abbreviation & Long-Form Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Label styling ─────────────────────────────────────────────────────────────
LABEL_STYLE = {
    "B-AC": {"bg": "#22c55e", "text": "#fff", "name": "Abbreviation"},
    "B-LF": {"bg": "#3b82f6", "text": "#fff", "name": "Long-Form (start)"},
    "I-LF": {"bg": "#06b6d4", "text": "#fff", "name": "Long-Form (cont.)"},
    "B-O":  {"bg": "#e5e7eb", "text": "#374151", "name": "Other"},
}


def token_badge(token: str, label: str) -> str:
    style = LABEL_STYLE.get(label, LABEL_STYLE["B-O"])
    return (
        f'<span style="'
        f'background:{style["bg"]};color:{style["text"]};'
        f'padding:3px 8px;border-radius:6px;margin:2px 2px;'
        f'display:inline-block;font-size:0.9rem;font-weight:500;">'
        f'{token}</span>'
    )


# ── Model loading (cached so Word2Vec only loads once) ────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    return AbbreviationDetector()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ About")
    st.markdown(
        """
        **Model:** RNN + Word2Vec (Google News 300-dim)
        **Optimizer:** RMSprop · lr 0.001
        **Dataset:** [PLOD-CW](https://huggingface.co/datasets/surrey-nlp/PLOD-CW)

        **Test-set performance**
        | Metric | Score |
        |--------|-------|
        | Accuracy | 87.4% |
        | Weighted F1 | 0.857 |
        | B-AC F1 | 0.63 |
        | B-LF F1 | 0.21 |

        **Labels**
        """
    )
    for label, s in LABEL_STYLE.items():
        st.markdown(
            f'<span style="background:{s["bg"]};color:{s["text"]};'
            f'padding:2px 8px;border-radius:4px;font-size:0.85rem;">'
            f'{label}</span>&nbsp;{s["name"]}',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Run `python train.py` first if the model isn't loaded yet.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Abbreviation & Long-Form Detector")
st.markdown("Enter a sentence below to identify abbreviations and their long-forms.")

# ── Model load check ──────────────────────────────────────────────────────────
model_ready = DEFAULT_MODEL_PATH.exists()

if not model_ready:
    st.error(
        "**No trained model found.**  \n"
        "Run `python train.py` from the `abbreviation_lf_detector/` directory "
        "to train and save the model, then refresh this page.",
        icon="🚨",
    )
    st.stop()

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model and Word2Vec embeddings… (first load may take ~30 s)"):
    detector = load_detector()

# ── Example sentences ─────────────────────────────────────────────────────────
EXAMPLES = [
    "EPI = Echo planar imaging .",
    "MRI ( Magnetic Resonance Imaging ) is widely used in clinical diagnosis .",
    "The WHO declared COVID-19 a pandemic in March 2020 .",
    "NLP stands for Natural Language Processing and is a subfield of AI .",
    "ATP ( adenosine triphosphate ) is the energy currency of the cell .",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
selected_example = ""
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"Example {i + 1}", use_container_width=True):
        selected_example = ex

# ── Text input ────────────────────────────────────────────────────────────────
default_text = selected_example if selected_example else ""
user_input = st.text_area(
    "Input text",
    value=default_text,
    height=100,
    placeholder="e.g.  EPI = Echo planar imaging .",
    label_visibility="collapsed",
)

detect_btn = st.button("Detect", type="primary", use_container_width=False)

# ── Detection ─────────────────────────────────────────────────────────────────
if detect_btn or (selected_example and not user_input == ""):
    text = (user_input or selected_example).strip()
    if not text:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing…"):
            result = detector.detect(text)

        st.markdown("---")

        # ── Token visualisation ───────────────────────────────────────────────
        st.subheader("Token Labels")
        badges_html = " ".join(token_badge(tok, lbl) for tok, lbl in result["tokens"])
        st.markdown(f'<div style="line-height:2.2;">{badges_html}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Summary columns ───────────────────────────────────────────────────
        col_abbr, col_lf = st.columns(2)

        with col_abbr:
            st.subheader("📌 Abbreviations")
            abbrevs = result["abbreviations"]
            if abbrevs:
                for a in abbrevs:
                    st.markdown(
                        f'<span style="background:#22c55e;color:#fff;padding:4px 10px;'
                        f'border-radius:6px;font-weight:600;">{a}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No abbreviations detected.")

        with col_lf:
            st.subheader("📖 Long-Forms")
            lforms = result["long_forms"]
            if lforms:
                for lf in lforms:
                    st.markdown(
                        f'<span style="background:#3b82f6;color:#fff;padding:4px 10px;'
                        f'border-radius:6px;font-weight:600;">{lf}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No long-forms detected.")

        st.markdown("---")

        # ── Raw data expander ─────────────────────────────────────────────────
        with st.expander("Raw token-label pairs"):
            rows = [{"Token": tok, "Label": lbl} for tok, lbl in result["tokens"]]
            st.dataframe(rows, use_container_width=True, hide_index=True)
