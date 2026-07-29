"""
Simple Streamlit UI for the from-scratch NumPy Seq2Seq + Bahdanau attention
summarizer. Pure demo/inference layer — all the model math still lives in
model.py/layers.py; this file just wires a checkpoint + the dataset vocab to
a text box and renders the result, including an attention heatmap over the
article text (word background intensity = how much the decoder attended to
that word when generating each summary word).

Run with:
  .venv/bin/streamlit run numpy_seq2seq/app.py
"""
import html

import numpy as np
import streamlit as st

from data import prepare_dataset, tokenize, encode
from model import Seq2SeqAttention

CHECKPOINT_PATH = "checkpoint.npz"
ENC_MAX_LEN = 60
DEC_MAX_LEN = 20


@st.cache_resource
def load_model():
    return Seq2SeqAttention.load(CHECKPOINT_PATH)


@st.cache_resource
def load_dataset():
    return prepare_dataset(enc_max_len=ENC_MAX_LEN, dec_max_len=DEC_MAX_LEN)


def summarize(model, stoi, itos, article_tokens):
    enc_ids = np.array(
        [encode(article_tokens, stoi, ENC_MAX_LEN, add_sos_eos=False)], dtype=np.int32
    )
    pred_ids, alphas = model.greedy_decode(enc_ids, max_len=DEC_MAX_LEN)

    enc_tokens_padded = [itos[i] for i in enc_ids[0]]
    words, word_alphas = [], []
    for t, tok_id in enumerate(pred_ids[0]):
        tok = itos[tok_id]
        if tok == "<eos>":
            break
        if tok == "<pad>":
            continue
        words.append(tok)
        word_alphas.append(alphas[0, t])
    return words, word_alphas, enc_tokens_padded


def render_attention_html(enc_tokens, weights):
    """Render the (truncated) article with each word's background tinted
    orange proportional to its attention weight. Alpha-blended over the
    page background so it reads fine in both light and dark themes.

    Raw softmax weights over ~60 positions are all fairly close together
    (e.g. 0.02-0.12) even when the model has a clear preference, so we
    min-max normalize across this article's weights rather than dividing
    by the max alone - that stretches the actual spread to the full
    0-1 opacity range instead of everything looking uniformly faint."""
    real = [(tok, w) for tok, w in zip(enc_tokens, weights) if tok != "<pad>"]
    ws = [w for _, w in real]
    lo, hi = min(ws, default=0.0), max(ws, default=1.0)
    span = max(hi - lo, 1e-8)
    spans = []
    for tok, w in real:
        alpha = min(1.0, (w - lo) / span)
        safe_tok = html.escape(tok)
        spans.append(
            f'<span style="background-color: rgba(255,99,71,{alpha:.2f}); '
            f'padding: 1px 3px; border-radius: 3px; margin: 1px; display: inline-block;">'
            f"{safe_tok}</span>"
        )
    return " ".join(spans)


st.set_page_config(page_title="NumPy Seq2Seq Summarizer", page_icon="📝")
st.title("📝 From-Scratch NumPy Seq2Seq Summarizer")
st.caption(
    "Encoder-decoder LSTM with Bahdanau attention, implemented with plain NumPy "
    "(manual forward + backward pass, no autograd). Trained on the BBC News Summary dataset."
)

try:
    model = load_model()
    ds = load_dataset()
except FileNotFoundError:
    st.error(
        f"No checkpoint found at `{CHECKPOINT_PATH}`. Train the model first:\n\n"
        "`.venv/bin/python numpy_seq2seq/train.py`"
    )
    st.stop()

stoi, itos, raw_val = ds["stoi"], ds["itos"], ds["raw_val"]

st.sidebar.header("Input")
mode = st.sidebar.radio("Article source", ["Validation example", "Paste your own text"])

reference_summary = None
if mode == "Validation example":
    idx = st.sidebar.number_input(
        "Validation example index", min_value=0, max_value=len(raw_val) - 1, value=0, step=1
    )
    article_tokens, summary_tokens = raw_val[idx]
    reference_summary = " ".join(summary_tokens)
    article_text_preview = " ".join(article_tokens[:ENC_MAX_LEN])
else:
    default_text = " ".join(raw_val[0][0])
    user_text = st.sidebar.text_area("Article text", value=default_text, height=200)
    article_tokens = tokenize(user_text)
    article_text_preview = " ".join(article_tokens[:ENC_MAX_LEN])

st.subheader("Article (first 60 tokens used by the encoder)")
st.write(article_text_preview if article_text_preview else "_(empty)_")

if st.sidebar.button("Summarize", type="primary") or "words" not in st.session_state:
    words, word_alphas, enc_tokens_padded = summarize(model, stoi, itos, article_tokens)
    st.session_state["words"] = words
    st.session_state["word_alphas"] = word_alphas
    st.session_state["enc_tokens_padded"] = enc_tokens_padded

words = st.session_state["words"]
word_alphas = st.session_state["word_alphas"]
enc_tokens_padded = st.session_state["enc_tokens_padded"]

st.subheader("Summary")
if reference_summary is not None:
    st.markdown(f"**Reference:** {' '.join(reference_summary.split()[:DEC_MAX_LEN])}")
st.markdown(f"**Predicted:** {' '.join(words) if words else '_(empty)_'}")

st.subheader("Attention heatmap")
if words:
    view = st.radio(
        "Show attention for", ["Average over all words"] + [f"'{w}' (step {i})" for i, w in enumerate(words)],
        horizontal=False,
    )
    if view == "Average over all words":
        weights = np.mean(word_alphas, axis=0)
    else:
        step = int(view.split("step ")[1].rstrip(")"))
        weights = word_alphas[step]
    st.markdown(render_attention_html(enc_tokens_padded, weights), unsafe_allow_html=True)
    st.caption("Darker/more saturated highlight = higher attention weight on that article word.")
else:
    st.info("Nothing generated yet — click Summarize.")
