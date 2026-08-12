"""
Simple Streamlit UI for the from-scratch NumPy Seq2Seq + Bahdanau attention
summarizer. Pure demo/inference layer - all the model math still lives in
model.py/layers.py; this file just wires a checkpoint + the dataset vocab to
a text box and renders the result, including an attention heatmap over the
article text (word background intensity = how much the decoder attended to
that word when generating each summary word).

Run from inside numpy_seq2seq/ with:
  ../.venv/bin/streamlit run app.py
"""
import html

import numpy as np
import streamlit as st

from data import prepare_dataset, tokenize, encode
from model import Seq2SeqAttention

CHECKPOINT_PATH = "checkpoint.npz"


@st.cache_resource
def load_model():
    return Seq2SeqAttention.load(CHECKPOINT_PATH)


@st.cache_resource
def load_dataset():
    # No length arguments: always take data.py's own defaults, so this UI can
    # never drift out of sync with what the model was trained on.
    return prepare_dataset()


def summarize(model, stoi, itos, article_tokens, enc_max_len, dec_max_len):
    enc_ids = np.array(
        [encode(article_tokens, stoi, enc_max_len, add_sos_eos=False)], dtype=np.int32
    )
    pred_ids, alphas = model.greedy_decode(enc_ids, max_len=dec_max_len)

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

    Raw softmax weights over ~120 positions are all fairly close together
    even when the model has a clear preference, so we min-max normalize
    across this article's weights rather than dividing by the max alone -
    that stretches the actual spread to the full 0-1 opacity range instead
    of everything looking uniformly faint.
    """
    real = [(tok, w) for tok, w in zip(enc_tokens, weights) if tok != "<pad>"]
    if not real:
        return "<em>(empty)</em>"
    ws = [w for _, w in real]
    lo, hi = min(ws), max(ws)
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
        "```\n../.venv/bin/python train.py\n```"
    )
    st.stop()

stoi, itos, raw_val = ds["stoi"], ds["itos"], ds["raw_val"]
# Sequence lengths come from the cached dataset, not hardcoded constants.
ENC_MAX_LEN = int(ds["enc_ids_train"].shape[1])
DEC_MAX_LEN = int(ds["dec_ids_train"].shape[1])

# A checkpoint trained against a different vocabulary would silently produce
# nonsense, so fail loudly instead.
if model.V != len(itos):
    st.error(
        f"Checkpoint/vocabulary mismatch: `{CHECKPOINT_PATH}` was trained with a "
        f"vocab of {model.V}, but the current dataset has {len(itos)}. "
        "Retrain, or delete the stale checkpoint and `data_cache.npz`."
    )
    st.stop()

st.sidebar.header("Input")
mode = st.sidebar.radio("Article source", ["Validation example", "Paste your own text"])

reference_summary = None
if mode == "Validation example":
    idx = st.sidebar.number_input(
        "Validation example index", min_value=0, max_value=len(raw_val) - 1, value=0, step=1
    )
    article_tokens, target_tokens = raw_val[idx]
    reference_summary = " ".join(target_tokens)
else:
    default_text = " ".join(raw_val[0][0])
    user_text = st.sidebar.text_area("Article text", value=default_text, height=200)
    article_tokens = tokenize(user_text)

if st.sidebar.button("Reload checkpoint"):
    # Streamlit caches the model for the session; this picks up a retrain.
    load_model.clear()
    load_dataset.clear()
    st.rerun()

st.subheader(f"Article (first {ENC_MAX_LEN} tokens are what the encoder reads)")
if article_tokens:
    st.write(" ".join(article_tokens[:ENC_MAX_LEN]))
    if len(article_tokens) > ENC_MAX_LEN:
        st.caption(f"({len(article_tokens) - ENC_MAX_LEN} further tokens truncated)")
else:
    st.info("Enter some article text in the sidebar.")
    st.stop()

# Summarize on every rerun. A single greedy decode is milliseconds, and this
# guarantees the summary shown always matches the article shown - caching it
# in session_state let the two drift apart when you changed the input.
words, word_alphas, enc_tokens_padded = summarize(
    model, stoi, itos, article_tokens, ENC_MAX_LEN, DEC_MAX_LEN
)

st.subheader("Summary")
if reference_summary:
    st.markdown(f"**Reference (the human-selected key sentence):** {reference_summary}")
st.markdown(f"**Predicted:** {' '.join(words) if words else '_(model produced nothing)_'}")

st.subheader("Attention heatmap")
if words:
    options = ["Average over all words"] + [f"{i}: '{w}'" for i, w in enumerate(words)]
    view = st.selectbox("Show attention while generating", options)
    if view == options[0]:
        weights = np.mean(word_alphas, axis=0)
    else:
        weights = word_alphas[int(view.split(":")[0])]
    st.markdown(render_attention_html(enc_tokens_padded, weights), unsafe_allow_html=True)
    st.caption(
        "Darker highlight = higher attention weight on that article word. "
        "Weights are min-max normalized per view so the spread is visible."
    )
