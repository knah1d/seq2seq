# From-Scratch NumPy Seq2Seq with Bahdanau Attention

A text summarizer for the BBC News Summary dataset, built with **only
NumPy** — no PyTorch/TensorFlow, no autograd. Every equation below is
implemented as explicit forward code *and* explicit backward (gradient)
code, so nothing is hidden behind a framework.

## Why each piece exists

Summarization is a sequence-to-sequence problem: read a long article
(encoder), then generate a short summary one word at a time (decoder).
A single fixed-size vector can't hold everything about a 60-word article
before you've even generated the first word — that's the bottleneck
Bahdanau attention removes: at every decoding step, the decoder looks back
at *all* encoder states and learns which words matter most right now.

## Architecture

```
article tokens --> [Embedding] --> [Encoder LSTM] --> H_enc (all hidden states)
                                                           |
                                                     [Bahdanau Attention]
                                                           |
prev summary token --> [Embedding] --+--> [Decoder LSTM] --> [Dense + Softmax] --> next word
                                      |          ^
                                context vector --+
```

### 1. Encoder — LSTM over the article

For each article token embedding `x_t`, an LSTM cell updates its hidden
state `h_t` *and* a separate cell state `c_t` (the longer-term memory):

```
i_t = sigmoid(x_t Wxi + h_{t-1} Whi + bi)   # input gate  - how much new info to write
f_t = sigmoid(x_t Wxf + h_{t-1} Whf + bf)   # forget gate - how much old memory to keep
o_t = sigmoid(x_t Wxo + h_{t-1} Who + bo)   # output gate - how much of memory to expose
g_t = tanh(x_t Wxg + h_{t-1} Whg + bg)      # candidate content to write into memory
c_t = f_t * c_{t-1} + i_t * g_t             # new cell (memory) state
h_t = o_t * tanh(c_t)                       # new hidden state
```

This is the same idea as a GRU (a gated running summary of everything seen
so far) but with memory (`c_t`) and the externally-visible state (`h_t`)
kept separate, and 4 gates instead of 3 — the extra gate is what lets the
LSTM learn to preserve information over longer stretches without it decaying.

We keep **every** `h_t` (not just the last one) as `H_enc`, because attention
needs to look at all of them. Padded article positions carry the previous
`(h, c)` forward unchanged, so the last stored state is always the true
final state regardless of padding.

### 2. Bridge

The decoder's initial hidden and cell states are learned projections of the
encoder's final states:
```
s_0 = tanh(h_T_enc @ W_bridge_h + b_bridge_h)
c_0 = tanh(c_T_enc @ W_bridge_c + b_bridge_c)
```

### 3. Bahdanau (additive) attention

At each decoder step `t`, given the previous decoder hidden state `s_{t-1}`
(the LSTM's `h`, not its `c`):

```
e_i     = v . tanh(Wa @ s_{t-1} + Ua @ H_enc[i])   for every encoder position i
alpha   = softmax(e)                                over encoder positions (padding masked out)
context = sum_i alpha_i * H_enc[i]
```

`alpha` is a probability distribution over the article's words — that's what
`generate.py` prints as a text heatmap so you can see what the model
"looked at" when it produced each summary word.

### 4. Decoder — LSTM + output projection

```
lstm_input  = concat(embed(prev_word), context)
s_t, cell_t = LSTM_cell(lstm_input, s_{t-1}, cell_{t-1})   # same LSTM equations as the encoder
logits      = concat(s_t, context) @ W_out + b_out
next_word   = softmax(logits)                               # cross-entropy loss during training
```

During training we use **teacher forcing** (the real previous summary word
is fed in, not the model's own guess). During generation (`greedy_decode` in
`model.py`) there is no reference summary, so the model's own argmax
prediction is fed back in as the next `prev_word`.

### 5. Backward pass (the "from scratch" part)

`model.py`'s `backward()` manually walks time backwards through the decoder,
then the encoder, accumulating gradients into a plain `dict` of NumPy
arrays — this is exactly backpropagation-through-time (BPTT), just written
out by hand instead of relying on autograd. Each LSTM cell has two incoming
gradients per timestep (one for `h_t`, one for `c_t`) that both have to be
threaded backward — `layers.py`'s `lstm_cell_backward` returns gradients for
`x_t`, `h_{t-1}`, *and* `c_{t-1}`. `layers.py` has one `*_backward` function
per forward function (LSTM cell, attention, dense, softmax+cross-entropy),
each derived from the chain rule.

**`gradcheck.py`** proves this is correct: it perturbs individual parameter
entries by a tiny `epsilon`, recomputes the loss, and checks that the
numerical derivative matches the analytic gradient from `backward()` to
~1e-6 relative error. Run it any time you change the math:
```
.venv/bin/python numpy_seq2seq/gradcheck.py
```

## Files

| File | Purpose |
|---|---|
| `data.py` | Reads `archive/BBC News Summary/`, tokenizes, builds a frequency-based vocabulary, encodes to fixed-length padded id arrays, caches to `data_cache.npz`. |
| `layers.py` | Embedding, LSTM cell, Bahdanau attention, dense layer, softmax+cross-entropy — each with forward *and* backward. |
| `model.py` | `Seq2SeqAttention`: wires the layers into encoder → bridge → attention → decoder; `forward`/`backward` for training, `greedy_decode` for inference. |
| `optim.py` | Adam optimizer + gradient-norm clipping, over the raw parameter dict. |
| `train.py` | Training loop: batches the data, calls forward/backward/optimizer step, prints loss/perplexity and sample summaries each epoch, saves a checkpoint. |
| `generate.py` | Loads a checkpoint, summarizes a validation example or your own `.txt` file, prints the predicted vs. reference summary and an attention heatmap. |
| `gradcheck.py` | Numerical vs. analytic gradient check (correctness proof for the hand-written backward pass). |

## How to run

```
python3 -m venv .venv && .venv/bin/pip install numpy   # one-time setup

.venv/bin/python numpy_seq2seq/gradcheck.py             # verify backward pass is correct
.venv/bin/python numpy_seq2seq/train.py --epochs 12     # train (~1 min/epoch on CPU)
.venv/bin/python numpy_seq2seq/generate.py --val_index 3
.venv/bin/python numpy_seq2seq/generate.py --article_file some_article.txt
```

## Defaults and why

- **Vocabulary**: 8000 most frequent words (rest map to `<unk>`) — keeps the
  output-layer matrix (`hidden*2 x vocab`) small enough to train fast.
- **Sequence lengths**: articles truncated to 60 tokens, summaries to 20 —
  pure-Python-loop BPTT over long sequences is slow; short sequences keep
  each epoch under a minute while still exercising every part of the model.
- **Hidden size 128 / embedding 96**: small enough to train quickly on CPU,
  large enough to actually learn (loss drops from ~9.0 at init towards
  ~3-4 after a dozen epochs on this dataset).
- **Single-layer, unidirectional LSTM, greedy decoding (no beam search)**:
  deliberately simple — the point of this project is to understand every
  equation, not to match a production summarizer's quality.

## Expected results / limitations

This is a small model trained briefly on ~2000 examples with plain NumPy —
it will **not** produce publication-quality summaries. What you should
expect and can show your instructor:
- Training loss/perplexity steadily decreasing over epochs.
- Generated summaries moving from repeated generic words (early epochs) to
  fragments that reuse relevant article vocabulary (later epochs).
- Attention weights that concentrate on topically relevant words rather
  than being uniform — this is the part worth walking through in
  `generate.py`'s printed heatmap, since it's the clearest visual evidence
  the attention mechanism is doing something meaningful.
