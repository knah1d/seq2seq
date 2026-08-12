# From-Scratch NumPy Seq2Seq with Bahdanau Attention

A text summarizer for the BBC News Summary dataset, built with **only
NumPy** — no PyTorch/TensorFlow, no autograd. Every equation below is
implemented as explicit forward code *and* explicit backward (gradient)
code, so nothing is hidden behind a framework.

## Why each piece exists

Summarization is a sequence-to-sequence problem: read a long article
(encoder), then generate a short summary one word at a time (decoder).
A single fixed-size vector can't hold everything about a 120-word article
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

## What we actually train on (the most important design decision)

The obvious approach — "input = first N words of the article, target = first
M words of the summary" — **cannot work on this dataset**, and measuring why
was the single biggest improvement to this project.

BBC "summaries" are **extractive**: they are real article sentences copied
verbatim, but **reordered**. So a fixed truncation of the summary grabs text
from anywhere in the article. Measured across all 2225 pairs:

| Measurement (with the naive 60-in / 20-out setup) | Value | Consequence |
|---|---|---|
| Target content-words present in the input window | **52.7%** | The model is asked to generate words it cannot see |
| Summaries whose first sentence starts *after* token 60 | **61.3%** | The answer is usually outside the input entirely |
| Targets that end at a sentence boundary | **6.1%** | `<eos>` lands at a random mid-sentence point, so the model **never learns when to stop** → run-on repetition |

That explains the classic failure output — `"the <unk> , said the <unk> ,
said the <unk> ..."`. It wasn't a bug in the math (`gradcheck.py` passed
throughout); the task itself was unsolvable as posed.

**The fix** (`build_examples` / `select_key_sentence` in `data.py`): for each
article, emit one training example per summary sentence that actually
appears **inside** the encoder's input window. Every target is then:

- a **complete sentence** ending in `.`, so `<eos>` means something and the
  model learns to stop;
- **reachable** — 94% of key sentences start within the first 60 tokens and
  99.3% within 120 (hence `enc_max_len=120`);
- **still genuine extractive summarization** — a sentence a human summarizer
  picked out as salient.

It also yields **~3× more training data** (2,003 articles → ~6,000 examples)
from the same dataset, which matters a lot: the model has ~3M parameters, so
data volume is the main thing limiting it. The train/val split happens at the
**article** level *before* augmenting, so no article's sentences leak across
the split, and validation keeps exactly one target per article so ROUGE stays
a clean per-article measure.

## Fighting overfitting

~3M parameters against ~6k training examples memorizes readily, so:

- **Dropout** (`dropout=0.3`, `layers.py`) on the encoder embeddings, decoder
  embeddings, and the output-layer input. Inverted dropout: scale survivors
  by `1/(1-p)` at train time, identity at eval time.
- **Weight decay** (`1e-5`, `optim.py`) on matrices but not biases —
  shrinking a bias just shifts the function without reducing capacity.
- **LR decay on plateau**: halve the learning rate after 2 epochs with no
  improvement, before giving up entirely.
- **Early stopping on validation ROUGE-L**, not loss (see below).

## Measuring it: ROUGE, and an honest baseline

`rouge.py` implements ROUGE-1/2/L (word, word-pair, and longest-common-
subsequence overlap F1) in ~40 lines of dependency-free Python.

Loss and summary quality **diverge** once a model overfits — loss measures
per-token confidence, ROUGE measures whether the finished summary overlaps
the reference. So the checkpoint is saved on **best validation ROUGE-L**.

`train.py` also prints a **lead-1 baseline** (just emit the article's first
sentence) before training starts. If the model can't beat that, it hasn't
learned anything useful. Strong lead baselines on news summarization are a
well-known real result, so keeping this number visible is honest science
rather than a weakness to hide.

## Files

| File | Purpose |
|---|---|
| `data.py` | Reads `archive/BBC News Summary/`, tokenizes, selects the **key-sentence targets** described above, builds the vocabulary, encodes to padded id arrays, caches to `data_cache.npz`. |
| `rouge.py` | ROUGE-1/2/L in pure Python (no dependencies). |
| `layers.py` | Embedding, dropout, LSTM cell, Bahdanau attention, dense layer, softmax+cross-entropy — each with forward *and* backward. |
| `model.py` | `Seq2SeqAttention`: wires the layers into encoder → bridge → attention → decoder; `forward`/`backward` for training, `greedy_decode` for inference. |
| `optim.py` | Adam optimizer + weight decay + gradient-norm clipping, over the raw parameter dict. |
| `train.py` | Training loop: batches the data, calls forward/backward/optimizer step, reports train/val loss and validation ROUGE each epoch, decays the LR on plateau, early-stops, and saves the best-ROUGE-L checkpoint. |
| `train.ipynb` | The training loop as a notebook (runs locally or in Colab — see below), with train/val loss and ROUGE-L plots. |
| `generate.py` | Loads a checkpoint, summarizes a validation example or your own `.txt` file, prints the predicted vs. reference summary and an attention heatmap. |
| `gradcheck.py` | Numerical vs. analytic gradient check (correctness proof for the hand-written backward pass). |
| `app.py` | Streamlit UI: pick or paste an article, see the generated summary and an attention heatmap. |

## How to run

All commands below are run from inside `numpy_seq2seq/`.

```
python3 -m venv ../.venv && ../.venv/bin/pip install numpy   # one-time setup
                                                             # (streamlit + matplotlib only for app.py / the notebook)

../.venv/bin/python gradcheck.py             # verify the backward pass (expect PASS, ~1e-5)
../.venv/bin/python gradcheck.py --dropout   # same, but also checks the dropout backward
../.venv/bin/python rouge.py                 # self-test of the ROUGE implementation

../.venv/bin/python train.py                 # train with the defaults below; early-stops on val ROUGE-L

../.venv/bin/python generate.py --val_index 3
../.venv/bin/python generate.py --article_file some_article.txt
../.venv/bin/streamlit run app.py            # optional web UI
```

Useful training flags: `--epochs`, `--dropout`, `--weight_decay`, `--lr`,
`--patience`, `--lr_decay_patience`, `--enc_max_len`, `--dec_max_len`.
Changing `--vocab_size`/`--enc_max_len`/`--dec_max_len` automatically
invalidates `data_cache.npz` and re-preprocesses.

### Running training in Google Colab instead

The dataset is tracked in this repo (see the note in `.gitignore`) specifically
so Colab can get everything in one step. Open `numpy_seq2seq/train.ipynb` in
Colab (upload it, or File → Open notebook → GitHub → this repo) and run the
first code cell — it detects Colab, `git clone`s this repo (or `git pull`s if
you re-run it later in the same session), and `cd`s into `numpy_seq2seq/`
automatically. Everything after that cell runs identically to local. A GPU
runtime gives no speedup here (no GPU calls anywhere in this NumPy code) —
CPU runtime is fine.

## Defaults and why

- **Vocabulary**: 8000 most frequent words (rest map to `<unk>`) — keeps the
  output-layer matrix (`hidden*2 x vocab`) small enough to train fast.
  Dropping to 5000 would nearly double the target `<unk>` rate (3.8% → 7.4%).
- **`enc_max_len=120`**: the smallest window that makes 99.3% of key-sentence
  targets reachable (60 only reaches 94%, and costs the model a lot of the
  content words it needs).
- **`dec_max_len=32`**: covers the 90th-percentile key-sentence length (29
  tokens) plus `<sos>`/`<eos>`.
- **Hidden size 128 / embedding 96**: small enough to train on CPU, large
  enough to learn. Note this is still ~3M parameters (the `hidden*2 × vocab`
  output layer dominates) against ~6k examples — hence the regularization.
- **Dropout 0.3, weight decay 1e-5**: see "Fighting overfitting" above.
- **Single-layer, unidirectional LSTM, greedy decoding (no beam search)**:
  deliberately simple — the point of this project is to understand every
  equation, not to match a production summarizer's quality.
- **Decoding guards**: `<pad>`/`<sos>`/`<unk>` logits are masked to `-inf` so
  they can never be emitted, and an immediate word repeat falls back to the
  second-best token. Cheap fixes for the most visible greedy-decoding
  artifacts.

## Expected results / limitations

Be realistic about the ceiling here: this is a ~3M-parameter model trained on
~6k examples with plain NumPy on a CPU. Production summarizers train on
100k–1M+ pairs (CNN/DailyMail is 287k). It will **not** produce
publication-quality summaries.

What you *should* see, and what is worth showing an instructor:

- **Training loss falling and validation ROUGE-L rising** over the first
  epochs, then plateauing — with early stopping picking the best point.
- **Complete sentences that stop on their own**, containing no `<unk>` and
  reusing article vocabulary. (Compare against the "before" failure mode
  documented above — that contrast *is* the story of this project.)
- **Attention that tracks different words across decoder steps** rather than
  collapsing onto one or two. `generate.py`'s heatmap is the clearest visual
  evidence the attention mechanism is doing real work.
- **A ROUGE-L number next to the lead-1 baseline.** Beating a lead baseline
  on news is genuinely hard — if the model lands near it, say so plainly.
  That is a real, well-documented result in the summarization literature,
  not a failed project.

**On overfitting:** training loss will keep dropping long past the point of
generalizing. Symptoms: validation summaries become fluent-sounding but
*topically wrong* for the specific article, and attention collapses onto the
same one or two words every step. That is why the checkpoint is chosen on
validation ROUGE-L and training early-stops. If you want to *demonstrate*
overfitting to an instructor, set `--patience 999` and watch train loss and
validation ROUGE-L diverge.

**The honest biggest lever, if you want better numbers:** more data, not more
tuning. The sentence-level augmentation already extracts ~3× more supervision
from this corpus; beyond that you'd need a larger dataset, which pure-NumPy
CPU training makes impractical.
