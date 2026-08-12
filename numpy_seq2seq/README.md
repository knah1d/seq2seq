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

## Choosing a task that is actually learnable (the core of this project)

The obvious setup — "input = first N words of the article, target = first M
words of the summary" — **cannot work on this dataset**. Finding out why, and
fixing it, mattered far more than any hyperparameter.

BBC "summaries" are **extractive**: real article sentences copied verbatim but
**reordered**. So truncating the summary grabs text from anywhere in the
article. Measured across all 2225 pairs with the naive 60-in / 20-out setup:

| Measurement | Value | Consequence |
|---|---|---|
| Target content-words present in the input window | **52.7%** | Asked to generate words it cannot see |
| Summaries whose first sentence starts after token 60 | **61.3%** | The answer is usually outside the input entirely |
| Targets ending at a sentence boundary | **6.1%** | `<eos>` lands mid-sentence, so the model **never learns when to stop** |

That produced the classic failure output, `"the <unk> , said the <unk> , said
the <unk> ..."`. It was never a bug in the math — `gradcheck.py` passed
throughout. The task itself was unsolvable as posed.

`data.py` now offers two well-posed tasks via `target_mode`:

| `target_mode` | Input | Target | lead-1 baseline | Notes |
|---|---|---|---|---|
| **`headline`** (default) | article body, headline removed | the headline (~5 tokens) | **ROUGE-L 0.19** | Abstractive. Beatable baseline, so a good score means something. |
| `lead_sentence` | whole article | its earliest-appearing summary sentence (~24 tokens) | **ROUGE-L 0.66** | Extractive and easy to learn, but the trivial baseline is near-unbeatable. |

For `headline` the headline is **stripped from the encoder input** — otherwise
the task degenerates into copying the first five tokens.

## Three experiments that failed, and why

These are the most informative results in the project, so they are recorded
rather than hidden.

**1. Truncated summary targets → unlearnable.** See the table above: the
answer was usually not in the input, and `<eos>` marked a random mid-sentence
point. Fixed by making every target a complete, reachable sentence.

**2. Sentence augmentation → actively worse.** Each summary holds ~8
extractive sentences, so emitting one training example per sentence tripled
the data (2003 → 6032). But it pairs the *same* encoder input with *several
different* "correct" targets. Cross-entropy answers a contradiction by hedging
toward a bland average, and training stalled: loss stuck at 5.75 moving
0.025/epoch, with train and val loss nearly equal — textbook underfitting.
**More data is worthless if the extra examples contradict each other.**
Still available as `prepare_dataset(augment=True)`.

**3. Making the extractive task harder → the model stopped reading the input.**
Skipping the article's opening sentence dropped the lead-1 baseline from 0.66
to 0.18 — a genuinely non-trivial task. But at this scale the model could not
learn it. After 15 epochs it emitted nearly identical text for completely
different articles:

```
Tate & Lyle (business):  "the blair , who was the first year , the first year , ..."
Halo 2 (video games):    "the blair said the first year , who was a first year , ..."
```

Validation ROUGE-L actually *fell* (0.146 → 0.099) while val_loss flattened at
0.019/epoch. Selecting which of ~8 sentences is salient *and* reproducing ~24
exact words through an 8000-way softmax needs a copy/pointer mechanism, or far
more than 2003 examples.

## Fighting overfitting

~3M parameters against ~2000 training examples memorizes readily, so:

- **Dropout** (`0.1`, `layers.py`) on encoder embeddings, decoder embeddings
  and the output-layer input. Inverted dropout: scale survivors by `1/(1-p)`
  at train time, identity at eval. Kept low — for most of this project the
  model was *under*fitting, and you do not regularize an underfitting model.
- **Weight decay** (`1e-5`, `optim.py`) on matrices but not biases.
- **LR decay on plateau**, halving after 3 epochs with no `val_loss` gain and
  never before epoch 8.
- **Early stopping** on `val_loss`.

`val_loss` drives decay and stopping because it is smooth; ROUGE-L only picks
which checkpoint to keep. An earlier version drove LR decay off ROUGE-L and
collapsed the learning rate to 2.5e-4 by epoch 11, before the model had
learned anything — noisy metrics make bad triggers.

## Measuring it: ROUGE, and an honest baseline

`rouge.py` implements ROUGE-1/2/L (word, word-pair, and longest-common-
subsequence overlap F1) in ~40 lines of dependency-free Python.

Loss and summary quality **diverge** once a model overfits, so the checkpoint
is saved on **best validation ROUGE-L**.

`train.py` prints a **lead-1 baseline** (emit the input's first sentence)
before training starts. If the model cannot beat it, it has not learned
anything useful. Strong lead baselines on news are a well-known real result,
so keeping the number visible is honest science, not a weakness.

`<unk>` is excluded from both the training loss and ROUGE scoring. The decoder
is forbidden from emitting `<unk>`, so training it to predict that token would
waste capacity, and leaving `<unk>` in the references would charge the model
for matches it can never make while letting the baseline score on them.

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

../.venv/bin/python train.py                              # headline task (default), ~25s/epoch
../.venv/bin/python train.py --target_mode lead_sentence  # the extractive comparison

../.venv/bin/python generate.py --val_index 3
../.venv/bin/python generate.py --article_file some_article.txt
../.venv/bin/streamlit run app.py            # optional web UI
```

Useful training flags: `--target_mode`, `--epochs`, `--dropout`,
`--weight_decay`, `--lr`, `--patience`, `--lr_decay_patience`,
`--enc_max_len`, `--dec_max_len`. Changing `--target_mode`, `--vocab_size`,
`--enc_max_len` or `--dec_max_len` automatically invalidates
`data_cache.npz` and re-preprocesses.

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

- **`target_mode=headline`**: the only framing here whose trivial baseline
  (ROUGE-L 0.19) is realistically beatable — see the task table above.
- **Vocabulary**: 8000 most frequent words (rest map to `<unk>`) — keeps the
  output-layer matrix (`hidden*2 x vocab`) small enough to train fast.
  Dropping to 5000 would nearly double the target `<unk>` rate (3.8% → 7.4%).
- **`enc_max_len=60`**: enough context for both tasks. Attention costs
  O(`dec_max_len` × `enc_max_len`) per step, so this is the main speed knob.
- **`dec_max_len`**: 10 for headlines (p90 length is 7), 40 for lead
  sentences (p90 is 34). `encode()` appends `<eos>` **only if the sequence
  fit** — appending it after a truncation is what taught the first version of
  this model to stop mid-sentence.
- **Hidden size 128 / embedding 96**: small enough to train on CPU, large
  enough to learn. Still ~3M parameters (the `hidden*2 × vocab` output layer
  dominates) against ~2000 examples — an awkward ratio, and the honest
  limiting factor on results.
- **Dropout 0.1, weight decay 1e-5**: see "Fighting overfitting" above.
- **Single-layer, unidirectional LSTM, greedy decoding (no beam search)**:
  deliberately simple — the point is to understand every equation, not to
  match a production summarizer.
- **Decoding guards**: `<pad>`/`<sos>`/`<unk>` logits are masked to `-inf` so
  they can never be emitted. A repeat-blocker exists
  (`greedy_decode(block_repeats=True)`) but is **off by default** — it only
  hides symptoms: when the decoder was ignoring the encoder entirely it turned
  `"the the the"` into `"the a the a"`, which read as a different bug and cost
  real debugging time.

## Expected results / limitations

Be realistic about the ceiling: ~3M parameters, ~2000 training examples, plain
NumPy on a CPU. Production summarizers train on 100k–1M+ pairs
(CNN/DailyMail is 287k). This will **not** produce publication-quality
summaries.

What you *should* see, and what is worth showing an instructor:

- **Training loss falling and validation ROUGE-L rising**, then plateauing,
  with early stopping picking the best point. A useful reference point: the
  unigram entropy of the targets is **6.40 nats** — that is the loss a model
  gets from learning word frequencies alone, so dropping below it is the first
  real milestone.
- **`R2` (bigram overlap) lifting off zero.** This is the signal that whole
  phrases, not just plausible words, are appearing.
- **Predictions that differ across articles.** Sounds trivial, but the failed
  experiment above produced near-identical output for a business article and a
  video-game article — always check this before trusting a ROUGE number.
- **Attention that tracks different words across decoder steps** rather than
  collapsing onto one or two. `generate.py`'s heatmap is the clearest visual
  evidence the mechanism works.
- **A ROUGE-L number next to the lead-1 baseline**, whichever way it falls.

**On overfitting:** training loss keeps dropping long past the point of
generalizing. Symptoms: validation summaries become fluent but *topically
wrong*, and attention collapses onto the same one or two words. Hence
checkpointing on validation ROUGE-L and early stopping. To *demonstrate*
overfitting deliberately, set `--patience 999` and watch train loss and
validation ROUGE-L diverge.

**The biggest remaining lever is data, not tuning.** 2000 examples against 3M
parameters is the binding constraint. Within this dataset that is close to
exhausted (sentence augmentation was tried and made things worse); going
further would mean a larger corpus, which pure-NumPy CPU training makes
impractical, or adding a copy/pointer mechanism so the model does not have to
regenerate source words through an 8000-way softmax.
