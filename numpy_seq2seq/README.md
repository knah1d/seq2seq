# From-Scratch NumPy Seq2Seq with Bahdanau Attention

A text summarizer for the BBC News Summary dataset, built with **only
NumPy** — no PyTorch/TensorFlow, no autograd. Every equation below is
implemented as explicit forward code *and* explicit backward (gradient)
code, so nothing is hidden behind a framework.

## Results

The project went through several task framings and one architectural
addition. Each row is a real run, not a projection:

| Attempt | val ROUGE-L | baseline | What happened |
|---|---|---|---|
| Naive truncated summary target | unusable | — | 61% of targets weren't even in the input window |
| Extractive, sentence augmentation | unusable | — | one input mapped to several contradictory targets; loss stalled at 5.75 |
| Extractive, skip opening sentence | 0.159 | 0.657 | model stopped conditioning on input entirely (identical output for unrelated articles) |
| Abstractive headline generation | 0.058 | 0.180 | 54% of headline words are singletons — an 8000-way softmax can't learn them |
| Extractive, lead sentence | 0.179 | 0.712 | learns topic/grammar, but can't reproduce specific words ("juan carlos ferrero" → "england s henman") |
| **+ pointer-generator** | **0.601** | 0.712 | model copies rare/proper nouns directly from the input |
| + repeat-blocking at decode | **0.628** | 0.712 | fixes a pointer-specific failure mode (see below) |

**Best config**: `--target_mode lead_sentence --pointer` (dropout 0.1, the
default — a 0.1/0.2/0.3 sweep found no further gain). Run it with:
```
python train.py --target_mode lead_sentence --pointer
```

The model does not beat the 0.712 lead-1 baseline, and on this task that
baseline is close to unbeatable by construction — the target *is* the
article's own lead sentence, so "copy the first sentence verbatim" scores
almost as well as a perfect model. What the run demonstrates instead is
the full mechanism working end-to-end: attention finding the right words,
a pointer copying names the model has never seen, and every one of those
backward passes verified against numerical gradients. See "Choosing a
task" and "Three experiments that failed" below for the reasoning behind
each row, and "The pointer-generator" for why it was worth adding.

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

### 5. The pointer-generator (`--pointer`)

Without it, the only way to produce a word is to pick it out of an 8000-way
softmax — which the model can never do reliably for a name it saw once or
twice in training (measured: 54% of headline words are singletons). Trained
without a pointer, the model learned topic and grammar but not entities:

```
article : ferrero eyes return to top form former world number one
           juan carlos ferrero insists he can get back to his best...
predicted: england s henman will be set to the first open in the first
           nations match in the dubai union championships .
```
Right sport-report *shape*, wrong everything specific.

A pointer lets the decoder **copy** a word straight out of the input instead
of generating it, reusing the attention weights it already computes as a
distribution over input *positions*:

```
p_gen    = sigmoid(w_h . h_t + w_c . context_t + w_x . x_t + b_gen)   # generate-vs-copy, per step
P_vocab  = softmax(logits)                                              # generate distribution
P_copy[w]= sum of alpha_i over input positions i where input_i == w    # copy distribution
P_final  = p_gen * P_vocab + (1 - p_gen) * P_copy
```

`p_gen` is learned, not fixed — the model decides per step whether this word
is better generated (`"the"`, `"said"`) or copied (`"ferrero"`). With it, the
same article decodes to:
```
former world number one juan carlos ferrero insists he can get back
to his best despite a tough start to 2005 .
```
one word off from the reference. Validation perplexity dropped from 270 to
23 in the same two epochs it took to establish this.

`layers.py`'s `pointer_ce_loss` implements the mixture loss and its backward
pass (gradients for the logits, the attention weights `alpha`, and `p_gen`);
the `d_alpha` gradient plugs straight into `attention_backward`'s existing
`d_alpha_extra` argument, which was added for exactly this.

### 6. Backward pass (the "from scratch" part)

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
~1e-5 relative error (every combination of tied/pointer/dropout passes).
Run it any time you change the math:
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

| `target_mode` | Input | Target | lead-1 baseline | Result |
|---|---|---|---|---|
| **`lead_sentence`** (default) | whole article | its earliest-appearing summary sentence (~24 tokens) | ROUGE-L 0.71 | 0.60 without a pointer, **0.63 with one** |
| `headline` | article body, headline removed | the headline (~5 tokens) | ROUGE-L 0.18 | 0.06 without a pointer, 0.17 with one - still short of baseline |

For `headline` the headline is **stripped from the encoder input** — otherwise
the task degenerates into copying the first five tokens.

`lead_sentence`'s baseline looks unbeatable because it nearly is - the task
is "reproduce the article's own lead sentence" and the baseline *is* the
article's own lead sentence. It was still the right task to invest in: with
a pointer the model gets to 0.63, and the same mechanism only got `headline`
to 0.17 against a much lower 0.18 bar. Copying is learnable at this scale;
generating from an 8000-way softmax over a singleton-heavy vocabulary is not
(see the headline row in "Three experiments that failed" below).

## Experiments that failed, and why

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
0.019/epoch — a plain (non-pointer) model at this scale cannot solve
*selection* and *exact reproduction* at once.

**4. Headline generation stayed hard even with the pointer.** Its lead-1
baseline (0.18) is far lower than `lead_sentence`'s (0.71), so it looked like
the easier task to beat on paper. It was not: without a pointer it scored
0.058 (the decoder emitted one near-identical headline for every article —
see "diversity" below); *with* a pointer it improved to 0.169, but still
fell short of the 0.180 baseline. The difference is span length. A
`lead_sentence` target is a ~24-word contiguous span the model can walk
through and copy piece by piece; a headline is a ~5-word *compression* that
reorders and drops words, so there is much less to literally point at.
Copying helps most when the target already resembles a copy.

**A methodology bug worth recording too: early stopping on the wrong
signal nearly hid the pointer's own success.** Validation loss on a ~210-
example set bottoms out after only 4-5 epochs and then *rises* as the model
starts memorizing specific training targets — but validation ROUGE-L kept
improving for another 25-30 epochs after that (e.g. the pointer run's best
ROUGE-L came at epoch 18, val_loss's minimum was epoch 4). Three early runs
were killed by patience keyed on `val_loss` alone, at epoch ~10, before the
model had learned anything close to its potential. `train.py` now requires
**both** `val_loss` and ROUGE-L to stall before decaying the LR or stopping
— a reminder that the metric you optimize for has to be the metric that
actually reflects the goal.

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

`evaluate_rouge` also reports **diversity**: the fraction of *distinct*
predictions across the validation set. This exists because a decoder that
has stopped reading its input emits nearly the same sentence for every
article — which happened (experiment 3 and the un-pointered `headline` run)
and was only caught by eyeballing samples. As a number it is unmissable:
healthy is `> 0.8`; the collapsed runs measured `0.05`–`0.19`.

### A pointer-specific bug, found and fixed for free (no retraining)

Pointer networks have a well-known failure mode — repeatedly copying the
same input word (`"...can can get to to his best..."`), documented in the
original pointer-generator paper (See et al., 2017), which adds a coverage
loss to fix it. `greedy_decode` already had a same-token repeat blocker
from earlier in the project, but it defaulted **off**, because at the time
it was hiding a different, worse bug: a decoder ignoring the encoder turned
`"the the the"` into `"the a the a"`, which read as a new bug rather than
the same one and cost real debugging time. Once diversity/ROUGE confirmed
the decoder genuinely reads its input, this default was revisited and
flipped — checked directly on an already-trained checkpoint, no retraining:

| `block_repeats` | validation predictions with a repeat | ROUGE-L |
|---|---|---|
| off | 186/210 | 0.601 |
| **on** (now the default) | **0/210** | **0.622** |

## Files

| File | Purpose |
|---|---|
| `data.py` | Reads `archive/BBC News Summary/`, tokenizes, dedupes near-identical articles, selects targets per `target_mode`, builds the vocabulary, encodes to padded id arrays, caches to `data_cache.npz`. |
| `rouge.py` | ROUGE-1/2/L in pure Python (no dependencies). |
| `layers.py` | Embedding, dropout, LSTM cell, Bahdanau attention, dense layer, softmax+cross-entropy, and the **pointer-generator mixture loss** — each with forward *and* backward. |
| `model.py` | `Seq2SeqAttention`: wires the layers into encoder → bridge → attention → decoder, optionally tied output weights and/or a pointer; `forward`/`backward` for training, `greedy_decode` for inference. |
| `optim.py` | Adam optimizer + weight decay + gradient-norm clipping, over the raw parameter dict. |
| `train.py` | Training loop: batches the data, calls forward/backward/optimizer step, reports train/val loss, validation ROUGE and prediction diversity each epoch, decays the LR and early-stops only when both loss and ROUGE-L have stalled, saves the best-ROUGE-L checkpoint. |
| `train.ipynb` | The training loop as a notebook (runs locally or in Colab — see below), with train/val loss and ROUGE-L plots. |
| `generate.py` | Loads a checkpoint, summarizes a validation example or your own `.txt` file, prints the predicted vs. reference summary and an attention heatmap. |
| `gradcheck.py` | Numerical vs. analytic gradient check (correctness proof for the hand-written backward pass). |
| `app.py` | Streamlit UI: pick or paste an article, see the generated summary and an attention heatmap. |

## How to run

All commands below are run from inside `numpy_seq2seq/`.

```
python3 -m venv ../.venv && ../.venv/bin/pip install numpy   # one-time setup
                                                             # (streamlit + matplotlib only for app.py / the notebook)

../.venv/bin/python gradcheck.py                     # verify the backward pass (expect PASS, ~1e-5)
../.venv/bin/python gradcheck.py --dropout           # same, but also checks the dropout backward
../.venv/bin/python gradcheck.py --pointer           # same, but also checks the pointer backward
../.venv/bin/python gradcheck.py --tie --pointer --dropout   # every path combined
../.venv/bin/python rouge.py                         # self-test of the ROUGE implementation

../.venv/bin/python train.py --pointer               # the best config found (see Results); ~2-2.5min/epoch
../.venv/bin/python train.py --target_mode headline --pointer   # the abstractive comparison

../.venv/bin/python generate.py --val_index 3
../.venv/bin/python generate.py --article_file some_article.txt
../.venv/bin/streamlit run app.py            # optional web UI
```

Useful training flags: `--target_mode`, `--pointer`, `--tie_weights`,
`--epochs`, `--dropout`, `--weight_decay`, `--lr`, `--patience`,
`--lr_decay_patience`, `--enc_max_len`, `--dec_max_len`. Changing
`--target_mode`, `--vocab_size`, `--enc_max_len` or `--dec_max_len`
automatically invalidates `data_cache.npz` and re-preprocesses.

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

- **`target_mode=lead_sentence`**: the task the model can actually learn.
  `headline`'s baseline is lower (0.18 vs 0.71), but its singleton-heavy
  5-word targets are much harder to generate than a 24-word span is to copy
  — see the Results table.
- **`--pointer`**: not on by default (it's an opt-in flag, matching
  `--tie_weights`), but it is the single biggest lever measured in this
  project (ROUGE-L 0.18 → 0.63 on `lead_sentence`) and the run command above
  includes it. Turn it off to see the pre-pointer failure mode directly.
- **Vocabulary**: 8000 most frequent words (rest map to `<unk>`) — keeps the
  output-layer matrix (`hidden*2 x vocab`) small enough to train fast.
  Dropping to 5000 would nearly double the target `<unk>` rate (3.8% → 7.4%).
- **`enc_max_len=60`**: enough context for both tasks. Attention costs
  O(`dec_max_len` × `enc_max_len`) per step, so this is the main speed knob.
- **`dec_max_len`**: 12 for headlines (covers 100% of headlines), 40 for lead
  sentences (p90 is 34). `encode()` appends `<eos>` **only if the sequence
  fit** — appending it after a truncation is what taught the first version of
  this model to stop mid-sentence.
- **Hidden size 128 / embedding 96**: small enough to train on CPU, large
  enough to learn. Still ~3M parameters (the `hidden*2 × vocab` output layer
  dominates) against ~2000 examples — an awkward ratio, and the reason a
  pointer helps as much as it does.
- **Dropout 0.1, weight decay 1e-5**: a 0.1/0.2/0.3 sweep on the pointer model
  found no further gain from more dropout (0.601/0.575/0.601 ROUGE-L) — see
  "Fighting overfitting" above.
- **Single-layer, unidirectional LSTM, greedy decoding (no beam search)**:
  deliberately simple — the point is to understand every equation, not to
  match a production summarizer.
- **Decoding guards**: `<pad>`/`<sos>`/`<unk>` logits are masked to `-inf` so
  they can never be emitted, and `block_repeats=True` blocks immediate word
  repeats (see "A pointer-specific bug" above — this is on by default now,
  for reasons that took two different bugs to establish).

## Expected results / limitations

Be realistic about the ceiling: ~3M parameters, ~2000 training examples, plain
NumPy on a CPU. Production summarizers train on 100k–1M+ pairs
(CNN/DailyMail is 287k). This will **not** produce publication-quality
summaries — but with `--pointer` on `lead_sentence` it produces summaries
that are frequently correct or one word off, per the Results table.

What you *should* see, and what is worth showing an instructor:

- **Training loss falling and validation ROUGE-L rising**, then plateauing.
  A useful early reference point: unigram entropy of the targets is
  **6.40 nats** (`lead_sentence`) — the loss from word frequencies alone, so
  dropping below it is the first real milestone. With `--pointer`, val
  perplexity should reach the 20s within ~5 epochs (it reached 270 *without*
  a pointer over 35 epochs on the same task).
- **`R2` (bigram overlap) lifting well off zero.** With the pointer this
  should reach ~0.5 on `lead_sentence` — whole phrases, not just plausible
  words, are appearing.
- **Diversity (fraction of distinct predictions) above 0.8.** Below that,
  don't trust the ROUGE number — check samples first (see "Experiments that
  failed").
- **Attention that tracks different words across decoder steps.**
  `generate.py`'s heatmap is the clearest visual evidence the mechanism
  works; with the pointer, watch for `p_gen` dropping (more copying) on
  proper nouns and numbers specifically.
- **A ROUGE-L number next to the lead-1 baseline**, whichever way it falls —
  see the Results table for what to expect on each task.

**On overfitting:** training loss keeps dropping long past the point of
generalizing (the pointer model reached train_loss 0.87 while val_loss rose
from 2.9 to 3.9). Hence checkpointing on validation ROUGE-L, and both loss
*and* ROUGE-L required to stall before early stopping. To *demonstrate*
overfitting deliberately, set `--patience 999` and watch train loss keep
falling long after validation ROUGE-L flattens.

**The pointer was the lever that mattered.** Tuning dropout, weight decay,
and the LR schedule on the plain model moved ROUGE-L by hundredths; adding
the pointer moved it by tenths, because it fixed the actual measured
bottleneck (rare/singleton target words) rather than adjusting regularization
around a model that structurally couldn't produce them.
