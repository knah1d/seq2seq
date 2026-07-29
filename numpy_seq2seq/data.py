
"""
Data loading and preprocessing for the BBC News Summary dataset.

Pipeline:
  1. Read every (article, summary) pair from archive/BBC News Summary/.
  2. Tokenize with a simple lowercase word tokenizer (no external libs).
  3. Build a vocabulary from the most frequent tokens in the training split.
  4. Encode each article/summary into fixed-length integer id sequences
     (truncated + padded), ready to be fed to the model in batches.
  5. Cache everything to a single .npz file so re-runs are instant.
"""
import glob
import os
import re
import numpy as np

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "archive", "BBC News Summary",
)
ARTICLES_DIR = os.path.join(DATA_ROOT, "News Articles")
SUMMARIES_DIR = os.path.join(DATA_ROOT, "Summaries")

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[.,!?;:]")


def tokenize(text):
    """Lowercase word/number/punctuation tokenizer. No external deps."""
    return _TOKEN_RE.findall(text.lower())


def load_pairs():
    """Read every (article_tokens, summary_tokens) pair from disk, sorted
    for reproducibility (categories then filenames)."""
    pairs = []
    categories = sorted(os.listdir(ARTICLES_DIR))
    for category in categories:
        article_dir = os.path.join(ARTICLES_DIR, category)
        summary_dir = os.path.join(SUMMARIES_DIR, category)
        filenames = sorted(os.listdir(article_dir))
        for fname in filenames:
            article_path = os.path.join(article_dir, fname)
            summary_path = os.path.join(summary_dir, fname)
            if not os.path.isfile(summary_path):
                continue
            with open(article_path, encoding="utf-8", errors="ignore") as f:
                article_text = f.read()
            with open(summary_path, encoding="utf-8", errors="ignore") as f:
                summary_text = f.read()
            pairs.append((tokenize(article_text), tokenize(summary_text)))
    return pairs


def build_vocab(token_lists, vocab_size):
    """Frequency-based vocabulary capped at vocab_size (including specials)."""
    counts = {}
    for tokens in token_lists:
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
    most_common = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    num_regular = vocab_size - len(SPECIAL_TOKENS)
    words = [w for w, _ in most_common[:num_regular]]
    itos = list(SPECIAL_TOKENS) + words
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi, max_len, add_sos_eos):
    """Convert tokens -> fixed-length id array (truncate + pad with PAD id).
    If add_sos_eos, wraps as <sos> tok1 tok2 ... <eos> before truncating."""
    ids = [stoi.get(t, stoi[UNK]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SOS]] + ids[: max_len - 2] + [stoi[EOS]]
    else:
        ids = ids[:max_len]
    pad_id = stoi[PAD]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids[:max_len]


def prepare_dataset(
    vocab_size=8000,
    enc_max_len=60,
    dec_max_len=20,
    val_fraction=0.1,
    seed=0,
    cache_path=None,
):
    """Build (or load from cache) the full encoded dataset + vocab.

    Returns a dict with keys:
      enc_ids_train, dec_ids_train, enc_ids_val, dec_ids_val  (int32 arrays)
      stoi, itos
      raw_val: list of (article_tokens, summary_tokens) for the val split,
               kept for human-readable inspection in generate.py
    """
    if cache_path is None:
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_cache.npz"
        )

    config_tag = f"{vocab_size}-{enc_max_len}-{dec_max_len}-{val_fraction}-{seed}"

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached["config_tag"]) == config_tag:
            return {
                "enc_ids_train": cached["enc_ids_train"],
                "dec_ids_train": cached["dec_ids_train"],
                "enc_ids_val": cached["enc_ids_val"],
                "dec_ids_val": cached["dec_ids_val"],
                "stoi": cached["stoi"].item(),
                "itos": list(cached["itos"]),
                "raw_val": list(cached["raw_val"]),
            }

    print("Preprocessing dataset from raw text files (first run only)...")
    pairs = load_pairs()

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]

    num_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:num_val]
    train_pairs = pairs[num_val:]

    stoi, itos = build_vocab(
        [tokens for tokens, _ in train_pairs] + [tokens for _, tokens in train_pairs],
        vocab_size,
    )

    def encode_split(split_pairs):
        enc_ids = np.array(
            [encode(a, stoi, enc_max_len, add_sos_eos=False) for a, _ in split_pairs],
            dtype=np.int32,
        )
        dec_ids = np.array(
            [encode(s, stoi, dec_max_len, add_sos_eos=True) for _, s in split_pairs],
            dtype=np.int32,
        )
        return enc_ids, dec_ids

    enc_ids_train, dec_ids_train = encode_split(train_pairs)
    enc_ids_val, dec_ids_val = encode_split(val_pairs)

    np.savez_compressed(
        cache_path,
        enc_ids_train=enc_ids_train,
        dec_ids_train=dec_ids_train,
        enc_ids_val=enc_ids_val,
        dec_ids_val=dec_ids_val,
        stoi=stoi,
        itos=np.array(itos),
        raw_val=np.array(val_pairs, dtype=object),
        config_tag=config_tag,
    )
    print(f"Cached preprocessed data to {cache_path}")

    return {
        "enc_ids_train": enc_ids_train,
        "dec_ids_train": dec_ids_train,
        "enc_ids_val": enc_ids_val,
        "dec_ids_val": dec_ids_val,
        "stoi": stoi,
        "itos": itos,
        "raw_val": val_pairs,
    }


if __name__ == "__main__":
    ds = prepare_dataset()
    print("train pairs:", len(ds["enc_ids_train"]))
    print("val pairs:", len(ds["enc_ids_val"]))
    print("vocab size:", len(ds["itos"]))
    print("sample encoder ids:", ds["enc_ids_train"][0][:15])
    print("sample decoder ids:", ds["dec_ids_train"][0])
