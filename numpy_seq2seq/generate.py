"""
Load a trained checkpoint and summarize an article: either a random
validation example or a raw .txt file you point it at. Also prints a
text-based heatmap of the Bahdanau attention weights so you can see which
input words the decoder focused on for each generated summary word.

Usage:
  .venv/bin/python numpy_seq2seq/generate.py
  .venv/bin/python numpy_seq2seq/generate.py --article_file path/to/some.txt
  .venv/bin/python numpy_seq2seq/generate.py --val_index 5
"""
import argparse

import numpy as np

from data import prepare_dataset, tokenize, encode, UNK
from model import Seq2SeqAttention

def attention_heatmap(alpha_row, enc_tokens, top_k=6):
    """alpha_row: (T_enc,) attention weights for one generated summary word.
    Returns the top_k most-attended article words with their weight, e.g.
    'climate(0.42) warning(0.18) change(0.11) ...' — no plotting library
    needed, and far more readable than printing all 60 positions."""
    pairs = [(tok, w) for tok, w in zip(enc_tokens, alpha_row) if tok != "<pad>"]
    pairs.sort(key=lambda tw: -tw[1])
    return " ".join(f"{tok}({w:.2f})" for tok, w in pairs[:top_k])


def summarize(model, stoi, itos, enc_max_len, dec_max_len, article_tokens):
    enc_ids = np.array([encode(article_tokens, stoi, enc_max_len, add_sos_eos=False)], dtype=np.int32)
    pred_ids, alphas = model.greedy_decode(enc_ids, max_len=dec_max_len)

    enc_tokens_padded = [itos[i] for i in enc_ids[0]]
    words = []
    print("\nAttention per generated word (top article words it looked at, with weight):")
    for t, tok_id in enumerate(pred_ids[0]):
        tok = itos[tok_id]
        if tok == "<eos>":
            break
        if tok == "<pad>":
            continue
        words.append(tok)
        print(f"  '{tok}': {attention_heatmap(alphas[0, t], enc_tokens_padded)}")
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    parser.add_argument("--article_file", type=str, default=None)
    parser.add_argument("--val_index", type=int, default=0)
    parser.add_argument("--enc_max_len", type=int, default=60)
    parser.add_argument("--dec_max_len", type=int, default=20)
    args = parser.parse_args()

    model = Seq2SeqAttention.load(args.checkpoint)
    ds = prepare_dataset(enc_max_len=args.enc_max_len, dec_max_len=args.dec_max_len)
    stoi, itos = ds["stoi"], ds["itos"]

    if args.article_file:
        with open(args.article_file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        article_tokens = tokenize(text)
        reference = None
    else:
        article_tokens, summary_tokens = ds["raw_val"][args.val_index]
        reference = " ".join(summary_tokens)

    print("Article (truncated to first", args.enc_max_len, "tokens):")
    print(" ".join(article_tokens[: args.enc_max_len]))

    prediction = summarize(model, stoi, itos, args.enc_max_len, args.dec_max_len, article_tokens)

    print("\n--- Summary ---")
    if reference is not None:
        print("Reference :", " ".join(reference.split()[: args.dec_max_len]))
    print("Predicted :", prediction)


if __name__ == "__main__":
    main()
