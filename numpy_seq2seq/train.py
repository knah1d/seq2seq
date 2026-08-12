"""
Train the from-scratch NumPy Seq2Seq + Bahdanau attention summarizer on the
BBC News Summary dataset.

Usage:
  .venv/bin/python numpy_seq2seq/train.py
  .venv/bin/python numpy_seq2seq/train.py --epochs 15 --batch_size 32
"""
import argparse
import time

import numpy as np

from data import prepare_dataset, PAD
from model import Seq2SeqAttention
from optim import Adam, clip_grads_


def iterate_batches(enc_ids, dec_ids, batch_size, rng, shuffle=True):
    n = enc_ids.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield enc_ids[idx], dec_ids[idx]


def ids_to_text(ids, itos):
    words = []
    for i in ids:
        tok = itos[i]
        if tok == "<eos>":
            break
        if tok in ("<pad>", "<sos>"):
            continue
        words.append(tok)
    return " ".join(words)


def evaluate(model, enc_ids, dec_ids, batch_size, rng):
    """Validation-set loss (no backward pass) - the honest signal for
    whether the model is generalizing, since training loss keeps dropping
    even after the model starts just memorizing the training set."""
    total_loss_tokens = 0.0
    total_tokens = 0.0
    for enc_batch, dec_batch in iterate_batches(enc_ids, dec_ids, batch_size, rng, shuffle=False):
        avg_loss, num_real, _ = model.forward(enc_batch, dec_batch)
        total_loss_tokens += avg_loss * num_real
        total_tokens += num_real
    return total_loss_tokens / max(total_tokens, 1.0)


def show_samples(model, ds, num_samples=2):
    itos = ds["itos"]
    enc_ids = ds["enc_ids_val"][:num_samples]
    ref_dec_ids = ds["dec_ids_val"][:num_samples]
    pred_ids, _ = model.greedy_decode(enc_ids, max_len=ds["dec_ids_val"].shape[1])
    for i in range(num_samples):
        print(f"  [val #{i}] article: {ids_to_text(enc_ids[i], itos)[:200]}...")
        print(f"            reference : {ids_to_text(ref_dec_ids[i], itos)}")
        print(f"            prediction: {ids_to_text(pred_ids[i], itos)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--enc_max_len", type=int, default=60)
    parser.add_argument("--dec_max_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=96)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5,
                         help="stop after this many epochs with no validation-loss improvement")
    args = parser.parse_args()

    ds = prepare_dataset(
        vocab_size=args.vocab_size,
        enc_max_len=args.enc_max_len,
        dec_max_len=args.dec_max_len,
    )
    print(f"train examples: {len(ds['enc_ids_train'])}, val examples: {len(ds['enc_ids_val'])}")
    print(f"vocab size: {len(ds['itos'])}")

    model = Seq2SeqAttention(
        vocab_size=len(ds["itos"]), emb_dim=args.emb_dim,
        hidden_size=args.hidden_size, seed=args.seed,
    )
    optimizer = Adam(model.params, lr=args.lr)
    rng = np.random.RandomState(args.seed)

    enc_ids_train, dec_ids_train = ds["enc_ids_train"], ds["dec_ids_train"]
    num_batches = int(np.ceil(len(enc_ids_train) / args.batch_size))

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        total_loss_tokens = 0.0
        total_tokens = 0.0
        for b, (enc_batch, dec_batch) in enumerate(
            iterate_batches(enc_ids_train, dec_ids_train, args.batch_size, rng), start=1
        ):
            avg_loss, num_real, cache = model.forward(enc_batch, dec_batch)
            grads = model.backward(cache)
            clip_grads_(grads, max_norm=args.clip_norm)
            optimizer.step(model.params, grads)

            total_loss_tokens += avg_loss * num_real
            total_tokens += num_real
            if b % 20 == 0 or b == num_batches:
                print(f"  epoch {epoch} batch {b}/{num_batches} "
                      f"running_loss={total_loss_tokens / total_tokens:.4f}")

        train_loss = total_loss_tokens / total_tokens
        train_ppl = float(np.exp(min(train_loss, 20)))

        val_loss = evaluate(model, ds["enc_ids_val"], ds["dec_ids_val"], args.batch_size, rng)
        val_ppl = float(np.exp(min(val_loss, 20)))

        elapsed = time.time() - epoch_start
        print(f"epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
              f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save(args.checkpoint)
            print(f"  -> new best val_loss, saved checkpoint to {args.checkpoint}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no val_loss improvement ({epochs_without_improvement}/{args.patience})")

        if epoch % args.sample_every == 0:
            show_samples(model, ds)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val_loss improvement for {args.patience} epochs in a row.")
            break

    print(f"Best val_loss={best_val_loss:.4f}, checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
