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
from rouge import rouge_scores


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
        avg_loss, num_real, _ = model.forward(enc_batch, dec_batch, training=False)
        total_loss_tokens += avg_loss * num_real
        total_tokens += num_real
    return total_loss_tokens / max(total_tokens, 1.0)


def ids_to_tokens(ids, itos):
    """Same as ids_to_text but returns a token list, for ROUGE."""
    return ids_to_text(ids, itos).split()


def evaluate_rouge(model, ds, batch_size, max_examples=222):
    """Greedy-decode the validation set and score it with ROUGE.

    Loss measures per-token confidence; ROUGE measures whether the finished
    summary actually overlaps the reference. They diverge once the model
    starts overfitting, which is why the checkpoint is chosen on ROUGE-L.
    """
    itos = ds["itos"]
    enc_ids = ds["enc_ids_val"][:max_examples]
    ref_ids = ds["dec_ids_val"][:max_examples]
    dec_max_len = ds["dec_ids_val"].shape[1]

    predictions, references = [], []
    for start in range(0, len(enc_ids), batch_size):
        enc_batch = enc_ids[start:start + batch_size]
        pred_batch, _ = model.greedy_decode(enc_batch, max_len=dec_max_len)
        for j in range(len(enc_batch)):
            predictions.append(ids_to_tokens(pred_batch[j], itos))
            references.append(ids_to_tokens(ref_ids[start + j], itos))
    return rouge_scores(predictions, references)


def lead_baseline_rouge(ds, max_examples=222):
    """Honest reference point: score the trivial 'just emit the article's
    first sentence' baseline. If the model cannot beat this, it has not
    learned anything useful."""
    itos = ds["itos"]
    dec_max_len = ds["dec_ids_val"].shape[1]
    predictions, references = [], []
    for i in range(min(max_examples, len(ds["enc_ids_val"]))):
        article = ids_to_tokens(ds["enc_ids_val"][i], itos)
        lead = []
        for tok in article[:dec_max_len]:
            lead.append(tok)
            if tok == ".":
                break
        predictions.append(lead)
        references.append(ids_to_tokens(ds["dec_ids_val"][i], itos))
    return rouge_scores(predictions, references)


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
    parser.add_argument("--dec_max_len", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=96)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6,
                         help="stop after this many epochs with no val_loss improvement")
    parser.add_argument("--lr_decay_patience", type=int, default=3,
                         help="halve the LR after this many epochs with no val_loss improvement")
    parser.add_argument("--lr_warmup", type=int, default=8,
                         help="never decay the LR before this epoch")
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
        hidden_size=args.hidden_size, dropout=args.dropout, seed=args.seed,
    )
    optimizer = Adam(model.params, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.RandomState(args.seed)

    baseline = lead_baseline_rouge(ds)
    print(f"lead-1 baseline (emit the article's first sentence): "
          f"R1={baseline['rouge1']:.3f} R2={baseline['rouge2']:.3f} RL={baseline['rougeL']:.3f}",
          flush=True)
    print("  -> the model needs to beat this to have learned anything useful.\n", flush=True)

    enc_ids_train, dec_ids_train = ds["enc_ids_train"], ds["dec_ids_train"]
    num_batches = int(np.ceil(len(enc_ids_train) / args.batch_size))

    # Two separate signals, because they are good at different jobs:
    #   val_loss  - smooth, so it is the reliable trigger for LR decay and
    #               early stopping.
    #   ROUGE-L   - what we actually care about, but noisy and near-zero
    #               early on, so it only decides which checkpoint to keep.
    # (Driving LR decay off ROUGE-L collapsed the LR to 2.5e-4 by epoch 11
    # on an earlier run, before the model had learned anything.)
    best_rouge_l = -1.0
    best_val_loss = float("inf")
    epochs_no_rouge_gain = 0
    epochs_no_val_gain = 0
    current_lr = args.lr

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        total_loss_tokens = 0.0
        total_tokens = 0.0
        for b, (enc_batch, dec_batch) in enumerate(
            iterate_batches(enc_ids_train, dec_ids_train, args.batch_size, rng), start=1
        ):
            avg_loss, num_real, cache = model.forward(enc_batch, dec_batch, training=True)
            grads = model.backward(cache)
            clip_grads_(grads, max_norm=args.clip_norm)
            optimizer.step(model.params, grads)

            total_loss_tokens += avg_loss * num_real
            total_tokens += num_real
            # flush=True: piping to `tee` makes stdout block-buffered, so
            # without this the run looks frozen for minutes at a time.
            if b % 20 == 0 or b == num_batches:
                so_far = time.time() - epoch_start
                eta = so_far / b * (num_batches - b)
                print(f"  epoch {epoch} batch {b}/{num_batches} "
                      f"loss={total_loss_tokens / total_tokens:.4f} "
                      f"({so_far:.0f}s elapsed, ~{eta:.0f}s left in epoch)", flush=True)

        train_loss = total_loss_tokens / total_tokens
        train_ppl = float(np.exp(min(train_loss, 20)))

        val_loss = evaluate(model, ds["enc_ids_val"], ds["dec_ids_val"], args.batch_size, rng)
        val_ppl = float(np.exp(min(val_loss, 20)))
        rouge = evaluate_rouge(model, ds, args.batch_size)

        elapsed = time.time() - epoch_start
        print(f"epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | "
              f"R1={rouge['rouge1']:.3f} R2={rouge['rouge2']:.3f} RL={rouge['rougeL']:.3f} "
              f"| lr={current_lr:.2e} ({elapsed:.1f}s)", flush=True)

        # Keep the checkpoint with the best ROUGE-L - that is the summary
        # quality we actually care about.
        if rouge["rougeL"] > best_rouge_l + 1e-4:
            best_rouge_l = rouge["rougeL"]
            epochs_no_rouge_gain = 0
            model.save(args.checkpoint)
            print(f"  -> new best ROUGE-L, saved checkpoint to {args.checkpoint}")
        else:
            epochs_no_rouge_gain += 1

        # Drive LR decay and early stopping off val_loss, which is smooth.
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_val_gain = 0
        else:
            epochs_no_val_gain += 1
            print(f"  -> no val_loss improvement ({epochs_no_val_gain}/{args.patience})")
            # Never decay during the warmup epochs: early loss is still
            # falling fast and cutting the LR then just stalls training.
            if (epoch > args.lr_warmup
                    and epochs_no_val_gain % args.lr_decay_patience == 0):
                current_lr *= 0.5
                optimizer.set_lr(current_lr)
                print(f"  -> lowered learning rate to {current_lr:.2e}")

        if epoch % args.sample_every == 0:
            show_samples(model, ds)

        if epochs_no_val_gain >= args.patience:
            print(f"Early stopping: no val_loss improvement for {args.patience} epochs in a row.")
            break

    print(f"\nBest val ROUGE-L={best_rouge_l:.4f} (lead-1 baseline {baseline['rougeL']:.4f}), "
          f"checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
