"""
ROUGE - the standard metric for summarization - in pure Python.

All three variants below are F1 scores over some notion of "overlap"
between the predicted summary and the reference summary:
  ROUGE-1  overlap of single words   (did we pick the right words?)
  ROUGE-2  overlap of word pairs     (did we get the phrasing right?)
  ROUGE-L  longest common subsequence (did we get the right order?)

Higher is better; 1.0 means an exact match. Loss tells you how confident
the model is about the next token, which is not the same as whether the
finished summary is any good - that is what these measure.
"""
from collections import Counter


def _f1(overlap, pred_total, ref_total):
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred, ref, n):
    """F1 over shared n-grams (clipped by count, so repeating a word does
    not let a prediction farm extra credit for it)."""
    pred_ngrams, ref_ngrams = _ngrams(pred, n), _ngrams(ref, n)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return _f1(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def lcs_length(a, b):
    """Length of the longest common subsequence (classic DP table)."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                cur.append(prev[j] + 1)
            else:
                cur.append(max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    """F1 over the longest common subsequence - rewards getting words in
    the right order without requiring them to be contiguous."""
    return _f1(lcs_length(pred, ref), len(pred), len(ref))


def rouge_scores(predictions, references):
    """Average ROUGE-1/2/L over a list of token-list pairs."""
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    return {
        "rouge1": sum(rouge_n(p, r, 1) for p, r in zip(predictions, references)) / n,
        "rouge2": sum(rouge_n(p, r, 2) for p, r in zip(predictions, references)) / n,
        "rougeL": sum(rouge_l(p, r) for p, r in zip(predictions, references)) / n,
    }


if __name__ == "__main__":
    ref = "the game has sold more than five million copies worldwide".split()
    for name, pred in [
        ("exact match      ", ref),
        ("close paraphrase ", "the game sold more than five million copies".split()),
        ("wrong topic      ", "the company said it was a new deal".split()),
    ]:
        s = rouge_scores([pred], [ref])
        print(f"{name} R1={s['rouge1']:.3f} R2={s['rouge2']:.3f} RL={s['rougeL']:.3f}")
