"""
Numerical gradient check: proof that the hand-derived backward pass in
model.py actually matches the true gradient of the loss.

For a handful of parameters, we perturb a single scalar entry by +eps/-eps,
recompute the loss both times, estimate the derivative as
  (loss(w+eps) - loss(w-eps)) / (2*eps)
and compare it against the analytic gradient produced by backward().
Run this any time the layers/model math changes.
"""
import sys

import numpy as np

from model import Seq2SeqAttention

# Pass --dropout to additionally verify the dropout backward pass. The RNG is
# reset before every forward so the *same* dropout mask is drawn each time -
# otherwise the loss would change for reasons unrelated to the perturbation
# and finite differences would be meaningless.
USE_DROPOUT = "--dropout" in sys.argv
DROPOUT_P = 0.3 if USE_DROPOUT else 0.0
TIE = "--tie" in sys.argv
PTR = "--pointer" in sys.argv

np.random.seed(0)

VOCAB, EMB, HID = 30, 8, 10
B, T_ENC, T_DEC = 3, 5, 4

model = Seq2SeqAttention(VOCAB, EMB, HID, dropout=DROPOUT_P, tie_weights=TIE,
                         use_pointer=PTR, seed=1)

enc_ids = np.random.randint(4, VOCAB, size=(B, T_ENC)).astype(np.int32)
enc_ids[0, -2:] = 0  # exercise the padding/masking path
dec_ids = np.random.randint(4, VOCAB, size=(B, T_DEC)).astype(np.int32)
dec_ids[:, 0] = 1  # <sos>
dec_ids[1, -1:] = 0  # exercise decoder padding in the loss mask


def forward_fixed_mask():
    """Forward with a deterministic dropout mask (see USE_DROPOUT above)."""
    model.rng = np.random.RandomState(999)
    return model.forward(enc_ids, dec_ids, training=USE_DROPOUT)


def loss_fn():
    loss, _, _ = forward_fixed_mask()
    return loss


print(f"tie_weights={TIE}  use_pointer={PTR}  dropout: {DROPOUT_P}"
      f"{'  (pass --dropout to enable)' if not USE_DROPOUT else ''}\n")

_, _, cache = forward_fixed_mask()
grads = model.backward(cache)

eps = 1e-5
max_rel_err = 0.0
checks = [
    ("W_proj", (0, 0)) if TIE else ("W_out", (0, 0)),
    ("W_proj", (3, 5)) if TIE else ("W_out", (3, 5)),
    ("W_emb", (5, 2)), ("W_emb", (11, 4)), ("b_out", (7,)),
    ("attn_Wa", (1, 1)), ("attn_v", (2,)),
    ("enc_Wxi", (0, 0)), ("enc_Whf", (1, 2)), ("enc_Wxo", (2, 1)), ("enc_Whg", (0, 3)),
    ("dec_Wxi", (2, 3)), ("dec_Whf", (0, 0)), ("dec_Wxo", (1, 1)), ("dec_Whg", (3, 2)),
    ("W_bridge_h", (1, 1)), ("W_bridge_c", (0, 2)),
]
if PTR:
    checks += [("w_gen_h", (0, 0)), ("w_gen_c", (4, 0)),
               ("w_gen_x", (2, 0)), ("b_gen", (0,))]

print(f"{'param':10s} {'index':10s} {'analytic':>14s} {'numeric':>14s} {'rel_err':>10s}")
for name, idx in checks:
    orig = model.params[name][idx]

    model.params[name][idx] = orig + eps
    loss_plus = loss_fn()
    model.params[name][idx] = orig - eps
    loss_minus = loss_fn()
    model.params[name][idx] = orig

    numeric_grad = (loss_plus - loss_minus) / (2 * eps)
    analytic_grad = grads[name][idx]
    # Below ~1e-6 both estimates are just float64/finite-difference noise
    # (the true gradient is ~0 there); skip the ratio so it can't blow up.
    if max(abs(numeric_grad), abs(analytic_grad)) < 1e-6:
        rel_err = 0.0
    else:
        denom = max(abs(numeric_grad), abs(analytic_grad))
        rel_err = abs(numeric_grad - analytic_grad) / denom
    max_rel_err = max(max_rel_err, rel_err)
    print(f"{name:10s} {str(idx):10s} {analytic_grad:14.8f} {numeric_grad:14.8f} {rel_err:10.2e}")

print()
if max_rel_err < 1e-4:
    print(f"PASS - max relative error {max_rel_err:.2e} (backward pass matches numerical gradient)")
else:
    print(f"FAIL - max relative error {max_rel_err:.2e} is too high, check the backward pass")
