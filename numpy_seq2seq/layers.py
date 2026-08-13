"""
Building blocks of the network, written by hand with NumPy only.

Every layer exposes a `forward(...)` that returns `(output, cache)` and a
`backward(d_output, cache)` that returns gradients w.r.t. its inputs and
parameters. Everything is batched: arrays carry a leading `batch` dimension
(or are shaped (batch, features)) so we process the whole dataset with
matrix multiplies instead of a Python loop per example.

Notation used throughout:
  B  = batch size
  H  = hidden size
  E  = embedding size
  V  = vocab size
  T  = a sequence length (encoder or decoder)
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_matrix(rng, shape, scale=None):
    """Xavier/Glorot-ish uniform init."""
    fan_in = shape[0]
    if scale is None:
        scale = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-scale, scale, size=shape).astype(np.float64)


# ----------------------------------------------------------------------
# Embedding lookup
# ----------------------------------------------------------------------
def embedding_forward(ids, W_emb):
    """ids: (B, T) int32 -> (B, T, E)"""
    out = W_emb[ids]
    cache = (ids, W_emb.shape)
    return out, cache


def embedding_backward(d_out, cache):
    """d_out: (B, T, E) -> gradient dict for W_emb (same shape as W_emb)."""
    ids, shape = cache
    d_W = np.zeros(shape, dtype=np.float64)
    np.add.at(d_W, ids, d_out)
    return d_W


# ----------------------------------------------------------------------
# Dropout (inverted): during training, randomly zero a fraction p of the
# activations and scale the survivors up by 1/(1-p) so the expected sum is
# unchanged. At eval time it is the identity, so no rescaling is needed.
# This stops the network leaning on any single unit and is our main defence
# against memorizing a ~2000-example training set.
# ----------------------------------------------------------------------
def dropout_forward(x, p, rng, training):
    if not training or p <= 0.0:
        return x, None
    keep = 1.0 - p
    mask = (rng.random_sample(x.shape) < keep).astype(np.float64) / keep
    return x * mask, mask


def dropout_backward(d_out, mask):
    """The same mask that scaled the forward pass scales the gradient."""
    return d_out if mask is None else d_out * mask


# ----------------------------------------------------------------------
# LSTM cell (single timestep). Standard formulation:
#   i_t = sigmoid(x_t Wxi + h_{t-1} Whi + bi)          input gate
#   f_t = sigmoid(x_t Wxf + h_{t-1} Whf + bf)          forget gate
#   o_t = sigmoid(x_t Wxo + h_{t-1} Who + bo)          output gate
#   g_t = tanh(x_t Wxg + h_{t-1} Whg + bg)             candidate cell content
#   c_t = f_t * c_{t-1} + i_t * g_t                    new cell state
#   h_t = o_t * tanh(c_t)                              new hidden state
# Unlike a GRU, the LSTM carries two states through time (h_t and c_t).
# ----------------------------------------------------------------------
def lstm_cell_params(rng, input_size, hidden_size, prefix):
    H, X = hidden_size, input_size
    p = {}
    for gate in "ifog":
        p[f"{prefix}_Wx{gate}"] = init_matrix(rng, (X, H))
        p[f"{prefix}_Wh{gate}"] = init_matrix(rng, (H, H))
        p[f"{prefix}_b{gate}"] = np.zeros(H, dtype=np.float64)
    # forget gate bias initialized to 1: standard LSTM trick so the cell
    # defaults to "remember" early in training instead of forgetting everything.
    p[f"{prefix}_bf"][:] = 1.0
    return p


def lstm_cell_forward(x_t, h_prev, c_prev, params, prefix):
    """x_t: (B, X), h_prev/c_prev: (B, H) -> h_t, c_t: (B, H)"""
    Wxi, Whi, bi = params[f"{prefix}_Wxi"], params[f"{prefix}_Whi"], params[f"{prefix}_bi"]
    Wxf, Whf, bf = params[f"{prefix}_Wxf"], params[f"{prefix}_Whf"], params[f"{prefix}_bf"]
    Wxo, Who, bo = params[f"{prefix}_Wxo"], params[f"{prefix}_Who"], params[f"{prefix}_bo"]
    Wxg, Whg, bg = params[f"{prefix}_Wxg"], params[f"{prefix}_Whg"], params[f"{prefix}_bg"]

    i = sigmoid(x_t @ Wxi + h_prev @ Whi + bi)
    f = sigmoid(x_t @ Wxf + h_prev @ Whf + bf)
    o = sigmoid(x_t @ Wxo + h_prev @ Who + bo)
    g = np.tanh(x_t @ Wxg + h_prev @ Whg + bg)

    c_t = f * c_prev + i * g
    tanh_c_t = np.tanh(c_t)
    h_t = o * tanh_c_t

    cache = (x_t, h_prev, c_prev, i, f, o, g, c_t, tanh_c_t, params, prefix)
    return h_t, c_t, cache


def lstm_cell_backward(d_h_t, d_c_t, cache):
    """Returns (d_x_t, d_h_prev, d_c_prev, grads_dict) for this timestep.
    d_c_t is the gradient on c_t flowing back from the *next* timestep only
    (c_t has no other consumer); d_h_t is the gradient on h_t from wherever
    h_t was used (attention query, output layer, next timestep's h_prev)."""
    x_t, h_prev, c_prev, i, f, o, g, c_t, tanh_c_t, params, prefix = cache

    # h_t = o * tanh(c_t)
    d_o = d_h_t * tanh_c_t
    d_c_t = d_c_t + d_h_t * o * (1.0 - tanh_c_t * tanh_c_t)

    # c_t = f * c_prev + i * g
    d_f = d_c_t * c_prev
    d_c_prev = d_c_t * f
    d_i = d_c_t * g
    d_g = d_c_t * i

    d_i_pre = d_i * i * (1.0 - i)
    d_f_pre = d_f * f * (1.0 - f)
    d_o_pre = d_o * o * (1.0 - o)
    d_g_pre = d_g * (1.0 - g * g)

    grads = {}
    d_x = np.zeros_like(x_t)
    d_h_prev = np.zeros_like(h_prev)
    for gate, d_pre in zip("ifog", (d_i_pre, d_f_pre, d_o_pre, d_g_pre)):
        Wx = params[f"{prefix}_Wx{gate}"]
        Wh = params[f"{prefix}_Wh{gate}"]
        grads[f"{prefix}_Wx{gate}"] = x_t.T @ d_pre
        grads[f"{prefix}_Wh{gate}"] = h_prev.T @ d_pre
        grads[f"{prefix}_b{gate}"] = d_pre.sum(axis=0)
        d_x += d_pre @ Wx.T
        d_h_prev += d_pre @ Wh.T

    return d_x, d_h_prev, d_c_prev, grads


# ----------------------------------------------------------------------
# Bahdanau (additive) attention.
#   e_i = v . tanh(Wa @ s_prev + Ua @ h_i)      for each encoder position i
#   alpha = softmax(e) masked over padded encoder positions
#   context = sum_i alpha_i * h_i
# To keep it fast, Ua @ H_enc is precomputed once per batch (it doesn't
# depend on the decoder step), and Wa @ s_prev is recomputed every step.
# ----------------------------------------------------------------------
def attention_params(rng, hidden_size, prefix="attn"):
    H = hidden_size
    return {
        f"{prefix}_Wa": init_matrix(rng, (H, H)),
        f"{prefix}_Ua": init_matrix(rng, (H, H)),
        f"{prefix}_v": init_matrix(rng, (H,)),
    }


def precompute_encoder_proj(H_enc, params, prefix="attn"):
    """H_enc: (B, T_enc, H) -> (B, T_enc, H) projected by Ua (reused every decoder step)."""
    Ua = params[f"{prefix}_Ua"]
    return H_enc @ Ua  # (B, T_enc, H)


def attention_forward(s_prev, H_enc, U_H_enc, enc_mask, params, prefix="attn"):
    """
    s_prev:  (B, H)         previous decoder hidden state
    H_enc:   (B, T_enc, H)  all encoder hidden states
    U_H_enc: (B, T_enc, H)  Ua @ H_enc, precomputed once per batch
    enc_mask:(B, T_enc)     1.0 for real tokens, 0.0 for padding
    Returns: context (B, H), alpha (B, T_enc), cache
    """
    Wa = params[f"{prefix}_Wa"]
    v = params[f"{prefix}_v"]

    Wa_s = s_prev @ Wa  # (B, H)
    scores_pre = np.tanh(Wa_s[:, None, :] + U_H_enc)  # (B, T_enc, H)
    e = scores_pre @ v  # (B, T_enc)

    e_masked = np.where(enc_mask > 0, e, -1e9)
    e_shift = e_masked - e_masked.max(axis=1, keepdims=True)
    exp_e = np.exp(e_shift) * enc_mask
    alpha = exp_e / (exp_e.sum(axis=1, keepdims=True) + 1e-12)  # (B, T_enc)

    context = np.sum(alpha[:, :, None] * H_enc, axis=1)  # (B, H)

    cache = (s_prev, H_enc, U_H_enc, enc_mask, Wa_s, scores_pre, alpha, params, prefix)
    return context, alpha, cache


def attention_backward(d_context, d_alpha_extra, cache):
    """
    d_context:      (B, H) gradient flowing into the context vector
    d_alpha_extra:  (B, T_enc) or None - extra gradient directly on alpha
                    (unused by the decoder but kept for generality/tests)
    Returns (d_s_prev, d_H_enc, grads_dict)
    """
    s_prev, H_enc, U_H_enc, enc_mask, Wa_s, scores_pre, alpha, params, prefix = cache
    Wa = params[f"{prefix}_Wa"]
    v = params[f"{prefix}_v"]
    B, T_enc, H = H_enc.shape

    # context = sum_i alpha_i * h_i
    d_alpha = np.sum(d_context[:, None, :] * H_enc, axis=2)  # (B, T_enc)
    d_H_enc = alpha[:, :, None] * d_context[:, None, :]  # (B, T_enc, H) direct path
    if d_alpha_extra is not None:
        d_alpha = d_alpha + d_alpha_extra

    # alpha = softmax(e) over axis=1 (masked)
    d_e = alpha * (d_alpha - np.sum(d_alpha * alpha, axis=1, keepdims=True))
    d_e = d_e * enc_mask

    # e = scores_pre @ v
    d_v = np.sum((d_e[:, :, None] * scores_pre).reshape(B * T_enc, H), axis=0)
    d_scores_pre = d_e[:, :, None] * v[None, None, :]  # (B, T_enc, H)

    # scores_pre = tanh(Wa_s[:,None,:] + U_H_enc)
    d_pre = d_scores_pre * (1.0 - scores_pre * scores_pre)  # (B, T_enc, H)
    d_Wa_s = d_pre.sum(axis=1)  # (B, H)
    d_U_H_enc = d_pre  # (B, T_enc, H)

    # Wa_s = s_prev @ Wa
    d_Wa = s_prev.T @ d_Wa_s
    d_s_prev = d_Wa_s @ Wa.T

    # U_H_enc = H_enc @ Ua
    Ua = params[f"{prefix}_Ua"]
    d_Ua = H_enc.reshape(B * T_enc, H).T @ d_U_H_enc.reshape(B * T_enc, H)
    d_H_enc += d_U_H_enc @ Ua.T

    grads = {f"{prefix}_Wa": d_Wa, f"{prefix}_Ua": d_Ua, f"{prefix}_v": d_v}
    return d_s_prev, d_H_enc, grads


# ----------------------------------------------------------------------
# Dense (linear) layer: y = x @ W + b
# ----------------------------------------------------------------------
def dense_forward(x, W, b):
    y = x @ W + b
    return y, (x, W)


def dense_backward(d_y, cache):
    x, W = cache
    d_x = d_y @ W.T
    d_W = x.T @ d_y
    d_b = d_y.sum(axis=0)
    return d_x, d_W, d_b


# ----------------------------------------------------------------------
# Softmax + masked cross-entropy loss over the vocabulary, per timestep.
# ----------------------------------------------------------------------
def pointer_ce_loss(logits, alpha, enc_ids, p_gen, targets, mask):
    """Cross-entropy for a pointer-generator decoder.

    The plain decoder can only *generate* a word by picking it out of an
    8000-way softmax, which it can never do for a name it saw once or twice.
    A pointer lets it instead *copy* a word straight out of the input, using
    the attention weights it already computes as a distribution over input
    positions:

        P_vocab = softmax(logits)                      # generate
        P_copy[w] = sum of alpha_i over positions i where input_i == w
        P_final  = p_gen * P_vocab + (1 - p_gen) * P_copy

    p_gen in [0,1] is predicted per step, so the model learns *when* to copy.
    Loss is -log P_final[target].

    Returns (loss_sum, num_real, d_logits, d_alpha, d_p_gen); d_alpha feeds
    straight into attention_backward's d_alpha_extra argument.
    """
    B, V = logits.shape

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    P_vocab = exp / exp.sum(axis=1, keepdims=True)

    # Scatter attention mass onto the vocabulary entries it points at.
    # add.at accumulates, so a word appearing twice in the input sums both.
    P_copy = np.zeros_like(P_vocab)
    np.add.at(P_copy, (np.arange(B)[:, None], enc_ids), alpha)

    P_final = p_gen * P_vocab + (1.0 - p_gen) * P_copy

    rows = np.arange(B)
    p_target = P_final[rows, targets] + 1e-9
    loss_sum = -np.sum(np.log(p_target) * mask)
    num_real = np.sum(mask)

    # dL/dP_final is nonzero only at the target column.
    g = -(mask / p_target)                                    # (B,)

    d_p_gen = (g * (P_vocab[rows, targets] - P_copy[rows, targets]))[:, None]

    d_P_vocab = np.zeros_like(P_vocab)
    d_P_vocab[rows, targets] = g * p_gen[:, 0]
    d_logits = P_vocab * (d_P_vocab - np.sum(d_P_vocab * P_vocab, axis=1, keepdims=True))

    # alpha_i contributed to P_copy[enc_ids_i], so it only gets gradient where
    # the input token at that position IS the target token.
    d_copy_at_target = g * (1.0 - p_gen[:, 0])                # (B,)
    d_alpha = np.where(enc_ids == targets[:, None],
                       d_copy_at_target[:, None], 0.0)

    return loss_sum, num_real, d_logits, d_alpha, d_p_gen


def pointer_final_probs(logits, alpha, enc_ids, p_gen):
    """Same mixture as above, for inference (no loss/gradients)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    P_vocab = exp / exp.sum(axis=1, keepdims=True)
    P_copy = np.zeros_like(P_vocab)
    np.add.at(P_copy, (np.arange(logits.shape[0])[:, None], enc_ids), alpha)
    return p_gen * P_vocab + (1.0 - p_gen) * P_copy


def softmax_ce_loss(logits, targets, mask):
    """
    logits:  (B, V)
    targets: (B,) int ids
    mask:    (B,) 1.0 for real tokens, 0.0 for padding
    Returns scalar-ish (loss_sum, num_real_tokens, d_logits (B, V))
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    B = logits.shape[0]
    correct_logprobs = -np.log(probs[np.arange(B), targets] + 1e-12)
    loss_sum = np.sum(correct_logprobs * mask)
    num_real = np.sum(mask)

    d_logits = probs.copy()
    d_logits[np.arange(B), targets] -= 1.0
    d_logits *= mask[:, None]

    return loss_sum, num_real, d_logits
