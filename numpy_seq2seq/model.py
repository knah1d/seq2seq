"""
Wires the layers in layers.py into a full Seq2Seq-with-Bahdanau-attention
model: an LSTM encoder, an additive-attention mechanism, and an LSTM decoder
with an output projection to vocabulary logits.

Everything is plain NumPy. `forward` runs teacher-forced training and
returns a loss plus a cache; `backward` walks that cache back through time
(manual BPTT) and returns a gradient for every parameter. `greedy_decode`
runs the same encoder/attention/decoder but feeds the model's own previous
prediction back in, for inference.

Unlike a GRU, an LSTM carries two states through time at every step: the
hidden state `h` (what attention queries and what feeds the output layer)
and the cell state `c` (the longer-term memory gated by input/forget/output
gates). Both have to be threaded through the encoder/decoder loops and both
need a gradient path in backward().
"""
import numpy as np

from layers import (
    embedding_forward, embedding_backward,
    lstm_cell_params, lstm_cell_forward, lstm_cell_backward,
    attention_params, precompute_encoder_proj, attention_forward, attention_backward,
    dense_forward, dense_backward,
    softmax_ce_loss, init_matrix,
)

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Seq2SeqAttention:
    def __init__(self, vocab_size, emb_dim=96, hidden_size=128, seed=0):
        self.V = vocab_size
        self.E = emb_dim
        self.H = hidden_size
        rng = np.random.RandomState(seed)

        params = {}
        params["W_emb"] = init_matrix(rng, (vocab_size, emb_dim), scale=0.1)
        params.update(lstm_cell_params(rng, emb_dim, hidden_size, prefix="enc"))
        params.update(lstm_cell_params(rng, emb_dim + hidden_size, hidden_size, prefix="dec"))
        params.update(attention_params(rng, hidden_size, prefix="attn"))
        params["W_bridge_h"] = init_matrix(rng, (hidden_size, hidden_size))
        params["b_bridge_h"] = np.zeros(hidden_size, dtype=np.float64)
        params["W_bridge_c"] = init_matrix(rng, (hidden_size, hidden_size))
        params["b_bridge_c"] = np.zeros(hidden_size, dtype=np.float64)
        params["W_out"] = init_matrix(rng, (hidden_size * 2, vocab_size))
        params["b_out"] = np.zeros(vocab_size, dtype=np.float64)
        self.params = params

    # ------------------------------------------------------------------
    # Encoder: run the LSTM left-to-right over the article tokens.
    # Padded positions carry the previous (h, c) forward unchanged (masked
    # update) so the final timestep always holds the last *real* state,
    # which seeds the decoder via the bridge layers.
    # ------------------------------------------------------------------
    def _encode(self, enc_ids):
        B, T_enc = enc_ids.shape
        H = self.H
        enc_mask = (enc_ids != PAD_ID).astype(np.float64)  # (B, T_enc)

        X_enc, emb_cache = embedding_forward(enc_ids, self.params["W_emb"])  # (B,T_enc,E)

        h_prev = np.zeros((B, H), dtype=np.float64)
        c_prev = np.zeros((B, H), dtype=np.float64)
        H_enc = np.zeros((B, T_enc, H), dtype=np.float64)
        step_caches = []
        for t in range(T_enc):
            mask_t = enc_mask[:, t]
            h_computed, c_computed, lstm_cache = lstm_cell_forward(
                X_enc[:, t, :], h_prev, c_prev, self.params, "enc"
            )
            h_t = mask_t[:, None] * h_computed + (1.0 - mask_t[:, None]) * h_prev
            c_t = mask_t[:, None] * c_computed + (1.0 - mask_t[:, None]) * c_prev
            H_enc[:, t, :] = h_t
            step_caches.append((mask_t, h_prev, c_prev, lstm_cache))
            h_prev, c_prev = h_t, c_t

        h_final, c_final = h_prev, c_prev
        s0_pre = h_final @ self.params["W_bridge_h"] + self.params["b_bridge_h"]
        s0 = np.tanh(s0_pre)
        c0_pre = c_final @ self.params["W_bridge_c"] + self.params["b_bridge_c"]
        c0 = np.tanh(c0_pre)

        U_H_enc = precompute_encoder_proj(H_enc, self.params, "attn")

        cache = {
            "enc_ids": enc_ids, "emb_cache": emb_cache, "step_caches": step_caches,
            "H_enc": H_enc, "h_final": h_final, "c_final": c_final,
            "s0_pre": s0_pre, "c0_pre": c0_pre,
            "enc_mask": enc_mask, "T_enc": T_enc, "B": B,
        }
        return H_enc, U_H_enc, enc_mask, s0, c0, cache

    def _encode_backward(self, d_H_enc_extra, d_s0, d_c0, cache):
        """d_H_enc_extra: (B,T_enc,H) gradient on every encoder hidden state
        (accumulated from attention at every decoder step).
        d_s0, d_c0: (B,H) gradients on the bridge outputs."""
        params = self.params
        H, T_enc, B = self.H, cache["T_enc"], cache["B"]

        d_s0_pre = d_s0 * (1.0 - np.tanh(cache["s0_pre"]) ** 2)
        d_W_bridge_h = cache["h_final"].T @ d_s0_pre
        d_b_bridge_h = d_s0_pre.sum(axis=0)
        d_h_final = d_s0_pre @ params["W_bridge_h"].T

        d_c0_pre = d_c0 * (1.0 - np.tanh(cache["c0_pre"]) ** 2)
        d_W_bridge_c = cache["c_final"].T @ d_c0_pre
        d_b_bridge_c = d_c0_pre.sum(axis=0)
        d_c_final = d_c0_pre @ params["W_bridge_c"].T

        d_H_enc = d_H_enc_extra.copy()
        d_H_enc[:, -1, :] += d_h_final

        grads = {
            "W_bridge_h": d_W_bridge_h, "b_bridge_h": d_b_bridge_h,
            "W_bridge_c": d_W_bridge_c, "b_bridge_c": d_b_bridge_c,
        }
        d_X_enc = np.zeros((B, T_enc, self.E), dtype=np.float64)

        d_h_future = np.zeros((B, H), dtype=np.float64)
        d_c_future = d_c_final
        for t in reversed(range(T_enc)):
            mask_t, h_prev, c_prev, lstm_cache = cache["step_caches"][t]
            d_h_t = d_H_enc[:, t, :] + d_h_future
            d_c_t = d_c_future

            d_h_computed = d_h_t * mask_t[:, None]
            d_h_prev_direct = d_h_t * (1.0 - mask_t[:, None])
            d_c_computed = d_c_t * mask_t[:, None]
            d_c_prev_direct = d_c_t * (1.0 - mask_t[:, None])

            d_x_t, d_h_prev_cell, d_c_prev_cell, lstm_grads = lstm_cell_backward(
                d_h_computed, d_c_computed, lstm_cache
            )
            d_X_enc[:, t, :] = d_x_t
            d_h_future = d_h_prev_direct + d_h_prev_cell
            d_c_future = d_c_prev_direct + d_c_prev_cell

            for k, v in lstm_grads.items():
                grads[k] = grads.get(k, 0.0) + v

        d_W_emb = embedding_backward(d_X_enc, cache["emb_cache"])
        grads["W_emb"] = grads.get("W_emb", 0.0) + d_W_emb
        return grads

    # ------------------------------------------------------------------
    # Teacher-forced decoding for training.
    # ------------------------------------------------------------------
    def forward(self, enc_ids, dec_ids):
        params = self.params
        B, T_dec = dec_ids.shape
        H_enc, U_H_enc, enc_mask, s0, c0, enc_cache = self._encode(enc_ids)

        dec_input_ids = dec_ids[:, :-1]
        dec_target_ids = dec_ids[:, 1:]
        dec_mask = (dec_target_ids != PAD_ID).astype(np.float64)
        T_step = dec_input_ids.shape[1]

        X_dec, dec_emb_cache = embedding_forward(dec_input_ids, params["W_emb"])

        h_prev, c_prev = s0, c0
        total_loss = 0.0
        total_real = 0.0
        step_caches = []
        for t in range(T_step):
            context_t, alpha_t, attn_cache = attention_forward(
                h_prev, H_enc, U_H_enc, enc_mask, params, "attn"
            )
            lstm_input = np.concatenate([X_dec[:, t, :], context_t], axis=1)
            h_t, c_t, lstm_cache = lstm_cell_forward(lstm_input, h_prev, c_prev, params, "dec")

            out_input = np.concatenate([h_t, context_t], axis=1)
            logits_t, dense_cache = dense_forward(out_input, params["W_out"], params["b_out"])

            loss_t, real_t, d_logits_t = softmax_ce_loss(
                logits_t, dec_target_ids[:, t], dec_mask[:, t]
            )
            total_loss += loss_t
            total_real += real_t

            step_caches.append((attn_cache, lstm_cache, dense_cache, d_logits_t))
            h_prev, c_prev = h_t, c_t

        cache = {
            "enc_cache": enc_cache, "H_enc": H_enc, "T_enc": H_enc.shape[1],
            "dec_emb_cache": dec_emb_cache, "step_caches": step_caches,
            "B": B, "T_step": T_step, "s0": s0, "c0": c0, "total_real": total_real,
        }
        avg_loss = total_loss / max(total_real, 1.0)
        return avg_loss, total_real, cache

    def backward(self, cache):
        params = self.params
        B, T_step, H = cache["B"], cache["T_step"], self.H
        T_enc = cache["T_enc"]

        grads = {}
        d_X_dec = np.zeros((B, T_step, self.E), dtype=np.float64)
        d_H_enc_accum = np.zeros((B, T_enc, H), dtype=np.float64)

        # forward() returns avg_loss = sum(loss_t) / total_real, so every
        # d_logits (a gradient of the *sum*) must be divided by the same
        # total_real to match the loss we actually reported/optimized.
        total_real = max(cache["total_real"], 1.0)

        d_h_next = np.zeros((B, H), dtype=np.float64)
        d_c_next = np.zeros((B, H), dtype=np.float64)
        for t in reversed(range(T_step)):
            attn_cache, lstm_cache, dense_cache, d_logits_t = cache["step_caches"][t]
            d_logits_t = d_logits_t / total_real

            d_out_input, d_W_out, d_b_out = dense_backward(d_logits_t, dense_cache)
            grads["W_out"] = grads.get("W_out", 0.0) + d_W_out
            grads["b_out"] = grads.get("b_out", 0.0) + d_b_out
            d_h_t_out, d_context_out = d_out_input[:, :H], d_out_input[:, H:]

            d_h_t = d_h_t_out + d_h_next
            d_c_t = d_c_next
            d_lstm_input, d_h_prev_cell, d_c_prev_cell, lstm_grads = lstm_cell_backward(
                d_h_t, d_c_t, lstm_cache
            )
            for k, v in lstm_grads.items():
                grads[k] = grads.get(k, 0.0) + v
            d_x_t, d_context_lstm = d_lstm_input[:, : self.E], d_lstm_input[:, self.E:]
            d_X_dec[:, t, :] = d_x_t

            d_context_total = d_context_out + d_context_lstm
            d_h_prev_attn, d_H_enc_t, attn_grads = attention_backward(
                d_context_total, None, attn_cache
            )
            for k, v in attn_grads.items():
                grads[k] = grads.get(k, 0.0) + v
            d_H_enc_accum += d_H_enc_t

            d_h_next = d_h_prev_cell + d_h_prev_attn
            d_c_next = d_c_prev_cell

        d_W_emb_dec = embedding_backward(d_X_dec, cache["dec_emb_cache"])
        grads["W_emb"] = grads.get("W_emb", 0.0) + d_W_emb_dec

        enc_grads = self._encode_backward(d_H_enc_accum, d_h_next, d_c_next, cache["enc_cache"])
        for k, v in enc_grads.items():
            grads[k] = grads.get(k, 0.0) + v

        # Any parameter untouched this batch (shouldn't happen, but keeps
        # the optimizer's dict keys stable) gets a zero gradient.
        for k in self.params:
            if k not in grads:
                grads[k] = np.zeros_like(self.params[k])
        return grads

    # ------------------------------------------------------------------
    # Inference: greedy decoding (argmax at every step, no teacher forcing).
    # ------------------------------------------------------------------
    def greedy_decode(self, enc_ids, max_len=20):
        params = self.params
        H_enc, U_H_enc, enc_mask, s0, c0, _ = self._encode(enc_ids)
        B = enc_ids.shape[0]

        h_prev, c_prev = s0, c0
        cur_ids = np.full((B,), SOS_ID, dtype=np.int32)
        outputs = np.zeros((B, max_len), dtype=np.int32)
        alphas = np.zeros((B, max_len, enc_ids.shape[1]), dtype=np.float64)
        finished = np.zeros((B,), dtype=bool)

        for t in range(max_len):
            x_t = params["W_emb"][cur_ids]
            context_t, alpha_t, _ = attention_forward(h_prev, H_enc, U_H_enc, enc_mask, params, "attn")
            lstm_input = np.concatenate([x_t, context_t], axis=1)
            h_t, c_t, _ = lstm_cell_forward(lstm_input, h_prev, c_prev, params, "dec")
            out_input = np.concatenate([h_t, context_t], axis=1)
            logits_t, _ = dense_forward(out_input, params["W_out"], params["b_out"])

            next_ids = np.argmax(logits_t, axis=1)
            next_ids = np.where(finished, PAD_ID, next_ids)
            outputs[:, t] = next_ids
            alphas[:, t, :] = alpha_t
            finished = finished | (next_ids == EOS_ID)

            cur_ids = next_ids
            h_prev, c_prev = h_t, c_t
            if finished.all():
                break
        return outputs, alphas

    def save(self, path):
        np.savez_compressed(path, **self.params, V=self.V, E=self.E, H=self.H)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["V"]), int(data["E"]), int(data["H"]))
        for k in model.params:
            model.params[k] = data[k]
        return model
