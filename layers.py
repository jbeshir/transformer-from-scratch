import torch
import torch as t
import torch.nn as nn

from config import Config

eps = 1e-10


# Layer Normalisation layer.
# Per https://arxiv.org/pdf/1607.06450.pdf
# Simple implementation of https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
# Normalises mean of activations to each element in the last dimension to 0 with variance of 1.
# Applies a learned multiplication and bias to each element,
# to also normalise mean of activations over the whole dataset.
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        (variance, mean) = t.var_mean(residual, keepdim=True, dim=-1, unbiased=False)
        std_dev = t.sqrt(variance + self.cfg.layer_norm_eps)
        normalised = t.subtract(residual, mean) / std_dev
        return normalised * self.w + self.b


# Embed layer.
# Returns an initial residual stream vector corresponding to each input token.
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))

        # Start out with initial parameters drawn from the normal distribution.
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        return self.W_E[tokens]


# Positional Embedding layer.
# Returns an initial residual stream vector corresponding to each token index in the input.
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))

        # Start out with initial parameters drawn from the normal distribution.
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # Looking only at the position embeddings up to the context length of this batch,
        # broadcast our learned position embedding out over every input in the batch.
        # We are completely ignoring the actual input values.
        batch, seq_len = tokens.shape
        return self.W_pos[:seq_len].expand((batch, seq_len, self.cfg.d_model))


# Attention layer.
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_key = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_query = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_values = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_output = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Bias_key = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Bias_query = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Bias_values = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Bias_output = nn.Parameter(t.zeros(cfg.d_model))

        # Start out with initial weights drawn from the normal distribution.
        nn.init.normal_(self.W_key, std=self.cfg.init_range)
        nn.init.normal_(self.W_query, std=self.cfg.init_range)
        nn.init.normal_(self.W_values, std=self.cfg.init_range)
        nn.init.normal_(self.W_output, std=self.cfg.init_range)

    def forward(self, residual):
        batch, seq_len, _ = residual.shape
        keys = t.add(
            t.einsum('bse,neh->bsnh', residual, self.W_key),
            self.Bias_key.expand(batch, seq_len, self.cfg.n_heads, self.cfg.d_head),
        )

        queries = t.add(
            t.einsum('bse,neh->bsnh', residual, self.W_query),
            self.Bias_query.expand(batch, seq_len, self.cfg.n_heads, self.cfg.d_head),
        )

        values = t.add(
            t.einsum('bse,neh->bsnh', residual, self.W_values),
            self.Bias_values.expand(batch, seq_len, self.cfg.n_heads, self.cfg.d_head),
        )

        scores = t.einsum('bknh,bqnh->bnqk', keys, queries) / (self.cfg.d_head ** 0.5)

        query_indices = t.arange(seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        key_indices = t.arange(seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        causal_mask = query_indices < key_indices
        scores[causal_mask.expand(batch, self.cfg.n_heads, seq_len, seq_len)] = eps

        probs = t.softmax(scores, 3)

        weighted_values = t.einsum('bnqk,bknh->bqnh', probs, values)

        head_outputs = t.einsum('bqnh,nhe->bqne', weighted_values, self.W_output),

        output = t.add(
            t.einsum('bqne->bqe', head_outputs),
            self.Bias_output.expand(batch, seq_len, self.cfg.d_model),
        )
        return output


# MLP layer.
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_input = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_output = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.Bias_input = nn.Parameter(t.zeros(cfg.d_mlp))
        self.Bias_output = nn.Parameter(t.zeros(cfg.d_model))

        # Start out with initial weights drawn from the normal distribution.
        nn.init.normal_(self.W_input, std=self.cfg.init_range)
        nn.init.normal_(self.W_output, std=self.cfg.init_range)

    def forward(self, residual):
        input = t.einsum('bse,em->bsm', residual, self.W_input) + self.Bias_input

        activation = nn.GELU()(input)

        output = t.einsum('bsm,me->bse', activation, self.W_output) + self.Bias_output
        return output

# Transformer Block.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, residual):
        mid = self.attn(self.ln1(residual)) + residual
        output = self.mlp(self.ln2(mid)) + mid
        return output

# Unembedding layer.
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        self.Bias_U = nn.Parameter(t.zeros(cfg.d_vocab))

        # Start out with initial parameters drawn from the normal distribution.
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

    def forward(self, residual):
        logits = t.einsum('bse,eu->bsu', residual, self.W_U) + self.Bias_U
        return logits

# Full Transformer, all layers.
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        embeddings = self.embed(tokens)
        pos_embeddings = self.pos_embed(tokens)

        residual = embeddings + pos_embeddings
        for block in self.blocks:
            residual = block.forward(residual)

        output = self.ln_final(residual)
        return self.unembed(output)

if __name__ == "__main__":
    def rand_float_test(cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = t.randn(shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        print("Output shape:", output.shape, "\n")


    def rand_int_test(cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = t.randint(100, 1000, shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        print("Output shape:", output.shape, "\n")


    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print("LayerNorm")
    rand_float_test(LayerNorm, [2, 4, 768])
    print("Embed")
    rand_int_test(Embed, [2, 4])
    print("PosEmbed")
    rand_int_test(PosEmbed, [2, 4])
    print("Attention")
    rand_float_test(Attention, [2, 4, 768])
    print("MLP")
    rand_float_test(MLP, [2, 4, 768])
    print("TransformerBlock")
    rand_float_test(TransformerBlock, [2, 4, 768])
    print("Unembed")
    rand_float_test(Unembed, [2, 4, 768])
    print("Transformer")
    rand_int_test(Transformer, [2, 4])
