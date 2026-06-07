"""Induction-head detection on GPT-2 small (TransformerLens).

Repeated-random-token test: a head is an induction head if, in the second copy
of a repeated random sequence, it attends to the token that *followed* the
matching token in the first copy. We score every (layer, head) by the mean
attention on that diagonal stripe and plot a heatmap.

Run: python src/induction_heads.py
"""
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)


def main():
    model = HookedTransformer.from_pretrained("gpt2")  # 12 layers x 12 heads
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    # [BOS] + rand(seq_len) + same rand(seq_len)
    seq_len, batch = 50, 8
    rand = torch.randint(0, model.cfg.d_vocab, (batch, seq_len))
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id)
    tokens = torch.cat([bos, rand, rand], dim=1)  # [batch, 1 + 2*seq_len]

    _, cache = model.run_with_cache(tokens)

    # induction stripe: a dest position attends to src = dest - seq_len
    induction = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, head, dest, src]
        stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=-(seq_len - 1))
        induction[layer] = stripe.mean(dim=(0, -1))

    # report top heads
    vals, idx = torch.topk(induction.flatten(), 5)
    print("Top induction heads (layer, head, score):")
    for v, i in zip(vals.tolist(), idx.tolist()):
        print(f"  L{i // n_heads}H{i % n_heads}: {v:.3f}")

    # heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(induction.numpy(), cmap="viridis", aspect="auto")
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.colorbar(label="induction score")
    plt.title("GPT-2 small — induction heads")
    plt.tight_layout()
    plt.savefig("report_induction.png", dpi=150, bbox_inches="tight")
    print("Saved report_induction.png")


if __name__ == "__main__":
    main()
