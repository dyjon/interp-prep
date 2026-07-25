"""IOI activation patching on GPT-2 small (TransformerLens).

Reproduces the core localisation result from Wang et al. 2022, "Interpretability
in the Wild": the indirect-object-identification circuit.

Setup. Each prompt names two people and then has one of them give something away;
the model should complete with the *other* name (the indirect object).

    clean:     "When Mary and John went to the store, John gave a drink to" -> " Mary"
    corrupted: "When Mary and John went to the store, Mary gave a drink to" -> " John"

The two differ only at the subject token, so they tokenise to the same length and
the correct answer flips. We score with the logit difference between the two names,
which is positive on clean prompts and negative on corrupted ones.

Patching. We run the corrupted prompt but splice in a clean activation at one
location, then measure how much of the clean logit difference comes back:

    0.0 = the patched site carries none of the signal
    1.0 = that site alone restores the clean behaviour

Sweeping the location over (layer, position) localises the computation in the
residual stream; sweeping over (layer, head) localises it to attention heads.

Run: python src/ioi_patching.py
"""
import functools

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer
from transformer_lens import utilities as utils

torch.set_grad_enabled(False)

NAME_PAIRS = [
    ("Mary", "John"),
    ("Alice", "Bob"),
    ("Tom", "James"),
    ("Anna", "Peter"),
    ("Sarah", "David"),
    ("Emma", "Michael"),
    ("Laura", "Kevin"),
    ("Julia", "Robert"),
]


def build_dataset(model):
    """Clean/corrupted prompt pairs that differ only at the subject token."""
    clean_prompts, corrupted_prompts, answers = [], [], []
    for a, b in NAME_PAIRS:
        # both names must be single tokens, or the prompts won't align
        if len(model.to_tokens(f" {a}", prepend_bos=False)[0]) != 1:
            continue
        if len(model.to_tokens(f" {b}", prepend_bos=False)[0]) != 1:
            continue
        clean_prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to")
        corrupted_prompts.append(f"When {a} and {b} went to the store, {a} gave a drink to")
        answers.append((f" {a}", f" {b}"))

    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupted_prompts)
    assert clean_tokens.shape == corrupted_tokens.shape, "prompt pairs must align"

    answer_tokens = torch.tensor(
        [[model.to_single_token(io), model.to_single_token(s)] for io, s in answers]
    )
    return clean_tokens, corrupted_tokens, answer_tokens


def logit_diff(logits, answer_tokens):
    """logit(indirect object) - logit(subject), averaged over prompts."""
    final = logits[:, -1, :]
    io = final.gather(1, answer_tokens[:, 0:1])
    s = final.gather(1, answer_tokens[:, 1:2])
    return (io - s).squeeze(1).mean()


def recovery(logits, answer_tokens, clean_ld, corrupted_ld):
    """Fraction of the clean logit difference restored by the patch."""
    ld = logit_diff(logits, answer_tokens)
    return ((ld - corrupted_ld) / (clean_ld - corrupted_ld)).item()


def patch_resid_at_pos(resid, hook, pos, clean_cache):
    resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return resid


def patch_head_z(z, hook, head, clean_cache):
    z[:, :, head, :] = clean_cache[hook.name][:, :, head, :]
    return z


def sweep_residual(model, corrupted_tokens, clean_cache, answer_tokens, clean_ld, corrupted_ld):
    """Patch the residual stream at every (layer, position)."""
    n_layers, n_pos = model.cfg.n_layers, corrupted_tokens.shape[1]
    scores = torch.zeros(n_layers, n_pos)
    for layer in range(n_layers):
        name = utils.get_act_name("resid_pre", layer)
        for pos in range(n_pos):
            hook_fn = functools.partial(patch_resid_at_pos, pos=pos, clean_cache=clean_cache)
            logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook_fn)])
            scores[layer, pos] = recovery(logits, answer_tokens, clean_ld, corrupted_ld)
    return scores


def sweep_heads(model, corrupted_tokens, clean_cache, answer_tokens, clean_ld, corrupted_ld):
    """Patch each attention head's output over all positions."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        name = utils.get_act_name("z", layer)
        for head in range(n_heads):
            hook_fn = functools.partial(patch_head_z, head=head, clean_cache=clean_cache)
            logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook_fn)])
            scores[layer, head] = recovery(logits, answer_tokens, clean_ld, corrupted_ld)
    return scores


def plot(scores, xlabel, xticklabels, title, path):
    plt.figure(figsize=(max(6, scores.shape[1] * 0.6), 5))
    limit = scores.abs().max().item()
    plt.imshow(scores.numpy(), cmap="RdBu", aspect="auto", vmin=-limit, vmax=limit)
    plt.xlabel(xlabel)
    plt.ylabel("layer")
    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), xticklabels, rotation=90, fontsize=7)
    plt.colorbar(label="logit difference recovered")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    model = HookedTransformer.from_pretrained("gpt2")
    clean_tokens, corrupted_tokens, answer_tokens = build_dataset(model)
    print(f"{clean_tokens.shape[0]} prompt pairs, {clean_tokens.shape[1]} tokens each")

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    clean_ld = logit_diff(clean_logits, answer_tokens)
    corrupted_ld = logit_diff(corrupted_logits, answer_tokens)
    print(f"clean logit diff:     {clean_ld:+.3f}  (should be positive)")
    print(f"corrupted logit diff: {corrupted_ld:+.3f}  (should be negative)")

    if clean_ld <= 0:
        print("WARNING: model does not solve the clean task. Check the prompts.")

    resid_scores = sweep_residual(
        model, corrupted_tokens, clean_cache, answer_tokens, clean_ld, corrupted_ld
    )
    token_labels = model.to_str_tokens(clean_tokens[0])
    plot(
        resid_scores,
        "position",
        token_labels,
        "IOI: residual stream patching",
        "report_ioi_resid.png",
    )

    head_scores = sweep_heads(
        model, corrupted_tokens, clean_cache, answer_tokens, clean_ld, corrupted_ld
    )
    plot(head_scores, "head", None, "IOI: attention head patching", "report_ioi_heads.png")

    vals, idx = torch.topk(head_scores.flatten(), 8)
    print("\nTop heads by logit difference recovered:")
    for v, i in zip(vals.tolist(), idx.tolist()):
        print(f"  L{i // model.cfg.n_heads}H{i % model.cfg.n_heads}: {v:+.3f}")


if __name__ == "__main__":
    main()
