"""Metalinguistic judgment on an instruction-tuned model.

Follow-up to metalinguistic_gap.py. On GPT-2 small the metalinguistic channel was dead:
only ~15% of probability mass landed on "Yes" or "No" and P(Yes) barely moved between
items, so a failure to introspect and a failure to follow the instruction were not
separable.

The obvious fix is a model that can follow an instruction at all. That introduces its own
confound, since instruction tuning changes the model in ways that are not limited to
making it answer questions, but it at least tells us whether the channel opens.

What to look at, in order:
  1. does Yes/No probability mass rise well above 15%
  2. does P(Yes) actually vary across items
  3. only if both hold, is the accuracy gap meaningful

Run: python src/metalinguistic_instruct.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metalinguistic_gap import MINIMAL_PAIRS

torch.set_grad_enabled(False)

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def sentence_logprob(model, tokenizer, sentence):
    ids = tokenizer(sentence, return_tensors="pt").input_ids
    logits = model(ids).logits
    logprobs = logits.log_softmax(dim=-1)
    targets = ids[0, 1:]
    scored = logprobs[0, :-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    return scored.mean().item()


def yes_probability(model, tokenizer, sentence, yes_ids, no_ids):
    chat = [
        {
            "role": "user",
            "content": f'Is the following sentence grammatical? Answer Yes or No.\n"{sentence}"',
        }
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids
    probs = model(ids).logits[0, -1].softmax(dim=-1)
    y = sum(probs[i].item() for i in yes_ids)
    n = sum(probs[i].item() for i in no_ids)
    return y / (y + n), y + n


def variants(tokenizer, word):
    """Token ids for a word with and without a leading space, upper and lower case."""
    out = set()
    for form in (word, word.lower(), f" {word}", f" {word.lower()}"):
        ids = tokenizer(form, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.add(ids[0])
    return sorted(out)


def main():
    print(f"loading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()

    yes_ids = variants(tokenizer, "Yes")
    no_ids = variants(tokenizer, "No")
    print(f"Yes token ids {yes_ids}, No token ids {no_ids}\n")

    implicit_correct = meta_correct = agree = 0
    yes_fractions, masses = [], []

    for good, bad, _ in MINIMAL_PAIRS:
        implicit_ok = sentence_logprob(model, tokenizer, good) > sentence_logprob(
            model, tokenizer, bad
        )
        yes_good, mass_good = yes_probability(model, tokenizer, good, yes_ids, no_ids)
        yes_bad, mass_bad = yes_probability(model, tokenizer, bad, yes_ids, no_ids)
        meta_ok = yes_good > yes_bad

        implicit_correct += implicit_ok
        meta_correct += meta_ok
        agree += implicit_ok == meta_ok
        yes_fractions += [yes_good, yes_bad]
        masses += [mass_good, mass_bad]

    n = len(MINIMAL_PAIRS)
    yf = torch.tensor(yes_fractions)
    mass = torch.tensor(masses)

    print("=" * 56)
    print(f"model: {MODEL}")
    print(f"implicit (sentence log prob):  {implicit_correct}/{n} = {implicit_correct/n:.0%}")
    print(f"metalinguistic (asked):        {meta_correct}/{n} = {meta_correct/n:.0%}")
    print(f"the two agree:                 {agree}/{n} = {agree/n:.0%}")
    print()
    print("Is the metalinguistic channel actually open?")
    print(f"  Yes/No probability mass:  mean {mass.mean():.4f}   (GPT-2 small was 0.1470)")
    print(f"  P(Yes) spread:            sd {yf.std():.3f}, range "
          f"[{yf.min():.3f}, {yf.max():.3f}]   (GPT-2 small sd was 0.080)")
    print()
    print("If the mass is much higher and P(Yes) varies, the channel is open and the")
    print("accuracy gap means something. If not, the confound survives model choice.")


if __name__ == "__main__":
    main()
