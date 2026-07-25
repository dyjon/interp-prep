"""Metalinguistic judgment vs string probability on GPT-2 small.

Hu et al. report that language models' direct metalinguistic judgments diverge from
what their string probabilities imply: the model behaves as though it knows a
grammatical contrast but cannot report it when asked.

This is a small check of that, and mainly a check of whether the comparison is even
identifiable at this model scale. Two ways of asking the same question:

  implicit      compare log P(grammatical sentence) against log P(ungrammatical one)
  metalinguistic  prompt "Is this sentence grammatical?" and compare P(" Yes") / P(" No")

If the metalinguistic side is at chance, that is ambiguous. It could mean the model
cannot introspect, or it could mean the model cannot follow the instruction at all.
The diagnostic for the second reading is whether the Yes/No answer varies across items
at all, or is effectively constant.

Run: python src/metalinguistic_gap.py
"""
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

# (grammatical, ungrammatical, phenomenon)
MINIMAL_PAIRS = [
    ("The keys are on the table.", "The keys is on the table.", "subject-verb agreement"),
    ("The dogs bark loudly.", "The dogs barks loudly.", "subject-verb agreement"),
    ("My sister writes letters.", "My sister write letters.", "subject-verb agreement"),
    ("The children were playing.", "The children was playing.", "subject-verb agreement"),
    ("These books are heavy.", "This books are heavy.", "determiner-noun agreement"),
    ("That car is expensive.", "Those car is expensive.", "determiner-noun agreement"),
    ("I bought three apples.", "I bought three apple.", "determiner-noun agreement"),
    ("She saw many birds.", "She saw many bird.", "determiner-noun agreement"),
    ("The boy hurt himself.", "The boy hurt herself.", "anaphor agreement"),
    ("The woman blamed herself.", "The woman blamed himself.", "anaphor agreement"),
    ("The men defended themselves.", "The men defended himself.", "anaphor agreement"),
    ("John cut himself shaving.", "John cut themselves shaving.", "anaphor agreement"),
    ("The teacher has arrived.", "The teacher have arrived.", "auxiliary agreement"),
    ("They have finished already.", "They has finished already.", "auxiliary agreement"),
    ("He does not understand.", "He do not understand.", "auxiliary agreement"),
    ("We were waiting outside.", "We was waiting outside.", "auxiliary agreement"),
    ("The cat slept on the mat.", "The cat slept on mat the.", "word order"),
    ("She quickly opened the door.", "She opened quickly the door.", "word order"),
    ("I know what he said.", "I know what said he.", "word order"),
    ("The letter was sent yesterday.", "The letter was send yesterday.", "verb form"),
    ("He has taken the bus.", "He has took the bus.", "verb form"),
    ("They are running fast.", "They are ran fast.", "verb form"),
    ("She wants to leave now.", "She wants to leaving now.", "verb form"),
    ("Nobody has seen anything.", "Nobody has saw anything.", "verb form"),
]

PROMPT = 'Question: Is the following sentence grammatical?\nSentence: "{sentence}"\nAnswer:'


def sentence_logprob(model, sentence):
    """Mean log probability per token, so length doesn't dominate the comparison."""
    tokens = model.to_tokens(sentence)
    logits = model(tokens)
    logprobs = logits.log_softmax(dim=-1)
    # predict token t+1 from position t
    targets = tokens[0, 1:]
    scored = logprobs[0, :-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    return scored.mean().item()


def yes_probability(model, sentence, yes_id, no_id):
    """P(Yes) / (P(Yes) + P(No)) after the metalinguistic prompt."""
    tokens = model.to_tokens(PROMPT.format(sentence=sentence))
    logits = model(tokens)[0, -1]
    probs = logits.softmax(dim=-1)
    y, n = probs[yes_id].item(), probs[no_id].item()
    return y / (y + n), y + n


def main():
    model = HookedTransformer.from_pretrained("gpt2")
    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")

    implicit_correct = 0
    meta_correct = 0
    agree = 0
    yes_fractions = []
    yesno_masses = []

    print(f"{'phenomenon':<28} {'implicit':>9} {'meta':>9}")
    print("-" * 50)

    for good, bad, phenomenon in MINIMAL_PAIRS:
        lp_good = sentence_logprob(model, good)
        lp_bad = sentence_logprob(model, bad)
        implicit_ok = lp_good > lp_bad

        yes_good, mass_good = yes_probability(model, good, yes_id, no_id)
        yes_bad, mass_bad = yes_probability(model, bad, yes_id, no_id)
        # the model "gets it right" metalinguistically if it is more willing to say Yes
        # about the grammatical sentence than the ungrammatical one
        meta_ok = yes_good > yes_bad

        implicit_correct += implicit_ok
        meta_correct += meta_ok
        agree += implicit_ok == meta_ok
        yes_fractions += [yes_good, yes_bad]
        yesno_masses += [mass_good, mass_bad]

        print(f"{phenomenon:<28} {'ok' if implicit_ok else '--':>9} {'ok' if meta_ok else '--':>9}")

    n = len(MINIMAL_PAIRS)
    yf = torch.tensor(yes_fractions)
    masses = torch.tensor(yesno_masses)

    print("\n" + "=" * 50)
    print(f"implicit (log prob of the sentence):  {implicit_correct}/{n} = {implicit_correct/n:.0%}")
    print(f"metalinguistic (asking the model):    {meta_correct}/{n} = {meta_correct/n:.0%}")
    print(f"the two methods agree on:             {agree}/{n} = {agree/n:.0%}")

    print("\nIs the metalinguistic channel doing anything at all?")
    print(f"  P(Yes | Yes or No), mean {yf.mean():.3f}, sd {yf.std():.3f}, "
          f"range [{yf.min():.3f}, {yf.max():.3f}]")
    print(f"  probability mass on Yes/No combined, mean {masses.mean():.4f}")
    print("  A near-constant P(Yes) or a tiny Yes/No mass means the model is not really")
    print("  answering the question, so a low metalinguistic score does not isolate")
    print("  a failure to introspect.")


if __name__ == "__main__":
    main()
