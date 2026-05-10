"""
Test-Time Compute Demo: Best-of-N Sampling (Self-Consistency)
=============================================================
Demonstrates how accuracy improves as you sample more candidate
answers and pick by majority vote — the core test-time scaling law.

Paper: arXiv:2501.02497  "A Survey of Test-Time Compute" (2025)
Author: Yashaswini Dinesh | CMPE258 Deep Learning | SJSU | Spring 2026
"""

import random
import json
from collections import Counter

# ── Simple GSM8K-style reasoning questions for demonstration ──────────────────
PROBLEMS = [
    {"q": "Janet has 24 apples. She gives 1/3 to her sister and eats 4. How many left?", "a": "12"},
    {"q": "A train travels 60 mph for 2.5 hours. How many miles?", "a": "150"},
    {"q": "5 workers build a wall in 8 days. How many days for 10 workers?", "a": "4"},
    {"q": "A store marks up cost by 40%. Item costs $35. Selling price?", "a": "49"},
    {"q": "Class has 30 students, 40% girls. How many boys?", "a": "18"},
    {"q": "Rectangle: length 12cm, width 8cm. Perimeter?", "a": "40"},
    {"q": "3 pens cost $1.20 total. Cost of 7 pens?", "a": "2.80"},
    {"q": "Start at 8:45am, work 6h 30min. Finish time?", "a": "3:15pm"},
    {"q": "Bag has 5 red, 3 blue, 2 green marbles. P(red)?", "a": "0.5"},
    {"q": "Phone: was $800, now 25% off. Sale price?", "a": "600"},
]


def simulate_model_sample(problem: dict, accuracy_rate: float = 0.65) -> str:
    """
    Simulate a single model sample.
    In a real experiment, replace this with an actual LLM API call.
    The accuracy_rate controls how often the model gets the right answer.
    """
    correct = problem["a"]
    
    # Simulate realistic model behavior:
    # - Right answer with probability = accuracy_rate
    # - Wrong answers are varied (not all the same) to model real LLM behavior
    if random.random() < accuracy_rate:
        return correct
    else:
        wrong_answers = _generate_wrong_answers(correct)
        return random.choice(wrong_answers)


def _generate_wrong_answers(correct: str) -> list:
    """Generate plausible-looking wrong answers."""
    try:
        val = float(correct.replace("$", "").replace("cm", "").replace("mph", ""))
        offsets = [val * 0.9, val * 1.1, val + 5, val - 5, val * 2, val / 2]
        return [str(round(o, 2)) for o in offsets if o != val]
    except Exception:
        return ["wrong_a", "wrong_b", "wrong_c", "wrong_d", "wrong_e"]


def best_of_n_accuracy(n_samples: int, base_accuracy: float = 0.65, n_problems: int = None) -> float:
    """
    Evaluate Best-of-N self-consistency accuracy over all problems.
    Returns fraction of problems answered correctly.
    """
    problems = PROBLEMS if n_problems is None else PROBLEMS[:n_problems]
    correct_count = 0

    for prob in problems:
        # Generate N samples
        samples = [simulate_model_sample(prob, base_accuracy) for _ in range(n_samples)]
        # Majority vote
        majority = Counter(samples).most_common(1)[0][0]
        if majority == prob["a"]:
            correct_count += 1

    return correct_count / len(problems)


def run_experiment(n_values: list = None, trials: int = 50, base_accuracy: float = 0.65):
    """
    Run the full Best-of-N scaling experiment.
    n_values: list of N to test (number of samples)
    trials: number of times to repeat each N (for variance estimation)
    """
    if n_values is None:
        n_values = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 60)
    print("Test-Time Compute Demo: Best-of-N Self-Consistency")
    print("Paper: arXiv:2501.02497 (2025)")
    print("=" * 60)
    print(f"\nBase model accuracy per sample: {base_accuracy:.0%}")
    print(f"Problems: {len(PROBLEMS)}, Trials per N: {trials}\n")
    print(f"{'N (samples)':>12} | {'Accuracy':>10} | {'Gain vs N=1':>12}")
    print("-" * 42)

    results = {}
    baseline = None

    for n in n_values:
        accs = [best_of_n_accuracy(n, base_accuracy) for _ in range(trials)]
        mean_acc = sum(accs) / len(accs)
        results[n] = mean_acc
        
        if baseline is None:
            baseline = mean_acc
            gain_str = "—  (baseline)"
        else:
            gain = mean_acc - baseline
            gain_str = f"+{gain:.1%}"

        print(f"{n:>12} | {mean_acc:>9.1%} | {gain_str:>12}")

    print("\n✅ Key finding: Accuracy increases with N, demonstrating the")
    print("   test-time scaling law described in arXiv:2501.02497.\n")

    # Save results
    import os
    os.makedirs("reproduction/results", exist_ok=True)
    with open("reproduction/results/results_summary.json", "w") as f:
        json.dump({"n_values": n_values, "accuracy": results, "base_accuracy": base_accuracy}, f, indent=2)
    print("📊 Results saved to reproduction/results/results_summary.json")

    # Try to plot
    try:
        import matplotlib.pyplot as plt
        ns = list(results.keys())
        accs = [results[n] * 100 for n in ns]

        plt.figure(figsize=(8, 5))
        plt.semilogx(ns, accs, 'o-', color='#00B4D8', linewidth=2.5, markersize=8, label='Self-Consistency (Majority Vote)')
        plt.axhline(y=accs[0], color='#8BAEC2', linewidth=1.5, linestyle='--', label='Greedy (N=1 baseline)')
        plt.fill_between(ns, accs[0], accs, alpha=0.1, color='#00B4D8')
        plt.xlabel('Number of Samples (N)', fontsize=13)
        plt.ylabel('Accuracy (%)', fontsize=13)
        plt.title('Test-Time Scaling: Accuracy vs. Inference Compute Budget\n(arXiv:2501.02497)', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reproduction/results/accuracy_vs_N.png", dpi=150)
        print("📈 Plot saved to reproduction/results/accuracy_vs_N.png")
    except ImportError:
        print("(Install matplotlib to generate plot: pip install matplotlib)")

    return results


def real_model_instructions():
    """Print instructions for running with a real LLM."""
    print("""
─── To run with a REAL language model ──────────────────────────────────
Replace the simulate_model_sample() function with an actual API call:

    from openai import OpenAI
    client = OpenAI()

    def simulate_model_sample(problem, **kwargs):
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer math problems concisely. Give only the number."},
                {"role": "user", "content": problem["q"]}
            ],
            temperature=0.8  # High temp for diverse samples
        )
        return resp.choices[0].message.content.strip()

Or use a HuggingFace model locally for a fully open-source demo.
────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    random.seed(42)
    results = run_experiment(
        n_values=[1, 2, 4, 8, 16, 32, 64],
        trials=30,
        base_accuracy=0.65
    )
    real_model_instructions()
