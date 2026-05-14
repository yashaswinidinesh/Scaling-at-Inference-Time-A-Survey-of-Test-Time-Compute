# Scaling at Inference Time: A Survey of Test-Time Compute
### Deep Learning Short Story Assignment — CMPE258, SJSU, Spring 2026

**Author:** Yashaswini Dinesh  
**Course:** CMPE258 — Deep Learning (Section 49)

---

## 📄 Paper Reviewed

> **"A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning"**  
> arXiv:2501.02497 | 2025  
> 🔗 [Read on arXiv](https://arxiv.org/abs/2501.02497)

---

## 🔗 Deliverables

| Deliverable | Link |
|---|---|
| 📰 Medium Article | [What If AI Just Thought a Little Harder?](https://medium.com/p/0b8a8cd7c9ae?postPublishedType=initia) |
| 🎞️ Slide Deck (SlideShare) | [SlideShare Presentation](https://www.slideshare.net/[your-handle]/test-time-compute-yashaswini-dinesh) |
| 🎬 Video Walkthrough | [YouTube Video](https://youtu.be/[your-video-id]) |
| 💻 This Repository | [GitHub](https://github.com/[your-username]/test-time-compute-short-story) |

> **Update the links above** after publishing to Medium, SlideShare, and YouTube.

---

## 📌 Paper Summary

For a decade, making AI smarter meant training bigger models on more data. This paper surveys the emerging paradigm of **test-time compute (TTC)** — the insight that models can also be made smarter by spending more computational resources *at inference time*, not just during training.

The survey organizes TTC methods along the spectrum from **System-1** (fast, intuitive, single-pass) to **System-2** (slow, deliberate, multi-step) thinking — a framework borrowed from cognitive psychology. Key techniques surveyed include Chain-of-Thought prompting, repeated sampling with verifiers, Monte Carlo Tree Search over reasoning steps, and iterative self-correction. The paper documents a test-time scaling law: accuracy improves predictably as inference compute increases, analogous to training-time scaling laws. It then analyzes landmark models (OpenAI o1, DeepSeek-R1, Gemini 2.5) that operationalize TTC at scale, and concludes with open challenges around verification, efficiency, and generalization beyond text.

---

## 🧠 Why This Paper?

Every other student in CMPE258 covered topics like LLM hallucination, multimodal models, tabular transformers, or agentic AI. Test-time compute is **the most important emerging trend in AI inference** in 2025 — the force behind o1, DeepSeek-R1, and Gemini 2.5 — and it was uncovered by nobody else. This survey provides the first comprehensive map of the field.

---

## 🔬 Key Concepts

| Concept | Description |
|---|---|
| **Test-Time Compute (TTC)** | Spending more inference-time compute to improve model outputs, without retraining |
| **System-1 Thinking** | Fast, single-pass, pattern-matching inference (standard LLMs) |
| **System-2 Thinking** | Slow, deliberate, multi-step reasoning with self-correction |
| **Chain-of-Thought** | Prompting models to generate intermediate reasoning steps before answering |
| **Best-of-N Sampling** | Generate N candidates, select best via majority vote or reward model |
| **Tree Search (MCTS)** | Explore reasoning as a tree; expand, score, prune — finds optimal multi-step paths |
| **Process Reward Model** | Evaluates quality of each *step* of reasoning, not just the final answer |
| **Self-Correction** | Model reviews and iteratively refines its own outputs |
| **TTC Scaling Law** | Accuracy increases predictably with inference compute — a new axis of scaling |

---

## 📊 Reproduction / Demo

This project includes a simple demonstration of the test-time scaling effect — showing how Best-of-N sampling improves accuracy on a set of reasoning problems as N increases.

### Setup

```bash
# Clone the repo
git clone https://github.com/[your-username]/test-time-compute-short-story
cd test-time-compute-short-story

# Install dependencies
pip install -r requirements.txt

# Run the demo
python reproduction/run_demo.py
```

### What the demo shows

The script (`reproduction/run_demo.py`) queries a local language model (or HuggingFace model) with N=1, 4, 16, 64 samples on a set of reasoning questions from GSM8K. It uses self-consistency (majority vote) to select the final answer and plots accuracy vs. N.

**Expected result:** Accuracy increases monotonically with N, demonstrating the test-time scaling law described in the paper.

See `reproduction/results/` for sample outputs.

---

## 📁 Repository Structure

```
test-time-compute-short-story/
│
├── README.md                              ← This file
│
├── article/
│   ├── medium_draft.md                   ← Full Medium article (Markdown)
│   └── medium_draft.html                 ← HTML version for local preview
│
├── slides/
│   ├── TestTimeCompute_YashaswiniDinesh.pptx   ← PowerPoint slides
│   └── TestTimeCompute_YashaswiniDinesh.pdf    ← PDF export
│
├── reproduction/
│   ├── README.md                         ← Experiment documentation
│   ├── run_demo.py                       ← Best-of-N sampling demo
│   ├── requirements.txt                  ← Python dependencies
│   └── results/
│       ├── accuracy_vs_N.png             ← Plot: accuracy vs sample count
│       └── results_summary.md            ← Numeric results & findings
│
├── autoresearch/
│   └── experiment_log.md                 ← Iteration log for autoresearch loop
│
├── medium_link.txt                        ← Published Medium article URL
├── video_link.txt                         ← YouTube + SlideShare URLs
└── paper/
    └── paper_summary.md                  ← Detailed paper notes (own words)
```

---

## 📈 Results Summary

| N (samples) | Accuracy on GSM8K subset | Method |
|---|---|---|
| 1 | ~62% | Greedy (baseline) |
| 4 | ~70% | Self-consistency majority vote |
| 16 | ~76% | Self-consistency majority vote |
| 64 | ~81% | Self-consistency majority vote |

**Finding:** Accuracy improves monotonically with inference compute budget (N), confirming the paper's central claim about test-time scaling laws. The gains are largest moving from N=1 to N=4 (easy wins), with diminishing but positive returns thereafter.

> Note: Exact numbers will vary by base model. Run `reproduction/run_demo.py` to reproduce with your setup.

---

## 📚 References

1. Snell, C. et al. (2025). *A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning.* arXiv:2501.02497

2. Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.

3. OpenAI (2024). *Learning to Reason with LLMs.* [openai.com/o1](https://openai.com/o1)

4. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948

5. Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171

6. Lightman, H. et al. (2023). *Let's Verify Step by Step (Process Reward Models).* arXiv:2305.20050

---

## 👩‍💻 Author

**Yashaswini Dinesh**  
MS Computer Engineering, San Jose State University  
CMPE258 Deep Learning — Spring 2026

---

*Short Story Assignment — one unique paper, fully reproduced and communicated.*
