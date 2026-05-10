# Paper Summary: A Survey of Test-Time Compute (arXiv:2501.02497)

## Citation
Snell, C. et al. (2025). A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning. arXiv:2501.02497

## Core Argument
Training-time scaling (bigger models + more data) is hitting limits.
Test-time compute is a new scaling axis: spending more compute at inference
consistently improves accuracy, following power-law scaling laws.

## Key Techniques
1. Chain-of-Thought — intermediate reasoning steps
2. Repeated sampling + self-consistency (majority vote)
3. Process Reward Models — score each reasoning step
4. Monte Carlo Tree Search over reasoning trajectories
5. Self-correction and iterative refinement

## Landmark Models
- OpenAI o1/o3: RL-trained long thinking chains
- DeepSeek-R1: Open-source GRPO, matches o1
- Gemini 2.5: Multimodal TTC reasoning
- QwQ/Qwen3: Tool-augmented reasoning
