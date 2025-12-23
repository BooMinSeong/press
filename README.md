# PRESS: Probabilistic Reasoning Entropy-based Study System

> Entropy-based LLM Math Problem Solving Analysis with Multiple Injection Prompts

## Overview

This project analyzes Large Language Model (LLM) uncertainty during mathematical problem-solving using entropy measurements from logprobs. It implements multiple injection prompt strategies to measure model confidence from different perspectives and evaluates their effectiveness in predicting solution correctness.

## Key Features

- **Entropy Tracking**: Token-level and step-level entropy calculation from vLLM logprobs
- **Multiple Injection Prompts**: 6 different prompts to measure model uncertainty:
  - Confidence score (1-10)
  - Correctness probability (0-100%)
  - Error likelihood (1-10)
  - Step quality (A-F)
  - Binary verification (Yes/No)
  - Revision need (1-10)
- **Comprehensive Analysis**: Correlation analysis, calibration studies, and visualization
- **Math Verification**: Uses `math_verify` package for robust answer checking

## Installation

```bash
# Clone the repository
cd press

# Install dependencies (using uv, pip, or your preferred package manager)
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

## Project Structure

```
press/
├── press/                      # Main package
│   ├── config.py              # Configuration (prompts, model settings)
│   ├── entropy.py             # Entropy calculation utilities
│   ├── inference.py           # Core inference pipeline
│   └── verification.py        # Answer parsing and verification
├── experiments/               # Experiment scripts
│   ├── run_experiment.py      # Main experiment runner
│   └── analyze_results.py     # Analysis and visualization
├── results/                   # Experiment results (generated)
└── pyproject.toml            # Project dependencies
```

## Usage

### 1. Run Experiment

```bash
# Run on full MATH-500 dataset
python experiments/run_experiment.py --output results/experiment_001.json

# Run on subset (e.g., first 50 problems)
python experiments/run_experiment.py \
    --output results/test_run.json \
    --num-problems 50

# Custom model and parameters
python experiments/run_experiment.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output results/qwen_experiment.json \
    --temperature 0.7 \
    --max-tokens 512
```

### 2. Analyze Results

```bash
# Generate analysis report and visualizations
python experiments/analyze_results.py results/experiment_001.json

# Specify output directory
python experiments/analyze_results.py \
    results/experiment_001.json \
    --output-dir results/analysis_001
```

This will generate:
- `analysis_report.txt`: Comprehensive text report
- `plots/entropy_distribution.png`: Entropy comparison (correct vs incorrect)
- `plots/injection_prompt_comparison.png`: Effectiveness of different injection prompts
- `plots/entropy_evolution.png`: Entropy changes across reasoning steps

## Output Data Structure

Each experiment generates a JSON file with the following structure:

```json
{
  "problem_id": "001",
  "problem_text": "...",
  "gold_answer": "42",
  "generated_solution": "...",
  "final_answer": "42",
  "is_correct": true,
  "steps": [
    {
      "step_number": 1,
      "text": "## Step 1: ...",
      "token_entropies": [0.5, 0.3, ...],
      "avg_entropy": 0.35,
      "max_entropy": 0.8,
      "injection_results": {
        "confidence_score": {
          "entropy": 1.2,
          "predicted_value": "8",
          "top_k_probs": {"8": 0.6, "7": 0.2, ...}
        },
        "correctness_prob": {...},
        ...
      }
    }
  ]
}
```

## Analysis Metrics

### A. Basic Performance
- Overall accuracy
- Average steps per problem
- Accuracy by problem difficulty

### B. Entropy Analysis
- Average entropy: Correct vs Incorrect solutions
- Entropy evolution across reasoning steps
- Correlation between entropy and correctness

### C. Injection Prompt Effectiveness
- Which injection prompts best predict correctness?
- Entropy difference between correct/incorrect solutions
- Calibration analysis (predicted confidence vs actual accuracy)

### D. Calibration Study
- Does `confidence_score=8` actually mean 80% accuracy?
- Calibration curves for each injection prompt

## System Prompt

The model uses a structured prompt that encourages:
- Concise solutions for simple problems (≤2 steps)
- Step-by-step reasoning for complex problems (≥3 steps)
- Final answer in `\boxed{answer}` format

See [press/config.py](press/config.py:10) for the full prompt.

## Injection Prompts

Six different injection prompts are used to measure uncertainty:

1. **Confidence Score**: Self-rated confidence (1-10)
2. **Correctness Probability**: Likelihood of correctness (0-100%)
3. **Error Likelihood**: Probability of error (1-10)
4. **Step Quality**: Grade quality (A-F)
5. **Binary Check**: Mathematical soundness (Yes/No)
6. **Revision Need**: Need for revision (1-10)

See [press/config.py](press/config.py:30) for implementation details.

## Dataset

Uses **HuggingFaceH4/MATH-500** - a curated subset of 500 problems from the MATH dataset, covering various difficulty levels and topics.

## Model

Default: `Qwen/Qwen2.5-1.5B-Instruct`

Can be changed via `--model` argument. Any vLLM-compatible model can be used.

## Requirements

- Python ≥3.12
- CUDA-capable GPU (recommended for vLLM)
- ~8GB GPU memory (for Qwen2.5-1.5B)

## Future Extensions

### 1. Injection Prompt Ensemble
Combine multiple injection prompts using weighted averaging based on predictive power.

### 2. Early Stopping Strategy
Stop generation when entropy exceeds threshold and regenerate with different parameters.

### 3. Step-wise Error Detection
Identify which reasoning step contains errors by tracking entropy spikes.

### 4. Injection Prompt Evolution
A/B testing to discover more effective injection prompts.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{press2024,
  title={PRESS: Probabilistic Reasoning Entropy-based Study System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/press}
}
```

## License

MIT License

## Acknowledgments

- vLLM team for the inference engine
- HuggingFace for the MATH-500 dataset
- math_verify package for robust answer verification
