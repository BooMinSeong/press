"""
Main experiment runner script.

Usage:
    python experiments/run_experiment.py --output results/experiment_001.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from vllm import LLM
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from press.config import (
    MODEL_NAME,
    SYSTEM_PROMPT,
    DATASET_NAME,
    DATASET_SPLIT,
    get_sampling_params,
)
from press.inference import solve_with_entropy_tracking
from press.verification import evaluate_result


def load_math_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT):
    """Load MATH-500 dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def save_results(results: list, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def run_experiment(
    model_name: str = MODEL_NAME,
    output_path: str = "results/experiment.json",
    num_problems: int = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """
    Run the full experiment with entropy tracking and multiple injection prompts.

    Args:
        model_name: vLLM model name
        output_path: Path to save results
        num_problems: Number of problems to process (None = all)
        temperature: Sampling temperature
        max_tokens: Maximum tokens per step
    """
    print("="*80)
    print("Entropy-based LLM Math Problem Solving Analysis")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Load model
    print("Loading model...")
    llm = LLM(model=model_name, trust_remote_code=True)
    print("Model loaded successfully!")
    print()

    # Load dataset
    dataset = load_math_dataset()

    # Limit number of problems if specified
    if num_problems:
        dataset = dataset.select(range(min(num_problems, len(dataset))))

    print(f"Processing {len(dataset)} problems...")
    print()

    # Get sampling parameters
    sampling_params = get_sampling_params(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Run experiments
    all_results = []

    for idx, problem in enumerate(tqdm(dataset, desc="Solving problems")):
        try:
            # Solve with entropy tracking
            result = solve_with_entropy_tracking(
                llm,
                problem,
                SYSTEM_PROMPT,
                sampling_params,
            )

            # Evaluate correctness
            eval_data = {
                'gold_answer': result['gold_answer'],
                'final_answer': result['final_answer'],
            }
            result['is_correct'] = evaluate_result(eval_data)

            all_results.append(result)

            # Save incrementally every 10 problems
            if (idx + 1) % 10 == 0:
                save_results(all_results, output_path)

        except Exception as e:
            print(f"\nError processing problem {idx}: {e}")
            continue

    # Final save
    save_results(all_results, output_path)

    # Print summary statistics
    print()
    print("="*80)
    print("Experiment Summary")
    print("="*80)
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('is_correct', False))
    accuracy = correct / total if total > 0 else 0

    print(f"Total problems: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    # Average steps per problem
    avg_steps = sum(len(r['steps']) for r in all_results) / total if total > 0 else 0
    print(f"Average steps per problem: {avg_steps:.2f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run entropy-based LLM math problem solving experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of problems to process (default: all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per step"
    )

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        output_path=args.output,
        num_problems=args.num_problems,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
