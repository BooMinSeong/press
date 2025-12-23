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
    num_samples: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """
    Run the full experiment with entropy tracking and multiple injection prompts.

    Args:
        model_name: vLLM model name
        output_path: Path to save results
        num_problems: Number of problems to process (None = all)
        num_samples: Number of trajectories to sample per problem
        temperature: Sampling temperature
        max_tokens: Maximum tokens per step
    """
    print("="*80)
    print("Entropy-based LLM Math Problem Solving Analysis")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Samples per problem: {num_samples}")
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
        # Initialize problem result with multiple trajectories
        problem_result = {
            'problem_id': problem.get('id', problem.get('problem_id', f'problem_{idx}')),
            'problem_text': problem['problem'],
            'gold_answer': problem['answer'],
            'trajectories': []
        }

        # Generate multiple trajectories for this problem
        for traj_id in range(num_samples):
            try:
                # Solve with entropy tracking
                trajectory = solve_with_entropy_tracking(
                    llm,
                    problem,
                    SYSTEM_PROMPT,
                    sampling_params,
                )

                # Add trajectory ID
                trajectory['trajectory_id'] = traj_id

                # Evaluate correctness
                eval_data = {
                    'gold_answer': trajectory['gold_answer'],
                    'final_answer': trajectory['final_answer'],
                }
                trajectory['is_correct'] = evaluate_result(eval_data)

                # Remove redundant fields (already in problem_result)
                trajectory.pop('problem_id', None)
                trajectory.pop('problem_text', None)
                trajectory.pop('gold_answer', None)

                problem_result['trajectories'].append(trajectory)

            except Exception as e:
                print(f"\nError processing problem {idx}, trajectory {traj_id}: {e}")
                continue

        all_results.append(problem_result)

        # Save incrementally every 10 problems
        if (idx + 1) % 10 == 0:
            save_results(all_results, output_path)

    # Final save
    save_results(all_results, output_path)

    # Print summary statistics
    print()
    print("="*80)
    print("Experiment Summary")
    print("="*80)

    total_problems = len(all_results)
    total_trajectories = sum(len(p['trajectories']) for p in all_results)

    # Count correct trajectories
    correct_trajectories = sum(
        1 for p in all_results
        for t in p['trajectories']
        if t.get('is_correct', False)
    )

    # Calculate accuracy
    accuracy = correct_trajectories / total_trajectories if total_trajectories > 0 else 0

    # Count problems with at least one correct trajectory
    problems_with_correct = sum(
        1 for p in all_results
        if any(t.get('is_correct', False) for t in p['trajectories'])
    )
    problem_accuracy = problems_with_correct / total_problems if total_problems > 0 else 0

    print(f"Total problems: {total_problems}")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Samples per problem: {num_samples}")
    print()
    print(f"Correct trajectories: {correct_trajectories}/{total_trajectories}")
    print(f"Trajectory-level accuracy: {accuracy:.2%}")
    print()
    print(f"Problems with ≥1 correct trajectory: {problems_with_correct}/{total_problems}")
    print(f"Problem-level accuracy (pass@{num_samples}): {problem_accuracy:.2%}")
    print()

    # Average steps per trajectory
    all_trajectories = [t for p in all_results for t in p['trajectories']]
    if all_trajectories:
        avg_steps = sum(len(t['steps']) for t in all_trajectories) / len(all_trajectories)
        print(f"Average steps per trajectory: {avg_steps:.2f}")

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
        "--num-samples",
        type=int,
        default=1,
        help="Number of trajectories to sample per problem (default: 1)"
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
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
