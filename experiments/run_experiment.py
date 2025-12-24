"""
Main experiment runner script.

Usage:
    python experiments/run_experiment.py --output results/experiment_001.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
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
    BEAM_WIDTH,
    BEAM_CANDIDATES_PER_BEAM,
    BEAM_SELECTION_METHOD,
    get_sampling_params,
)
from press.inference import solve_with_entropy_tracking, solve_with_beam_search
from press.verification import evaluate_result


def load_math_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT):
    """Load MATH-500 dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def get_beam_score(beam: dict, method: str = "avg") -> float:
    """
    Calculate score for beam selection based on aggregation method.

    Args:
        beam: Beam dictionary with 'steps' and 'avg_entropy' fields
        method: Aggregation method - "avg" or "last"

    Returns:
        Score for beam selection (lower is better)
    """
    if method == "avg":
        # Use average entropy across all steps
        return beam['avg_entropy']
    elif method == "last":
        # Use entropy from the last step
        if beam['steps']:
            return beam['steps'][-1]['selection_entropy']
        else:
            # Fallback to avg_entropy if no steps
            return beam['avg_entropy']
    else:
        raise ValueError(f"Unknown beam selection method: {method}")


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
    use_beam_search: bool = False,
    beam_width: int = None,
    candidates_per_beam: int = None,
    beam_selection_method: str = None,
):
    """
    Run the full experiment with entropy tracking and multiple injection prompts.

    Args:
        model_name: vLLM model name
        output_path: Path to save results
        num_problems: Number of problems to process (None = all)
        num_samples: Number of trajectories to sample per problem (ignored if use_beam_search=True)
        temperature: Sampling temperature
        max_tokens: Maximum tokens per step
        use_beam_search: Whether to use beam search instead of multi-trajectory sampling
        beam_width: Number of beams to maintain (only for beam search)
        candidates_per_beam: Number of candidates per beam (only for beam search)
        beam_selection_method: How to select best beam - "avg" or "last" (only for beam search)
    """
    if beam_selection_method is None:
        beam_selection_method = BEAM_SELECTION_METHOD

    print("="*80)
    print("Entropy-based LLM Math Problem Solving Analysis")
    print("="*80)
    print(f"Model: {model_name}")
    if use_beam_search:
        print(f"Mode: Beam Search")
        print(f"Beam width: {beam_width or BEAM_WIDTH}")
        print(f"Candidates per beam: {candidates_per_beam or BEAM_CANDIDATES_PER_BEAM}")
        print(f"Selection method: {beam_selection_method}")
    else:
        print(f"Mode: Multi-trajectory sampling")
        print(f"Samples per problem: {num_samples}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Load model
    print("Loading model...")
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=num_gpus)
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
        if use_beam_search:
            # Beam search mode
            try:
                result = solve_with_beam_search(
                    llm,
                    problem,
                    SYSTEM_PROMPT,
                    sampling_params,
                    beam_width,
                    candidates_per_beam,
                )

                # Evaluate each beam
                for beam_idx, beam in enumerate(result['beams']):
                    eval_data = {
                        'gold_answer': result['gold_answer'],
                        'final_answer': beam['final_answer'],
                    }
                    beam['is_correct'] = evaluate_result(eval_data)
                    beam['beam_id'] = beam_idx

                # Select best beam based on selection method
                if result['beams']:
                    best_beam_idx = min(
                        range(len(result['beams'])),
                        key=lambda i: get_beam_score(result['beams'][i], beam_selection_method)
                    )
                    result['selected_beam_id'] = best_beam_idx
                    result['selected_beam_correct'] = result['beams'][best_beam_idx]['is_correct']
                    result['selection_method'] = beam_selection_method
                else:
                    result['selected_beam_id'] = None
                    result['selected_beam_correct'] = False
                    result['selection_method'] = beam_selection_method

                all_results.append(result)

            except Exception as e:
                print(f"\nError processing problem {idx} with beam search: {e}")
                continue

        else:
            # Multi-trajectory sampling mode
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

    if use_beam_search:
        # Beam search statistics
        total_beams = sum(len(p['beams']) for p in all_results)

        # Count correct beams
        correct_beams = sum(
            1 for p in all_results
            for b in p['beams']
            if b.get('is_correct', False)
        )

        # Calculate accuracy
        accuracy = correct_beams / total_beams if total_beams > 0 else 0

        # Count problems with at least one correct beam
        problems_with_correct = sum(
            1 for p in all_results
            if any(b.get('is_correct', False) for b in p['beams'])
        )
        problem_accuracy = problems_with_correct / total_problems if total_problems > 0 else 0

        # Count correct selected beams (entropy-based selection)
        correct_selected = sum(
            1 for p in all_results
            if p.get('selected_beam_correct', False)
        )
        selected_accuracy = correct_selected / total_problems if total_problems > 0 else 0

        print(f"Total problems: {total_problems}")
        print(f"Total beams: {total_beams}")
        print(f"Beam width: {beam_width or BEAM_WIDTH}")
        print(f"Selection method: {beam_selection_method}")
        print()
        print(f"Correct beams: {correct_beams}/{total_beams}")
        print(f"Beam-level accuracy: {accuracy:.2%}")
        print()
        print(f"Selected beams (method={beam_selection_method}): {correct_selected}/{total_problems}")
        print(f"Selected beam accuracy: {selected_accuracy:.2%}")
        print()
        print(f"Problems with ≥1 correct beam: {problems_with_correct}/{total_problems}")
        print(f"Problem-level accuracy (pass@{beam_width or BEAM_WIDTH}): {problem_accuracy:.2%}")
        print()

        # Average steps per beam
        all_beams = [b for p in all_results for b in p['beams']]
        if all_beams:
            avg_steps = sum(len(b['steps']) for b in all_beams) / len(all_beams)
            print(f"Average steps per beam: {avg_steps:.2f}")

            # Average entropy per beam
            avg_beam_entropy = sum(b['avg_entropy'] for b in all_beams) / len(all_beams)
            print(f"Average entropy per beam: {avg_beam_entropy:.4f}")

    else:
        # Multi-trajectory statistics
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
    parser.add_argument(
        "--use-beam-search",
        action="store_true",
        help="Use beam search instead of multi-trajectory sampling"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help=f"Number of beams to maintain (default: {BEAM_WIDTH})"
    )
    parser.add_argument(
        "--candidates-per-beam",
        type=int,
        default=None,
        help=f"Number of candidates to sample from each beam (default: {BEAM_CANDIDATES_PER_BEAM})"
    )
    parser.add_argument(
        "--beam-selection-method",
        type=str,
        default=None,
        choices=["avg", "last"],
        help=f"How to select best beam: 'avg' (average entropy) or 'last' (last step entropy) (default: {BEAM_SELECTION_METHOD})"
    )

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        output_path=args.output,
        num_problems=args.num_problems,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_beam_search=args.use_beam_search,
        beam_width=args.beam_width,
        candidates_per_beam=args.candidates_per_beam,
        beam_selection_method=args.beam_selection_method,
    )


if __name__ == "__main__":
    main()
