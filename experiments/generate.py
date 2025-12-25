"""
Generation script - runs model inference without evaluation.

This script performs model inference and saves raw results without computing correctness.
Use evaluate.py to evaluate the generated results separately.

Usage:
    python experiments/generate.py --output results/generation.json
"""

import argparse
from pathlib import Path

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
    get_sampling_params,
)
from press.inference import solve_with_entropy_tracking, solve_with_beam_search
from experiments.utils import save_results


def load_math_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT):
    """Load MATH-500 dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def run_generation(
    model_name: str = MODEL_NAME,
    output_path: str = "results/generation.json",
    num_problems: int = None,
    num_samples: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    use_beam_search: bool = False,
    beam_width: int = None,
    candidates_per_beam: int = None,
):
    """
    Run generation without evaluation.

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
    """
    print("="*80)
    print("Entropy-based LLM Math Problem Solving - Generation Only")
    print("="*80)
    print(f"Model: {model_name}")
    if use_beam_search:
        print(f"Mode: Beam Search")
        print(f"Beam width: {beam_width or BEAM_WIDTH}")
        print(f"Candidates per beam: {candidates_per_beam or BEAM_CANDIDATES_PER_BEAM}")
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

    # Run generation
    all_results = []

    for idx, problem in enumerate(tqdm(dataset, desc="Generating solutions")):
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

                # Note: No evaluation here - just store raw generation results
                all_results.append(result)

            except Exception as e:
                print(f"\nError processing problem {idx} with beam search: {e}")
                continue

        else:
            # Multi-trajectory sampling mode
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

                    # Remove redundant fields (already in problem_result)
                    trajectory.pop('problem_id', None)
                    trajectory.pop('problem_text', None)
                    trajectory.pop('gold_answer', None)

                    # Note: No evaluation here - just store raw generation
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

    # Print summary statistics (without evaluation metrics)
    print()
    print("="*80)
    print("Generation Summary")
    print("="*80)

    total_problems = len(all_results)

    if use_beam_search:
        # Beam search statistics (no correctness)
        total_beams = sum(len(p['beams']) for p in all_results)

        print(f"Total problems: {total_problems}")
        print(f"Total beams generated: {total_beams}")
        print(f"Beam width: {beam_width or BEAM_WIDTH}")
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
        # Multi-trajectory statistics (no correctness)
        total_trajectories = sum(len(p['trajectories']) for p in all_results)

        print(f"Total problems: {total_problems}")
        print(f"Total trajectories generated: {total_trajectories}")
        print(f"Samples per problem: {num_samples}")
        print()

        # Average steps per trajectory
        all_trajectories = [t for p in all_results for t in p['trajectories']]
        if all_trajectories:
            avg_steps = sum(len(t['steps']) for t in all_trajectories) / len(all_trajectories)
            print(f"Average steps per trajectory: {avg_steps:.2f}")

    print()
    print(f"Generation complete! Results saved to: {output_path}")
    print(f"Run evaluate.py to compute correctness metrics.")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Generate solutions with entropy tracking (no evaluation)"
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
        default="results/generation.json",
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

    args = parser.parse_args()

    run_generation(
        model_name=args.model,
        output_path=args.output,
        num_problems=args.num_problems,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_beam_search=args.use_beam_search,
        beam_width=args.beam_width,
        candidates_per_beam=args.candidates_per_beam,
    )


if __name__ == "__main__":
    main()
