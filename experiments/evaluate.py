"""
Evaluation script - evaluates generated solutions.

This script reads generation results and computes correctness metrics.
Can be run multiple times with different evaluation methods without re-running inference.

Usage:
    python experiments/evaluate.py results/generation.json --output results/evaluated.json
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from press.config import BEAM_SELECTION_METHOD
from press.verification import evaluate_result
from experiments.utils import (
    load_results,
    save_results,
    get_beam_score,
    detect_result_type,
    print_beam_search_statistics,
    print_trajectory_statistics,
)


def evaluate_beam_search_results(
    results: List[Dict[str, Any]],
    beam_selection_method: str = "avg"
) -> List[Dict[str, Any]]:
    """
    Evaluate beam search results.

    Args:
        results: List of beam search results from generation
        beam_selection_method: How to select best beam - "avg" or "last"

    Returns:
        Results with is_correct fields added
    """
    evaluated_results = []

    for result in tqdm(results, desc="Evaluating beams"):
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

        evaluated_results.append(result)

    return evaluated_results


def evaluate_trajectory_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate multi-trajectory results.

    Args:
        results: List of problem results with trajectories from generation

    Returns:
        Results with is_correct fields added
    """
    evaluated_results = []

    for result in tqdm(results, desc="Evaluating trajectories"):
        for trajectory in result['trajectories']:
            eval_data = {
                'gold_answer': result['gold_answer'],
                'final_answer': trajectory['final_answer'],
            }
            trajectory['is_correct'] = evaluate_result(eval_data)

        evaluated_results.append(result)

    return evaluated_results


def run_evaluation(
    input_path: str,
    output_path: str = None,
    beam_selection_method: str = None,
    inplace: bool = False,
):
    """
    Evaluate generated results.

    Args:
        input_path: Path to generation results JSON
        output_path: Path to save evaluated results (default: input_path with _evaluated suffix)
        beam_selection_method: How to select best beam (only for beam search)
        inplace: Whether to update the input file instead of creating a new one
    """
    if beam_selection_method is None:
        beam_selection_method = BEAM_SELECTION_METHOD

    print("="*80)
    print("Evaluating Generated Solutions")
    print("="*80)
    print(f"Input: {input_path}")
    print()

    # Load results
    print("Loading generation results...")
    results = load_results(input_path)
    print(f"Loaded {len(results)} problems")

    # Detect result type
    result_type = detect_result_type(results)
    print(f"Result type: {result_type}")
    print()

    # Evaluate
    print("Evaluating correctness...")
    if result_type == "beam_search":
        print(f"Beam selection method: {beam_selection_method}")
        evaluated_results = evaluate_beam_search_results(results, beam_selection_method)
        beam_width = len(results[0]['beams']) if results and results[0]['beams'] else None
    elif result_type == "multi_trajectory":
        evaluated_results = evaluate_trajectory_results(results)
        num_samples = len(results[0]['trajectories']) if results and results[0]['trajectories'] else 1
    else:
        print(f"Error: Unknown result type")
        return

    # Add evaluation metadata
    evaluation_metadata = {
        'evaluated_at': datetime.now().isoformat(),
        'evaluation_method': 'math_verify',
        'beam_selection_method': beam_selection_method if result_type == "beam_search" else None,
    }

    # Optionally add metadata to results (as a top-level field if results is a list)
    # For now, we'll just print it
    print()
    print(f"Evaluation method: {evaluation_metadata['evaluation_method']}")
    print(f"Evaluated at: {evaluation_metadata['evaluated_at']}")

    # Determine output path
    if inplace:
        final_output_path = input_path
    elif output_path:
        final_output_path = output_path
    else:
        input_file = Path(input_path)
        final_output_path = str(input_file.parent / f"{input_file.stem}_evaluated{input_file.suffix}")

    # Save results
    save_results(evaluated_results, final_output_path)

    # Print statistics
    print()
    print("="*80)
    print("Evaluation Results")
    print("="*80)

    if result_type == "beam_search":
        print_beam_search_statistics(evaluated_results, beam_width, beam_selection_method)
    elif result_type == "multi_trajectory":
        print_trajectory_statistics(evaluated_results, num_samples)

    print()
    print(f"Evaluated results saved to: {final_output_path}")

    return evaluated_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated solutions"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file with generation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: input_evaluated.json)"
    )
    parser.add_argument(
        "--beam-selection-method",
        type=str,
        default=None,
        choices=["avg", "last"],
        help=f"How to select best beam: 'avg' (average entropy) or 'last' (last step entropy) (default: {BEAM_SELECTION_METHOD})"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Update the input file instead of creating a new one"
    )

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        beam_selection_method=args.beam_selection_method,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()
