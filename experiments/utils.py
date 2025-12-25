"""
Shared utility functions for experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load results from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of result dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: list, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


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


def print_beam_search_statistics(all_results: List[Dict[str, Any]], beam_width: int = None, beam_selection_method: str = "avg"):
    """
    Print statistics for beam search results.

    Args:
        all_results: List of beam search results
        beam_width: Beam width used (for display)
        beam_selection_method: Selection method used
    """
    total_problems = len(all_results)
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
    if beam_width:
        print(f"Beam width: {beam_width}")
    print(f"Selection method: {beam_selection_method}")
    print()
    print(f"Correct beams: {correct_beams}/{total_beams}")
    print(f"Beam-level accuracy: {accuracy:.2%}")
    print()
    print(f"Selected beams (method={beam_selection_method}): {correct_selected}/{total_problems}")
    print(f"Selected beam accuracy: {selected_accuracy:.2%}")
    print()
    print(f"Problems with ≥1 correct beam: {problems_with_correct}/{total_problems}")
    if beam_width:
        print(f"Problem-level accuracy (pass@{beam_width}): {problem_accuracy:.2%}")
    else:
        print(f"Problem-level accuracy: {problem_accuracy:.2%}")
    print()

    # Average steps per beam
    all_beams = [b for p in all_results for b in p['beams']]
    if all_beams:
        avg_steps = sum(len(b['steps']) for b in all_beams) / len(all_beams)
        print(f"Average steps per beam: {avg_steps:.2f}")

        # Average entropy per beam
        avg_beam_entropy = sum(b['avg_entropy'] for b in all_beams) / len(all_beams)
        print(f"Average entropy per beam: {avg_beam_entropy:.4f}")


def print_trajectory_statistics(all_results: List[Dict[str, Any]], num_samples: int = 1):
    """
    Print statistics for multi-trajectory results.

    Args:
        all_results: List of problem results with trajectories
        num_samples: Number of samples per problem (for display)
    """
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


def detect_result_type(results: List[Dict[str, Any]]) -> str:
    """
    Detect whether results are from beam search or multi-trajectory mode.

    Args:
        results: List of result dictionaries

    Returns:
        "beam_search" or "multi_trajectory"
    """
    if not results:
        return "unknown"

    first_result = results[0]
    if 'beams' in first_result:
        return "beam_search"
    elif 'trajectories' in first_result:
        return "multi_trajectory"
    else:
        return "unknown"
