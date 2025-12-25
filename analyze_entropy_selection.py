#!/usr/bin/env python3
"""
Analyze pass@k results by selecting the trajectory with lowest total entropy per problem.
"""
import json
import sys
from pathlib import Path


def calculate_trajectory_entropy(trajectory):
    """Calculate total entropy for a trajectory by summing avg_entropy of all steps."""
    total_entropy = 0.0
    for step in trajectory.get("steps", []):
        avg_entropy = step.get("avg_entropy", 0.0)
        total_entropy += avg_entropy
    return total_entropy


def analyze_entropy_selection(input_file):
    """
    Analyze accuracy when selecting the trajectory with lowest total entropy per problem.

    Args:
        input_file: Path to the pass@k JSON file

    Returns:
        Dictionary with analysis results
    """
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_problems = len(data)
    correct_selections = 0

    results = []

    for problem in data:
        problem_id = problem.get("problem_id", "unknown")
        trajectories = problem.get("trajectories", [])

        if not trajectories:
            print(f"Warning: No trajectories for {problem_id}")
            continue

        # Calculate total entropy for each trajectory
        trajectory_entropies = []
        for idx, trajectory in enumerate(trajectories):
            total_entropy = calculate_trajectory_entropy(trajectory)
            is_correct = trajectory.get("is_correct", False)

            trajectory_entropies.append({
                "trajectory_idx": idx,
                "total_entropy": total_entropy,
                "is_correct": is_correct
            })

        # Find trajectory with minimum entropy
        min_entropy_trajectory = min(trajectory_entropies, key=lambda x: x["total_entropy"])

        if min_entropy_trajectory["is_correct"]:
            correct_selections += 1

        # Get the selected trajectory's text
        selected_trajectory = trajectories[min_entropy_trajectory["trajectory_idx"]]
        selected_text = selected_trajectory.get("generated_solution", "")

        results.append({
            "problem_id": problem_id,
            "selected_trajectory_idx": min_entropy_trajectory["trajectory_idx"],
            "selected_entropy": min_entropy_trajectory["total_entropy"],
            "is_correct": min_entropy_trajectory["is_correct"],
            "selected_text": selected_text,
            "num_trajectories": len(trajectories),
            "all_trajectories": trajectory_entropies
        })

    accuracy = correct_selections / total_problems if total_problems > 0 else 0.0

    return {
        "total_problems": total_problems,
        "correct_selections": correct_selections,
        "accuracy": accuracy,
        "details": results
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_entropy_selection.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    print(f"Analyzing {input_file}...")
    print("=" * 80)

    results = analyze_entropy_selection(input_file)

    print(f"\nResults:")
    print(f"  Total problems: {results['total_problems']}")
    print(f"  Correct selections: {results['correct_selections']}")
    print(f"  Accuracy: {results['accuracy']:.2%}")

    # Show some examples
    print(f"\n{'='*80}")
    print("Sample results (first 5 problems):")
    print(f"{'='*80}")
    for detail in results['details'][:5]:
        print(f"\nProblem: {detail['problem_id']}")
        print(f"  Selected trajectory: {detail['selected_trajectory_idx']} (entropy: {detail['selected_entropy']:.4f})")
        print(f"  Is correct: {detail['is_correct']}")
        print(f"  Total trajectories: {detail['num_trajectories']}")

        # Show entropy distribution
        entropies = sorted(detail['all_trajectories'], key=lambda x: x['total_entropy'])
        print(f"  Entropy range: {entropies[0]['total_entropy']:.4f} - {entropies[-1]['total_entropy']:.4f}")

        # Show selected trajectory text (truncated if too long)
        selected_text = detail.get('selected_text', '')
        if selected_text:
            truncated_text = selected_text[:200] + "..." if len(selected_text) > 200 else selected_text
            print(f"  Selected text: {truncated_text}")

    # Save detailed results
    output_file = Path(input_file).parent / f"{Path(input_file).stem}_entropy_selection.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
