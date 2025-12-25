#!/usr/bin/env python3
"""
Analyze entropy characteristics of correct vs incorrect trajectories by problem difficulty.

This script analyzes:
1. Per-problem trajectory correctness patterns
2. Per-problem accuracy (pass rate)
3. Entropy comparison between correct and incorrect trajectories
4. Statistical analysis by problem difficulty (based on pass rate)

Usage:
    uv run python analyze_correctness_entropy.py results/pass_at_16.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def calculate_trajectory_entropy(trajectory: Dict[str, Any], method: str = 'mean') -> float:
    """
    Calculate entropy for a trajectory.

    Args:
        trajectory: Trajectory dictionary with steps
        method: 'mean' (average of step entropies) or 'sum' (total entropy)

    Returns:
        Calculated entropy value
    """
    steps = trajectory.get('steps', [])
    if not steps:
        return np.nan

    entropies = [step.get('avg_entropy', np.nan) for step in steps]
    entropies = [e for e in entropies if not np.isnan(e)]

    if not entropies:
        return np.nan

    if method == 'mean':
        return np.mean(entropies)
    elif method == 'sum':
        return np.sum(entropies)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_injection_entropy(trajectory: Dict[str, Any],
                                injection_type: str = 'correctness_prob',
                                method: str = 'mean') -> float:
    """
    Calculate injection prompt entropy for a trajectory.

    Args:
        trajectory: Trajectory dictionary with steps
        injection_type: Type of injection prompt (e.g., 'correctness_prob', 'confidence_score')
        method: 'mean' or 'sum'

    Returns:
        Calculated injection entropy value
    """
    steps = trajectory.get('steps', [])
    if not steps:
        return np.nan

    entropies = []
    for step in steps:
        injection_results = step.get('injection_results', {})
        if injection_type in injection_results:
            entropy = injection_results[injection_type].get('entropy', np.nan)
            if not np.isnan(entropy):
                entropies.append(entropy)

    if not entropies:
        return np.nan

    if method == 'mean':
        return np.mean(entropies)
    elif method == 'sum':
        return np.sum(entropies)
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_problem(problem: Dict[str, Any],
                    entropy_method: str = 'mean',
                    injection_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a single problem's trajectories.

    Args:
        problem: Problem dictionary with trajectories
        entropy_method: 'mean' or 'sum' for entropy calculation
        injection_type: Optional injection prompt type to analyze

    Returns:
        Dictionary with analysis results
    """
    problem_id = problem.get('problem_id', 'unknown')
    trajectories = problem.get('trajectories', [])

    if not trajectories:
        return {
            'problem_id': problem_id,
            'num_trajectories': 0,
            'num_correct': 0,
            'num_incorrect': 0,
            'pass_rate': 0.0,
            'status': 'no_trajectories'
        }

    # Separate correct and incorrect trajectories
    correct_trajectories = [t for t in trajectories if t.get('is_correct', False)]
    incorrect_trajectories = [t for t in trajectories if not t.get('is_correct', False)]

    num_correct = len(correct_trajectories)
    num_incorrect = len(incorrect_trajectories)
    pass_rate = num_correct / len(trajectories) if trajectories else 0.0

    # Determine status
    if num_correct == 0:
        status = 'all_incorrect'
    elif num_incorrect == 0:
        status = 'all_correct'
    else:
        status = 'mixed'

    # Calculate entropies
    result = {
        'problem_id': problem_id,
        'num_trajectories': len(trajectories),
        'num_correct': num_correct,
        'num_incorrect': num_incorrect,
        'pass_rate': pass_rate,
        'status': status,
    }

    # Calculate average token entropy statistics
    if num_correct > 0:
        correct_entropies = [calculate_trajectory_entropy(t, entropy_method)
                           for t in correct_trajectories]
        correct_entropies = [e for e in correct_entropies if not np.isnan(e)]

        if correct_entropies:
            result['correct_entropy_mean'] = np.mean(correct_entropies)
            result['correct_entropy_median'] = np.median(correct_entropies)
            result['correct_entropy_std'] = np.std(correct_entropies, ddof=1) if len(correct_entropies) > 1 else 0.0
            result['correct_entropy_min'] = np.min(correct_entropies)
            result['correct_entropy_max'] = np.max(correct_entropies)

    if num_incorrect > 0:
        incorrect_entropies = [calculate_trajectory_entropy(t, entropy_method)
                             for t in incorrect_trajectories]
        incorrect_entropies = [e for e in incorrect_entropies if not np.isnan(e)]

        if incorrect_entropies:
            result['incorrect_entropy_mean'] = np.mean(incorrect_entropies)
            result['incorrect_entropy_median'] = np.median(incorrect_entropies)
            result['incorrect_entropy_std'] = np.std(incorrect_entropies, ddof=1) if len(incorrect_entropies) > 1 else 0.0
            result['incorrect_entropy_min'] = np.min(incorrect_entropies)
            result['incorrect_entropy_max'] = np.max(incorrect_entropies)

    # Calculate injection entropy statistics if specified
    if injection_type:
        if num_correct > 0:
            correct_inj_entropies = [calculate_injection_entropy(t, injection_type, entropy_method)
                                   for t in correct_trajectories]
            correct_inj_entropies = [e for e in correct_inj_entropies if not np.isnan(e)]

            if correct_inj_entropies:
                result[f'correct_{injection_type}_mean'] = np.mean(correct_inj_entropies)
                result[f'correct_{injection_type}_median'] = np.median(correct_inj_entropies)
                result[f'correct_{injection_type}_std'] = np.std(correct_inj_entropies, ddof=1) if len(correct_inj_entropies) > 1 else 0.0

        if num_incorrect > 0:
            incorrect_inj_entropies = [calculate_injection_entropy(t, injection_type, entropy_method)
                                     for t in incorrect_trajectories]
            incorrect_inj_entropies = [e for e in incorrect_inj_entropies if not np.isnan(e)]

            if incorrect_inj_entropies:
                result[f'incorrect_{injection_type}_mean'] = np.mean(incorrect_inj_entropies)
                result[f'incorrect_{injection_type}_median'] = np.median(incorrect_inj_entropies)
                result[f'incorrect_{injection_type}_std'] = np.std(incorrect_inj_entropies, ddof=1) if len(incorrect_inj_entropies) > 1 else 0.0

    # Statistical test if both correct and incorrect exist
    if status == 'mixed':
        correct_entropies = [calculate_trajectory_entropy(t, entropy_method)
                           for t in correct_trajectories]
        incorrect_entropies = [calculate_trajectory_entropy(t, entropy_method)
                             for t in incorrect_trajectories]

        correct_entropies = [e for e in correct_entropies if not np.isnan(e)]
        incorrect_entropies = [e for e in incorrect_entropies if not np.isnan(e)]

        if len(correct_entropies) > 0 and len(incorrect_entropies) > 0:
            # t-test
            try:
                t_stat, p_value = stats.ttest_ind(correct_entropies, incorrect_entropies)
                result['ttest_statistic'] = t_stat
                result['ttest_pvalue'] = p_value
            except:
                pass

            # Cohen's d (effect size)
            try:
                mean_correct = np.mean(correct_entropies)
                mean_incorrect = np.mean(incorrect_entropies)
                std_correct = np.std(correct_entropies, ddof=1)
                std_incorrect = np.std(incorrect_entropies, ddof=1)

                n_correct = len(correct_entropies)
                n_incorrect = len(incorrect_entropies)

                pooled_std = np.sqrt(
                    ((n_correct - 1) * std_correct**2 + (n_incorrect - 1) * std_incorrect**2) /
                    (n_correct + n_incorrect - 2)
                )

                if pooled_std > 0:
                    cohens_d = (mean_correct - mean_incorrect) / pooled_std
                    result['cohens_d'] = cohens_d
            except:
                pass

    return result


def analyze_all_problems(data: List[Dict[str, Any]],
                        entropy_method: str = 'mean',
                        injection_type: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze all problems and return as DataFrame.

    Args:
        data: List of problem dictionaries
        entropy_method: 'mean' or 'sum' for entropy calculation
        injection_type: Optional injection prompt type to analyze

    Returns:
        DataFrame with per-problem analysis
    """
    results = []
    for problem in data:
        result = analyze_problem(problem, entropy_method, injection_type)
        results.append(result)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print(f"Total problems: {len(df)}")
    print(f"Problems with mixed results: {len(df[df['status'] == 'mixed'])}")
    print(f"Problems with all correct: {len(df[df['status'] == 'all_correct'])}")
    print(f"Problems with all incorrect: {len(df[df['status'] == 'all_incorrect'])}")
    print()

    print("Pass rate distribution:")
    print(f"  Mean: {df['pass_rate'].mean():.2%}")
    print(f"  Median: {df['pass_rate'].median():.2%}")
    print(f"  Std: {df['pass_rate'].std():.2%}")
    print(f"  Min: {df['pass_rate'].min():.2%}")
    print(f"  Max: {df['pass_rate'].max():.2%}")
    print()

    # Entropy comparison for mixed problems
    mixed_df = df[df['status'] == 'mixed'].copy()
    if len(mixed_df) > 0:
        print("=" * 80)
        print("ENTROPY COMPARISON (Mixed Problems Only)")
        print("=" * 80)
        print()

        print(f"Number of problems with both correct and incorrect: {len(mixed_df)}")
        print()

        # Average token entropy
        print("Average Token Entropy:")
        print("-" * 40)
        print(f"  Correct trajectories:")
        print(f"    Mean: {mixed_df['correct_entropy_mean'].mean():.4f} ± {mixed_df['correct_entropy_mean'].std():.4f}")
        print(f"    Median: {mixed_df['correct_entropy_median'].mean():.4f}")
        print()
        print(f"  Incorrect trajectories:")
        print(f"    Mean: {mixed_df['incorrect_entropy_mean'].mean():.4f} ± {mixed_df['incorrect_entropy_mean'].std():.4f}")
        print(f"    Median: {mixed_df['incorrect_entropy_median'].mean():.4f}")
        print()

        # Difference
        mean_diff = mixed_df['incorrect_entropy_mean'].mean() - mixed_df['correct_entropy_mean'].mean()
        print(f"  Difference (Incorrect - Correct): {mean_diff:.4f}")
        print(f"    {'Incorrect has HIGHER entropy (expected)' if mean_diff > 0 else 'Incorrect has LOWER entropy (unexpected)'}")
        print()

        # Statistical significance
        if 'cohens_d' in mixed_df.columns:
            avg_cohens_d = mixed_df['cohens_d'].mean()
            print(f"  Average Cohen's d: {avg_cohens_d:.4f}")
            print(f"    Effect size: ", end="")
            if abs(avg_cohens_d) < 0.2:
                print("negligible")
            elif abs(avg_cohens_d) < 0.5:
                print("small")
            elif abs(avg_cohens_d) < 0.8:
                print("medium")
            else:
                print("large")
            print()

        # Significant differences
        if 'ttest_pvalue' in mixed_df.columns:
            significant = mixed_df[mixed_df['ttest_pvalue'] < 0.05]
            print(f"  Problems with significant difference (p < 0.05): {len(significant)}/{len(mixed_df)}")
            print()


def create_visualizations(df: pd.DataFrame, output_dir: str = "results/correctness_analysis"):
    """Create visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    # 1. Pass rate distribution
    print("  - Pass rate distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['pass_rate'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pass Rate')
    ax.set_ylabel('Number of Problems')
    ax.set_title('Distribution of Problem Pass Rates')
    ax.axvline(df['pass_rate'].mean(), color='red', linestyle='--', label=f'Mean: {df["pass_rate"].mean():.2%}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'pass_rate_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Entropy comparison for mixed problems
    mixed_df = df[df['status'] == 'mixed'].copy()
    if len(mixed_df) > 0:
        print("  - Entropy comparison (correct vs incorrect)")

        # Prepare data for box plot
        entropy_data = []
        for _, row in mixed_df.iterrows():
            entropy_data.append({
                'Problem': row['problem_id'],
                'Correctness': 'Correct',
                'Entropy (Mean)': row['correct_entropy_mean'],
                'Entropy (Median)': row['correct_entropy_median']
            })
            entropy_data.append({
                'Problem': row['problem_id'],
                'Correctness': 'Incorrect',
                'Entropy (Mean)': row['incorrect_entropy_mean'],
                'Entropy (Median)': row['incorrect_entropy_median']
            })

        plot_df = pd.DataFrame(entropy_data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Mean entropy
        sns.boxplot(data=plot_df, x='Correctness', y='Entropy (Mean)', ax=axes[0], palette='Set2')
        sns.stripplot(data=plot_df, x='Correctness', y='Entropy (Mean)', ax=axes[0],
                     color='black', alpha=0.5, size=4)
        axes[0].set_title('Mean Entropy: Correct vs Incorrect')
        axes[0].set_ylabel('Mean Entropy per Trajectory')

        # Median entropy
        sns.boxplot(data=plot_df, x='Correctness', y='Entropy (Median)', ax=axes[1], palette='Set2')
        sns.stripplot(data=plot_df, x='Correctness', y='Entropy (Median)', ax=axes[1],
                     color='black', alpha=0.5, size=4)
        axes[1].set_title('Median Entropy: Correct vs Incorrect')
        axes[1].set_ylabel('Median Entropy per Trajectory')

        plt.suptitle('Entropy Comparison: Correct vs Incorrect Trajectories', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Entropy vs pass rate
        print("  - Entropy vs pass rate")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Correct entropy vs pass rate
        if 'correct_entropy_mean' in mixed_df.columns:
            axes[0].scatter(mixed_df['pass_rate'], mixed_df['correct_entropy_mean'],
                          alpha=0.6, s=100, color='green')
            axes[0].set_xlabel('Pass Rate')
            axes[0].set_ylabel('Mean Entropy (Correct Trajectories)')
            axes[0].set_title('Correct Trajectory Entropy vs Problem Difficulty')

            # Add trend line
            z = np.polyfit(mixed_df['pass_rate'], mixed_df['correct_entropy_mean'], 1)
            p = np.poly1d(z)
            axes[0].plot(mixed_df['pass_rate'], p(mixed_df['pass_rate']),
                        "r--", alpha=0.5, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            axes[0].legend()

        # Incorrect entropy vs pass rate
        if 'incorrect_entropy_mean' in mixed_df.columns:
            axes[1].scatter(mixed_df['pass_rate'], mixed_df['incorrect_entropy_mean'],
                          alpha=0.6, s=100, color='red')
            axes[1].set_xlabel('Pass Rate')
            axes[1].set_ylabel('Mean Entropy (Incorrect Trajectories)')
            axes[1].set_title('Incorrect Trajectory Entropy vs Problem Difficulty')

            # Add trend line
            z = np.polyfit(mixed_df['pass_rate'], mixed_df['incorrect_entropy_mean'], 1)
            p = np.poly1d(z)
            axes[1].plot(mixed_df['pass_rate'], p(mixed_df['pass_rate']),
                        "r--", alpha=0.5, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            axes[1].legend()

        plt.suptitle('Entropy vs Problem Difficulty (Pass Rate as Proxy)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_vs_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Effect size by problem
        if 'cohens_d' in mixed_df.columns:
            print("  - Effect size by problem")

            fig, ax = plt.subplots(figsize=(12, 6))

            sorted_mixed = mixed_df.sort_values('cohens_d')
            colors = ['red' if d < 0 else 'green' for d in sorted_mixed['cohens_d']]

            ax.barh(range(len(sorted_mixed)), sorted_mixed['cohens_d'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(sorted_mixed)))
            ax.set_yticklabels(sorted_mixed['problem_id'])
            ax.set_xlabel("Cohen's d (Correct - Incorrect)")
            ax.set_title("Effect Size by Problem\n(Negative = Incorrect has higher entropy)")
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'effect_size_by_problem.png', dpi=300, bbox_inches='tight')
            plt.close()

    print()
    print(f"Visualizations saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze entropy characteristics of correct vs incorrect trajectories"
    )
    parser.add_argument("input_file", type=str, help="Path to pass@k JSON file")
    parser.add_argument("--entropy-method", type=str, default="mean",
                       choices=["mean", "sum"],
                       help="Method to calculate trajectory entropy (default: mean)")
    parser.add_argument("--injection-type", type=str, default=None,
                       help="Optional: Analyze specific injection prompt type")
    parser.add_argument("--output-dir", type=str, default="results/correctness_analysis",
                       help="Output directory for results")

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Load data
    print(f"Loading data from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")
    print()

    # Analyze all problems
    print("Analyzing problems...")
    df = analyze_all_problems(data, args.entropy_method, args.injection_type)

    # Print summary
    print_summary(df)

    # Create visualizations
    create_visualizations(df, args.output_dir)

    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "problem_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    print()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
