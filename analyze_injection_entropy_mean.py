#!/usr/bin/env python3
"""
Analyze injection entropy patterns for different injection prompts using MEAN method.
Compares the discriminatory power of different injection types with length normalization.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats


# Injection types to analyze
INJECTION_TYPES = [
    'confidence_score',
    'correctness_prob',
    'error_likelihood',
    'step_quality',
    'binary_check',
    'revision_need'
]


def calculate_trajectory_injection_entropy(trajectory: Dict[str, Any], injection_type: str) -> float:
    """
    Calculate MEAN injection entropy for a trajectory (length-normalized).

    Args:
        trajectory: Trajectory data with steps
        injection_type: Type of injection (e.g., 'confidence_score')

    Returns:
        Mean of injection entropies across all steps
    """
    entropies = []

    for step in trajectory.get('steps', []):
        injection_results = step.get('injection_results', {})
        if injection_type in injection_results:
            entropy = injection_results[injection_type].get('entropy', np.nan)
            if not np.isnan(entropy):
                entropies.append(entropy)

    if not entropies:
        return np.nan

    return np.mean(entropies)


def analyze_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single problem's injection entropy patterns.

    Returns per-problem statistics for each injection type.
    """
    problem_id = problem['problem_id']
    trajectories = problem['trajectories']

    # Separate correct and incorrect trajectories
    correct_trajectories = [t for t in trajectories if t.get('is_correct', False)]
    incorrect_trajectories = [t for t in trajectories if not t.get('is_correct', False)]

    num_correct = len(correct_trajectories)
    num_incorrect = len(incorrect_trajectories)
    num_total = len(trajectories)
    pass_rate = num_correct / num_total if num_total > 0 else 0.0

    # Determine status
    if num_correct == 0:
        status = 'all_incorrect'
    elif num_incorrect == 0:
        status = 'all_correct'
    else:
        status = 'mixed'

    result = {
        'problem_id': problem_id,
        'num_trajectories': num_total,
        'num_correct': num_correct,
        'num_incorrect': num_incorrect,
        'pass_rate': pass_rate,
        'status': status
    }

    # Analyze each injection type
    for injection_type in INJECTION_TYPES:
        # Calculate entropies for all trajectories
        correct_entropies = [
            calculate_trajectory_injection_entropy(t, injection_type)
            for t in correct_trajectories
        ]
        correct_entropies = [e for e in correct_entropies if not np.isnan(e)]

        incorrect_entropies = [
            calculate_trajectory_injection_entropy(t, injection_type)
            for t in incorrect_trajectories
        ]
        incorrect_entropies = [e for e in incorrect_entropies if not np.isnan(e)]

        # Statistics for correct trajectories
        if correct_entropies:
            result[f'{injection_type}_correct_mean'] = np.mean(correct_entropies)
            result[f'{injection_type}_correct_median'] = np.median(correct_entropies)
            result[f'{injection_type}_correct_std'] = np.std(correct_entropies)
            result[f'{injection_type}_correct_min'] = np.min(correct_entropies)
            result[f'{injection_type}_correct_max'] = np.max(correct_entropies)
        else:
            result[f'{injection_type}_correct_mean'] = None
            result[f'{injection_type}_correct_median'] = None
            result[f'{injection_type}_correct_std'] = None
            result[f'{injection_type}_correct_min'] = None
            result[f'{injection_type}_correct_max'] = None

        # Statistics for incorrect trajectories
        if incorrect_entropies:
            result[f'{injection_type}_incorrect_mean'] = np.mean(incorrect_entropies)
            result[f'{injection_type}_incorrect_median'] = np.median(incorrect_entropies)
            result[f'{injection_type}_incorrect_std'] = np.std(incorrect_entropies)
            result[f'{injection_type}_incorrect_min'] = np.min(incorrect_entropies)
            result[f'{injection_type}_incorrect_max'] = np.max(incorrect_entropies)
        else:
            result[f'{injection_type}_incorrect_mean'] = None
            result[f'{injection_type}_incorrect_median'] = None
            result[f'{injection_type}_incorrect_std'] = None
            result[f'{injection_type}_incorrect_min'] = None
            result[f'{injection_type}_incorrect_max'] = None

        # Statistical comparison (only for mixed problems)
        if status == 'mixed' and len(correct_entropies) > 0 and len(incorrect_entropies) > 0:
            # T-test
            try:
                t_stat, p_value = stats.ttest_ind(correct_entropies, incorrect_entropies)
                result[f'{injection_type}_ttest_statistic'] = t_stat
                result[f'{injection_type}_ttest_pvalue'] = p_value
            except:
                result[f'{injection_type}_ttest_statistic'] = None
                result[f'{injection_type}_ttest_pvalue'] = None

            # Cohen's d effect size
            try:
                mean_diff = np.mean(correct_entropies) - np.mean(incorrect_entropies)
                pooled_std = np.sqrt(
                    ((len(correct_entropies) - 1) * np.var(correct_entropies) +
                     (len(incorrect_entropies) - 1) * np.var(incorrect_entropies)) /
                    (len(correct_entropies) + len(incorrect_entropies) - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                result[f'{injection_type}_cohens_d'] = cohens_d
            except:
                result[f'{injection_type}_cohens_d'] = None

            # Entropy difference (incorrect - correct)
            result[f'{injection_type}_entropy_diff'] = (
                np.mean(incorrect_entropies) - np.mean(correct_entropies)
            )
        else:
            result[f'{injection_type}_ttest_statistic'] = None
            result[f'{injection_type}_ttest_pvalue'] = None
            result[f'{injection_type}_cohens_d'] = None
            result[f'{injection_type}_entropy_diff'] = None

    return result


def categorize_pass_rate(pass_rate: float) -> str:
    """Categorize problem difficulty based on pass rate."""
    if pass_rate == 0:
        return "0% (All Incorrect)"
    elif pass_rate == 1.0:
        return "100% (All Correct)"
    elif pass_rate < 0.25:
        return "1-24% (Very Hard)"
    elif pass_rate < 0.5:
        return "25-49% (Hard)"
    elif pass_rate < 0.75:
        return "50-74% (Medium)"
    else:
        return "75-100% (Easy)"


def compare_injection_discriminators(df: pd.DataFrame, output_dir: Path):
    """
    Compare the discriminatory power of different injection types.
    """
    # Filter to only mixed problems
    mixed_df = df[df['status'] == 'mixed'].copy()

    print(f"\nAnalyzing {len(mixed_df)} problems with mixed results...")

    # For each injection type, calculate discrimination metrics
    injection_stats = []

    for injection_type in INJECTION_TYPES:
        entropy_diff_col = f'{injection_type}_entropy_diff'
        pvalue_col = f'{injection_type}_ttest_pvalue'
        cohens_d_col = f'{injection_type}_cohens_d'

        # Get valid data (non-null values)
        valid_data = mixed_df[
            mixed_df[entropy_diff_col].notna() &
            mixed_df[pvalue_col].notna() &
            mixed_df[cohens_d_col].notna()
        ].copy()

        if len(valid_data) == 0:
            continue

        # Count discrimination patterns
        expected_pattern = (valid_data[entropy_diff_col] > 0).sum()  # Incorrect > Correct
        opposite_pattern = (valid_data[entropy_diff_col] <= 0).sum()  # Correct > Incorrect

        expected_pct = (expected_pattern / len(valid_data)) * 100
        opposite_pct = (opposite_pattern / len(valid_data)) * 100

        # Count significant differences
        significant = (valid_data[pvalue_col] < 0.05).sum()
        significant_pct = (significant / len(valid_data)) * 100

        # Effect size statistics
        mean_abs_cohens_d = valid_data[cohens_d_col].abs().mean()
        median_abs_cohens_d = valid_data[cohens_d_col].abs().median()

        # Mean entropy difference
        mean_entropy_diff = valid_data[entropy_diff_col].mean()
        median_entropy_diff = valid_data[entropy_diff_col].median()

        injection_stats.append({
            'injection_type': injection_type,
            'num_problems': len(valid_data),
            'expected_pattern_count': expected_pattern,
            'expected_pattern_pct': expected_pct,
            'opposite_pattern_count': opposite_pattern,
            'opposite_pattern_pct': opposite_pct,
            'significant_count': significant,
            'significant_pct': significant_pct,
            'mean_abs_cohens_d': mean_abs_cohens_d,
            'median_abs_cohens_d': median_abs_cohens_d,
            'mean_entropy_diff': mean_entropy_diff,
            'median_entropy_diff': median_entropy_diff
        })

    stats_df = pd.DataFrame(injection_stats)

    # Save comparison results
    stats_df.to_csv(output_dir / 'injection_comparison_mean.csv', index=False)

    return stats_df


def analyze_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze injection entropy discrimination by difficulty category.

    Returns DataFrame with per-difficulty, per-injection-type statistics.
    """
    mixed_df = df[df['status'] == 'mixed'].copy()

    difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']

    results = []

    for difficulty in difficulty_order:
        diff_df = mixed_df[mixed_df['difficulty_category'] == difficulty]

        if len(diff_df) == 0:
            continue

        for injection_type in INJECTION_TYPES:
            entropy_diff_col = f'{injection_type}_entropy_diff'
            pvalue_col = f'{injection_type}_ttest_pvalue'
            cohens_d_col = f'{injection_type}_cohens_d'

            # Get valid data
            valid_data = diff_df[
                diff_df[entropy_diff_col].notna() &
                diff_df[pvalue_col].notna() &
                diff_df[cohens_d_col].notna()
            ].copy()

            if len(valid_data) == 0:
                continue

            # Calculate statistics
            expected_pattern = (valid_data[entropy_diff_col] > 0).sum()
            opposite_pattern = (valid_data[entropy_diff_col] <= 0).sum()
            expected_pct = (expected_pattern / len(valid_data)) * 100

            significant = (valid_data[pvalue_col] < 0.05).sum()
            significant_pct = (significant / len(valid_data)) * 100

            mean_entropy_diff = valid_data[entropy_diff_col].mean()
            median_entropy_diff = valid_data[entropy_diff_col].median()
            mean_abs_cohens_d = valid_data[cohens_d_col].abs().mean()

            results.append({
                'difficulty_category': difficulty,
                'injection_type': injection_type,
                'num_problems': len(valid_data),
                'expected_pattern_count': expected_pattern,
                'expected_pattern_pct': expected_pct,
                'opposite_pattern_count': opposite_pattern,
                'significant_count': significant,
                'significant_pct': significant_pct,
                'mean_entropy_diff': mean_entropy_diff,
                'median_entropy_diff': median_entropy_diff,
                'mean_abs_cohens_d': mean_abs_cohens_d
            })

    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, injection_stats_df: pd.DataFrame,
                   difficulty_stats_df: pd.DataFrame, output_path: Path):
    """Generate comprehensive text report with MEAN method results."""
    mixed_df = df[df['status'] == 'mixed']

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INJECTION ENTROPY DISCRIMINATION ANALYSIS (MEAN METHOD)\n")
        f.write("=" * 80 + "\n\n")

        f.write("MEASUREMENT METHOD: Mean (length-normalized)\n")
        f.write("-" * 80 + "\n")
        f.write("This analysis uses MEAN entropy instead of SUM entropy.\n")
        f.write("Entropies are averaged across steps, removing length bias.\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total problems: {len(df)}\n")
        f.write(f"Mixed problems (both correct/incorrect): {len(mixed_df)}\n")
        f.write(f"All correct problems: {(df['status'] == 'all_correct').sum()}\n")
        f.write(f"All incorrect problems: {(df['status'] == 'all_incorrect').sum()}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INJECTION TYPE COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write("Ranking by discriminatory power (percentage showing expected pattern):\n")
        f.write("-" * 80 + "\n\n")

        # Sort by expected pattern percentage
        sorted_stats = injection_stats_df.sort_values('expected_pattern_pct', ascending=False)

        for idx, row in sorted_stats.iterrows():
            f.write(f"{row['injection_type']}:\n")
            f.write(f"  Expected pattern (Incorrect > Correct): {row['expected_pattern_count']} ({row['expected_pattern_pct']:.1f}%)\n")
            f.write(f"  Opposite pattern (Correct > Incorrect): {row['opposite_pattern_count']} ({row['opposite_pattern_pct']:.1f}%)\n")
            f.write(f"  Statistically significant: {row['significant_count']} ({row['significant_pct']:.1f}%)\n")
            f.write(f"  Mean |Cohen's d|: {row['mean_abs_cohens_d']:.4f}\n")
            f.write(f"  Mean entropy diff: {row['mean_entropy_diff']:.4f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Find best and worst discriminators
        best = sorted_stats.iloc[0]
        worst = sorted_stats.iloc[-1]

        f.write(f"BEST DISCRIMINATOR: {best['injection_type']}\n")
        f.write(f"  - {best['expected_pattern_pct']:.1f}% show expected pattern\n")
        f.write(f"  - {best['significant_pct']:.1f}% statistically significant\n")
        f.write(f"  - Mean effect size: {best['mean_abs_cohens_d']:.4f}\n\n")

        f.write(f"WORST DISCRIMINATOR: {worst['injection_type']}\n")
        f.write(f"  - {worst['expected_pattern_pct']:.1f}% show expected pattern\n")
        f.write(f"  - {worst['significant_pct']:.1f}% statistically significant\n")
        f.write(f"  - Mean effect size: {worst['mean_abs_cohens_d']:.4f}\n\n")

        # Overall assessment
        f.write("\nOVERALL ASSESSMENT:\n")
        f.write("-" * 80 + "\n")

        avg_expected = sorted_stats['expected_pattern_pct'].mean()
        avg_significant = sorted_stats['significant_pct'].mean()

        f.write(f"Average expected pattern across all injection types: {avg_expected:.1f}%\n")
        f.write(f"Average statistical significance: {avg_significant:.1f}%\n\n")

        if avg_expected < 60:
            f.write("CONCLUSION: Injection entropies show WEAK discrimination overall.\n")
        elif avg_expected < 75:
            f.write("CONCLUSION: Injection entropies show MODERATE discrimination.\n")
        else:
            f.write("CONCLUSION: Injection entropies show STRONG discrimination.\n")

        f.write(f"\nNOTE: Mean method (length-normalized) should be compared with\n")
        f.write(f"Sum method results to understand the impact of length bias.\n")

        # Difficulty-based analysis
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("ANALYSIS BY PROBLEM DIFFICULTY\n")
        f.write("=" * 80 + "\n\n")

        difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']

        for difficulty in difficulty_order:
            diff_data = difficulty_stats_df[difficulty_stats_df['difficulty_category'] == difficulty]

            if len(diff_data) == 0:
                continue

            f.write(f"\n{difficulty}\n")
            f.write("-" * 80 + "\n\n")

            # Sort by expected pattern percentage for this difficulty
            diff_sorted = diff_data.sort_values('expected_pattern_pct', ascending=False)

            for idx, row in diff_sorted.iterrows():
                f.write(f"{row['injection_type']}:\n")
                f.write(f"  Problems analyzed: {row['num_problems']}\n")
                f.write(f"  Expected pattern: {row['expected_pattern_count']} ({row['expected_pattern_pct']:.1f}%)\n")
                f.write(f"  Opposite pattern: {row['opposite_pattern_count']}\n")
                f.write(f"  Statistically significant: {row['significant_count']} ({row['significant_pct']:.1f}%)\n")
                f.write(f"  Mean entropy diff: {row['mean_entropy_diff']:.4f}\n")
                f.write(f"  Mean |Cohen's d|: {row['mean_abs_cohens_d']:.4f}\n\n")

    print(f"\nReport saved to: {output_path}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_injection_entropy_mean.py <input_json>")
        print("Example: python analyze_injection_entropy_mean.py results/pass_at_32.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = input_path.parent / f"{input_path.stem}_injection_analysis_mean"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("INJECTION ENTROPY ANALYSIS (MEAN METHOD)")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Method: MEAN (length-normalized)\n")

    # Load data
    print("Loading data...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems")

    # Analyze each problem
    print("\nAnalyzing problems...")
    results = []
    for problem in data:
        result = analyze_problem(problem)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add difficulty categories
    df['difficulty_category'] = df['pass_rate'].apply(categorize_pass_rate)

    # Save detailed results
    csv_path = output_dir / 'problem_injection_analysis_mean.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Compare injection types
    print("\nComparing injection types...")
    injection_stats_df = compare_injection_discriminators(df, output_dir)

    # Analyze by difficulty
    print("\nAnalyzing by difficulty...")
    difficulty_stats_df = analyze_by_difficulty(df)
    difficulty_csv_path = output_dir / 'difficulty_comparison_mean.csv'
    difficulty_stats_df.to_csv(difficulty_csv_path, index=False)
    print(f"Difficulty comparison saved to: {difficulty_csv_path}")

    # Generate report
    print("\nGenerating report...")
    report_path = output_dir / 'injection_analysis_report_mean.txt'
    generate_report(df, injection_stats_df, difficulty_stats_df, report_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey files generated:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    print(f"  - {output_dir}/injection_comparison_mean.csv")
    print(f"  - {difficulty_csv_path}")


if __name__ == "__main__":
    main()
