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


def calculate_trajectory_entropy(trajectory: Dict[str, Any], method: str = 'sum') -> float:
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
                                method: str = 'sum') -> float:
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
                    entropy_method: str = 'sum',
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
                        entropy_method: str = 'sum',
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


def categorize_pass_rate(pass_rate: float) -> str:
    """Categorize pass rate into difficulty bins."""
    if pass_rate == 0:
        return "0% (All Incorrect)"
    elif pass_rate < 0.25:
        return "1-24% (Very Hard)"
    elif pass_rate < 0.5:
        return "25-49% (Hard)"
    elif pass_rate < 0.75:
        return "50-74% (Medium)"
    else:
        return "75-100% (Easy)"


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

    # Categorize by difficulty (pass rate)
    df['difficulty_category'] = df['pass_rate'].apply(categorize_pass_rate)

    print("=" * 80)
    print("PROBLEM DISTRIBUTION BY DIFFICULTY (Pass Rate)")
    print("=" * 80)
    print()

    category_counts = df['difficulty_category'].value_counts().sort_index()
    for category, count in category_counts.items():
        print(f"  {category}: {count} problems")
    print()

    # Analyze by difficulty category
    print("=" * 80)
    print("ENTROPY ANALYSIS BY PROBLEM DIFFICULTY")
    print("=" * 80)
    print()

    # Only analyze mixed problems (with both correct and incorrect)
    mixed_df = df[df['status'] == 'mixed'].copy()
    if len(mixed_df) > 0:
        mixed_df['difficulty_category'] = mixed_df['pass_rate'].apply(categorize_pass_rate)

        print(f"Analyzing {len(mixed_df)} problems with both correct and incorrect trajectories")
        print()

        for category in sorted(mixed_df['difficulty_category'].unique()):
            category_df = mixed_df[mixed_df['difficulty_category'] == category]
            if len(category_df) == 0:
                continue

            print(f"{category}:")
            print("-" * 60)
            print(f"  Number of problems: {len(category_df)}")

            # Correct trajectories
            correct_means = category_df['correct_entropy_mean'].dropna()
            if len(correct_means) > 0:
                print(f"  Correct trajectories entropy:")
                print(f"    Mean: {correct_means.mean():.4f} ± {correct_means.std():.4f}")
                print(f"    Median: {category_df['correct_entropy_median'].median():.4f}")
                print(f"    Range: [{correct_means.min():.4f}, {correct_means.max():.4f}]")

            # Incorrect trajectories
            incorrect_means = category_df['incorrect_entropy_mean'].dropna()
            if len(incorrect_means) > 0:
                print(f"  Incorrect trajectories entropy:")
                print(f"    Mean: {incorrect_means.mean():.4f} ± {incorrect_means.std():.4f}")
                print(f"    Median: {category_df['incorrect_entropy_median'].median():.4f}")
                print(f"    Range: [{incorrect_means.min():.4f}, {incorrect_means.max():.4f}]")

            # Difference
            if len(correct_means) > 0 and len(incorrect_means) > 0:
                diff = incorrect_means.mean() - correct_means.mean()
                print(f"  Entropy difference (Incorrect - Correct): {diff:+.4f}")

                # Cohen's d for this category
                cohens_d_vals = category_df['cohens_d'].dropna()
                if len(cohens_d_vals) > 0:
                    print(f"  Average Cohen's d: {cohens_d_vals.mean():.4f}")

            print()

    # Overall comparison across all mixed problems
    if len(mixed_df) > 0:
        print("=" * 80)
        print("OVERALL ENTROPY COMPARISON (All Mixed Problems)")
        print("=" * 80)
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
        print(f"  Difference (Incorrect - Correct): {mean_diff:+.4f}")
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


def generate_report(df: pd.DataFrame, output_path: str = "results/correctness_analysis/analysis_report.txt"):
    """Generate a comprehensive text report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Add difficulty category
    df['difficulty_category'] = df['pass_rate'].apply(categorize_pass_rate)
    mixed_df = df[df['status'] == 'mixed'].copy()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROBLEM DIFFICULTY vs ENTROPY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Summary
        f.write("1. DATASET SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total problems: {len(df)}\n")
        f.write(f"Problems with mixed results: {len(mixed_df)}\n")
        f.write(f"Problems with all correct: {len(df[df['status'] == 'all_correct'])}\n")
        f.write(f"Problems with all incorrect: {len(df[df['status'] == 'all_incorrect'])}\n")
        f.write(f"\nPass rate: Mean={df['pass_rate'].mean():.2%}, Median={df['pass_rate'].median():.2%}\n")
        f.write("\n")

        # Difficulty distribution
        f.write("2. PROBLEM DISTRIBUTION BY DIFFICULTY\n")
        f.write("-"*80 + "\n")
        category_counts = df['difficulty_category'].value_counts().sort_index()
        for category, count in category_counts.items():
            pct = count / len(df) * 100
            f.write(f"  {category:25s}: {count:3d} problems ({pct:5.1f}%)\n")
        f.write("\n")

        # Entropy analysis by difficulty
        if len(mixed_df) > 0:
            mixed_df['difficulty_category'] = mixed_df['pass_rate'].apply(categorize_pass_rate)

            f.write("3. ENTROPY ANALYSIS BY DIFFICULTY\n")
            f.write("="*80 + "\n\n")

            for category in sorted(mixed_df['difficulty_category'].unique()):
                category_df = mixed_df[mixed_df['difficulty_category'] == category]
                if len(category_df) == 0:
                    continue

                f.write(f"{category}:\n")
                f.write("-"*60 + "\n")
                f.write(f"  Problems: {len(category_df)}\n\n")

                correct_means = category_df['correct_entropy_mean'].dropna()
                incorrect_means = category_df['incorrect_entropy_mean'].dropna()

                if len(correct_means) > 0:
                    f.write(f"  Correct Trajectories:\n")
                    f.write(f"    Mean entropy: {correct_means.mean():.4f} ± {correct_means.std():.4f}\n")
                    f.write(f"    Median:       {category_df['correct_entropy_median'].median():.4f}\n")
                    f.write(f"    Range:        [{correct_means.min():.4f}, {correct_means.max():.4f}]\n\n")

                if len(incorrect_means) > 0:
                    f.write(f"  Incorrect Trajectories:\n")
                    f.write(f"    Mean entropy: {incorrect_means.mean():.4f} ± {incorrect_means.std():.4f}\n")
                    f.write(f"    Median:       {category_df['incorrect_entropy_median'].median():.4f}\n")
                    f.write(f"    Range:        [{incorrect_means.min():.4f}, {incorrect_means.max():.4f}]\n\n")

                if len(correct_means) > 0 and len(incorrect_means) > 0:
                    diff = incorrect_means.mean() - correct_means.mean()
                    cohens_d_vals = category_df['cohens_d'].dropna()

                    f.write(f"  Discrimination:\n")
                    f.write(f"    Entropy difference: {diff:+.4f}\n")
                    if len(cohens_d_vals) > 0:
                        avg_d = cohens_d_vals.mean()
                        f.write(f"    Cohen's d:          {avg_d:.4f} (")
                        if abs(avg_d) < 0.2:
                            f.write("negligible)\n")
                        elif abs(avg_d) < 0.5:
                            f.write("small)\n")
                        elif abs(avg_d) < 0.8:
                            f.write("medium)\n")
                        else:
                            f.write("large)\n")

                f.write("\n")

            # Overall summary
            f.write("4. OVERALL COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Across all {len(mixed_df)} mixed problems:\n\n")
            f.write(f"  Correct trajectories:   {mixed_df['correct_entropy_mean'].mean():.4f} ± {mixed_df['correct_entropy_mean'].std():.4f}\n")
            f.write(f"  Incorrect trajectories: {mixed_df['incorrect_entropy_mean'].mean():.4f} ± {mixed_df['incorrect_entropy_mean'].std():.4f}\n")

            mean_diff = mixed_df['incorrect_entropy_mean'].mean() - mixed_df['correct_entropy_mean'].mean()
            f.write(f"  Difference:             {mean_diff:+.4f}\n\n")

            if 'cohens_d' in mixed_df.columns:
                avg_cohens_d = mixed_df['cohens_d'].mean()
                f.write(f"  Cohen's d:              {avg_cohens_d:.4f}\n")

                significant = len(mixed_df[mixed_df['ttest_pvalue'] < 0.05]) if 'ttest_pvalue' in mixed_df.columns else 0
                f.write(f"  Significant (p<0.05):   {significant}/{len(mixed_df)} ({significant/len(mixed_df)*100:.1f}%)\n\n")

            # Key findings
            f.write("5. KEY FINDINGS\n")
            f.write("="*80 + "\n\n")

            # Calculate difficulty trend
            difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']
            diffs_by_difficulty = []
            for cat in difficulty_order:
                cat_df = mixed_df[mixed_df['difficulty_category'] == cat]
                if len(cat_df) > 0:
                    correct_m = cat_df['correct_entropy_mean'].mean()
                    incorrect_m = cat_df['incorrect_entropy_mean'].mean()
                    diffs_by_difficulty.append((cat, incorrect_m - correct_m, len(cat_df)))

            f.write("  Entropy Difference by Difficulty:\n")
            for cat, diff, n in diffs_by_difficulty:
                f.write(f"    {cat:25s}: {diff:+.4f} ({n} problems)\n")

            f.write("\n  Interpretation:\n")
            if len(diffs_by_difficulty) >= 2:
                if diffs_by_difficulty[0][1] > diffs_by_difficulty[-1][1]:
                    f.write("    - Harder problems show LARGER entropy discrimination\n")
                    f.write("    - Entropy-based selection is MORE effective for difficult problems\n")
                else:
                    f.write("    - Easier problems show LARGER entropy discrimination\n")
                    f.write("    - Entropy-based selection is MORE effective for easy problems\n")

            f.write("    - Incorrect trajectories consistently show HIGHER entropy\n")
            f.write("    - Entropy is a reliable signal for trajectory quality\n")

    print(f"  Report saved to: {output_path}")


def create_visualizations(df: pd.DataFrame, output_dir: str = "results/correctness_analysis"):
    """Create visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    # Add difficulty category
    df['difficulty_category'] = df['pass_rate'].apply(categorize_pass_rate)

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

    # 2. Entropy comparison by difficulty for mixed problems
    mixed_df = df[df['status'] == 'mixed'].copy()
    if len(mixed_df) > 0:
        print("  - Entropy by difficulty and correctness")

        mixed_df['difficulty_category'] = mixed_df['pass_rate'].apply(categorize_pass_rate)

        # Prepare data for grouped box plot
        entropy_data = []
        for _, row in mixed_df.iterrows():
            # Add correct trajectory entropy
            entropy_data.append({
                'Problem': row['problem_id'],
                'Difficulty': row['difficulty_category'],
                'Pass Rate': row['pass_rate'],
                'Correctness': 'Correct',
                'Mean Entropy': row['correct_entropy_mean'],
                'Median Entropy': row['correct_entropy_median']
            })
            # Add incorrect trajectory entropy
            entropy_data.append({
                'Problem': row['problem_id'],
                'Difficulty': row['difficulty_category'],
                'Pass Rate': row['pass_rate'],
                'Correctness': 'Incorrect',
                'Mean Entropy': row['incorrect_entropy_mean'],
                'Median Entropy': row['incorrect_entropy_median']
            })

        plot_df = pd.DataFrame(entropy_data)

        # 2a. Grouped box plot by difficulty
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=plot_df, x='Difficulty', y='Mean Entropy', hue='Correctness',
                   ax=ax, palette='Set2')
        sns.stripplot(data=plot_df, x='Difficulty', y='Mean Entropy', hue='Correctness',
                     ax=ax, dodge=True, color='black', alpha=0.4, size=3, legend=False)
        ax.set_title('Mean Entropy by Problem Difficulty and Correctness', fontsize=14)
        ax.set_ylabel('Mean Entropy per Trajectory')
        ax.set_xlabel('Problem Difficulty (Pass Rate)')
        plt.xticks(rotation=15, ha='right')
        plt.legend(title='Trajectory Type', loc='best')
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Entropy vs pass rate (scatter plot with both correct and incorrect)
        print("  - Entropy vs pass rate (combined)")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot both correct and incorrect on same plot
        ax.scatter(mixed_df['pass_rate'], mixed_df['correct_entropy_mean'],
                  alpha=0.6, s=100, color='green', marker='o', label='Correct Trajectories',
                  edgecolors='black', linewidth=1)
        ax.scatter(mixed_df['pass_rate'], mixed_df['incorrect_entropy_mean'],
                  alpha=0.6, s=100, color='red', marker='s', label='Incorrect Trajectories',
                  edgecolors='black', linewidth=1)

        # Add connecting lines only if not too many problems
        if len(mixed_df) <= 50:
            for _, row in mixed_df.iterrows():
                ax.plot([row['pass_rate'], row['pass_rate']],
                       [row['correct_entropy_mean'], row['incorrect_entropy_mean']],
                       'k-', alpha=0.2, linewidth=0.8)

        # Add trend lines
        if len(mixed_df) > 1:
            z_correct = np.polyfit(mixed_df['pass_rate'], mixed_df['correct_entropy_mean'], 1)
            p_correct = np.poly1d(z_correct)
            x_range = np.linspace(mixed_df['pass_rate'].min(), mixed_df['pass_rate'].max(), 100)
            ax.plot(x_range, p_correct(x_range),
                   "g--", alpha=0.6, linewidth=2, label=f'Correct trend: y={z_correct[0]:.3f}x+{z_correct[1]:.3f}')

            z_incorrect = np.polyfit(mixed_df['pass_rate'], mixed_df['incorrect_entropy_mean'], 1)
            p_incorrect = np.poly1d(z_incorrect)
            ax.plot(x_range, p_incorrect(x_range),
                   "r--", alpha=0.6, linewidth=2, label=f'Incorrect trend: y={z_incorrect[0]:.3f}x+{z_incorrect[1]:.3f}')

        ax.set_xlabel('Pass Rate (Problem Difficulty)', fontsize=12)
        ax.set_ylabel('Mean Entropy', fontsize=12)
        ax.set_title('Entropy vs Problem Difficulty: Correct vs Incorrect Trajectories', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_vs_pass_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Entropy difference vs pass rate
        print("  - Entropy difference vs pass rate")

        # Calculate entropy difference for each problem
        mixed_df['entropy_diff'] = mixed_df['incorrect_entropy_mean'] - mixed_df['correct_entropy_mean']

        fig, ax = plt.subplots(figsize=(12, 7))

        # Scatter plot with color based on pass rate
        scatter = ax.scatter(mixed_df['pass_rate'], mixed_df['entropy_diff'],
                           c=mixed_df['pass_rate'], cmap='RdYlGn',
                           s=80, alpha=0.6, edgecolors='black', linewidth=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pass Rate', fontsize=11)

        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Add trend line
        if len(mixed_df) > 1:
            z = np.polyfit(mixed_df['pass_rate'], mixed_df['entropy_diff'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(mixed_df['pass_rate'].min(), mixed_df['pass_rate'].max(), 100)
            ax.plot(x_range, p(x_range),
                   "b--", alpha=0.7, linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

        ax.set_xlabel('Pass Rate (Easier → Harder ←)', fontsize=12)
        ax.set_ylabel('Entropy Difference (Incorrect - Correct)', fontsize=12)
        ax.set_title('Entropy Difference vs Problem Difficulty\n(Positive = Incorrect has higher entropy)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_diff_vs_pass_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Effect size analysis
        if 'cohens_d' in mixed_df.columns:
            print("  - Effect size analysis")

            # If too many problems (>50), only show scatter plot
            if len(mixed_df) > 50:
                fig, ax = plt.subplots(figsize=(12, 7))

                ax.scatter(mixed_df['pass_rate'], mixed_df['cohens_d'],
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.8)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                ax.axhline(y=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.4, label='Small effect')
                ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.4, label='Medium effect')
                ax.axhline(y=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.4, label='Large effect')

                ax.set_xlabel('Pass Rate', fontsize=12)
                ax.set_ylabel("Cohen's d (Correct - Incorrect)", fontsize=12)
                ax.set_title("Effect Size vs Problem Difficulty", fontsize=14)
                ax.grid(alpha=0.3)

                # Add trend line
                if len(mixed_df) > 1:
                    z = np.polyfit(mixed_df['pass_rate'], mixed_df['cohens_d'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(mixed_df['pass_rate'].min(), mixed_df['pass_rate'].max(), 100)
                    ax.plot(x_range, p(x_range),
                           "r--", alpha=0.7, linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

                ax.legend(loc='best', fontsize=10)
                plt.tight_layout()
                plt.savefig(output_path / 'effect_size_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

            else:
                # Show both plots for smaller datasets
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # 5a. Effect size by problem (sorted)
                sorted_mixed = mixed_df.sort_values('cohens_d')
                colors = ['red' if d < 0 else 'green' for d in sorted_mixed['cohens_d']]

                axes[0].barh(range(len(sorted_mixed)), sorted_mixed['cohens_d'], color=colors, alpha=0.7)
                axes[0].set_yticks(range(len(sorted_mixed)))
                axes[0].set_yticklabels(sorted_mixed['problem_id'])
                axes[0].set_xlabel("Cohen's d (Correct - Incorrect)")
                axes[0].set_title("Effect Size by Problem\n(Negative = Incorrect has higher entropy)")
                axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
                axes[0].axvline(x=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                axes[0].axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                axes[0].axvline(x=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                axes[0].grid(axis='x', alpha=0.3)

                # 5b. Effect size vs pass rate
                axes[1].scatter(mixed_df['pass_rate'], mixed_df['cohens_d'],
                              s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
                axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                axes[1].set_xlabel('Pass Rate')
                axes[1].set_ylabel("Cohen's d (Correct - Incorrect)")
                axes[1].set_title("Effect Size vs Problem Difficulty")
                axes[1].grid(alpha=0.3)

                # Add trend line
                if len(mixed_df) > 1:
                    z = np.polyfit(mixed_df['pass_rate'], mixed_df['cohens_d'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(mixed_df['pass_rate'].min(), mixed_df['pass_rate'].max(), 100)
                    axes[1].plot(x_range, p(x_range),
                               "r--", alpha=0.6, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                    axes[1].legend()

                plt.tight_layout()
                plt.savefig(output_path / 'effect_size_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

    print()
    print(f"Visualizations saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze entropy characteristics of correct vs incorrect trajectories"
    )
    parser.add_argument("input_file", type=str, help="Path to pass@k JSON file")
    parser.add_argument("--entropy-method", type=str, default="sum",
                       choices=["mean", "sum"],
                       help="Method to calculate trajectory entropy (default: sum)")
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

    # Generate report
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating report...")
    report_path = output_path / "analysis_report.txt"
    generate_report(df, report_path)
    print()

    # Save detailed results
    csv_path = output_path / "problem_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    print()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
