#!/usr/bin/env python3
"""
Analyze injection entropy patterns for different injection prompts.
Compares the discriminatory power of different injection types.
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
    Calculate total injection entropy for a trajectory by summing across all steps.

    Args:
        trajectory: Trajectory data with steps
        injection_type: Type of injection (e.g., 'confidence_score')

    Returns:
        Sum of injection entropies across all steps
    """
    total_entropy = 0.0

    for step in trajectory.get('steps', []):
        injection_results = step.get('injection_results', {})
        if injection_type in injection_results:
            entropy = injection_results[injection_type].get('entropy', 0.0)
            total_entropy += entropy

    return total_entropy


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
        incorrect_entropies = [
            calculate_trajectory_injection_entropy(t, injection_type)
            for t in incorrect_trajectories
        ]

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
        return "75-99% (Easy)"


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
    stats_df.to_csv(output_dir / 'injection_comparison.csv', index=False)

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
    """Generate comprehensive text report."""
    mixed_df = df[df['status'] == 'mixed']

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INJECTION ENTROPY DISCRIMINATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

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
            f.write("Most injection types cannot reliably distinguish correct from incorrect\n")
            f.write("trajectories on a per-problem basis.\n")
        elif avg_expected < 75:
            f.write("CONCLUSION: Injection entropies show MODERATE discrimination.\n")
            f.write("Some injection types may be useful, but reliability varies.\n")
        else:
            f.write("CONCLUSION: Injection entropies show STRONG discrimination.\n")
            f.write("Multiple injection types can reliably distinguish trajectories.\n")

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

        # Summary across difficulties
        f.write("\n" + "=" * 80 + "\n")
        f.write("DIFFICULTY COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for injection_type in INJECTION_TYPES:
            inj_data = difficulty_stats_df[difficulty_stats_df['injection_type'] == injection_type]

            if len(inj_data) == 0:
                continue

            f.write(f"\n{injection_type}:\n")
            f.write("-" * 80 + "\n")

            for idx, row in inj_data.iterrows():
                f.write(f"  {row['difficulty_category']}: "
                       f"{row['expected_pattern_pct']:.1f}% expected "
                       f"(n={row['num_problems']}, "
                       f"sig={row['significant_pct']:.1f}%)\n")

            # Calculate trend
            avg_by_difficulty = inj_data.groupby('difficulty_category')['expected_pattern_pct'].mean()
            if len(avg_by_difficulty) > 1:
                very_hard_pct = inj_data[inj_data['difficulty_category'] == '1-24% (Very Hard)']['expected_pattern_pct'].values
                easy_pct = inj_data[inj_data['difficulty_category'].str.contains('Easy')]['expected_pattern_pct'].values

                if len(very_hard_pct) > 0 and len(easy_pct) > 0:
                    diff_change = easy_pct[0] - very_hard_pct[0]
                    if abs(diff_change) > 10:
                        trend = "increases" if diff_change > 0 else "decreases"
                        f.write(f"  → Discrimination {trend} for easier problems ({diff_change:+.1f}% change)\n")

            f.write("\n")

    print(f"\nReport saved to: {output_path}")


def create_visualizations(df: pd.DataFrame, injection_stats_df: pd.DataFrame,
                         difficulty_stats_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations comparing injection types."""
    mixed_df = df[df['status'] == 'mixed']

    # 1. Comparison of injection types
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Expected pattern percentage
    ax = axes[0, 0]
    sorted_stats = injection_stats_df.sort_values('expected_pattern_pct', ascending=True)
    colors = ['steelblue' if x > 50 else 'coral' for x in sorted_stats['expected_pattern_pct']]
    ax.barh(sorted_stats['injection_type'], sorted_stats['expected_pattern_pct'], color=colors, alpha=0.8)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% (Random)')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Percentage of Problems Showing Expected Pattern\n(Incorrect > Correct Entropy)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Statistical significance percentage
    ax = axes[0, 1]
    sorted_stats = injection_stats_df.sort_values('significant_pct', ascending=True)
    ax.barh(sorted_stats['injection_type'], sorted_stats['significant_pct'], color='green', alpha=0.7)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Percentage of Problems with Significant Difference\n(p < 0.05)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Mean absolute effect size
    ax = axes[1, 0]
    sorted_stats = injection_stats_df.sort_values('mean_abs_cohens_d', ascending=True)
    ax.barh(sorted_stats['injection_type'], sorted_stats['mean_abs_cohens_d'], color='purple', alpha=0.7)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.2)')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Medium (0.5)')
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.9, label='Large (0.8)')
    ax.set_xlabel("Mean |Cohen's d|", fontsize=12)
    ax.set_title('Mean Absolute Effect Size Across Problems',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Mean entropy difference
    ax = axes[1, 1]
    sorted_stats = injection_stats_df.sort_values('mean_entropy_diff', ascending=True)
    colors = ['steelblue' if x > 0 else 'coral' for x in sorted_stats['mean_entropy_diff']]
    ax.barh(sorted_stats['injection_type'], sorted_stats['mean_entropy_diff'], color=colors, alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.set_xlabel('Mean Entropy Difference (Incorrect - Correct)', fontsize=12)
    ax.set_title('Average Entropy Difference Across Problems',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / 'injection_comparison_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

    # 2. Distribution comparison for each injection type
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, injection_type in enumerate(INJECTION_TYPES):
        ax = axes[idx]
        entropy_diff_col = f'{injection_type}_entropy_diff'

        valid_data = mixed_df[mixed_df[entropy_diff_col].notna()]

        if len(valid_data) > 0:
            ax.hist(valid_data[entropy_diff_col], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No discrimination')
            ax.axvline(x=valid_data[entropy_diff_col].median(), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {valid_data[entropy_diff_col].median():.3f}')

            expected = (valid_data[entropy_diff_col] > 0).sum()
            opposite = (valid_data[entropy_diff_col] <= 0).sum()

            ax.text(0.05, 0.95, f'Expected: {expected} ({expected/len(valid_data)*100:.1f}%)\n'
                                f'Opposite: {opposite} ({opposite/len(valid_data)*100:.1f}%)',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Entropy Diff (Incorrect - Correct)', fontsize=10)
            ax.set_ylabel('Number of Problems', fontsize=10)
            ax.set_title(f'{injection_type.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'injection_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

    # 3. Difficulty-based heatmap
    difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']

    # Create pivot table for heatmap
    pivot_data = difficulty_stats_df.pivot(
        index='injection_type',
        columns='difficulty_category',
        values='expected_pattern_pct'
    )

    # Reorder columns
    available_difficulties = [d for d in difficulty_order if d in pivot_data.columns]
    pivot_data = pivot_data[available_difficulties]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=100, cbar_kws={'label': 'Expected Pattern %'},
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Problem Difficulty', fontsize=13, fontweight='bold')
    ax.set_ylabel('Injection Type', fontsize=13, fontweight='bold')
    ax.set_title('Injection Entropy Discrimination by Problem Difficulty\n(% Problems Showing Expected Pattern: Incorrect > Correct)',
                 fontsize=14, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'difficulty_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

    # 4. Line plot showing trends across difficulties
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, injection_type in enumerate(INJECTION_TYPES):
        ax = axes[idx]

        inj_data = difficulty_stats_df[difficulty_stats_df['injection_type'] == injection_type]
        inj_data = inj_data.sort_values('difficulty_category')

        # Map difficulty to numeric for x-axis
        difficulty_map = {d: i for i, d in enumerate(difficulty_order)}
        inj_data['diff_num'] = inj_data['difficulty_category'].map(difficulty_map)
        inj_data = inj_data.sort_values('diff_num')

        # Plot expected pattern percentage
        ax.plot(inj_data['diff_num'], inj_data['expected_pattern_pct'],
               marker='o', linewidth=2, markersize=8, label='Expected pattern %')

        # Add 50% reference line (random baseline)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Random)')

        # Annotate points with values
        for _, row in inj_data.iterrows():
            ax.annotate(f"{row['expected_pattern_pct']:.1f}%",
                       (row['diff_num'], row['expected_pattern_pct']),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=9)

        ax.set_xticks(range(len(available_difficulties)))
        ax.set_xticklabels([d.split('(')[1].replace(')', '') for d in available_difficulties],
                          rotation=15, ha='right')
        ax.set_xlabel('Difficulty', fontsize=11)
        ax.set_ylabel('Expected Pattern %', fontsize=11)
        ax.set_title(f'{injection_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'difficulty_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_injection_entropy.py <input_json>")
        print("Example: python analyze_injection_entropy.py results/pass_at_32.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = input_path.parent / f"{input_path.stem}_injection_analysis"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("INJECTION ENTROPY ANALYSIS")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    print()

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
    csv_path = output_dir / 'problem_injection_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Compare injection types
    print("\nComparing injection types...")
    injection_stats_df = compare_injection_discriminators(df, output_dir)

    # Analyze by difficulty
    print("\nAnalyzing by difficulty...")
    difficulty_stats_df = analyze_by_difficulty(df)
    difficulty_csv_path = output_dir / 'difficulty_comparison.csv'
    difficulty_stats_df.to_csv(difficulty_csv_path, index=False)
    print(f"Difficulty comparison saved to: {difficulty_csv_path}")

    # Generate report
    print("\nGenerating report...")
    report_path = output_dir / 'injection_analysis_report.txt'
    generate_report(df, injection_stats_df, difficulty_stats_df, report_path)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, injection_stats_df, difficulty_stats_df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey files generated:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    print(f"  - {output_dir}/injection_comparison.csv")
    print(f"  - {difficulty_csv_path}")
    print(f"  - {output_dir}/injection_comparison_overview.png")
    print(f"  - {output_dir}/injection_distributions.png")
    print(f"  - {output_dir}/difficulty_heatmap.png")
    print(f"  - {output_dir}/difficulty_trends.png")


if __name__ == "__main__":
    main()
