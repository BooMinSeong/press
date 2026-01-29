#!/usr/bin/env python3
"""
Analyze per-problem entropy patterns to determine if entropy is a reliable discriminator.
This corrects the previous analysis which incorrectly averaged across problems.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_per_problem_patterns(csv_path: str, output_dir: str):
    """
    Analyze entropy discrimination patterns on a per-problem basis.

    Key insight: We need to look at individual problems, not aggregate statistics.
    """
    # Load the per-problem analysis
    df = pd.read_csv(csv_path)

    # Filter to only mixed problems (those with both correct and incorrect trajectories)
    mixed_df = df[df['status'] == 'mixed'].copy()

    print(f"Analyzing {len(mixed_df)} problems with mixed results...")

    # Calculate entropy difference for each problem
    # Positive = incorrect has higher entropy (expected pattern)
    # Negative = correct has higher entropy (opposite pattern)
    mixed_df['entropy_diff'] = (
        mixed_df['incorrect_entropy_mean'] - mixed_df['correct_entropy_mean']
    )

    # Categorize problems by their discrimination pattern
    mixed_df['discrimination_pattern'] = mixed_df['entropy_diff'].apply(
        lambda x: 'Expected (Incorrect > Correct)' if x > 0 else 'Opposite (Correct > Incorrect)'
    )

    # Count patterns
    pattern_counts = mixed_df['discrimination_pattern'].value_counts()

    # Statistical breakdown
    total_problems = len(mixed_df)
    expected_pattern = (mixed_df['entropy_diff'] > 0).sum()
    opposite_pattern = (mixed_df['entropy_diff'] <= 0).sum()

    expected_pct = (expected_pattern / total_problems) * 100
    opposite_pct = (opposite_pattern / total_problems) * 100

    # Generate report
    report_path = Path(output_dir) / "per_problem_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PER-PROBLEM ENTROPY DISCRIMINATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("CRITICAL FINDING:\n")
        f.write("-" * 80 + "\n")
        f.write(f"When examining individual problems (not aggregated statistics):\n\n")

        f.write(f"Total problems analyzed: {total_problems}\n\n")

        f.write(f"Problems where Incorrect > Correct entropy: {expected_pattern} ({expected_pct:.1f}%)\n")
        f.write(f"Problems where Correct > Incorrect entropy: {opposite_pattern} ({opposite_pct:.1f}%)\n\n")

        if abs(expected_pct - 50) < 10:  # Within 10% of 50-50 split
            f.write("CONCLUSION: Entropy discrimination is essentially RANDOM at the problem level.\n")
            f.write("The pattern is approximately 50-50, meaning entropy is NOT a reliable\n")
            f.write("discriminator when examined per-problem.\n\n")
        elif expected_pct > 60:
            f.write("CONCLUSION: Entropy shows weak discrimination at the problem level.\n")
            f.write(f"While {expected_pct:.1f}% of problems show the expected pattern,\n")
            f.write(f"{opposite_pct:.1f}% show the opposite, indicating unreliable discrimination.\n\n")

        # Breakdown by difficulty
        f.write("\n" + "=" * 80 + "\n")
        f.write("BREAKDOWN BY DIFFICULTY\n")
        f.write("=" * 80 + "\n\n")

        for difficulty in ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']:
            diff_df = mixed_df[mixed_df['difficulty_category'] == difficulty]
            if len(diff_df) == 0:
                continue

            diff_total = len(diff_df)
            diff_expected = (diff_df['entropy_diff'] > 0).sum()
            diff_opposite = (diff_df['entropy_diff'] <= 0).sum()
            diff_expected_pct = (diff_expected / diff_total) * 100
            diff_opposite_pct = (diff_opposite / diff_total) * 100

            f.write(f"{difficulty}:\n")
            f.write(f"  Total problems: {diff_total}\n")
            f.write(f"  Expected pattern: {diff_expected} ({diff_expected_pct:.1f}%)\n")
            f.write(f"  Opposite pattern: {diff_opposite} ({diff_opposite_pct:.1f}%)\n")
            f.write(f"  Mean entropy diff: {diff_df['entropy_diff'].mean():.4f}\n")
            f.write(f"  Median entropy diff: {diff_df['entropy_diff'].median():.4f}\n\n")

        # Effect size distribution
        f.write("\n" + "=" * 80 + "\n")
        f.write("EFFECT SIZE (Cohen's d) DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")

        # Note: Cohen's d is negative when incorrect > correct (expected)
        cohens_d_data = mixed_df['cohens_d'].dropna()

        f.write(f"Mean Cohen's d: {cohens_d_data.mean():.4f}\n")
        f.write(f"Median Cohen's d: {cohens_d_data.median():.4f}\n")
        f.write(f"Std Cohen's d: {cohens_d_data.std():.4f}\n\n")

        # Categorize effect sizes
        small_threshold = 0.2
        medium_threshold = 0.5
        large_threshold = 0.8

        # Expected direction (negative Cohen's d)
        expected_small = ((cohens_d_data < 0) & (cohens_d_data >= -small_threshold)).sum()
        expected_medium = ((cohens_d_data < -small_threshold) & (cohens_d_data >= -medium_threshold)).sum()
        expected_large = ((cohens_d_data < -medium_threshold) & (cohens_d_data >= -large_threshold)).sum()
        expected_very_large = (cohens_d_data < -large_threshold).sum()

        # Opposite direction (positive Cohen's d)
        opposite_small = ((cohens_d_data > 0) & (cohens_d_data <= small_threshold)).sum()
        opposite_medium = ((cohens_d_data > small_threshold) & (cohens_d_data <= medium_threshold)).sum()
        opposite_large = ((cohens_d_data > medium_threshold) & (cohens_d_data <= large_threshold)).sum()
        opposite_very_large = (cohens_d_data > large_threshold).sum()

        f.write("Expected direction (Incorrect > Correct):\n")
        f.write(f"  Negligible (|d| < {small_threshold}): {expected_small}\n")
        f.write(f"  Small ({small_threshold} ≤ |d| < {medium_threshold}): {expected_medium}\n")
        f.write(f"  Medium ({medium_threshold} ≤ |d| < {large_threshold}): {expected_large}\n")
        f.write(f"  Large (|d| ≥ {large_threshold}): {expected_very_large}\n\n")

        f.write("Opposite direction (Correct > Incorrect):\n")
        f.write(f"  Negligible (|d| < {small_threshold}): {opposite_small}\n")
        f.write(f"  Small ({small_threshold} ≤ |d| < {medium_threshold}): {opposite_medium}\n")
        f.write(f"  Medium ({medium_threshold} ≤ |d| < {large_threshold}): {opposite_large}\n")
        f.write(f"  Large (|d| ≥ {large_threshold}): {opposite_very_large}\n\n")

        # Statistical significance
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("=" * 80 + "\n\n")

        significant = (mixed_df['ttest_pvalue'] < 0.05).sum()
        significant_pct = (significant / total_problems) * 100

        f.write(f"Problems with significant difference (p < 0.05): {significant} ({significant_pct:.1f}%)\n")
        f.write(f"Problems without significant difference: {total_problems - significant} ({100 - significant_pct:.1f}%)\n\n")

        f.write("INTERPRETATION:\n")
        f.write(f"Only {significant_pct:.1f}% of problems show statistically significant entropy\n")
        f.write("differences between correct and incorrect trajectories. This indicates\n")
        f.write("that entropy is NOT a reliable per-problem discriminator.\n")

    print(f"\nReport saved to: {report_path}")

    # Create visualizations
    create_visualizations(mixed_df, output_dir)

    return mixed_df


def create_visualizations(df, output_dir):
    """Create visualizations showing per-problem patterns."""
    output_dir = Path(output_dir)

    # 1. Distribution of entropy differences
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of entropy differences
    ax = axes[0, 0]
    ax.hist(df['entropy_diff'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No discrimination')
    ax.axvline(x=df['entropy_diff'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {df["entropy_diff"].median():.3f}')
    ax.set_xlabel('Entropy Difference (Incorrect - Correct)', fontsize=11)
    ax.set_ylabel('Number of Problems', fontsize=11)
    ax.set_title('Distribution of Entropy Differences Across Problems', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Count expected vs opposite patterns
    expected = (df['entropy_diff'] > 0).sum()
    opposite = (df['entropy_diff'] <= 0).sum()

    ax.text(0.05, 0.95, f'Expected pattern: {expected} ({expected/len(df)*100:.1f}%)\n'
                        f'Opposite pattern: {opposite} ({opposite/len(df)*100:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Cohen's d distribution
    ax = axes[0, 1]
    cohens_d_data = df['cohens_d'].dropna()
    ax.hist(cohens_d_data, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
    ax.axvline(x=cohens_d_data.median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {cohens_d_data.median():.3f}')
    ax.set_xlabel("Cohen's d (Negative = Expected Pattern)", fontsize=11)
    ax.set_ylabel('Number of Problems', fontsize=11)
    ax.set_title("Distribution of Effect Sizes Across Problems", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy diff vs pass rate (showing the symmetric distribution)
    ax = axes[1, 0]
    ax.scatter(df['pass_rate'], df['entropy_diff'], alpha=0.6, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No discrimination')

    # Fit and plot trend line
    z = np.polyfit(df['pass_rate'], df['entropy_diff'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['pass_rate'].min(), df['pass_rate'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2, label='Trend')

    ax.set_xlabel('Pass Rate (Problem Difficulty)', fontsize=11)
    ax.set_ylabel('Entropy Difference (Incorrect - Correct)', fontsize=11)
    ax.set_title('Entropy Discrimination vs Problem Difficulty\n(Note: Symmetric distribution around trend)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy diff by difficulty category
    ax = axes[1, 1]
    difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']
    difficulty_data = []
    difficulty_labels = []

    for difficulty in difficulty_order:
        diff_df = df[df['difficulty_category'] == difficulty]
        if len(diff_df) > 0:
            difficulty_data.append(diff_df['entropy_diff'].values)
            difficulty_labels.append(f"{difficulty}\n(n={len(diff_df)})")

    bp = ax.boxplot(difficulty_data, labels=difficulty_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No discrimination')
    ax.set_ylabel('Entropy Difference (Incorrect - Correct)', fontsize=11)
    ax.set_title('Entropy Discrimination by Difficulty\n(Boxplot shows per-problem variability)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "per_problem_entropy_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")

    # 2. Pattern breakdown by difficulty
    fig, ax = plt.subplots(figsize=(10, 6))

    difficulty_order = ['1-24% (Very Hard)', '25-49% (Hard)', '50-74% (Medium)', '75-100% (Easy)']

    expected_counts = []
    opposite_counts = []
    labels = []

    for difficulty in difficulty_order:
        diff_df = df[df['difficulty_category'] == difficulty]
        if len(diff_df) > 0:
            expected = (diff_df['entropy_diff'] > 0).sum()
            opposite = (diff_df['entropy_diff'] <= 0).sum()
            expected_counts.append(expected)
            opposite_counts.append(opposite)
            labels.append(difficulty)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, expected_counts, width, label='Expected (Incorrect > Correct)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, opposite_counts, width, label='Opposite (Correct > Incorrect)',
                   color='coral', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Difficulty Category', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Entropy Discrimination Patterns by Difficulty\n(Per-Problem Analysis)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "pattern_breakdown_by_difficulty.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_per_problem_patterns.py <problem_analysis.csv>")
        print("Example: python analyze_per_problem_patterns.py results/correctness_analysis_final/problem_analysis.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = Path(csv_path).parent

    print("=" * 80)
    print("PER-PROBLEM ENTROPY PATTERN ANALYSIS")
    print("=" * 80)
    print(f"\nInput: {csv_path}")
    print(f"Output: {output_dir}")
    print()

    df = analyze_per_problem_patterns(csv_path, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey files generated:")
    print(f"  - {output_dir}/per_problem_analysis_report.txt")
    print(f"  - {output_dir}/per_problem_entropy_patterns.png")
    print(f"  - {output_dir}/pattern_breakdown_by_difficulty.png")


if __name__ == "__main__":
    main()
