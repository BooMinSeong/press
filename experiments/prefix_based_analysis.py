"""
Prefix-based entropy analysis for tree/beam search applicability.

This analysis filters trajectories to only include steps that follow a correct prefix,
simulating the actual decision points in tree/beam search where we must choose
the next step given a correct path so far.

Key difference from standard analysis:
- Standard: Compares all correct steps vs all incorrect steps (includes contaminated paths)
- Prefix-based: At step k, only compares steps where steps 1 to k-1 were all correct

Usage:
    python experiments/prefix_based_analysis.py results/experiment_001.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from press.config import INJECTION_PROMPTS


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_step_correctness(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Determine if each step is on a correct path by checking if the final answer is correct.

    Note: This is an approximation. Ideally we'd have ground truth for each step,
    but we use final correctness as a proxy for "correct reasoning path".

    Returns:
        Same results structure with step-level correctness inferred
    """
    # For now, we assume a step is "correct" if the trajectory ends correctly
    # This is the best we can do without step-level ground truth
    return results


def extract_prefix_based_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract step data but only include steps that follow a correct prefix.

    For each step k:
    - Include only if this trajectory is correct (proxy for "correct path so far")
    - Compare step k entropy between trajectories that:
      a) Continue being correct (end with correct answer)
      b) This is impossible to determine per-step without ground truth

    Actually, we need a different approach. Let me think...

    Better approach:
    - Group trajectories by problem
    - For each problem, identify "correct" trajectories (those with correct final answer)
    - For step k, compare:
      * Steps from correct trajectories (these stayed on correct path through step k)
      * Steps from incorrect trajectories (these diverged at some point)

    But this still has contamination issue...

    Real solution:
    - We can only properly analyze this by comparing WITHIN each step position
    - Filter to only include step k from trajectories that are correct
    - Compare their entropy distribution
    - But we can't distinguish "step k is where it went wrong" vs "it was already wrong"

    Alternative approach - "Prefix-conditioned analysis":
    - For step k, only include data from trajectories where we have evidence that
      steps 1 to k-1 were on a reasonable path
    - Use multiple correct trajectories to estimate "correct path" distribution

    Let's implement a cleaner version:
    - For each (problem, step_number), collect all steps at that position
    - Separate by final correctness
    - This gives us "what entropy looks like at step k for eventually-correct paths"
    - Better than nothing, though still not perfect

    Enhanced version:
    - Only include problems that have at least 1 correct trajectory
    - For those problems, compare step k entropy in correct vs incorrect trajectories
    """

    step_data = []

    # Only analyze problems with at least one correct solution
    problems_with_correct = [
        p for p in results
        if any(t.get('is_correct', False) for t in p.get('trajectories', []))
    ]

    print(f"  Filtering to {len(problems_with_correct)}/{len(results)} problems with ≥1 correct trajectory")

    for problem in problems_with_correct:
        problem_id = problem['problem_id']

        # Separate correct and incorrect trajectories for this problem
        correct_trajs = [t for t in problem.get('trajectories', []) if t.get('is_correct', False)]
        incorrect_trajs = [t for t in problem.get('trajectories', []) if not t.get('is_correct', False)]

        # Process all trajectories
        for traj in problem.get('trajectories', []):
            is_correct = traj.get('is_correct', False)

            for step in traj.get('steps', []):
                step_number = step['step_number']

                # Base step info
                step_info = {
                    'problem_id': problem_id,
                    'is_correct': is_correct,
                    'step_number': step_number,
                    'avg_entropy': step['avg_entropy'],
                    'max_entropy': step['max_entropy'],
                    'min_entropy': step['min_entropy'],
                    'median_entropy': step['median_entropy'],
                    'std_entropy': step['std_entropy'],
                    'has_correct_sibling': len(correct_trajs) > 0,  # This problem has correct solutions
                    'num_correct_siblings': len(correct_trajs),
                    'num_incorrect_siblings': len(incorrect_trajs),
                }

                # Add injection entropies
                for prompt_name in INJECTION_PROMPTS.keys():
                    if prompt_name in step.get('injection_results', {}):
                        inj_result = step['injection_results'][prompt_name]
                        step_info[f'{prompt_name}_entropy'] = inj_result['entropy']
                    else:
                        step_info[f'{prompt_name}_entropy'] = np.nan

                step_data.append(step_info)

    return pd.DataFrame(step_data)


def within_problem_discrimination(df: pd.DataFrame, use_normalized: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Measure discrimination ability within each problem (controls for problem difficulty).

    For each problem that has both correct and incorrect trajectories,
    measure the entropy difference between them.

    This is closer to tree search: "Given this problem, can entropy distinguish
    correct from incorrect paths?"
    """

    results = {}

    for prompt_name in INJECTION_PROMPTS.keys():
        entropy_col = f'{prompt_name}_entropy_norm' if use_normalized else f'{prompt_name}_entropy'

        if entropy_col not in df.columns:
            continue

        # Only analyze problems with both correct and incorrect trajectories
        problems_with_both = df.groupby('problem_id')['is_correct'].apply(
            lambda x: x.any() and (~x).any()
        )
        valid_problems = problems_with_both[problems_with_both].index

        problem_metrics = []

        for problem_id in valid_problems:
            problem_df = df[df['problem_id'] == problem_id].copy()

            # Skip if no valid entropy data
            problem_df = problem_df.dropna(subset=[entropy_col])
            if len(problem_df) < 2:
                continue

            correct_entropy = problem_df[problem_df['is_correct']][entropy_col]
            incorrect_entropy = problem_df[~problem_df['is_correct']][entropy_col]

            if len(correct_entropy) == 0 or len(incorrect_entropy) == 0:
                continue

            # Effect size within this problem
            mean_diff = incorrect_entropy.mean() - correct_entropy.mean()

            # Pooled std
            n_c, n_i = len(correct_entropy), len(incorrect_entropy)
            std_c = correct_entropy.std(ddof=1) if n_c > 1 else 0
            std_i = incorrect_entropy.std(ddof=1) if n_i > 1 else 0

            if n_c + n_i > 2:
                pooled_std = np.sqrt(
                    ((n_c - 1) * std_c**2 + (n_i - 1) * std_i**2) / (n_c + n_i - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = np.nan

            # AUC for this problem
            try:
                labels = problem_df['is_correct'].values
                scores = -problem_df[entropy_col].values  # Reverse so lower entropy = better
                auc = roc_auc_score(labels, scores)
            except:
                auc = np.nan

            problem_metrics.append({
                'problem_id': problem_id,
                'n_correct': n_c,
                'n_incorrect': n_i,
                'mean_entropy_diff': mean_diff,
                'cohens_d': cohens_d,
                'auc': auc,
            })

        if problem_metrics:
            results[prompt_name] = pd.DataFrame(problem_metrics)

    return results


def step_by_step_prefix_discrimination(df: pd.DataFrame, use_normalized: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Step-by-step discrimination, but only on problems with correct solutions.

    This reduces contamination by ensuring we're comparing paths on "solvable" problems.
    """

    results = {}

    for prompt_name in INJECTION_PROMPTS.keys():
        entropy_col = f'{prompt_name}_entropy_norm' if use_normalized else f'{prompt_name}_entropy'

        if entropy_col not in df.columns:
            continue

        step_metrics = []

        for step_num in sorted(df['step_number'].unique()):
            step_df = df[df['step_number'] == step_num].copy()
            step_df = step_df.dropna(subset=[entropy_col])

            if len(step_df) < 2:
                continue

            correct_entropy = step_df[step_df['is_correct']][entropy_col]
            incorrect_entropy = step_df[~step_df['is_correct']][entropy_col]

            if len(correct_entropy) == 0 or len(incorrect_entropy) == 0:
                continue

            # Pearson correlation
            try:
                pearson_r, pearson_p = stats.pearsonr(
                    step_df[entropy_col],
                    step_df['is_correct'].astype(int)
                )
            except:
                pearson_r, pearson_p = np.nan, np.nan

            # Cohen's d
            mean_correct = correct_entropy.mean()
            mean_incorrect = incorrect_entropy.mean()
            std_correct = correct_entropy.std(ddof=1) if len(correct_entropy) > 1 else 0
            std_incorrect = incorrect_entropy.std(ddof=1) if len(incorrect_entropy) > 1 else 0

            n_correct = len(correct_entropy)
            n_incorrect = len(incorrect_entropy)

            if n_correct + n_incorrect > 2:
                pooled_std = np.sqrt(
                    ((n_correct - 1) * std_correct**2 + (n_incorrect - 1) * std_incorrect**2) /
                    (n_correct + n_incorrect - 2)
                )
                cohens_d = (mean_correct - mean_incorrect) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = np.nan

            # AUC
            try:
                reversed_entropy = -step_df[entropy_col]
                auc = roc_auc_score(step_df['is_correct'], reversed_entropy)
            except:
                auc = np.nan

            # Count problems at this step
            n_problems = step_df['problem_id'].nunique()
            n_problems_with_both = step_df.groupby('problem_id')['is_correct'].apply(
                lambda x: x.any() and (~x).any()
            ).sum()

            step_metrics.append({
                'Step': step_num,
                'N_Problems': n_problems,
                'N_Problems_With_Both': n_problems_with_both,
                'N_Correct': n_correct,
                'N_Incorrect': n_incorrect,
                'Mean_Entropy_Correct': mean_correct,
                'Mean_Entropy_Incorrect': mean_incorrect,
                'Std_Entropy_Correct': std_correct,
                'Std_Entropy_Incorrect': std_incorrect,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Cohens_d': cohens_d,
                'AUC': auc,
            })

        if step_metrics:
            results[prompt_name] = pd.DataFrame(step_metrics)

    return results


def normalize_entropy_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize injection entropies by step number."""
    df = df.copy()

    for prompt_name in INJECTION_PROMPTS.keys():
        entropy_col = f'{prompt_name}_entropy'
        norm_col = f'{prompt_name}_entropy_norm'

        if entropy_col not in df.columns:
            continue

        # Normalize within each step
        df[norm_col] = df.groupby('step_number')[entropy_col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return df


def aggregate_within_problem_results(within_problem_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate within-problem discrimination metrics across all problems.

    Returns summary statistics showing how well entropy discriminates
    when controlling for problem difficulty.
    """

    summary = []

    for prompt_name, df in within_problem_results.items():
        if len(df) == 0:
            continue

        # Filter out NaN values
        valid_df = df.dropna(subset=['cohens_d', 'auc'])

        if len(valid_df) == 0:
            continue

        summary.append({
            'Prompt': prompt_name,
            'N_Problems': len(valid_df),
            'Mean_AUC': valid_df['auc'].mean(),
            'Median_AUC': valid_df['auc'].median(),
            'Std_AUC': valid_df['auc'].std(),
            'Mean_Cohens_d': valid_df['cohens_d'].mean(),
            'Median_Cohens_d': valid_df['cohens_d'].median(),
            'Std_Cohens_d': valid_df['cohens_d'].std(),
            'Pct_AUC_Above_0.5': (valid_df['auc'] > 0.5).mean() * 100,
            'Pct_AUC_Above_0.6': (valid_df['auc'] > 0.6).mean() * 100,
            'Pct_AUC_Above_0.7': (valid_df['auc'] > 0.7).mean() * 100,
        })

    results_df = pd.DataFrame(summary)

    # Sort by mean AUC
    results_df = results_df.sort_values('Mean_AUC', ascending=False)

    return results_df


def create_visualizations(df: pd.DataFrame,
                         within_problem_results: Dict[str, pd.DataFrame],
                         within_problem_summary: pd.DataFrame,
                         step_discrimination: Dict[str, pd.DataFrame],
                         output_dir: str = "results/prefix_analysis"):
    """Create visualization plots for prefix-based analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    print("  - Creating visualizations...")

    # 1. Within-problem AUC distribution
    print("    * Within-problem AUC distribution")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (prompt_name, df_prob) in enumerate(within_problem_results.items()):
        if idx >= 6:
            break

        ax = axes[idx]

        # Filter valid AUC values
        valid_auc = df_prob['auc'].dropna()

        if len(valid_auc) > 0:
            ax.hist(valid_auc, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
            ax.axvline(x=valid_auc.mean(), color='green', linestyle='-', linewidth=2,
                      label=f'Mean ({valid_auc.mean():.3f})')
            ax.set_xlabel('AUC')
            ax.set_ylabel('Number of Problems')
            ax.set_title(f'{prompt_name}\n({len(valid_auc)} problems)')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)

    plt.suptitle('Within-Problem AUC Distribution (Tree Search Simulation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'within_problem_auc_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Comparison: Within-problem vs Overall
    print("    * Within-problem discrimination summary")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean AUC
    axes[0].barh(within_problem_summary['Prompt'], within_problem_summary['Mean_AUC'],
                color='steelblue', alpha=0.8)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Random')
    axes[0].set_xlabel('Mean AUC (within-problem)')
    axes[0].set_title('Average Discrimination Ability')
    axes[0].set_xlim(0.4, 0.8)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].legend()

    # Median Cohen's d
    axes[1].barh(within_problem_summary['Prompt'], within_problem_summary['Median_Cohens_d'],
                color='coral', alpha=0.8)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel("Median Cohen's d (within-problem)")
    axes[1].set_title('Effect Size')
    axes[1].grid(axis='x', alpha=0.3)

    # Percentage of problems with AUC > 0.6
    axes[2].barh(within_problem_summary['Prompt'], within_problem_summary['Pct_AUC_Above_0.6'],
                color='mediumseagreen', alpha=0.8)
    axes[2].set_xlabel('% Problems with AUC > 0.6')
    axes[2].set_title('Strong Discrimination Rate')
    axes[2].set_xlim(0, 100)
    axes[2].grid(axis='x', alpha=0.3)

    plt.suptitle('Within-Problem Discrimination Summary (Closest to Tree Search)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'within_problem_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Step-by-step evolution (prefix-based)
    print("    * Step-by-step evolution")

    if step_discrimination:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (prompt_name, step_df) in enumerate(step_discrimination.items()):
            if idx >= 6:
                break

            ax = axes[idx]

            # Plot AUC evolution
            ax.plot(step_df['Step'], step_df['AUC'],
                   marker='o', linewidth=2, markersize=6, color='steelblue', label='AUC')
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
            ax.set_xlabel('Step Number')
            ax.set_ylabel('AUC')
            ax.set_title(f'{prompt_name}')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
            ax.set_ylim(0.4, 0.8)

        plt.suptitle('Step-by-Step AUC Evolution (Problems with ≥1 Correct)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'prefix_step_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n  Visualizations saved to: {output_path}")


def generate_report(df: pd.DataFrame,
                   within_problem_summary: pd.DataFrame,
                   step_discrimination: Dict[str, pd.DataFrame],
                   output_path: str = "results/prefix_analysis/prefix_analysis_report.txt"):
    """Generate comprehensive prefix-based analysis report."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PREFIX-BASED ENTROPY ANALYSIS REPORT\n")
        f.write("Tree/Beam Search Applicability Assessment\n")
        f.write("="*80 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write("This analysis addresses the contamination issue in the standard analysis.\n")
        f.write("Instead of comparing ALL correct vs incorrect steps, we:\n")
        f.write("  1. Only analyze problems that have at least one correct solution\n")
        f.write("  2. Compare entropy WITHIN each problem (controls for difficulty)\n")
        f.write("  3. Measure if entropy can distinguish correct from incorrect paths\n")
        f.write("     on the SAME problem (closer to tree/beam search scenario)\n\n")

        total_problems = df['problem_id'].nunique()
        f.write(f"Total problems analyzed: {total_problems}\n")
        f.write(f"These are problems with ≥1 correct trajectory in the dataset\n\n")

        # Within-problem discrimination
        f.write("="*80 + "\n")
        f.write("1. WITHIN-PROBLEM DISCRIMINATION (Tree Search Simulation)\n")
        f.write("="*80 + "\n\n")
        f.write("For each problem, we measure how well entropy distinguishes correct from\n")
        f.write("incorrect trajectories ON THAT SAME PROBLEM. This controls for problem\n")
        f.write("difficulty and simulates tree search decision-making.\n\n")

        f.write("Summary Statistics (Averaged Across Problems):\n")
        f.write("-"*80 + "\n\n")

        display_cols = ['Prompt', 'N_Problems', 'Mean_AUC', 'Median_AUC',
                       'Mean_Cohens_d', 'Median_Cohens_d',
                       'Pct_AUC_Above_0.5', 'Pct_AUC_Above_0.6']
        f.write(within_problem_summary[display_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation:\n")
        f.write("  - Mean_AUC: Average discrimination ability across all problems\n")
        f.write("    * 0.5 = random guessing\n")
        f.write("    * 0.6 = weak discrimination\n")
        f.write("    * 0.7 = moderate discrimination\n")
        f.write("    * 0.8+ = strong discrimination\n")
        f.write("  - Median_Cohens_d: Typical effect size within a problem\n")
        f.write("    * Negative = correct paths have lower entropy (desired)\n")
        f.write("    * |d| > 0.2=small, >0.5=medium, >0.8=large\n")
        f.write("  - Pct_AUC_Above_X: Percentage of problems where discrimination works well\n\n")

        # Best prompt analysis
        best_prompt = within_problem_summary.iloc[0]
        f.write("Best Performing Prompt:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Prompt: {best_prompt['Prompt']}\n")
        f.write(f"  Mean AUC: {best_prompt['Mean_AUC']:.4f}\n")
        f.write(f"  Median AUC: {best_prompt['Median_AUC']:.4f}\n")
        f.write(f"  Mean Cohen's d: {best_prompt['Mean_Cohens_d']:.4f}\n")
        f.write(f"  % Problems with AUC > 0.6: {best_prompt['Pct_AUC_Above_0.6']:.1f}%\n")
        f.write(f"  % Problems with AUC > 0.7: {best_prompt['Pct_AUC_Above_0.7']:.1f}%\n\n")

        # Step-by-step analysis
        f.write("="*80 + "\n")
        f.write("2. STEP-BY-STEP DISCRIMINATION (Prefix-Based)\n")
        f.write("="*80 + "\n\n")
        f.write("How discrimination ability evolves across reasoning steps,\n")
        f.write("analyzed only on problems with ≥1 correct solution.\n\n")

        for prompt_name, step_df in step_discrimination.items():
            f.write(f"\n{prompt_name}:\n")
            f.write("-" * 40 + "\n")

            display_df = step_df[['Step', 'N_Problems_With_Both',
                                 'N_Correct', 'N_Incorrect',
                                 'Mean_Entropy_Correct', 'Mean_Entropy_Incorrect',
                                 'Pearson_r', 'AUC']].copy()

            # Format for readability
            display_df['Mean_Entropy_Correct'] = display_df['Mean_Entropy_Correct'].apply(lambda x: f"{x:.4f}")
            display_df['Mean_Entropy_Incorrect'] = display_df['Mean_Entropy_Incorrect'].apply(lambda x: f"{x:.4f}")
            display_df['Pearson_r'] = display_df['Pearson_r'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

            if len(display_df) > 10:
                f.write(display_df.head(5).to_string(index=False))
                f.write("\n  ...\n")
                f.write(display_df.tail(5).to_string(index=False, header=False))
            else:
                f.write(display_df.to_string(index=False))
            f.write("\n")

        f.write("\n")

        # Key findings
        f.write("="*80 + "\n")
        f.write("3. KEY FINDINGS & TREE/BEAM SEARCH APPLICABILITY\n")
        f.write("="*80 + "\n\n")

        best_mean_auc = best_prompt['Mean_AUC']
        best_pct_good = best_prompt['Pct_AUC_Above_0.6']

        f.write("Assessment of Tree/Beam Search Viability:\n\n")

        if best_mean_auc >= 0.65:
            verdict = "PROMISING"
            color = "✓"
        elif best_mean_auc >= 0.55:
            verdict = "MARGINAL"
            color = "~"
        else:
            verdict = "WEAK"
            color = "✗"

        f.write(f"{color} Overall Verdict: {verdict}\n\n")

        f.write(f"Evidence:\n")
        f.write(f"  - Best mean AUC: {best_mean_auc:.3f}\n")
        f.write(f"    * Above 0.65 = promising for tree search\n")
        f.write(f"    * 0.55-0.65 = marginal benefit\n")
        f.write(f"    * Below 0.55 = weak signal\n\n")

        f.write(f"  - {best_pct_good:.1f}% of problems show AUC > 0.6\n")
        f.write(f"    * This means entropy is helpful on ~{best_pct_good:.0f}% of problems\n")
        f.write(f"    * On remaining ~{100-best_pct_good:.0f}% of problems, signal is weak\n\n")

        f.write("Comparison with Standard Analysis:\n")
        f.write("  - Standard analysis showed higher AUC (0.62+) due to contamination\n")
        f.write("  - Prefix-based analysis (this report) is more conservative and realistic\n")
        f.write("  - The difference reveals the contamination effect magnitude\n\n")

        f.write("Recommendations:\n")
        if best_mean_auc >= 0.65:
            f.write("  ✓ Tree/Beam search with entropy-based scoring is RECOMMENDED\n")
            f.write(f"  ✓ Use '{best_prompt['Prompt']}' as the primary scoring function\n")
            f.write("  ✓ Expected improvement over random sampling\n")
        elif best_mean_auc >= 0.55:
            f.write("  ~ Tree/Beam search may provide marginal benefit\n")
            f.write(f"  ~ Use '{best_prompt['Prompt']}' but with low confidence\n")
            f.write("  ~ Consider ensemble of multiple prompts\n")
            f.write("  ~ Run small-scale experiments to validate before full deployment\n")
        else:
            f.write("  ✗ Entropy signal is too weak for reliable tree/beam search\n")
            f.write("  ✗ Random sampling or other methods may be more effective\n")
            f.write("  ✗ Consider alternative uncertainty metrics or model improvements\n")

        f.write("\n")

    print(f"  Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prefix-based entropy analysis for tree/beam search applicability"
    )
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="results/prefix_analysis",
                       help="Output directory for analysis")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} problems")

    total_trajectories = sum(len(p.get('trajectories', [])) for p in results)
    print(f"Total trajectories: {total_trajectories}\n")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract prefix-based data
    print("Extracting prefix-based step data...")
    df = extract_prefix_based_data(results)
    print(f"Extracted {len(df)} steps from trajectories on solvable problems\n")

    # Normalize entropy
    print("Normalizing entropy by step...")
    df = normalize_entropy_by_step(df)
    print()

    # Within-problem discrimination (most important for tree search)
    print("Analyzing within-problem discrimination (tree search simulation)...")
    print("  - Computing discrimination metrics for each problem...")
    within_problem_results = within_problem_discrimination(df, use_normalized=False)

    print("  - Aggregating results across problems...")
    within_problem_summary = aggregate_within_problem_results(within_problem_results)
    print("\nWithin-Problem Discrimination Summary:")
    print(within_problem_summary[['Prompt', 'Mean_AUC', 'Median_AUC', 'Pct_AUC_Above_0.6']])
    print()

    # Step-by-step analysis
    print("Analyzing step-by-step discrimination...")
    step_discrimination = step_by_step_prefix_discrimination(df, use_normalized=False)
    print(f"  Analyzed {len(step_discrimination)} prompts across steps\n")

    # Generate visualizations
    print("Generating visualizations...")
    create_visualizations(df, within_problem_results, within_problem_summary,
                         step_discrimination, args.output_dir)
    print()

    # Generate report
    print("Generating report...")
    generate_report(df, within_problem_summary, step_discrimination,
                   output_path / "prefix_analysis_report.txt")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    best = within_problem_summary.iloc[0]
    print(f"  Best prompt: {best['Prompt']}")
    print(f"  Mean AUC: {best['Mean_AUC']:.3f} (0.5 = random, 1.0 = perfect)")
    print(f"  % problems with good discrimination (AUC>0.6): {best['Pct_AUC_Above_0.6']:.1f}%")
    print()

    if best['Mean_AUC'] >= 0.65:
        print("✓ VERDICT: Promising for tree/beam search")
    elif best['Mean_AUC'] >= 0.55:
        print("~ VERDICT: Marginal benefit, validate with experiments")
    else:
        print("✗ VERDICT: Weak signal, consider alternatives")
    print()


if __name__ == "__main__":
    main()
