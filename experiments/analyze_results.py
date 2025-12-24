"""
Step-by-step injection entropy analysis for distinguishing correct/incorrect trajectories.

Focus: Analyze how well injection entropy at each reasoning step discriminates
       between correct and incorrect solution trajectories in Best-of-N sampling.

Usage:
    python experiments/analyze_results.py results/experiment_001.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

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


def basic_performance_analysis(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze basic performance metrics.
    Supports both multi-trajectory ('trajectories') and beam search ('beams') formats.

    Returns:
        DataFrame with performance statistics
    """
    # Flatten trajectories/beams from all problems
    all_trajectories = []
    for problem in results:
        # Support both 'trajectories' (multi-trajectory) and 'beams' (beam search)
        trajectories = problem.get('trajectories', problem.get('beams', []))

        for traj in trajectories:
            all_trajectories.append({
                'problem_id': problem.get('problem_id', problem.get('id', 'unknown')),
                'is_correct': traj.get('is_correct', False),
                'num_steps': len(traj.get('steps', [])),
            })

    total = len(all_trajectories)
    if total == 0:
        return pd.DataFrame({'Value': ['No trajectories/beams found']}, index=['Error'])

    correct = sum(1 for t in all_trajectories if t['is_correct'])
    accuracy = correct / total if total > 0 else 0

    # Steps distribution
    correct_steps = [t['num_steps'] for t in all_trajectories if t['is_correct']]
    incorrect_steps = [t['num_steps'] for t in all_trajectories if not t['is_correct']]

    # Count unique problems
    unique_problems = len(results)
    problems_with_correct = sum(
        1 for p in results
        if any(t.get('is_correct', False) for t in p.get('trajectories', p.get('beams', [])))
    )

    # Trajectories/beams per problem
    trajectories_per_problem = np.mean([
        len(p.get('trajectories', p.get('beams', []))) for p in results
    ])

    # Determine if this is beam search or multi-trajectory
    is_beam_search = 'beams' in results[0] if results else False
    traj_label = 'Beams' if is_beam_search else 'Trajectories'

    stats_dict = {
        'Total Problems': unique_problems,
        f'Total {traj_label}': total,
        f'{traj_label} per Problem': f"{trajectories_per_problem:.1f}",
        f'Correct {traj_label}': correct,
        f'Incorrect {traj_label}': total - correct,
        f'{traj_label} Accuracy': f"{accuracy:.2%}",
        'Problems with ≥1 Correct': f"{problems_with_correct}/{unique_problems}",
        'Problem Success Rate': f"{problems_with_correct/unique_problems:.2%}" if unique_problems > 0 else "N/A",
        'Avg Steps (All)': f"{np.mean([t['num_steps'] for t in all_trajectories]):.2f}",
        'Avg Steps (Correct)': f"{np.mean(correct_steps):.2f}" if correct_steps else "N/A",
        'Avg Steps (Incorrect)': f"{np.mean(incorrect_steps):.2f}" if incorrect_steps else "N/A",
    }

    return pd.DataFrame([stats_dict]).T.rename(columns={0: 'Value'})


def extract_step_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract step-level data from all trajectories/beams.
    Supports both multi-trajectory ('trajectories') and beam search ('beams') formats.

    Returns:
        DataFrame with columns: problem_id, trajectory_idx, is_correct, step_number,
                                avg_entropy, and injection entropies
    """
    step_data = []

    for problem in results:
        problem_id = problem.get('problem_id', problem.get('id', 'unknown'))

        # Support both 'trajectories' (multi-trajectory) and 'beams' (beam search)
        trajectories = problem.get('trajectories', problem.get('beams', []))

        for traj_idx, traj in enumerate(trajectories):
            is_correct = traj.get('is_correct', False)

            for step in traj.get('steps', []):
                step_number = step['step_number']

                # Base step info
                step_info = {
                    'problem_id': problem_id,
                    'trajectory_idx': traj_idx,
                    'is_correct': is_correct,
                    'step_number': step_number,
                    'avg_entropy': step.get('avg_entropy', np.nan),
                    'max_entropy': step.get('max_entropy', np.nan),
                    'min_entropy': step.get('min_entropy', np.nan),
                    'median_entropy': step.get('median_entropy', np.nan),
                    'std_entropy': step.get('std_entropy', np.nan),
                }

                # Add selection_entropy for beam search
                if 'selection_entropy' in step:
                    step_info['selection_entropy'] = step['selection_entropy']

                # Add injection entropies
                # For beam search, only selection injection is available
                if 'selection_injection_result' in step:
                    # Beam search format
                    inj_result = step['selection_injection_result']
                    # Get the injection prompt name from config
                    from press.config import BEAM_SELECTION_INJECTION
                    step_info[f'{BEAM_SELECTION_INJECTION}_entropy'] = inj_result['entropy']
                elif 'injection_results' in step:
                    # Multi-trajectory format
                    for prompt_name in INJECTION_PROMPTS.keys():
                        if prompt_name in step.get('injection_results', {}):
                            inj_result = step['injection_results'][prompt_name]
                            step_info[f'{prompt_name}_entropy'] = inj_result['entropy']
                        else:
                            step_info[f'{prompt_name}_entropy'] = np.nan

                step_data.append(step_info)

    return pd.DataFrame(step_data)


def normalize_entropy_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize injection entropies by step to remove step number effect.

    For each step and each injection prompt:
        normalized_entropy = (raw_entropy - mean_at_step) / std_at_step

    This removes the confounding effect of "being closer to the answer"
    at later steps, isolating the true discrimination ability.

    Adds columns: {prompt_name}_entropy_norm

    Returns:
        DataFrame with additional normalized entropy columns
    """
    df = df.copy()

    for prompt_name in INJECTION_PROMPTS.keys():
        entropy_col = f'{prompt_name}_entropy'
        norm_col = f'{prompt_name}_entropy_norm'

        if entropy_col not in df.columns:
            continue

        # Group by step_number and normalize within each step
        df[norm_col] = df.groupby('step_number')[entropy_col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return df


def step_by_step_discrimination(df: pd.DataFrame, use_normalized: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Analyze how well entropy discriminates correct/incorrect at each step.

    Args:
        df: DataFrame with step data
        use_normalized: If True, use normalized entropy; if False, use raw entropy

    Returns:
        Dictionary mapping prompt names to DataFrames with step-by-step metrics
    """
    results = {}

    # Analyze each injection prompt
    for prompt_name in INJECTION_PROMPTS.keys():
        # Select entropy column based on normalization
        if use_normalized:
            entropy_col = f'{prompt_name}_entropy_norm'
        else:
            entropy_col = f'{prompt_name}_entropy'

        if entropy_col not in df.columns:
            continue

        # Group by step number
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

            # Cohen's d (effect size)
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

            # AUC (lower entropy = more correct, so we reverse)
            try:
                # Reverse entropy so higher = better (for AUC calculation)
                reversed_entropy = -step_df[entropy_col]
                auc = roc_auc_score(step_df['is_correct'], reversed_entropy)
            except:
                auc = np.nan

            step_metrics.append({
                'Step': step_num,
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


def injection_prompt_effectiveness(df: pd.DataFrame, use_normalized: bool = False) -> pd.DataFrame:
    """
    Compare overall effectiveness of each injection prompt.

    Args:
        df: DataFrame with step data
        use_normalized: If True, use normalized entropy; if False, use raw entropy

    Returns:
        DataFrame with effectiveness metrics for each prompt
    """
    effectiveness_metrics = []

    for prompt_name in INJECTION_PROMPTS.keys():
        # Select entropy column based on normalization
        if use_normalized:
            entropy_col = f'{prompt_name}_entropy_norm'
        else:
            entropy_col = f'{prompt_name}_entropy'

        if entropy_col not in df.columns:
            continue

        # Remove NaN values
        valid_df = df.dropna(subset=[entropy_col])

        if len(valid_df) < 2:
            continue

        correct_entropy = valid_df[valid_df['is_correct']][entropy_col]
        incorrect_entropy = valid_df[~valid_df['is_correct']][entropy_col]

        if len(correct_entropy) == 0 or len(incorrect_entropy) == 0:
            continue

        # Overall Pearson correlation
        try:
            pearson_r, pearson_p = stats.pearsonr(
                valid_df[entropy_col],
                valid_df['is_correct'].astype(int)
            )
        except:
            pearson_r, pearson_p = np.nan, np.nan

        # Overall Cohen's d
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

        # Overall AUC
        try:
            reversed_entropy = -valid_df[entropy_col]
            auc = roc_auc_score(valid_df['is_correct'], reversed_entropy)
        except:
            auc = np.nan

        effectiveness_metrics.append({
            'Prompt': prompt_name,
            'Sample_Size': len(valid_df),
            'Mean_Entropy_Correct': mean_correct,
            'Mean_Entropy_Incorrect': mean_incorrect,
            'Entropy_Difference': mean_incorrect - mean_correct,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Cohens_d': cohens_d,
            'AUC': auc,
        })

    results_df = pd.DataFrame(effectiveness_metrics)

    # Sort by absolute Pearson r (descending)
    results_df = results_df.sort_values('Pearson_r', ascending=False, key=abs)

    return results_df


def create_visualizations(df: pd.DataFrame, step_discrimination: Dict[str, pd.DataFrame],
                         effectiveness: pd.DataFrame, output_dir: str = "results/plots",
                         use_normalized: bool = False):
    """
    Create visualization plots focusing on step-by-step entropy discrimination.

    Args:
        df: DataFrame with step data
        step_discrimination: Step-by-step discrimination results
        effectiveness: Overall effectiveness results
        output_dir: Output directory for plots
        use_normalized: If True, use normalized entropy; if False, use raw entropy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Determine entropy column suffix
    entropy_suffix = '_entropy_norm' if use_normalized else '_entropy'
    title_prefix = 'Normalized' if use_normalized else 'Raw'

    print("  - Creating visualizations...")

    # 1. Overall entropy distribution: Correct vs Incorrect
    print("    * Entropy distribution (correct vs incorrect)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, prompt_name in enumerate(INJECTION_PROMPTS.keys()):
        if idx >= 6:
            break

        entropy_col = f'{prompt_name}{entropy_suffix}'
        if entropy_col not in df.columns:
            continue

        plot_df = df.dropna(subset=[entropy_col]).copy()
        plot_df['Correctness'] = plot_df['is_correct'].map({True: 'Correct', False: 'Incorrect'})

        ax = axes[idx]
        sns.violinplot(data=plot_df, x='Correctness', y=entropy_col, ax=ax, palette='Set2')
        ax.set_title(f'{prompt_name}')
        ax.set_ylabel(f'{title_prefix} Entropy')
        ax.set_xlabel('')

    plt.suptitle(f'{title_prefix} Injection Entropy Distribution: Correct vs Incorrect Trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'entropy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Step-by-step entropy evolution
    print("    * Step-by-step entropy evolution")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, prompt_name in enumerate(INJECTION_PROMPTS.keys()):
        if idx >= 6:
            break

        entropy_col = f'{prompt_name}{entropy_suffix}'
        if entropy_col not in df.columns:
            continue

        plot_df = df.dropna(subset=[entropy_col]).copy()
        plot_df['Correctness'] = plot_df['is_correct'].map({True: 'Correct', False: 'Incorrect'})

        ax = axes[idx]
        sns.lineplot(data=plot_df, x='step_number', y=entropy_col,
                    hue='Correctness', ax=ax, errorbar='se', palette='Set1')
        ax.set_title(f'{prompt_name}')
        ax.set_ylabel(f'Mean {title_prefix} Entropy')
        ax.set_xlabel('Step Number')
        ax.legend(title='', loc='best')

    plt.suptitle(f'{title_prefix} Entropy Evolution Across Steps: Correct vs Incorrect', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'entropy_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Injection prompt effectiveness comparison
    print("    * Injection prompt effectiveness")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pearson r
    axes[0].barh(effectiveness['Prompt'], effectiveness['Pearson_r'], color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Pearson Correlation (r)')
    axes[0].set_title('Correlation with Correctness')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)

    # Cohen's d
    axes[1].barh(effectiveness['Prompt'], effectiveness['Cohens_d'], color='coral', alpha=0.8)
    axes[1].set_xlabel("Cohen's d")
    axes[1].set_title('Effect Size')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(axis='x', alpha=0.3)

    # AUC
    axes[2].barh(effectiveness['Prompt'], effectiveness['AUC'], color='mediumseagreen', alpha=0.8)
    axes[2].set_xlabel('AUC')
    axes[2].set_title('Discrimination Ability')
    axes[2].axvline(x=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].set_xlim(0, 1)

    plt.suptitle(f'{title_prefix} Injection Prompt Effectiveness Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'injection_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Step-by-step correlation evolution
    print("    * Step-by-step correlation evolution")

    if step_discrimination:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (prompt_name, step_df) in enumerate(step_discrimination.items()):
            if idx >= 6:
                break

            ax = axes[idx]

            # Plot Pearson r
            ax.plot(step_df['Step'], step_df['Pearson_r'],
                   marker='o', linewidth=2, markersize=6, label='Pearson r', color='steelblue')

            # Add shaded area for significance
            significant = step_df['Pearson_p'] < 0.05
            if significant.any():
                ax.fill_between(step_df['Step'],
                               step_df['Pearson_r'].where(significant),
                               0, alpha=0.2, color='steelblue', label='p < 0.05')

            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Step Number')
            ax.set_ylabel('Pearson r')
            ax.set_title(f'{prompt_name}')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8)

        plt.suptitle(f'{title_prefix} Step-by-Step Correlation Evolution (Pearson r)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'step_correlation_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Step-by-step effect size evolution
    print("    * Step-by-step effect size evolution")

    if step_discrimination:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (prompt_name, step_df) in enumerate(step_discrimination.items()):
            if idx >= 6:
                break

            ax = axes[idx]

            # Plot Cohen's d
            ax.plot(step_df['Step'], step_df['Cohens_d'],
                   marker='s', linewidth=2, markersize=6, color='coral')

            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            # Positive thresholds (incorrect > correct entropy)
            ax.axhline(y=0.2, color='green', linestyle=':', linewidth=0.5, alpha=0.3)
            ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=0.5, alpha=0.3)
            ax.axhline(y=0.8, color='red', linestyle=':', linewidth=0.5, alpha=0.3)
            # Negative thresholds (correct < incorrect entropy, desired)
            ax.axhline(y=-0.2, color='green', linestyle=':', linewidth=0.5, alpha=0.5, label='Small')
            ax.axhline(y=-0.5, color='orange', linestyle=':', linewidth=0.5, alpha=0.5, label='Medium')
            ax.axhline(y=-0.8, color='red', linestyle=':', linewidth=0.5, alpha=0.5, label='Large')

            ax.set_xlabel('Step Number')
            ax.set_ylabel("Cohen's d")
            ax.set_title(f'{prompt_name}')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8)

        plt.suptitle(f"{title_prefix} Step-by-Step Effect Size Evolution (Cohen's d)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'step_effect_size_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n  Visualizations saved to: {output_path}")


def generate_report(results: List[Dict[str, Any]], df: pd.DataFrame,
                   step_discrimination_raw: Dict[str, pd.DataFrame],
                   effectiveness_raw: pd.DataFrame,
                   step_discrimination_norm: Dict[str, pd.DataFrame],
                   effectiveness_norm: pd.DataFrame,
                   output_path: str = "results/analysis/analysis_report.txt"):
    """Generate a comprehensive text report with both raw and normalized entropy analysis."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STEP-BY-STEP INJECTION ENTROPY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Basic Performance
        f.write("1. BASIC PERFORMANCE ANALYSIS\n")
        f.write("-"*80 + "\n")
        perf_stats = basic_performance_analysis(results)
        f.write(perf_stats.to_string())
        f.write("\n\n")

        # RAW ENTROPY ANALYSIS
        f.write("2. RAW ENTROPY ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("Analysis using absolute entropy values (includes step number effect).\n")
        f.write("Note: Later steps are naturally closer to the answer, which may lower entropy.\n\n")

        f.write("2.1 INJECTION PROMPT EFFECTIVENESS (RAW ENTROPY)\n")
        f.write("-"*80 + "\n")
        f.write("How well each injection prompt discriminates correct/incorrect trajectories:\n\n")

        display_cols = ['Prompt', 'Sample_Size', 'Entropy_Difference',
                       'Pearson_r', 'Pearson_p', 'Cohens_d', 'AUC']
        f.write(effectiveness_raw[display_cols].to_string(index=False))
        f.write("\n\n")
        f.write("Interpretation:\n")
        f.write("  - Pearson r: Correlation between entropy and correctness (0=False, 1=True)\n")
        f.write("               Negative = correct trajectories have lower entropy (desired)\n")
        f.write("  - Pearson p: Statistical significance (p < 0.05 = significant)\n")
        f.write("  - Cohen's d: (Mean_Correct - Mean_Incorrect) / pooled_std\n")
        f.write("               Negative = correct trajectories have lower entropy (desired)\n")
        f.write("               |d| > 0.2=small, >0.5=medium, >0.8=large effect\n")
        f.write("  - AUC: Discrimination ability (>0.5=better than random, 1.0=perfect)\n")
        f.write("\n\n")

        # Step-by-Step Raw Entropy Analysis
        f.write("2.2 STEP-BY-STEP DISCRIMINATION (RAW ENTROPY)\n")
        f.write("-"*80 + "\n")
        f.write("How discrimination ability evolves across reasoning steps.\n")
        f.write("Pearson r measures correlation between entropy and correctness at each step.\n\n")

        for prompt_name, step_df in step_discrimination_raw.items():
            f.write(f"\n{prompt_name}:\n")
            f.write("-" * 40 + "\n")

            # Show key statistics for each step
            display_df = step_df[['Step', 'N_Correct', 'N_Incorrect',
                                 'Mean_Entropy_Correct', 'Mean_Entropy_Incorrect',
                                 'Pearson_r', 'Pearson_p', 'Cohens_d', 'AUC']].copy()

            # Format for readability
            display_df['Mean_Entropy_Correct'] = display_df['Mean_Entropy_Correct'].apply(lambda x: f"{x:.4f}")
            display_df['Mean_Entropy_Incorrect'] = display_df['Mean_Entropy_Incorrect'].apply(lambda x: f"{x:.4f}")
            display_df['Pearson_r'] = display_df['Pearson_r'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['Pearson_p'] = display_df['Pearson_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['Cohens_d'] = display_df['Cohens_d'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

            if len(display_df) > 10:
                f.write(display_df.head(5).to_string(index=False))
                f.write("\n  ...\n")
                f.write(display_df.tail(5).to_string(index=False, header=False))
            else:
                f.write(display_df.to_string(index=False))
            f.write("\n")

        f.write("\n")

        # NORMALIZED ENTROPY ANALYSIS
        f.write("3. NORMALIZED ENTROPY ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("Analysis using step-normalized entropy (step number effect removed).\n")
        f.write("Normalized entropy = (raw_entropy - mean_at_step) / std_at_step\n")
        f.write("This isolates pure discrimination ability independent of step proximity to answer.\n\n")

        f.write("3.1 INJECTION PROMPT EFFECTIVENESS (NORMALIZED ENTROPY)\n")
        f.write("-"*80 + "\n\n")
        f.write(effectiveness_norm[display_cols].to_string(index=False))
        f.write("\n\n")

        f.write("3.2 STEP-BY-STEP DISCRIMINATION (NORMALIZED ENTROPY)\n")
        f.write("-"*80 + "\n\n")

        for prompt_name, step_df in step_discrimination_norm.items():
            f.write(f"\n{prompt_name}:\n")
            f.write("-" * 40 + "\n")

            # Show key statistics for each step
            display_df = step_df[['Step', 'N_Correct', 'N_Incorrect',
                                 'Mean_Entropy_Correct', 'Mean_Entropy_Incorrect',
                                 'Pearson_r', 'Pearson_p', 'Cohens_d', 'AUC']].copy()

            # Format for readability
            display_df['Mean_Entropy_Correct'] = display_df['Mean_Entropy_Correct'].apply(lambda x: f"{x:.4f}")
            display_df['Mean_Entropy_Incorrect'] = display_df['Mean_Entropy_Incorrect'].apply(lambda x: f"{x:.4f}")
            display_df['Pearson_r'] = display_df['Pearson_r'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['Pearson_p'] = display_df['Pearson_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['Cohens_d'] = display_df['Cohens_d'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

            if len(display_df) > 10:
                f.write(display_df.head(5).to_string(index=False))
                f.write("\n  ...\n")
                f.write(display_df.tail(5).to_string(index=False, header=False))
            else:
                f.write(display_df.to_string(index=False))
            f.write("\n")

        f.write("\n")

        # Key Findings Summary
        f.write("4. COMPARISON & KEY FINDINGS\n")
        f.write("="*80 + "\n\n")

        # Best overall prompt (Raw)
        best_prompt_raw = effectiveness_raw.iloc[0]
        best_prompt_norm = effectiveness_norm.iloc[0]

        f.write("Best Discrimination (Raw Entropy):\n")
        f.write(f"  - Prompt: {best_prompt_raw['Prompt']}\n")
        f.write(f"  - Pearson r: {best_prompt_raw['Pearson_r']:.4f}\n")
        f.write(f"  - Cohen's d: {best_prompt_raw['Cohens_d']:.4f}\n")
        f.write(f"  - AUC: {best_prompt_raw['AUC']:.4f}\n")
        f.write("\n")

        f.write("Best Discrimination (Normalized Entropy, step effect removed):\n")
        f.write(f"  - Prompt: {best_prompt_norm['Prompt']}\n")
        f.write(f"  - Pearson r: {best_prompt_norm['Pearson_r']:.4f}\n")
        f.write(f"  - Cohen's d: {best_prompt_norm['Cohens_d']:.4f}\n")
        f.write(f"  - AUC: {best_prompt_norm['AUC']:.4f}\n")
        f.write("\n")

        # Comparison note
        f.write("Interpretation:\n")
        f.write("  - RAW entropy shows overall discrimination including step proximity effect\n")
        f.write("  - NORMALIZED entropy shows pure discrimination ability at each step\n")
        f.write("  - If raw >> normalized: discrimination is mainly due to step proximity\n")
        f.write("  - If raw ≈ normalized: genuine discrimination ability independent of step\n")
        f.write("\n")

    print(f"  Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze step-by-step injection entropy for distinguishing correct/incorrect trajectories"
    )
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                       help="Output directory for analysis")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} problems")

    total_trajectories = sum(len(p.get('trajectories', [])) for p in results)
    print(f"Total trajectories: {total_trajectories}")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract step data
    print("Extracting step-level data...")
    df = extract_step_data(results)
    print(f"Extracted {len(df)} steps from all trajectories")
    print()

    # Normalize entropy by step
    print("Normalizing entropy by step (removing step number effect)...")
    df = normalize_entropy_by_step(df)
    print()

    # Run analyses
    print("Running analyses...")

    print("  - Basic performance analysis")
    perf_stats = basic_performance_analysis(results)
    print(perf_stats)
    print()

    print("  - RAW ENTROPY: Overall injection prompt effectiveness")
    effectiveness_raw = injection_prompt_effectiveness(df, use_normalized=False)
    print(effectiveness_raw[['Prompt', 'Pearson_r', 'Cohens_d', 'AUC']])
    print()

    print("  - RAW ENTROPY: Step-by-step discrimination analysis")
    step_discrimination_raw = step_by_step_discrimination(df, use_normalized=False)
    print(f"    Analyzed {len(step_discrimination_raw)} prompts across steps")
    print()

    print("  - NORMALIZED ENTROPY: Overall injection prompt effectiveness")
    effectiveness_norm = injection_prompt_effectiveness(df, use_normalized=True)
    print(effectiveness_norm[['Prompt', 'Pearson_r', 'Cohens_d', 'AUC']])
    print()

    print("  - NORMALIZED ENTROPY: Step-by-step discrimination analysis")
    step_discrimination_norm = step_by_step_discrimination(df, use_normalized=True)
    print(f"    Analyzed {len(step_discrimination_norm)} prompts across steps")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    print("  - Raw entropy visualizations")
    create_visualizations(df, step_discrimination_raw, effectiveness_raw,
                         output_path / "plots_raw", use_normalized=False)
    print("  - Normalized entropy visualizations")
    create_visualizations(df, step_discrimination_norm, effectiveness_norm,
                         output_path / "plots_normalized", use_normalized=True)
    print()

    # Generate report
    print("Generating report...")
    generate_report(results, df,
                   step_discrimination_raw, effectiveness_raw,
                   step_discrimination_norm, effectiveness_norm,
                   output_path / "analysis_report.txt")
    print()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
