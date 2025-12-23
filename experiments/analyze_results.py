"""
Analysis and visualization script for experiment results.

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

    Returns:
        DataFrame with performance statistics
    """
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    accuracy = correct / total if total > 0 else 0

    # Calculate average steps
    avg_steps = np.mean([len(r['steps']) for r in results])

    # Steps distribution for correct vs incorrect
    correct_steps = [len(r['steps']) for r in results if r.get('is_correct', False)]
    incorrect_steps = [len(r['steps']) for r in results if not r.get('is_correct', False)]

    stats = {
        'Total Problems': total,
        'Correct': correct,
        'Incorrect': total - correct,
        'Accuracy': f"{accuracy:.2%}",
        'Avg Steps': f"{avg_steps:.2f}",
        'Avg Steps (Correct)': f"{np.mean(correct_steps):.2f}" if correct_steps else "N/A",
        'Avg Steps (Incorrect)': f"{np.mean(incorrect_steps):.2f}" if incorrect_steps else "N/A",
    }

    return pd.DataFrame([stats]).T.rename(columns={0: 'Value'})


def entropy_analysis(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Analyze entropy patterns.

    Returns:
        Dictionary of analysis DataFrames
    """
    # Collect entropy data
    entropy_data = []

    for result in results:
        is_correct = result.get('is_correct', False)

        for step in result['steps']:
            entropy_data.append({
                'problem_id': result['problem_id'],
                'is_correct': is_correct,
                'step_number': step['step_number'],
                'avg_entropy': step['avg_entropy'],
                'max_entropy': step['max_entropy'],
                'min_entropy': step['min_entropy'],
                'median_entropy': step['median_entropy'],
                'std_entropy': step['std_entropy'],
            })

    df = pd.DataFrame(entropy_data)

    # Group by correctness
    correct_entropy = df[df['is_correct']]['avg_entropy']
    incorrect_entropy = df[~df['is_correct']]['avg_entropy']

    entropy_stats = pd.DataFrame({
        'Correct': [
            correct_entropy.mean(),
            correct_entropy.median(),
            correct_entropy.std(),
            correct_entropy.max(),
            correct_entropy.min(),
        ],
        'Incorrect': [
            incorrect_entropy.mean(),
            incorrect_entropy.median(),
            incorrect_entropy.std(),
            incorrect_entropy.max(),
            incorrect_entropy.min(),
        ]
    }, index=['Mean', 'Median', 'Std', 'Max', 'Min'])

    return {
        'full_data': df,
        'summary': entropy_stats,
    }


def injection_prompt_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze injection prompt effectiveness.

    Returns:
        Dictionary with analysis results for each injection prompt
    """
    analyses = {}

    for prompt_name in INJECTION_PROMPTS.keys():
        # Collect injection entropy data
        injection_data = []

        for result in results:
            is_correct = result.get('is_correct', False)

            for step in result['steps']:
                if prompt_name in step['injection_results']:
                    inj_result = step['injection_results'][prompt_name]
                    injection_data.append({
                        'problem_id': result['problem_id'],
                        'is_correct': is_correct,
                        'step_number': step['step_number'],
                        'injection_entropy': inj_result['entropy'],
                        'predicted_value': inj_result['predicted_value'],
                    })

        df = pd.DataFrame(injection_data)

        # Calculate correlation with correctness
        correct_entropy = df[df['is_correct']]['injection_entropy']
        incorrect_entropy = df[~df['is_correct']]['injection_entropy']

        analyses[prompt_name] = {
            'data': df,
            'avg_entropy_correct': correct_entropy.mean() if len(correct_entropy) > 0 else 0,
            'avg_entropy_incorrect': incorrect_entropy.mean() if len(incorrect_entropy) > 0 else 0,
            'entropy_difference': (incorrect_entropy.mean() - correct_entropy.mean()) if len(correct_entropy) > 0 and len(incorrect_entropy) > 0 else 0,
        }

    return analyses


def calibration_analysis(results: List[Dict[str, Any]], prompt_name: str = 'confidence_score') -> pd.DataFrame:
    """
    Analyze calibration: predicted confidence vs actual accuracy.

    Args:
        results: Experiment results
        prompt_name: Which injection prompt to analyze

    Returns:
        DataFrame with calibration data
    """
    calibration_data = {}

    for result in results:
        is_correct = result.get('is_correct', False)

        for step in result['steps']:
            if prompt_name in step['injection_results']:
                pred_value = step['injection_results'][prompt_name]['predicted_value']

                # Try to parse as numeric
                try:
                    # Extract first number from string
                    import re
                    numbers = re.findall(r'\d+', pred_value)
                    if numbers:
                        pred_value = numbers[0]
                except:
                    pass

                if pred_value not in calibration_data:
                    calibration_data[pred_value] = []

                calibration_data[pred_value].append(is_correct)

    # Calculate accuracy for each predicted value
    calibration_results = []
    for pred_value, correctness_list in calibration_data.items():
        accuracy = sum(correctness_list) / len(correctness_list)
        calibration_results.append({
            'Predicted Value': pred_value,
            'Count': len(correctness_list),
            'Actual Accuracy': f"{accuracy:.2%}",
            'Accuracy (numeric)': accuracy,
        })

    df = pd.DataFrame(calibration_results)
    df = df.sort_values('Predicted Value')

    return df


def create_visualizations(results: List[Dict[str, Any]], output_dir: str = "results/plots"):
    """
    Create visualization plots.

    Args:
        results: Experiment results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # 1. Entropy distribution: Correct vs Incorrect
    entropy_data = []
    for result in results:
        is_correct = result.get('is_correct', False)
        for step in result['steps']:
            entropy_data.append({
                'Correctness': 'Correct' if is_correct else 'Incorrect',
                'Average Entropy': step['avg_entropy'],
            })

    df = pd.DataFrame(entropy_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Correctness', y='Average Entropy')
    plt.title('Entropy Distribution: Correct vs Incorrect Solutions')
    plt.tight_layout()
    plt.savefig(output_path / 'entropy_distribution.png', dpi=300)
    plt.close()

    # 2. Injection prompt comparison
    injection_analyses = injection_prompt_analysis(results)

    prompt_names = list(injection_analyses.keys())
    avg_entropy_correct = [injection_analyses[p]['avg_entropy_correct'] for p in prompt_names]
    avg_entropy_incorrect = [injection_analyses[p]['avg_entropy_incorrect'] for p in prompt_names]

    x = np.arange(len(prompt_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, avg_entropy_correct, width, label='Correct', alpha=0.8)
    ax.bar(x + width/2, avg_entropy_incorrect, width, label='Incorrect', alpha=0.8)

    ax.set_xlabel('Injection Prompt')
    ax.set_ylabel('Average Entropy')
    ax.set_title('Injection Prompt Entropy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'injection_prompt_comparison.png', dpi=300)
    plt.close()

    # 3. Entropy evolution across steps
    step_entropy_data = []
    for result in results:
        is_correct = result.get('is_correct', False)
        for step in result['steps']:
            step_entropy_data.append({
                'Step Number': step['step_number'],
                'Average Entropy': step['avg_entropy'],
                'Correctness': 'Correct' if is_correct else 'Incorrect',
            })

    df_steps = pd.DataFrame(step_entropy_data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_steps, x='Step Number', y='Average Entropy', hue='Correctness', errorbar='sd')
    plt.title('Entropy Evolution Across Reasoning Steps')
    plt.tight_layout()
    plt.savefig(output_path / 'entropy_evolution.png', dpi=300)
    plt.close()

    print(f"Visualizations saved to: {output_path}")


def generate_report(results: List[Dict[str, Any]], output_path: str = "results/analysis_report.txt"):
    """Generate a comprehensive text report."""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ENTROPY-BASED LLM MATH PROBLEM SOLVING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Basic Performance
        f.write("1. BASIC PERFORMANCE ANALYSIS\n")
        f.write("-"*80 + "\n")
        perf_stats = basic_performance_analysis(results)
        f.write(perf_stats.to_string())
        f.write("\n\n")

        # Entropy Analysis
        f.write("2. ENTROPY ANALYSIS\n")
        f.write("-"*80 + "\n")
        entropy_results = entropy_analysis(results)
        f.write(entropy_results['summary'].to_string())
        f.write("\n\n")

        # Injection Prompt Analysis
        f.write("3. INJECTION PROMPT ANALYSIS\n")
        f.write("-"*80 + "\n")
        injection_analyses = injection_prompt_analysis(results)

        for prompt_name, analysis in injection_analyses.items():
            f.write(f"\n{prompt_name}:\n")
            f.write(f"  Avg Entropy (Correct):   {analysis['avg_entropy_correct']:.4f}\n")
            f.write(f"  Avg Entropy (Incorrect): {analysis['avg_entropy_incorrect']:.4f}\n")
            f.write(f"  Difference:              {analysis['entropy_difference']:.4f}\n")

        f.write("\n")

        # Calibration Analysis
        f.write("4. CALIBRATION ANALYSIS (confidence_score)\n")
        f.write("-"*80 + "\n")
        calib_df = calibration_analysis(results, 'confidence_score')
        f.write(calib_df.to_string(index=False))
        f.write("\n\n")

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="results/analysis", help="Output directory for analysis")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run analyses
    print("Running analyses...")

    print("  - Basic performance analysis")
    perf_stats = basic_performance_analysis(results)
    print(perf_stats)
    print()

    print("  - Entropy analysis")
    entropy_results = entropy_analysis(results)
    print(entropy_results['summary'])
    print()

    print("  - Injection prompt analysis")
    injection_analyses = injection_prompt_analysis(results)
    for name, analysis in injection_analyses.items():
        print(f"    {name}: Δ={analysis['entropy_difference']:.4f}")
    print()

    print("  - Calibration analysis")
    calib_df = calibration_analysis(results)
    print(calib_df)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    create_visualizations(results, output_path / "plots")
    print()

    # Generate report
    print("Generating report...")
    generate_report(results, output_path / "analysis_report.txt")
    print()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
