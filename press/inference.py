"""
Core inference pipeline with entropy tracking and multi-injection prompts.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from vllm import LLM

from .config import INJECTION_PROMPTS, VERIFICATION_PARAMS, MAX_STEPS
from .entropy import calculate_entropy, calculate_token_entropies, get_entropy_statistics
from .verification import parse_boxed_answer, is_solution_finished


def solve_with_entropy_tracking(
    llm: LLM,
    problem: Dict[str, Any],
    system_prompt: str,
    sampling_params,
) -> Dict[str, Any]:
    """
    Solve a math problem with detailed entropy tracking and multiple injection prompts.

    Each reasoning step is generated, then all injection prompts are applied to measure
    model uncertainty from different perspectives.

    Args:
        llm: vLLM instance
        problem: Problem dictionary with 'id', 'problem', 'answer' keys
        system_prompt: System prompt for problem solving
        sampling_params: Sampling parameters for generation

    Returns:
        Dictionary containing:
            - problem_id: Problem identifier
            - problem_text: Original problem text
            - gold_answer: Gold standard answer
            - steps: List of step information with entropies
            - generated_solution: Full generated text
            - final_answer: Extracted final answer
    """
    results = {
        'problem_id': problem.get('id', problem.get('problem_id', 'unknown')),
        'problem_text': problem['problem'],
        'gold_answer': problem['answer'],
        'steps': [],
        'final_answer': None,
        'generated_solution': '',
    }

    # Initialize prompt
    current_prompt = system_prompt + "\n\n" + problem['problem'] + "\n\n"
    generated_text = ""
    step_count = 0

    # Generate solution step by step
    while not is_solution_finished(generated_text, MAX_STEPS) and step_count < MAX_STEPS:
        step_count += 1

        # 1) Generate until "\n\n" (one step/paragraph)
        outputs = llm.generate([current_prompt], sampling_params)
        output = outputs[0]
        step_text = output.outputs[0].text

        # 2) Collect token-level logprobs & calculate entropy
        token_logprobs = output.outputs[0].logprobs
        if token_logprobs:
            token_entropies = calculate_token_entropies(token_logprobs)
            entropy_stats = get_entropy_statistics(token_entropies)
        else:
            token_entropies = []
            entropy_stats = get_entropy_statistics([])

        # 3) Apply ALL injection prompts
        injection_results = {}
        for prompt_name, injection_prompt in INJECTION_PROMPTS.items():
            injection_result = apply_injection_prompt(
                llm,
                current_prompt,
                step_text,
                injection_prompt
            )
            injection_results[prompt_name] = injection_result

        # 4) Store step information
        step_info = {
            'step_number': step_count,
            'text': step_text,
            'token_entropies': token_entropies,
            'avg_entropy': entropy_stats['mean'],
            'max_entropy': entropy_stats['max'],
            'min_entropy': entropy_stats['min'],
            'median_entropy': entropy_stats['median'],
            'std_entropy': entropy_stats['std'],
            'injection_results': injection_results,
        }
        results['steps'].append(step_info)

        # 5) Continue generation
        generated_text += step_text + "\n\n"
        current_prompt += step_text + "\n\n"

    # 6) Parse final answer
    results['generated_solution'] = generated_text.strip()
    results['final_answer'] = parse_boxed_answer(generated_text)

    return results


def apply_injection_prompt(
    llm: LLM,
    current_prompt: str,
    step_text: str,
    injection_prompt: str
) -> Dict[str, Any]:
    """
    Apply a single injection prompt and measure entropy.

    Args:
        llm: vLLM instance
        current_prompt: Current conversation prompt
        step_text: Text of the current step
        injection_prompt: Injection prompt to apply

    Returns:
        Dictionary with entropy, predicted value, and top-k probabilities
    """
    # Construct verification prompt
    verification_prompt = current_prompt + step_text + "\n\n" + injection_prompt

    # Generate with greedy decoding
    outputs = llm.generate([verification_prompt], VERIFICATION_PARAMS)
    output = outputs[0]

    # Get the first token's logprobs (this is the confidence score/answer)
    if output.outputs[0].logprobs and len(output.outputs[0].logprobs) > 0:
        first_token_logprobs = output.outputs[0].logprobs[0]
        injection_entropy = calculate_entropy(first_token_logprobs)

        # Store top-k probabilities
        top_k_probs = {
            str(token_id): float(np.exp(logprob))
            for token_id, logprob in first_token_logprobs.items()
        }
        # Normalize
        total_prob = sum(top_k_probs.values())
        top_k_probs = {k: v/total_prob for k, v in top_k_probs.items()}
    else:
        injection_entropy = 0.0
        top_k_probs = {}

    # Get predicted value
    predicted_token = output.outputs[0].text.strip()

    return {
        'entropy': injection_entropy,
        'predicted_value': predicted_token,
        'top_k_probs': top_k_probs,
    }


def batch_solve_problems(
    llm: LLM,
    problems: List[Dict[str, Any]],
    system_prompt: str,
    sampling_params,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Solve multiple problems with entropy tracking.

    Args:
        llm: vLLM instance
        problems: List of problem dictionaries
        system_prompt: System prompt
        sampling_params: Sampling parameters
        verbose: Whether to print progress

    Returns:
        List of result dictionaries
    """
    results = []

    for idx, problem in enumerate(problems):
        if verbose:
            print(f"Processing problem {idx+1}/{len(problems)}...")

        result = solve_with_entropy_tracking(
            llm,
            problem,
            system_prompt,
            sampling_params
        )
        results.append(result)

    return results
