"""
Core inference pipeline with entropy tracking and multi-injection prompts.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from vllm import LLM

from math_verify import parse

from .config import INJECTION_PROMPTS, VERIFICATION_PARAMS, MAX_STEPS, BEAM_WIDTH, BEAM_CANDIDATES_PER_BEAM, BEAM_SELECTION_INJECTION
from .entropy import calculate_entropy, calculate_token_entropies, get_entropy_statistics, _extract_logprob_value
from .verification import is_solution_finished


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

    # Parse returns [sympy_obj, str_fallback] or empty list on failure
    if generated_text.strip():
        result = parse(generated_text)
        if isinstance(result, list) and len(result) >= 2:
            results['final_answer'] = result[1]  # Return string fallback
        elif isinstance(result, list) and len(result) == 1:
            results['final_answer'] = str(result[0])  # Convert sympy to string
        else:
            results['final_answer'] = None
    else:
        results['final_answer'] = None

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

        # Store top-k probabilities with decoded tokens as keys
        top_k_probs = {}
        for token_id, logprob in first_token_logprobs.items():
            # Extract decoded token text
            if hasattr(logprob, 'decoded_token'):
                token_text = logprob.decoded_token
            else:
                # Fallback to token_id if decoded_token not available
                token_text = str(token_id)

            prob = float(np.exp(_extract_logprob_value(logprob)))
            top_k_probs[token_text] = prob

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


def solve_with_beam_search(
    llm: LLM,
    problem: Dict[str, Any],
    system_prompt: str,
    sampling_params,
    beam_width: int = None,
    candidates_per_beam: int = None,
) -> Dict[str, Any]:
    """
    Solve a math problem using beam search guided by step_quality injection entropy.

    At each step, generates multiple candidates from each beam, evaluates them using
    the step_quality injection prompt, and keeps the top beam_width candidates with
    lowest entropy.

    Args:
        llm: vLLM instance
        problem: Problem dictionary with 'id', 'problem', 'answer' keys
        system_prompt: System prompt for problem solving
        sampling_params: Sampling parameters for generation
        beam_width: Number of beams to maintain (default: BEAM_WIDTH from config)
        candidates_per_beam: Number of candidates to sample from each beam (default: BEAM_CANDIDATES_PER_BEAM)

    Returns:
        Dictionary containing:
            - problem_id: Problem identifier
            - problem_text: Original problem text
            - gold_answer: Gold standard answer
            - beams: List of beam results (each with steps, solution, answer, avg_entropy)
    """
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if candidates_per_beam is None:
        candidates_per_beam = BEAM_CANDIDATES_PER_BEAM

    problem_id = problem.get('id', problem.get('problem_id', 'unknown'))
    problem_text = problem['problem']
    gold_answer = problem['answer']

    # Initialize: single beam with empty generation
    initial_prompt = system_prompt + "\n\n" + problem_text + "\n\n"
    beams = [{
        'prompt': initial_prompt,
        'generated_text': '',
        'steps': [],
        'cumulative_entropy': 0.0,
    }]

    step_count = 0

    # Beam search loop
    while step_count < MAX_STEPS:
        step_count += 1

        # Check if all beams are finished
        if all(is_solution_finished(beam['generated_text'], MAX_STEPS) for beam in beams):
            break

        all_candidates = []

        # Generate candidates from each beam
        for beam in beams:
            # Skip if this beam is already finished
            if is_solution_finished(beam['generated_text'], MAX_STEPS):
                # Keep the finished beam as-is with lowest entropy (highest priority)
                all_candidates.append({
                    'beam': beam,
                    'step_text': '',
                    'step_info': None,
                    'selection_entropy': 0.0,  # Finished beams have highest priority
                    'finished': True,
                })
                continue

            # Generate multiple candidates from this beam
            prompts = [beam['prompt']] * candidates_per_beam
            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                step_text = output.outputs[0].text

                # Calculate token-level entropies
                token_logprobs = output.outputs[0].logprobs
                if token_logprobs:
                    token_entropies = calculate_token_entropies(token_logprobs)
                    entropy_stats = get_entropy_statistics(token_entropies)
                else:
                    token_entropies = []
                    entropy_stats = get_entropy_statistics([])

                # Apply step_quality injection prompt
                injection_prompt = INJECTION_PROMPTS[BEAM_SELECTION_INJECTION]
                injection_result = apply_injection_prompt(
                    llm,
                    beam['prompt'],
                    step_text,
                    injection_prompt
                )
                selection_entropy = injection_result['entropy']

                # Create step info
                step_info = {
                    'step_number': step_count,
                    'text': step_text,
                    'token_entropies': token_entropies,
                    'avg_entropy': entropy_stats['mean'],
                    'max_entropy': entropy_stats['max'],
                    'min_entropy': entropy_stats['min'],
                    'median_entropy': entropy_stats['median'],
                    'std_entropy': entropy_stats['std'],
                    'selection_entropy': selection_entropy,
                    'selection_injection_result': injection_result,
                }

                # Create new beam candidate
                new_beam = {
                    'prompt': beam['prompt'] + step_text + "\n\n",
                    'generated_text': beam['generated_text'] + step_text + "\n\n",
                    'steps': beam['steps'] + [step_info],
                    'cumulative_entropy': beam['cumulative_entropy'] + selection_entropy,
                }

                all_candidates.append({
                    'beam': new_beam,
                    'step_text': step_text,
                    'step_info': step_info,
                    'selection_entropy': selection_entropy,
                    'finished': False,
                })

        # Select top beam_width candidates by lowest selection_entropy
        all_candidates.sort(key=lambda x: x['selection_entropy'])
        beams = [candidate['beam'] for candidate in all_candidates[:beam_width]]

    # Process final results for all beams
    beam_results = []
    for beam in beams:
        generated_solution = beam['generated_text'].strip()

        # Parse final answer
        if generated_solution:
            result = parse(generated_solution)
            if isinstance(result, list) and len(result) >= 2:
                final_answer = result[1]
            elif isinstance(result, list) and len(result) == 1:
                final_answer = str(result[0])
            else:
                final_answer = None
        else:
            final_answer = None

        # Calculate average entropy for this beam
        if beam['steps']:
            avg_entropy = beam['cumulative_entropy'] / len(beam['steps'])
        else:
            avg_entropy = 0.0

        beam_results.append({
            'steps': beam['steps'],
            'generated_solution': generated_solution,
            'final_answer': final_answer,
            'cumulative_entropy': beam['cumulative_entropy'],
            'avg_entropy': avg_entropy,
        })

    return {
        'problem_id': problem_id,
        'problem_text': problem_text,
        'gold_answer': gold_answer,
        'beams': beam_results,
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


def batch_solve_problems_beam_search(
    llm: LLM,
    problems: List[Dict[str, Any]],
    system_prompt: str,
    sampling_params,
    beam_width: int = None,
    candidates_per_beam: int = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Solve multiple problems using beam search with entropy tracking.

    Args:
        llm: vLLM instance
        problems: List of problem dictionaries
        system_prompt: System prompt
        sampling_params: Sampling parameters
        beam_width: Number of beams to maintain
        candidates_per_beam: Number of candidates to sample from each beam
        verbose: Whether to print progress

    Returns:
        List of result dictionaries
    """
    results = []

    for idx, problem in enumerate(problems):
        if verbose:
            print(f"Processing problem {idx+1}/{len(problems)} with beam search...")

        result = solve_with_beam_search(
            llm,
            problem,
            system_prompt,
            sampling_params,
            beam_width,
            candidates_per_beam
        )
        results.append(result)

    return results
