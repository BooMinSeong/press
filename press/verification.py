"""
Answer parsing and verification utilities using math_verify package.
"""

from math_verify import parse, verify


def evaluate_result(data: dict) -> bool:
    """
    Evaluate if the generated solution contains the correct answer.
    Uses math_verify package for robust mathematical comparison.

    Args:
        data: Dictionary containing 'gold_answer' and 'final_answer' keys

    Returns:
        True if correct, False otherwise
    """
    # Parse gold answer
    gold_result = parse("\\boxed{" + data["gold_answer"] + "}")
    if isinstance(gold_result, list) and len(gold_result) > 0:
        gold_answer = gold_result[0]
    else:
        return False

    # Parse prediction (final_answer is a string from inference.py)
    prediction = data["final_answer"]
    if prediction:
        pred_result = parse(prediction)
        if isinstance(pred_result, list) and len(pred_result) > 0:
            prediction = pred_result[0]
        else:
            return False
    else:
        return False

    return verify(gold_answer, prediction)


def is_solution_finished(text: str, max_steps: int = 10) -> bool:
    """
    Check if the solution generation is finished.

    Args:
        text: Current generated text
        max_steps: Maximum allowed steps

    Returns:
        True if finished, False otherwise
    """
    # Check if final answer is present
    if "Therefore, the final answer is:" in text:
        return True

    # Check if boxed answer is present
    if "\\boxed{" in text:
        # Make sure it's not just in an intermediate step
        lines = text.strip().split('\n')
        if len(lines) > 0:
            last_lines = '\n'.join(lines[-5:])  # Check last 5 lines
            if "Therefore" in last_lines or "final answer" in last_lines.lower():
                return True

    # Check if we've exceeded max steps
    step_count = text.count("## Step")
    if step_count >= max_steps:
        return True

    return False
