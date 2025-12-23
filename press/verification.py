"""
Answer parsing and verification utilities using math_verify package.
"""

import re
from typing import Optional
from math_verify import parse, verify


def parse_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from LaTeX \\boxed{...} notation.

    Args:
        text: Generated solution text

    Returns:
        Extracted answer string, or None if not found
    """
    # Look for \boxed{...}
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    # Fallback: try to find "final answer is: X"
    pattern2 = r'final answer is:?\s*\$?\\boxed\{([^}]+)\}'
    match2 = re.search(pattern2, text, re.IGNORECASE)

    if match2:
        return match2.group(1).strip()

    return None


def evaluate_result(data: dict) -> bool:
    """
    Evaluate if the generated solution contains the correct answer.
    Uses math_verify package for robust mathematical comparison.

    Args:
        data: Dictionary containing 'gold_answer' and 'final_answer' keys

    Returns:
        True if correct, False otherwise
    """
    gold_answer = data["gold_answer"]
    gold_answer = parse("\\boxed{" + gold_answer + "}")

    prediction = data["final_answer"]
    prediction = parse(prediction) if prediction else None

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
