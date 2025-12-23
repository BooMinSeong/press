"""
Configuration for system prompts, injection prompts, and model parameters.
"""

from vllm import SamplingParams

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# System prompt for math problem solving
SYSTEM_PROMPT = """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""

# Multiple injection prompts for uncertainty estimation
INJECTION_PROMPTS = {
    "confidence_score": "Reviewing my work so far with self-doubt, I rate my confidence from 1-10 as: $\\boxed{",

    "correctness_prob": "The probability that this step is correct (0-100%) is: $\\boxed{",

    "error_likelihood": "The likelihood of an error in this step on a scale of 1-10 is: $\\boxed{",

    "step_quality": "I would grade the quality of this step from A to F as: $\\boxed{",

    "binary_check": "Is this step mathematically sound? (Yes/No): $\\boxed{",

    "revision_need": "Does this step need revision? (1=definitely, 10=definitely not): $\\boxed{",
}

# Sampling parameters for generation
def get_sampling_params(temperature=0.7, max_tokens=512, stop_sequences=None):
    """
    Get sampling parameters for vLLM generation.

    Args:
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        stop_sequences: List of stop sequences (default: ["\\n\\n"])

    Returns:
        SamplingParams object
    """
    if stop_sequences is None:
        stop_sequences = ["\n\n"]

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
        logprobs=5,  # Return top-5 logprobs
    )

# Verification sampling parameters (for injection prompts)
VERIFICATION_PARAMS = SamplingParams(
    temperature=0.0,  # Greedy decoding
    max_tokens=3,     # Only need the score/answer
    logprobs=5,       # Top-5 logprobs
)

# Dataset configuration
DATASET_NAME = "HuggingFaceH4/MATH-500"
DATASET_SPLIT = "test"

# Experiment settings
MAX_STEPS = 10  # Maximum reasoning steps per problem
BATCH_SIZE = 1  # Process one problem at a time for detailed tracking
