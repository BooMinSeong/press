#!/bin/bash
#SBATCH --job-name=toy-bs32 # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80,n68
#SBATCH --gres=gpu:4                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=12:00:00                  # Time limit hrs:min:sec


# 환경 설정 - bashrc를 먼저 실행하여 모듈 충돌 방지

# 모듈 설정 - bashrc 이후에 필요한 모듈만 로드
module purge
module load gnu14/14.2.0
module load cuda/13.1

# source ~/.bashrc
set -x -e
source .venv/bin/activate



# 컴파일러 명시 설정
export CC=$(which gcc)
export CXX=$(which g++)

# ============================================================================
# NEW APPROACH: Separate generation and evaluation for better flexibility
# ============================================================================

# Step 1: Generate solutions (expensive, run once)
python experiments/generate.py \
  --num-problems 500 \
  --num-samples 32 \
  --temperature 1.0 \
  --output results/pass_at_32_generation.json

# Step 2: Evaluate generated solutions (free, can run multiple times)
python experiments/evaluate.py \
  results/pass_at_32_generation.json \
  --output results/pass_at_32.json

# Step 3: Analyze results
python experiments/analyze_results.py \
    results/pass_at_32.json \
    --output-dir results/analysis_pass_at_32


# Beam search experiment
# Step 1: Generate with beam search
python experiments/generate.py \
  --use-beam-search \
  --output results/beam_search_032_generation.json \
  --num-problems 500 \
  --temperature 1.0 \
  --beam-width 4 \
  --candidates-per-beam 8

# Step 2: Evaluate beams
python experiments/evaluate.py \
  results/beam_search_032_generation.json \
  --output results/beam_search_032.json

# Step 3: Analyze results
python experiments/analyze_results.py \
    results/beam_search_032.json \
    --output-dir results/analysis_beam_search_032


# ============================================================================
# OLD APPROACH (DEPRECATED): Combined generation + evaluation
# Kept for reference - can be deleted after verifying new approach works
# ============================================================================

# python experiments/run_experiment.py \
#   --num-problems 500 \
#   --num-samples 32 \
#   --temperature 1.0 \
#   --output results/pass_at_32.json
#
# python experiments/analyze_results.py \
#     results/pass_at_32.json \
#     --output-dir results/analysis_pass_at_32
#
# python experiments/run_experiment.py \
#   --use-beam-search \
#   --output results/beam_search_032.json \
#   --num-problems 500 \
#   --temperature 1.0 \
#   --beam-width 4 \
#   --candidates-per-beam 8 \
#
# python experiments/analyze_results.py \
#     results/beam_search_032.json \
#     --output-dir results/analysis_beam_search_032