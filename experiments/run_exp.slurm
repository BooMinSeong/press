#!/bin/bash
#SBATCH --job-name=toy-q7b # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80,n68
#SBATCH --gres=gpu:4                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=16             # CPU cores/threads (4 GPUs * 4 cores each)

#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec


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

# CUDA 가시성 설정 - SLURM 멀티프로세싱 이슈 해결
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================================================
# NEW APPROACH: Separate generation and evaluation for better flexibility
# ============================================================================

# # Step 1: Generate solutions (expensive, run once)
# python experiments/generate.py \
#   --num-problems 500 \
#   --num-samples 32 \
#   --temperature 1.0 \
#   --output results/pass_32_7B.json

# # Step 2: Evaluate generated solutions (free, can run multiple times)
# python experiments/evaluate.py \
#   results/pass_32_7B.json \
#   --output results/pass_32_7B.json

# # Step 3: Analyze results
# python experiments/analyze_results.py \
#     results/pass_32_7B.json \
#     --output-dir results/analysis_pass_32_7B  


# # Beam search experiment
# # Step 1: Generate with beam search
# python experiments/generate.py \
#   --use-beam-search \
#   --output results/beam_search_032_generation.json \
#   --num-problems 500 \
#   --temperature 1.0 \
#   --beam-width 4 \
#   --candidates-per-beam 8

# # Step 2: Evaluate beams
# python experiments/evaluate.py \
#   results/beam_search_032_generation.json \
#   --output results/beam_search_032.json

# # Step 3: Analyze results
# python experiments/analyze_results.py \
#     results/beam_search_032.json \
#     --output-dir results/analysis_beam_search_032


# ============================================================================
# OLD APPROACH (DEPRECATED): Combined generation + evaluation
# Kept for reference - can be deleted after verifying new approach works
# ============================================================================

python experiments/run_experiment.py \
  --num-problems 500 \
  --num-samples 32 \
  --temperature 1.0 \
  --output results/pass_at_32_7B.json

python experiments/analyze_results.py \
    results/pass_at_32_7B.json \
    --output-dir results/analysis_pass_at_32_7B
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
