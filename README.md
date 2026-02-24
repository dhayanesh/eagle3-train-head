# GPT-OSS-20B Eagle3 Summarization Training

This project documents an end-to-end Eagle3 training run for summarization with:

- Verifier model: `openai/gpt-oss-20b`
- Hardware: `2 x NVIDIA H100 SXM`
- Data source: XSum (`EdinburghNLP/xsum`, train split)
- Final training format: chat-style JSONL for Speculators data generation

## Repository Layout

- `download_data.py`: Downloads XSum train split and writes `data/raw_summaries.jsonl`
- `prep_summarization_data.py`: Converts raw samples to chat-format `data/summarization_train.jsonl`
- `speculators/scripts/data_generation_offline.py`: Generates verifier hidden-state training `.pt` files
- `speculators/scripts/build_vocab_mapping.py`: Builds `d2t.npy` and `t2d.npy`
- `speculators/scripts/train.py`: Trains Eagle3

## Dataset Details

The pipeline uses XSum articles and summaries:

- Input file: `data/raw_summaries.jsonl` with records like:
  - `{"article": "...", "summary": "..."}`
- Processed file: `data/summarization_train.jsonl` with records like:
  - `{"conversations": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`

Current local counts:

- `data/raw_summaries.jsonl`: `204045` rows
- `data/summarization_train.jsonl`: `204017` rows

## Setup

Create a Python environment, then install Speculators with data-generation extras:

```bash
cd speculators
pip install -e ".[datagen]"
```

## End-to-End Commands Used

Run dataset download and preprocessing from repo root:

```bash
python download_data.py
python prep_summarization_data.py
```

Then run generation + vocab mapping + training from `speculators/`:

```bash
cd speculators

mkdir -p ../log
ts=$(date +"%Y%m%d_%H%M%S")

python scripts/data_generation_offline.py \
  --target-model-path openai/gpt-oss-20b \
  --train-data-path ../data/summarization_train.jsonl \
  --output-dir ../output/gpt_oss_20b_summarization_eagle3/gen_50k_b2 \
  --seq-length 1024 \
  --tensor-parallel-size 2 \
  --batch-size 1 \
  --max-samples 50000 \
  > "../log/data_generation_50k_b2_${ts}.out.log" \
  2> "../log/data_generation_50k_b2_${ts}.err.log"

python scripts/build_vocab_mapping.py \
  --token-freq-path ./token_freq.pt \
  --draft-vocab-size 32000 \
  --target-model-path openai/gpt-oss-20b \
  --output-path ../output/gpt_oss_20b_summarization_eagle3/vocab_50k_b2

PYTHONPATH=./src torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --verifier-name-or-path openai/gpt-oss-20b \
  --data-path ../output/gpt_oss_20b_summarization_eagle3/gen_50k_b2 \
  --save-path ../output/gpt_oss_20b_summarization_eagle3/checkpoints_50k_b2 \
  --epochs 10 \
  --lr 2e-5 \
  --logger tensorboard \
  --total-seq-len 1024 \
  --data-format-version 1 \
  --log-dir ../output/gpt_oss_20b_summarization_eagle3/logs_50k_b2 \
  --run-name gpt_oss_20b_xsum_eagle3_50k_b2 \
  --num-layers 1 \
  --d2t-path ../output/gpt_oss_20b_summarization_eagle3/vocab_50k_b2/d2t.npy \
  --t2d-path ../output/gpt_oss_20b_summarization_eagle3/vocab_50k_b2/t2d.npy \
  --ttt-steps 3 \
  --ttt-step-loss-decay 1.0 \
  --no-resume-from-checkpoint
```

## 3-Step Training Process Summary

1. `data_generation_offline.py`: Generates training hidden states with vLLM (and preprocesses data if needed).
2. `build_vocab_mapping.py`: Creates draft/target vocabulary mappings (`d2t`, `t2d`) from token frequencies.
3. `train.py`: Trains Eagle3 on generated hidden-state data and vocab mappings.
