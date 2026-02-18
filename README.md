# distill_videoLLM

本仓库是 CaTeR 风格的可复现实验工程。当前文档聚焦 **MSRVTT-QA** 单数据集，覆盖以下实验：

- Prompt 消融：`none / random / frame / event`
- 蒸馏策略消融：`none / hard / soft / hybrid`
- LLM 生成器消融：`gpt4 / vicuna / mistral / llama3`（基于固定视觉摘要）

## 1. 环境

```bash
cd cater_repro_minimal
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据准备（MSRVTT-QA）

### 2.1 原始数据

你需要本地具备：

- `MSRVTT-QA` 标注：`train_qa.json / val_qa.json / test_qa.json`
- `MSRVTT` 视频：`video*.mp4`

建议目录：

```text
data/msrvtt_raw/msrvtt_qa/MSRVTT-QA/
  ├─ train_qa.json
  ├─ val_qa.json
  ├─ test_qa.json
  └─ video/
      ├─ video0.mp4
      └─ ...
```

### 2.2 抽帧（224）

```bash
python scripts/extract_msrvtt_frames.py \
  --video_dir data/msrvtt_raw/msrvtt_qa/MSRVTT-QA/video \
  --out_dir data/msrvtt_frames \
  --num_frames 8 \
  --image_size 224
```

### 2.3 生成训练 JSONL（含 event prompts）

```bash
export OPENAI_API_KEY=你的key

python scripts/prepare_msrvtt_jsonl.py \
  --qa_dir data/msrvtt_raw/msrvtt_qa/MSRVTT-QA \
  --frames_root data/msrvtt_frames \
  --out_dir data/msrvtt_full_jsonl \
  --num_choices 5 \
  --num_frames 8 \
  --openai_model gpt-4o-mini
```

可选做小规模快速迭代：

```bash
python scripts/make_mini_split.py \
  --input_dir data/msrvtt_full_jsonl \
  --output_dir data/msrvtt_mini \
  --train_n 2000 --val_n 400 --test_n 400
```

## 3. 生成 Teacher Cache（Video-LLaVA）

示例（以 mini 为例）：

```bash
mkdir -p data/teacher_cache

for s in train val test; do
  python scripts/build_teacher_cache.py \
    --input_jsonl data/msrvtt_mini/${s}.jsonl \
    --output_jsonl data/teacher_cache/${s}.videollava.redo.jsonl \
    --teacher_name videollava \
    --model_id LanguageBind/Video-LLaVA-7B-hf \
    --device cuda:0 \
    --dtype fp16 \
    --num_video_frames 8 \
    --max_new_tokens 8 \
    --d_model 512
done
```

说明：

- `*.videollava.redo.jsonl` 包含 `teacher_hard_idx` 与 `teacher_embedding`。
- 当前 real teacher 已实现：`videollava`。`flamingo/chatunivl` 仍是 TODO。

## 4. 训练与评估（单次）

```bash
python train.py \
  --data_dir data/teacher_cache \
  --train_file train.videollava.redo.jsonl \
  --val_file val.videollava.redo.jsonl \
  --dataset_name msrvtt_qa \
  --tokenizer clip \
  --clip_model_id openai/clip-vit-base-patch16 \
  --image_size 224 \
  --num_frames 8 \
  --d_model 512 \
  --prompt_variant event \
  --distill_mode hybrid \
  --epochs 5 \
  --batch_size 16 \
  --out_dir runs/msrvtt_single

python eval.py \
  --data_dir data/teacher_cache \
  --split_file test.videollava.redo.jsonl \
  --checkpoint runs/msrvtt_single/best.pt
```

## 5. 一键跑三类消融 + 自动记录结果

运行：

```bash
bash scripts/run_msrvtt_suite.sh
```

默认会执行：

- Prompt 消融：`none/random/frame/event`（固定 `hybrid`）
- Distill 消融：`none/hard/soft/hybrid`（固定 `event`）
- LLM 生成器消融：`gpt4/vicuna/mistral/llama3`（读取 `train.prompt_<tag>.jsonl` 等文件）

结果自动写入：

```text
runs/msrvtt_suite/results.csv
```

## 6. LLM 生成器消融（重点）

### 6.1 为什么 Vicuna/Mistral/Llama-3 也能做 prompt 消融？

这些模型本身是文本 LLM，不直接看帧。做法是两阶段：

1. 先用多模态模型得到“固定视觉摘要”（本工程里可直接复用已有 `event_prompts`）。
2. 再让文本 LLM 基于该摘要重写/生成 event-level prompts。

这样做能比较“语言生成器能力”对最终性能的影响，同时尽量固定视觉信息来源。

### 6.2 生成不同 LLM 的 prompt 版本

先准备 `gpt4` 对照版本（直接复制）：

```bash
for s in train val test; do
  cp data/teacher_cache/${s}.videollava.redo.jsonl data/teacher_cache/${s}.prompt_gpt4.jsonl
done
```

再生成文本 LLM 版本（示例）：

```bash
# Vicuna
for s in train val test; do
  python scripts/regenerate_prompts_text_llm.py \
    --input_jsonl data/teacher_cache/${s}.videollava.redo.jsonl \
    --output_jsonl data/teacher_cache/${s}.prompt_vicuna.jsonl \
    --model_id lmsys/vicuna-7b-v1.5 \
    --source_prompt_field event_prompts \
    --max_new_tokens 256
done

# Mistral
for s in train val test; do
  python scripts/regenerate_prompts_text_llm.py \
    --input_jsonl data/teacher_cache/${s}.videollava.redo.jsonl \
    --output_jsonl data/teacher_cache/${s}.prompt_mistral.jsonl \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --source_prompt_field event_prompts \
    --max_new_tokens 256
done

# Llama-3
for s in train val test; do
  python scripts/regenerate_prompts_text_llm.py \
    --input_jsonl data/teacher_cache/${s}.videollava.redo.jsonl \
    --output_jsonl data/teacher_cache/${s}.prompt_llama3.jsonl \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --source_prompt_field event_prompts \
    --max_new_tokens 256
done
```

然后再执行：

```bash
bash scripts/run_msrvtt_suite.sh
```

## 7. 需要记录的实验参数与结果

建议固定并记录：

- 数据版本：`msrvtt_mini` 或全量；每个 split 样本数
- `num_frames`、`image_size`
- `tokenizer`、`clip_model_id`
- `prompt_variant`
- `distill_mode`、`lambda_hard`、`lambda_soft`
- teacher 设置：`teacher_name/model_id/dtype/num_video_frames`
- prompt 生成器：`gpt4 / vicuna / mistral / llama3`
- 训练超参：`epochs / batch_size / lr / seed`
- 指标：`val top1`、`test top1`

`scripts/run_msrvtt_suite.sh` 会把大部分关键信息写到 `results.csv`。

## 8. 当前与论文差异（透明说明）

- 当前仅覆盖 MSRVTT-QA，不含 TVQA 和 ActivityNet-QA 全流程。
- LLM 生成器消融采用“固定视觉摘要 + 文本 LLM 重写”策略；不是每个文本 LLM直接看原始帧。
- 如果要更严格对齐论文，需要补齐 TVQA/ActivityNet、字幕上下文、以及更多 teacher 后端。
