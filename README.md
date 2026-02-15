# distill_videoLLM

# CaTeR 复现项目（Mini Suite）

这个仓库提供了一个可运行的 CaTeR 风格实验框架，并支持论文常见对比开关：

- 学生模型：冻结 CLIP 风格双编码器 + 事件级 prompt 跨模态注意力 + 时序适配器
- 蒸馏模式：`none / hard / soft / hybrid`
- Prompt 消融：`none / random / frame / event`
- Teacher 消融：`videollava / flamingo / chatunivl`（仅真实 teacher 路径）

目标是先把论文流程跑通，并在小数据上完成结构化对比实验。

## 论文数据集与官方划分

- TVQA：`80k / 7k / 7k`
- ActivityNet-QA：`50k / 5k / 3k`
- MSRVTT-QA：`9k / 500 / 500`

可使用 `scripts/make_mini_split.py` 从完整 jsonl 划分中抽取小子集，方便快速迭代。

## 安装环境

```bash
cd cater_repro_minimal
source .venv/bin/activate
pip install -r requirements.txt
```

如果当前环境网络受限，可使用已有环境：

```bash
./setup_env.sh
source ../parallel_minimal/.venv/bin/activate
```

## 数据格式（jsonl）

每一行至少应包含如下字段：

```json
{
  "id": "sample-id",
  "dataset": "tvqa|activitynet_qa|msrvtt_qa|toy",
  "question": "question text",
  "choices": ["a", "b", "c", "d", "e"],
  "answer_idx": 0,
  "event_prompts": ["...", "..."],
  "frame_prompts": ["...", "..."],
  "frame_paths": ["path/to/frame1.jpg", "..."],
  "teacher_embedding": [0.1, -0.2],
  "teacher_hard_idx": 0
}
```

说明：
- `frame_paths` 可选；如果没有该字段，会退回到 toy 的伪帧渲染模式
- `teacher_embedding` 和 `teacher_hard_idx` 可后续用 `build_teacher_cache.py` 回填

## 快速冒烟测试

```bash
cd cater_repro_minimal
PATH="../parallel_minimal/.venv/bin:$PATH" bash run_smoke.sh
```

## 生成 toy 数据

```bash
python scripts/make_toy_data.py --out_dir data/toy --d_model 256
```

## 训练与评估

```bash
python train.py \
  --data_dir data/toy \
  --train_file train.jsonl \
  --val_file val.jsonl \
  --d_model 256 \
  --prompt_variant event \
  --distill_mode hybrid \
  --out_dir runs/exp_event_hybrid

python eval.py \
  --data_dir data/toy \
  --split_file test.jsonl \
  --checkpoint runs/exp_event_hybrid/best.pt
```

## 生成 Teacher Cache（真实）

下面是生成 real teacher cache（用于 teacher 消融）的示例：

```bash
python scripts/build_teacher_cache.py \
  --input_jsonl data/tvqa_mini/train.jsonl \
  --output_jsonl data/teacher_cache/train.videollava.jsonl \
  --teacher_name videollava \
  --model_id LanguageBind/Video-LLaVA-7B-hf \
  --device cuda:0 \
  --dtype fp16 \
  --num_video_frames 8 \
  --max_new_tokens 8 \
  --d_model 512
```

`--teacher_name` 支持：`videollava`、`flamingo`、`chatunivl`。  
当前 real 推理已实现：`videollava`；`flamingo/chatunivl` 仍需你接入各自推理代码。

## 接入真实 Video-LLaVA

`VideoLLaVA` real 已接在 `cater/teachers.py`。

前置条件：

1. 每条 jsonl 样本必须提供 `frame_paths`（RGB 帧路径列表）
2. 环境需安装依赖：`transformers`、`accelerate`、`decord/av`（可选 `bitsandbytes`）
3. 建议使用 GPU

示例命令：

```bash
python scripts/build_teacher_cache.py \
  --input_jsonl data/tvqa_mini/train.jsonl \
  --output_jsonl data/tvqa_mini_teacher/train.videollava.real.jsonl \
  --teacher_name videollava \
  --model_id LanguageBind/Video-LLaVA-7B-hf \
  --device cuda:0 \
  --dtype fp16 \
  --num_video_frames 8 \
  --max_new_tokens 8 \
  --d_model 512
```

低显存模式（可选）：

```bash
python scripts/build_teacher_cache.py ... --load_in_4bit
```

使用 real teacher cache 训练学生模型：

```bash
python train.py \
  --data_dir data/tvqa_mini_teacher \
  --train_file train.videollava.real.jsonl \
  --val_file val.videollava.real.jsonl \
  --prompt_variant event \
  --distill_mode hybrid \
  --d_model 512 \
  --out_dir runs/tvqa_videollava_real
```

## 从 MSRVTT-QA 原始数据开始（推荐流程）

1. 准备视频文件：把 `MSR-VTT` 的 `.mp4` 放到  
`data/msrvtt_raw/msrvtt_qa/MSRVTT-QA/video/`

2. 抽帧（均匀采样）：

```bash
python scripts/extract_msrvtt_frames.py \
  --video_dir data/msrvtt_raw/msrvtt_qa/MSRVTT-QA/video \
  --out_dir data/msrvtt_frames \
  --num_frames 8 \
  --image_size 224
```

3. 转换标注为训练 jsonl（多选格式）并直接用 OpenAI 生成 event-level prompts：

先设置 OpenAI Key：

```bash
export OPENAI_API_KEY=你的key
```

```bash
python scripts/prepare_msrvtt_jsonl.py \
  --qa_dir data/msrvtt_raw/msrvtt_qa/MSRVTT-QA \
  --frames_root data/msrvtt_frames \
  --out_dir data/msrvtt_full_jsonl \
  --num_choices 5 \
  --num_frames 8 \
  --openai_model gpt-4o-mini
```

4. 先做 mini 子集（可选）：

```bash
python scripts/make_mini_split.py \
  --input_dir data/msrvtt_full_jsonl \
  --output_dir data/msrvtt_mini \
  --train_n 2000 --val_n 400 --test_n 400
```

5. 基于 LLM prompts 生成 real Video-LLaVA teacher cache：

```bash
python scripts/build_teacher_cache.py \
  --input_jsonl data/msrvtt_mini/train.jsonl \
  --output_jsonl data/teacher_cache/train.videollava.jsonl \
  --teacher_name videollava \
  --model_id LanguageBind/Video-LLaVA-7B-hf \
  --device cuda:0 \
  --dtype fp16 \
  --num_video_frames 8 \
  --max_new_tokens 8 \
  --d_model 512
```

## 运行 mini 论文实验套件

该脚本会自动跑三组实验：

1. Prompt 消融（`none/random/frame/event`）
2. 蒸馏策略消融（`none/hard/soft/hybrid`）
3. Teacher 消融（`videollava/flamingo/chatunivl`）

```bash
PYTHON_BIN=../parallel_minimal/.venv/bin/python \
DATA_DIR=./data/teacher_cache \
D_MODEL=256 \
EPOCHS=2 \
bash scripts/run_paper_mini.sh
```

## Real Teacher 支持状态

- `Video-LLaVA` real 后端已实现（模型加载 + hard/soft cache 提取）
- `Flamingo` 与 `Chat-UniVL` 的 real 后端当前仍为 TODO 脚手架
