# MBTI 性格预测系统

基于 **RoBERTa + 四维独立分类头** 的英文文本 MBTI 人格类型预测系统。输入一段英文文本，输出 E/I、S/N、T/F、J/P 四维概率、关键词归因、注意力热力图数据、以及自然语言人格解读。


---

## 目录

- [环境配置](#环境配置)
- [项目架构](#项目架构)
- [各模块说明](#各模块说明)
- [训练结果](#训练结果)
- [API 接口](#api-接口)
- [使用指南](#使用指南)
- [改进清单](#改进清单)
- [附录](#附录)

---

## 环境配置

### 1. Conda 环境

```bash
conda create -n mbti_pred python=3.10 -y
conda activate mbti_pred
```

### 2. 安装 PyTorch（CUDA 12.8）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型权重

从 [hf-mirror.com/FacebookAI/roberta-base](https://hf-mirror.com/FacebookAI/roberta-base/tree/main) 下载全部文件到 `models/roberta-base/`。

### 开发环境

| 组件 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| CUDA | 13.2 (Driver 595.79) |
| PyTorch | 2.11.0+cu128 |
| Python | 3.10 |
| 模型 | roberta-base (125M) |

---

## 项目架构

![系统架构图](architecture.webp)

```
MBTI_pred/
├── config/
│   └── default.yaml              # 所有可调超参数集中配置
├── data/
│   ├── getdata.py                # 数据集下载脚本
│   ├── preprocess.py             # 文本清洗 + 标签拆分 + 8:2 划分
│   ├── dataset.py                # PyTorch Dataset 封装
│   ├── train.csv / test.csv      # 预处理后数据 (gitignored)
│   └── label_map.json            # 标签映射表
├── models/
│   └── roberta-base/             # 本地 RoBERTa 权重 (gitignored)
├── checkpoints/
│   └── baseline/                 # 基准模型 (max_len=256, 3epoch)
│       ├── encoder.pt
│       └── classifier.pt
├── output/                       # 训练输出（每 epoch 独立目录）
│   └── <timestamp>/
│       ├── epoch_1/  (logs + 模型)
│       ├── epoch_2/
│       ├── best/                 # 最佳 epoch 模型副本
│       └── training_info.json
├── src/
│   ├── data/                     # 数据层
│   │   ├── dataset.py            # MBTIDataset + create_dataloaders
│   ├── representation/           # 表征层
│   │   ├── encoder.py            # RoBERTaEncoder + 4 种 Pooling 策略
│   │   └── doc.md                # 架构文档
│   ├── model/                    # 任务层
│   │   ├── classifier.py         # MBTIClassifier（logits 输出 + FP16 安全）
│   │   └── trainer.py            # MBTITrainer（训练 + 评估 + 早停）
│   ├── explanation/              # 解释层
│   │   ├── attribution.py        # Integrated Gradients 归因
│   │   ├── attention.py          # 注意力权重提取
│   │   └── interpreter.py        # 中/英 NLG 模板解读
│   └── app/                      # 应用层
│       └── api.py                # FastAPI 推理服务
├── train.py                      # 训练入口
├── eval.py                       # 评估入口
├── explain.py                    # 解释管线（输出 JSON 供前端）
├── api_server.py                 # API 启动入口
└── readme.md
```

---

## 各模块说明

### 数据层 (`src/data/`)

| 脚本 | 功能 | 用法 |
|------|------|------|
| `data/preprocess.py` | 文本清洗 → 标签拆分(16类→4维) → 8:2分层采样 | `python data/preprocess.py` |
| `src/data/dataset.py` | PyTorch Dataset，在线 tokenize，返回 `input_ids / attention_mask / labels` | `MBTIDataset("data/train.csv", tokenizer)` |

### 表征层 (`src/representation/`)

```
RoBERTaEncoder
├── AutoTokenizer (Byte-level BPE)
├── AutoModel (12层 × 12头 Transformer)
└── Pooling: cls | mean [默认] | max | attention
```

| 特性 | 说明 |
|------|------|
| 模型 | `roberta-base` (125M, 768维) |
| 默认 Pooling | Mean — 信息利用充分，长度鲁棒 |
| 冻结选项 | `freeze_backbone=True` → 静态特征提取（Baseline） |
| 本地缓存 | 权重缓存至 `models/`，不依赖网络 |

### 任务层 (`src/model/`)

![模型架构图](modelframe.png)

```
特征向量 (768-dim)
    │
    ├── EI Head   (768 → 512 → 1 logit)
    ├── SN Head   (768 → 512 → 1 logit)
    ├── TF Head   (768 → 512 → 1 logit)
    └── JP Head   (768 → 512 → 1 logit)
```

| 特性 | 说明 |
|------|------|
| 架构 | **无共享层**，四路独立，每个 head 仅 112K 参数 |
| 损失函数 | 加权 BCEWithLogitsLoss，支持 `pos_weight` 缓解不均衡 |
| FP16 | 原生兼容混合精度训练 |
| 按 epoch 调度 | CosineAnnealingLR epoch 级衰减，避免 LR 过早见底 |

### 解释层 (`src/explanation/`)

| 模块 | 技术 | 输出 |
|------|------|------|
| `AttributionAnalyzer` | Integrated Gradients (Captum) | 四维 token 贡献分数 |
| `AttentionExtractor` | 12层 × 12头注意力权重 | 热力图矩阵 + `<s>` token 关注分布 |
| `MBTIInterpreter` | 规则模板引擎 | 中/英文人格描述文本 |

### 应用层 (`src/app/`)

| 文件 | 功能 |
|------|------|
| `src/app/api.py` | FastAPI 服务，启动加载模型，`POST /api/predict` |
| `api_server.py` | 一键启动：`python api_server.py` |

---

## 训练结果

数据集: zeyadkhalid/MBTI-500 (106,067 条 → 训练 84,853 / 测试 21,214)

### Baseline vs HP 优化对比

| | Baseline | HP 最优 | 提升 |
|------|:---:|:---:|:---:|
| 模型 | roberta-base | **roberta-base** | — |
| Pooling | mean | **cls** | — |
| head_hidden | 64 | **512** | — |
| max_length | 256 | **512** | — |
| batch_size | 16 | **32** | — |
| epochs | 3 | 7 (best) | — |
| **Overall Acc** | 80.6% | **86.0%** | **+5.4%** |
| Mean Acc | 93.2% | **95.1%** | +1.9% |
| Mean F1 | 91.3% | **93.7%** | +2.4% |

### HP 最优逐维度指标

| 维度 | Acc | F1 | AUC | MCC | 提升亮点 |
|------|-----|-----|-----|-----|------|
| **EI** | 94.7% | 0.885 | 0.976 | 0.836 | F1 +4.9% ← 最难维度大幅突破 |
| **SN** | 97.9% | 0.989 | 0.983 | 0.844 | 接近饱和 |
| **TF** | 95.6% | 0.966 | 0.988 | 0.903 | 全维最强 |
| **JP** | 92.2% | 0.907 | 0.975 | 0.834 | F1 +3.0% |
| **Mean** | **95.1%** | **0.937** | **0.980** | **0.854** | — |

### 关键超参

| 参数 | Baseline | HP 最优 | 来源 |
|------|:---:|:---:|------|
| pooling | mean | **cls** | HP 搜索 |
| head_hidden | 64 | **512** | HP 搜索 |
| dropout | 0.2 | 0.2 | — |
| encoder_lr | 2e-5 | 2.84e-5 | HP 搜索 |
| classifier_lr | 1e-4 | 1e-4 | — |
| weight_decay | 0.01 | 0.01 | — |

### 可视化评估

`python eval.py` 生成 4 张分析图 → `eval_output/`：

#### 混淆矩阵

![Confusion Matrices](eval_output/best/confusion_matrices.png)

TF 维度对角线最亮（区分度最高），EI 假阳性偏高（模型倾向判 E）。

#### ROC 曲线

![ROC Curves](eval_output/best/roc_curves.png)

四维 AUC 均 > 0.975，区分能力极强。

#### 置信度分布

![Confidence Distribution](eval_output/best/confidence_dist.png)

正确预测集中在高置信区（右侧），错误预测集中在 0.5 附近——模型"自知其不知"。

#### 雷达图

![Radar Chart](eval_output/best/radar.png)

TF 维度 Accuracy/F1/AUC 最均衡（最大面积），EI 的 F1 相对偏低。

---

## API 接口

### 启动服务

```bash
# 一键启动（自动打开浏览器）
python api_server.py

# 自定义端口 / 不打开浏览器
python api_server.py --port 3000 --no-browser
```

启动后自动：检测端口 → 加载模型到 GPU → 启动 FastAPI → 打开 `http://localhost:8000`

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/predict` | 预测 + 解释 |

### POST /api/predict

**Request**

```json
{
  "text": "I enjoy spending quiet evenings alone with a book..."
}
```

**Response**

```json
{
  "prediction": {
    "mbti_type": "ISFJ",
    "probabilities": {
      "EI": { "positive": 0.09, "negative": 0.91 },
      "SN": { "positive": 0.98, "negative": 0.02 },
      "TF": { "positive": 0.35, "negative": 0.65 },
      "JP": { "positive": 0.58, "negative": 0.42 }
    },
    "confidence": 0.56
  },
  "keywords": {
    "EI": [{"token": "alone", "score": 0.02}, ...],
    "SN": [...], "TF": [...], "JP": [...]
  },
  "interpretation": {
    "EI": "你在内向维度上倾向明显（I: 91%）...",
    "SN": "你在感觉维度上倾向明显（S: 98%）...",
    "TF": "你在情感维度上倾向明显（F: 65%）...",
    "JP": "你在判断维度上倾向明显（J: 58%）...",
    "summary": "你是一个 ISFJ（守卫者）—— 温暖、细致的守护者..."
  }
}
```

### 前端调用示例

```javascript
const res = await fetch("http://localhost:8000/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: userInput }),
});
const data = await res.json();
// data.prediction.mbti_type    → "ISFJ"
// data.prediction.confidence   → 0.56
// data.interpretation.summary  → "你是一个 ISFJ..."
// data.keywords.EI             → 关键词 + 贡献分数
```

### 模型加载策略

模型在 FastAPI **启动时加载一次**到 GPU 显存，所有请求复用同一份权重，无冷启动延迟，并发安全（模型只读推理）。

---

## 使用指南

### 数据处理

```bash
# 下载数据集
python data/getdata.py

# 预处理（清洗 + 标签拆分 + 8:2 划分）
python data/preprocess.py
```

### 训练

```bash
# 使用默认配置
python -u train.py

# 指定配置或覆盖参数
python -u train.py --cfg config/default.yaml --epochs 5 --bs 8
```

配置文件 `config/default.yaml` 包含所有可调超参：模型路径、pooling 策略、学习率、损失权重、早停等。

### 评估

```bash
# 自动找最新 checkpoint
python eval.py

# 指定 checkpoint + 错误分析
python eval.py --ckpt output/20260605_135902/best --error-analysis
```

### 解释输出

```bash
# 生成 JSON 解释数据
python explain.py "I enjoy quiet evenings..." --lang zh

# 从文件读入
python explain.py --file input.txt --lang en
```

---

## 改进清单

### 相比原始设计的主要优化

| 方面 | 原设计 | 实际实现 | 原因 |
|------|--------|---------|------|
| **模型** | RoBERTa-wwm-ext (中文) | roberta-base (英文) | 数据集为英文论坛帖子 |
| **架构** | 共享层 768→256 + 4 Heads | 直接 4 Heads (768→64→1) | 减少参数，简化梯度流 |
| **输出** | Sigmoid → BCE | Logits → BCEWithLogitsLoss | FP16 混合精度安全 |
| **调度器** | WarmRestarts (step 级) | Cosine LR (epoch 级) | LR 不会过早衰减 |
| **配置** | CLI 参数散落 | `config/default.yaml` 集中管理 | 可读性、可复现 |
| **输出目录** | 全局单一 | 每 epoch 独立目录 | 每轮可追溯、可对比 |
| **不均衡** | 无 | `pos_weight` + `dim_weights` | SN 91% 严重不均衡 |
| **解释层** | 计划中 | 完整实现 | Attribution + Attention + NLG |

### 训练技巧

- **分层学习率**：encoder `2e-5`，classifier `1e-4`（头部适应快于骨干）
- **不均衡处理**：SN 维度 pos_weight=10，dim_weight=0.35 重点优化
- **梯度裁剪**：`max_grad_norm=1.0` 防止 RoBERTa 梯度爆炸
- **按 epoch 调度 LR**：CosineAnnealingLR，每 epoch 衰减一次

---

## 项目进度

### 已完成 ✅

| 模块 | 内容 | 状态 |
|------|------|------|
| 数据层 | 文本清洗、标签拆分(16类→4维)、8:2 分层采样、PyTorch Dataset | ✅ |
| 表征层 | RoBERTa-base 本地加载、4 种 Pooling 策略（默认 Mean） | ✅ |
| 任务层 | 四维独立分类头（logits 输出 + FP16 安全）、加权 BCEWithLogitsLoss | ✅ |
| 训练器 | 分层 LR、epoch 级 Cosine 调度、早停、每 epoch 独立目录存储 | ✅ |
| 配置系统 | `config/default.yaml` 集中管理所有超参，CLI 可覆盖 | ✅ |
| 评估系统 | 6 项指标 + 4 张可视化图（混淆矩阵/ROC/置信度/雷达图） | ✅ |
| 解释层 | Integrated Gradients 归因 + 注意力提取 + 中/英 NLG 解读 | ✅ |
| API 服务 | FastAPI 启动加载模型、`POST /api/predict` 完整响应 | ✅ |
| CLI 脚本 | train / eval / explain / api_server 四个入口 | ✅ |

### 测试集最终指标

| 指标 | 值 |
|------|-----|
| Mean Accuracy | **93.2%** |
| Mean F1 | **91.3%** |
| Mean AUC | **97.4%** |
| Exact Match | **80.6%** |
| Macro MCC | **0.810** |

### 后续优化方向

**模型层面：**

| 事项 | 说明 | 优先级 |
|------|------|:---:|
| roberta-large | 355M 参数 (1024 维)，预期额外 +1-2% | High |
| 针对性调优 EI / JP | 两个最难维度的 pos_weight 精调 | Medium |
| 用户级数据分割 | 同一用户多帖跨 train/test 可能虚高 ~1-2% | Medium |
| 全量数据 HP 搜索 | 当前用 30-50% 子集，全量可能翻出更优参数 | Medium |
| LoRA 微调 | 减少可训参数、加速训练 | Low |
| AB 对比实验 | TF-IDF+LR vs RoBERTa-base vs RoBERTa-large | Low |

**前端层面（已规划，待实施）：**

详见 [docs/frontend-improvement-plan.md](docs/frontend-improvement-plan.md)。核心改进：

- 新增 Model Comparison 视图：Baseline vs HP 最优模型参数/指标/图表交互式对比（Canvas/SVG）
- 新增 Hyperparameter Search 视图：20 个 trial 的散点图 + 平行坐标可视化
- 4 个 PNG 静态评估图替换为 Canvas 交互式图表（hover tooltip、数据驱动渲染）
- 学术报告风格（Tableau 色板、等宽数字、克制配色）
- 零外部依赖，纯 HTML/CSS/JS 单文件

---

## 附录

### A：MBTI 四维说明

| 维度 | 字母 | 含义 |
|------|------|------|
| E / I | Extraversion / Introversion | 外向 / 内向 |
| S / N | Sensing / Intuition | 感觉（具体） / 直觉（抽象） |
| T / F | Thinking / Feeling | 思考（逻辑） / 情感（价值观） |
| J / P | Judging / Perceiving | 判断（计划） / 感知（灵活） |

### B：16 种类型速查

| 类型 | 别称 | 类型 | 别称 |
|------|------|------|------|
| INTJ | 建筑师 | INTP | 逻辑学家 |
| ENTJ | 指挥官 | ENTP | 辩论家 |
| INFJ | 提倡者 | INFP | 调停者 |
| ENFJ | 主人公 | ENFP | 竞选者 |
| ISTJ | 物流师 | ISFJ | 守卫者 |
| ESTJ | 总经理 | ESFJ | 执政官 |
| ISTP | 鉴赏家 | ISFP | 探险家 |
| ESTP | 企业家 | ESFP | 表演者 |
