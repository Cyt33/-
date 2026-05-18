# OpenVLA 模型架构详细分析报告

## 组员二：模型架构 + 核心算法复现

---

## 1. OpenVLA 概述

OpenVLA（Open Vision-Language-Action Model）是一个开源的视觉-语言-动作多模态模型，专门为机器人操控任务设计。该模型基于视觉语言模型（VLM）架构，通过大规模互联网数据预训练学习视觉理解和语言理解能力，然后针对机器人控制任务进行微调。

### 1.1 核心创新点

1. **多模态融合**：将视觉感知、语言理解和动作预测统一到一个模型中
2. **动作离散化**：将连续机器人动作离散化为token，实现语言模型的动作生成能力
3. **大规模预训练**：在Open X-Embodiment数据集的97万条轨迹上训练
4. **开源可扩展**：完全开源，支持LoRA微调和全量微调

### 1.2 技术规格

| 参数 | 值 |
|------|------|
| 模型规模 | 7B 参数 |
| 视觉编码器 | DINOv2 + SigLIP 融合编码器 |
| 语言模型 | Llama-2 7B |
| 图像分辨率 | 224×224 |
| 动作维度 | 7-DoF |
| 动作离散化 | 256 bins |
| 训练数据 | 970K 轨迹（Open-X Magic Soup++） |

---

## 2. 模型整体架构

OpenVLA采用"视觉编码器 + 投影层 + 语言模型骨干 + 动作预测头"的四阶段架构。整体流程为：图像输入 → 视觉特征提取 → 多模态特征融合 → 语言模型推理 → 动作预测输出。

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenVLA 整体架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入层                                                          │
│   ├── 图像输入 (224×224×3)                                        │
│   └── 文本指令 "In: What action should the robot take to..."     │
│          ↓                              ↓                         │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │  视觉编码器       │         │  文本编码器       │               │
│   │  (Vision Encoder)│         │  (LLM Tokenizer) │               │
│   │                 │         │                 │               │
│   │  DINOv2 + SigLIP│         │  Llama-2 Tokenizer│              │
│   │  融合编码器      │         │                 │               │
│   └────────┬────────┘         └────────┬────────┘               │
│            │                          │                          │
│            ↓                          ↓                          │
│   ┌─────────────────────────────────────────────┐               │
│   │            投影层 (Projector)                 │               │
│   │  将视觉特征映射到语言模型特征空间              │               │
│   │  MLP: 线性层 → GELU → 线性层                 │               │
│   └────────┬───────────────────────────────────┘               │
│            │                                                  │
│            ↓                                                  │
│   ┌─────────────────────────────────────────────┐               │
│   │        语言模型骨干 (LLM Backbone)            │               │
│   │                                             │               │
│   │  Llama-2 7B                                 │               │
│   │  ├── 32层 Transformer                       │               │
│   │  ├── 隐藏维度 4096                          │               │
│   │  └── 32个注意力头                          │               │
│   └────────┬───────────────────────────────────┘               │
│            │                                                  │
│            ↓                                                  │
│   ┌─────────────────────────────────────────────┐               │
│   │        动作预测头 (Action Prediction Head)    │               │
│   │                                             │               │
│   │  ├── 动作词汇表扩展                          │               │
│   │  ├── 256个动作bin离散化                     │               │
│   │  └── 动作反归一化                            │               │
│   └────────┬───────────────────────────────────┘               │
│            │                                                  │
│            ↓                                                  │
│   输出层                                                          │
│   └── 7-DoF 机器人动作向量                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 三大核心模块详解

### 3.1 模块一：视觉编码器 (Vision Encoder)

#### 3.1.1 架构设计

视觉编码器采用DINOv2和SigLIP的融合架构，这是OpenVLA的关键创新之一。这种设计结合了两种视觉编码器的优势：

**DINOv2组件**：
- 自监督学习得到的视觉特征
- 擅长捕捉细粒度的局部结构信息
- 提供丰富的空间位置特征

**SigLIP组件**：
- 对比学习训练的视觉编码器
- 擅长语义级别的图像理解
- 提供强大的文本-图像对齐能力

#### 3.1.2 技术实现

在代码中，视觉编码器通过`PrismaticVisionBackbone`类实现（[modeling_prismatic.py:L63-123](file:///e:-/prismatic/extern/hf/modeling_prismatic.py#L63-123)）：

```python
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,  # 是否使用融合视觉编码器
        image_sizes: List[int],           # 图像尺寸列表
        timm_model_ids: List[str],        # TIMM模型ID
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        # 创建主视觉编码器（DINOv2）
        self.featurizer = timm.create_model(
            timm_model_ids[0],  # "vit_large_patch14_reg4_dinov2.lvd142m"
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
        )
        
        # 如果使用融合架构，创建第二个编码器（SigLIP）
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],  # "vit_so400m_patch14_siglip_224"
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
            )
```

#### 3.1.3 特征提取流程

```python
def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    if not self.use_fused_vision_backbone:
        # 单编码器模式
        return self.featurizer(pixel_values)
    else:
        # 融合编码器模式
        # 将图像按通道分割为两部分
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        
        # 分别提取特征
        patches = self.featurizer(img)           # DINOv2特征
        patches_fused = self.fused_featurizer(img_fused)  # SigLIP特征
        
        # 沿通道维度拼接
        return torch.cat([patches, patches_fused], dim=2)
```

#### 3.1.4 输出特征

- **特征维度**：融合后为 `embed_dim = 1024 + 1024 = 2048`
- **序列长度**：14×14 = 196个patch tokens
- **输出形状**：`[batch_size, 196, 2048]`

#### 3.1.5 LayerScale补丁

为确保与HuggingFace框架的兼容性，代码对TIMM的LayerScale模块进行了适配：

```python
def ls_apply_patch(ls_module: LayerScale):
    # 将gamma参数重命名为scale_factor
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma
```

---

### 3.2 模块二：语言模型骨干 (LLM Backbone)

#### 3.2.1 架构选择

OpenVLA使用Llama-2 7B作为语言模型骨干，这是一个基于Transformer的因果语言模型。选择Llama-2的原因包括：

1. **强大的语言理解能力**：在多项语言理解基准上表现优异
2. **高效的推理速度**：相比同规模的GPT模型，推理速度更快
3. **开放的许可**：相比GPT系列更易于商业应用
4. **良好的多模态兼容性**：架构设计适合与视觉编码器结合

#### 3.2.2 模型规格

| 规格参数 | 值 |
|---------|------|
| 参数量 | 7B |
| 层数 | 32 |
| 隐藏维度 | 4096 |
| 注意力头数 | 32 |
| 上下文长度 | 4096 tokens |
| 词汇表大小 | 32000 |

#### 3.2.3 多模态融合机制

语言模型的核心职责是实现视觉和文本信息的多模态融合。在`PrismaticForConditionalGeneration`中，多模态融合通过以下步骤实现（[modeling_prismatic.py:L362-415](file:///e:-/prismatic/extern/hf/modeling_prismatic.py#L362-415)）：

**步骤1：视觉特征投影**

```python
# 提取视觉patch特征
patch_features = self.vision_backbone(pixel_values)

# 通过投影层映射到语言模型特征空间
projected_patch_embeddings = self.projector(patch_features)
# 输出维度: [batch_size, 196, 4096]
```

**步骤2：构建多模态输入**

```python
# 获取文本嵌入
input_embeddings = self.get_input_embeddings()(input_ids)
# 形状: [batch_size, seq_len, 4096]

# 将视觉嵌入插入到文本嵌入中
# OpenVLA默认在<BOS> token之后插入视觉特征
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],      # <BOS> token
    projected_patch_embeddings,       # 196个视觉tokens
    input_embeddings[:, 1:, :]        # 原始文本tokens
], dim=1)
# 最终形状: [batch_size, 197+seq_len, 4096]
```

**步骤3：构造注意力掩码**

```python
# 为视觉patch创建注意力掩码（全为True，表示可见）
projected_patch_attention_mask = torch.full(
    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    fill_value=True,
    dtype=attention_mask.dtype,
    device=attention_mask.device,
)

# 拼接注意力掩码
multimodal_attention_mask = torch.cat([
    attention_mask[:, :1],           # <BOS>掩码
    projected_patch_attention_mask,  # 视觉patch掩码
    attention_mask[:, 1:]            # 文本掩码
], dim=1)
```

**步骤4：构造训练标签**

```python
# 视觉patch的标签设为IGNORE_INDEX，不参与损失计算
projected_patch_labels = torch.full(
    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    fill_value=IGNORE_INDEX,  # -100
    dtype=labels.dtype,
    device=labels.device,
)

multimodal_labels = torch.cat([
    labels[:, :1],           # <BOS>标签
    projected_patch_labels,  # 视觉标签（忽略）
    labels[:, 1:]            # 文本标签
], dim=1)
```

**步骤5：语言模型前向传播**

```python
language_model_output = self.language_model(
    input_ids=None,
    attention_mask=multimodal_attention_mask,
    inputs_embeds=multimodal_embeddings,
    labels=multimodal_labels,
    use_cache=use_cache,
)
```

#### 3.2.4 投影层设计

投影层（Projector）负责将视觉特征映射到语言模型的特征空间。OpenVLA采用两阶段MLP设计：

```python
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        self.vision_dim, self.llm_dim = vision_dim, llm_dim
        
        if not use_fused_vision_backbone:
            # 单视觉编码器：简单MLP
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            # 融合视觉编码器：更大的MLP
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()
```

---

### 3.3 模块三：动作预测头 (Action Prediction Head)

#### 3.3.1 动作离散化原理

OpenVLA的核心创新之一是将连续机器人动作离散化为token序列。这种设计基于以下考虑：

1. **语言模型的适应性**：语言模型天然适合处理token序列
2. **多模态学习的便利性**：动作可以与文本token统一处理
3. **动作空间的表达**：256个bin足以表达大多数机器人动作范围

#### 3.3.2 离散化实现

在`ActionTokenizer`类中实现动作离散化（[action_tokenizer.py:L13-48](file:///e:-/prismatic/vla/action_tokenizer.py#L13-48)）：

```python
class ActionTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        
        # 创建均匀bin划分
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        # bin_centers: 相邻bin边界的中心点
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # 计算动作token在词汇表中的起始位置
        # 假设词汇表末尾的token使用频率最低
        self.action_token_begin_idx = int(
            self.tokenizer.vocab_size - (self.n_bins + 1)
        )
```

#### 3.3.3 动作编码流程

连续动作到离散token的转换：

```python
def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
    # 1. 裁剪到有效范围
    action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
    
    # 2. 离散化：找到动作值属于哪个bin
    discretized_action = np.digitize(action, self.bins)
    
    # 3. 映射到词汇表中的token ID
    # 公式：token_id = vocab_size - bin_index
    action_token_ids = self.tokenizer.vocab_size - discretized_action
    
    # 4. 解码为token
    return self.tokenizer.decode(list(action_token_ids))
```

#### 3.3.4 动作预测推理

在`OpenVLAForActionPrediction`类中实现动作预测（[modeling_prismatic.py:L506-536](file:///e:-/prismatic/extern/hf/modeling_prismatic.py#L506-536)）：

```python
def predict_action(
    self, 
    input_ids: Optional[torch.LongTensor] = None, 
    unnorm_key: Optional[str] = None, 
    **kwargs: str
) -> np.ndarray:
    # 1. 添加空动作token（用于对齐训练时的输入）
    if not torch.all(input_ids[:, -1] == 29871):  # 29871是":" token
        input_ids = torch.cat([
            input_ids,
            torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)
        ], dim=1)
    
    # 2. 生成动作token
    generated_ids = self.generate(
        input_ids, 
        max_new_tokens=self.get_action_dim(unnorm_key),
        **kwargs
    )
    
    # 3. 提取预测的动作token
    predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):].cpu().numpy()
    
    # 4. 将token ID转换回bin索引
    discretized_actions = self.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, 
        a_min=0, 
        a_max=self.bin_centers.shape[0] - 1
    )
    
    # 5. 获取归一化动作值
    normalized_actions = self.bin_centers[discretized_actions]
    
    # 6. 反归一化到实际动作范围
    action_norm_stats = self.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    
    return actions
```

#### 3.3.5 动作归一化与反归一化

OpenVLA使用数据集统计信息对动作进行归一化和反归一化：

```python
# 归一化：将动作从[min, max]映射到[-1, 1]
normalized = 2.0 * (action - action_low) / (action_high - action_low) - 1.0

# 反归一化：从[-1, 1]映射回原始范围
action = 0.5 * (normalized + 1) * (action_high - action_low) + action_low
```

---

## 4. 跨模态融合原理

### 4.1 融合架构设计

OpenVLA采用"早期融合"策略，在输入层将视觉和文本信息进行融合。这种设计的优势包括：

1. **统一的表示学习**：视觉和文本在相同的特征空间中进行处理
2. **高效的特征交互**：两种模态可以充分交互
3. **端到端优化**：整个模型可以联合优化

### 4.2 信息流动过程

```
图像输入 (224×224×3)
    ↓
┌────────────────────────┐
│   视觉编码器           │
│   DINOv2 + SigLIP     │
│   输出: [B, 196, 2048] │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│   投影层               │
│   MLP: 2048→4096→4096 │
│   输出: [B, 196, 4096] │
└────────┬───────────────┘
         ↓
    ┌────┴────┐
    │ 融合点   │
    └────┬────┘
         ↓
┌────────────────────────┐
│   语言模型骨干          │
│   Llama-2 7B           │
│   32层Transformer      │
│   输入: [B, 197+len, 4096]
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│   动作预测头            │
│   词汇表扩展            │
│   输出: [B, 7] (7-DoF) │
└────────────────────────┘
```

### 4.3 注意力机制

语言模型中的自注意力机制使得视觉token和文本token可以相互关注：

1. **视觉→视觉注意力**：视觉token之间的空间关系建模
2. **文本→文本注意力**：语言上下文理解
3. **视觉→文本注意力**：文本指令指导视觉关注
4. **文本→视觉注意力**：根据文本选择相关的视觉区域

---

## 5. 模型推理流程

### 5.1 完整推理代码

基于现有代码，完整的模型推理流程如下（参考[experiments/robot/openvla_utils.py:L127-170](file:///e:-/experiments/robot/openvla_utils.py#L127-170)）：

```python
def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """生成机器人动作"""
    
    # 1. 图像预处理
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    
    # 可选的中心裁剪（用于数据增强匹配）
    if center_crop:
        image = center_crop_and_resize(image, crop_scale=0.9)
    
    # 2. 构建提示词
    if "openvla-v01" in base_vla_name:
        prompt = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    # 3. 输入处理
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    
    # 4. 动作预测
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    
    return action
```

### 5.2 提示词模板

OpenVLA使用特定的提示词模板来引导模型生成动作：

**标准模板（OpenVLA v1.0）**：
```
In: What action should the robot take to {task_instruction}?
Out:
```

**兼容模板（OpenVLA v0.1）**：
```
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: What action should the robot take to {task_instruction}?
ASSISTANT:
```

### 5.3 推理配置选项

```python
# 基础加载
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# 可选：使用Flash Attention加速
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

# 可选：8位量化
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    load_in_8bit=True,
)

# 可选：4位量化
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    load_in_4bit=True,
)
```

---

## 6. 模型轻量化处理

### 6.1 量化技术概述

OpenVLA支持多种量化技术以降低显存占用：

| 量化方式 | 精度损失 | 显存减少 | 适用场景 |
|---------|---------|---------|---------|
| BF16（默认） | 无 | 1x | 最高精度需求 |
| FP16 | 极小 | 1x | 兼容性优先 |
| INT8 | 较小 | ~2x | 平衡场景 |
| INT4 | 中等 | ~4x | 最低显存 |

### 6.2 量化实现代码

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# INT8量化配置
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# INT4量化配置
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 加载量化模型
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quantization_config_8bit,  # 或 4bit
    torch_dtype=torch.bfloat16,
)
```

### 6.3 显存占用对比

| 模型格式 | 参数量 | 精度 | 显存占用 | 批处理大小 |
|---------|-------|------|---------|-----------|
| BF16 | 7B | 16-bit | ~14GB | 1 |
| FP16 | 7B | 16-bit | ~14GB | 1 |
| INT8 | 7B | 8-bit | ~7GB | 2-3 |
| INT4 | 7B | 4-bit | ~4GB | 4-5 |

### 6.4 CPU卸载策略

对于显存有限的场景，可以使用CPU卸载：

```python
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    device_map="auto",  # 自动分配设备
    offload_folder="offload",  # 卸载路径
    torch_dtype=torch.float32,
)
```

---

## 7. 核心技术原理总结

### 7.1 视觉特征提取

OpenVLA使用DINOv2和SigLIP的融合编码器提取视觉特征：

1. **DINOv2**：自监督学习提供的细粒度局部特征
2. **SigLIP**：对比学习提供的高级语义特征
3. **融合策略**：通道拼接，保持两种特征的互补性

### 7.2 文本指令编码

文本指令通过Llama-2的分词器和嵌入层进行编码：

1. **分词**：使用SentencePiece分词器
2. **嵌入**：映射到4096维向量空间
3. **位置编码**：旋转位置编码（RoPE）

### 7.3 跨模态融合

融合在两个层面进行：

1. **特征层面**：视觉特征通过投影层映射到语言模型空间
2. **注意力层面**：Transformer的自注意力机制实现深层交互

### 7.4 动作预测

动作预测采用离散化方法：

1. **动作离散化**：将连续动作划分为256个bin
2. **Token映射**：将bin索引映射到词汇表的末尾token
3. **自回归生成**：使用语言模型的生成能力自回归预测动作token
4. **反归一化**：根据数据集统计信息将离散动作转换为实际值

---

## 8. 数据流完整示意

```
输入阶段
═══════════════════════════════════════════════════════

图像 (224×224×3 RGB)
    ↓
视觉编码器
    ├── DINOv2特征提取 (ViT-L/14, 1024维)
    └── SigLIP特征提取 (So400M, 1024维)
    ↓
特征融合 (沿通道维度拼接)
    ↓
投影层 (MLP: 2048→8192→4096)
    ↓
视觉嵌入 (196 tokens, 4096维)
    ↓

文本 "In: What action should the robot take to pick up the cup?"
    ↓
Llama-2分词器
    ↓
文本嵌入 (seq_len tokens, 4096维)
    ↓

融合阶段
═══════════════════════════════════════════════════════

<BOS> | [VISUAL_1]...[VISUAL_196] | In: What action should...?
    ↓
32层Llama-2 Transformer
    ├── 自注意力（融合视觉和文本信息）
    ├── 前馈网络
    └── LayerNorm
    ↓
最后一层隐藏状态
    ↓

动作生成阶段
═══════════════════════════════════════════════════════

动作预测头
    ├── 词汇表扩展（+256个动作token）
    └── 动作头输出层
    ↓
自回归生成
    ↓
动作Token序列 [A_1, A_2, ..., A_7]
    ↓
Token→动作反归一化
    ↓

输出阶段
═══════════════════════════════════════════════════════

7-DoF动作向量
├── gripper_open: 0.82
├── x_translation: -0.15
├── y_translation: 0.03
├── z_translation: 0.45
├── roll: 0.01
├── pitch: -0.02
└── yaw: 0.05
```

---

## 9. 总结

OpenVLA通过巧妙的多模态融合设计，将强大的视觉语言模型能力迁移到机器人控制任务中。其核心贡献包括：

1. **创新的架构设计**：融合DINOv2和SigLIP的视觉编码器，提供丰富的视觉表示
2. **动作离散化**：将连续动作空间映射到语言模型词汇表，实现端到端的动作生成
3. **大规模预训练**：在97万条真实机器人轨迹上训练，获得强大的泛化能力
4. **开源可扩展**：完全开源，支持多种微调方式，便于定制和应用

这一工作证明了视觉语言模型的预训练范式可以成功应用于机器人学习任务，为通用机器人智能开辟了新的道路。

---

**报告撰写人**：组员二  
**撰写日期**：2026-05-17  
**版本**：v1.0
