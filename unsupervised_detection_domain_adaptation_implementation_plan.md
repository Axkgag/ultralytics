# 无监督目标检测域适应与噪声标注鲁棒训练实施计划

## 1. 目标与范围

在现有 Ultralytics `detect` 训练路径中增加可选的无监督训练策略，采用 EMA Mean Teacher、伪标签和弱/强增强实现以下两种场景：

1. 外部目标域适应：源域有标注，`target_data` 指向无标注目标域图片。
2. 源域内部自训练：未提供 `target_data` 时，每个 epoch 按比例将源域训练图片随机划分为监督集合和无监督集合，用于降低错误标注、漏标和噪声标注的持续影响。

检测任务仍然是 `detect`，不新增顶层 task、模型家族或检测头。`unsup=False` 时必须执行原训练路径，不创建目标域数据集、不运行教师推理，也不改变现有损失、日志、验证、保存和导出行为。

第一版仅支持普通目标检测，不同时支持 `segment`、`pose`、`obb`、`distill_model` 或 `torch.compile`。在检测流程稳定后再按真实需求扩展，避免一次性引入任务组合复杂度。

## 2. 核心算法

```text
源域监督图片 + GT ───────────────────────────────┐
                                                 ├─→ 学生单次前向 → Lsup
目标域弱增强图片 → EMA 教师 → NMS/阈值过滤 → 伪标签 │
目标域强增强图片 ────────────────────────────────┘                → Lunsup

Ltotal = Lsup + λ(t) × Lunsup

θteacher ← α × θteacher + (1 - α) × θstudent
```

- 学生模型使用用户传入的预训练权重初始化。
- 教师直接复用训练器已有的 `ModelEMA.ema`，不再创建第三份模型。
- 教师始终为 `eval()`、禁用梯度，并在学生完成 optimizer step 后由现有 EMA 路径更新。
- 教师只在目标图片的弱增强视图上生成伪标签。
- 学生在保持相同几何坐标的强增强视图上学习伪标签。
- 源域监督损失始终保留，防止模型遗忘已有检测能力。

## 3. 配置接口

在 `ultralytics/cfg/default.yaml` 中增加以下最小配置：

```yaml
unsup: False # (bool) enable EMA teacher unsupervised detection training
target_data: # (str, optional) unlabeled target images; source train is split when omitted
unsup_ratio: 0.25 # (float) unsupervised share of the logical batch and internal source split
unsup_weight: 1.0 # (float) maximum unsupervised detection loss weight
unsup_warmup_epochs: 5 # (int) source-only epochs before pseudo-label training
pseudo_conf: 0.7 # (float) minimum teacher confidence for pseudo labels
pseudo_iou: 0.7 # (float) teacher NMS IoU threshold
```

### 3.1 配置语义

| 配置                               | 行为                                                        |
| ---------------------------------- | ----------------------------------------------------------- |
| `unsup=False`                      | 完全执行当前监督训练路径                                    |
| `unsup=True, target_data=<path>`   | 所有源域数据保留监督身份，外部目标域数据用于无监督训练      |
| `unsup=True, target_data=None`     | 每个 epoch 将源域随机划分为监督集合和无监督集合             |
| `target_data` 非空且 `unsup=False` | 配置错误，直接提示用户启用 `unsup`                          |
| `unsup=True, distill_model` 非空   | 第一版不支持，直接报错                                      |
| `unsup=True, compile` 非空或为真   | 第一版不支持，直接报错                                      |
| `unsup=True, batch<=0`             | 第一版要求显式正整数 batch，避免自动 batch 低估教师推理开销 |

### 3.2 参数校验

- `0 < unsup_ratio < 1`。
- `unsup_weight >= 0`。
- `unsup_warmup_epochs >= 0` 且小于总 epoch 数。
- `0 < pseudo_conf <= 1`。
- `0 < pseudo_iou <= 1`。
- 内部划分时，监督和无监督集合在每个 rank 上都必须至少提供一个样本。
- `target_data` 只接受现有图片目录、图片列表 TXT 或 `build_yolo_dataset` 已支持的图像输入形式；目标图片不要求标签文件。
- 外部目标域必须与源域共享相同的类别集合和类别顺序。开放集域适应不在第一版范围内。

## 4. 两种数据模式

### 4.1 外部目标域模式

```text
data.train ─────────────→ 监督 DataLoader
target_data ────────────→ 无监督 DataLoader
data.val ───────────────→ 原有验证 DataLoader
```

- 源域训练集不做拆分。
- 目标域使用固定 letterbox 弱增强，不读取标签参与损失。
- `unsup_ratio` 决定逻辑 batch 中目标图片所占比例。
- 训练 epoch 长度以源域监督 DataLoader 为准；目标域 DataLoader 到末尾后以确定性顺序循环。
- 推荐将 `data.val` 配置为小规模有标注目标域验证集，以目标域 mAP 选择最佳 checkpoint；如果没有目标域验证标签，只能验证训练稳定性，不能证明域适应收益。

### 4.2 源域内部随机划分模式

设源域训练图片数为 `N`，无监督比例为 `r`：

```text
Nu = round(N × r)
Ns = N - Nu

epoch e:
    permutation = randperm(N, seed + e)
    unsupervised_indices = permutation[:Nu]
    supervised_indices = permutation[Nu:]
```

要求：

- 每个 epoch 重新划分。
- 同一 epoch 内两个集合互斥，合集覆盖全部训练图片。
- 同一张图片可以在不同 epoch 切换监督/无监督身份。
- 无监督身份下完全丢弃原始 `cls`、`bboxes` 等 GT，不允许进入损失或伪标签筛选。
- 划分由 `seed + epoch` 决定，断点恢复后必须得到相同结果。
- DDP 所有 rank 先得到同一份全局划分，再由 sampler 分配各自样本，不能在各 rank 独立随机划分。

该模式能够把一条错误标注直接参与监督训练的期望频率从 `100%` 降为 `1 - unsup_ratio`，并在其无监督 epoch 使用 EMA 教师预测替代原标注。它只能稀释随机噪声，不能保证纠正系统性类别错误，因此仍需要一个小型干净验证集。

## 5. 逻辑 Batch 与 epoch 长度

`batch` 表示一次 optimizer step 的学生总图片数。设每个 rank 的逻辑 batch 为 `B`：

```text
Bu = round(B × unsup_ratio)
Bs = B - Bu
```

例如 `batch=16, unsup_ratio=0.25`：

```text
监督子 batch:   Bs = 12
无监督子 batch: Bu = 4
学生总前向:     16
```

内部随机划分时，索引数量和 batch 大小使用相同比例，因此监督、无监督 DataLoader 的 step 数近似相同，每张图片每个 epoch 原则上只消费一次。由于取整或 DDP 补齐导致长度相差一个 step 时，只允许循环较短 DataLoader 的最后一个确定性分片，不重新随机抽样。

外部目标域模式以监督 DataLoader 长度定义 epoch，目标域按需循环。`self.batch_size`、梯度累积和 weight decay 仍基于逻辑总 batch `B`，不按 `Bs` 重算。

## 6. 数据加载与可复现划分

### 6.1 数据集实例

内部划分模式为相同的源域图片构建两个数据集实例：

1. 监督实例：普通检测训练增强，但第一版禁用多图增强。
2. 无监督实例：非增强数据集，只执行固定 letterbox，输出教师弱视图。

两个实例必须校验 `im_files` 顺序完全一致。索引划分通过 sampler 完成，不复制图片、标签或生成临时 TXT 文件。

外部目标域模式只额外构建无监督实例。目标目录没有标签时沿用当前验证数据集允许空标签的行为，但日志应明确显示目标集按无标注数据使用，而不是将缺失标签误报为训练错误。

### 6.2 Sampler

在现有数据加载 owner 中实现最小的 epoch-aware 索引 sampler：

- 输入数据集长度、角色 `supervised/unsupervised`、比例、seed、rank 和 world size。
- `set_epoch(epoch)` 后使用同一全局 permutation 生成两个互补索引集合。
- 按现有 `DistributedSampler` 的补齐/截断规则切分到各 rank。
- 单卡和 DDP 下都暴露确定的 `__len__`。
- `build_dataloader` 允许传入显式 sampler；未传入时保持当前逻辑不变。

当前 `InfiniteDataLoader` 使用持久 worker，角色索引必须由主进程 sampler 控制，不能只修改 worker 内部 dataset 的 Python 列表。

### 6.3 多图增强标签泄漏

内部划分模式第一版自动关闭：

```yaml
mosaic: 0.0
mixup: 0.0
cutmix: 0.0
```

原因是 Mosaic/MixUp/CutMix 会绕过当前 sampler，从完整 dataset 随机读取额外图片和原标签，可能把当前 epoch 的无监督样本及其噪声标签混入监督分支。

外部目标域模式不拆分源域，可以继续使用源域 Mosaic/MixUp/CutMix。后续如需在内部模式恢复 Mosaic，应让多图增强只能从当前监督索引池取样，而不是放宽标签隔离规则。

## 7. 弱增强、强增强与坐标一致性

### 7.1 教师弱视图

- 固定 resize/letterbox 到当前训练尺寸。
- 不使用 Mosaic、MixUp、CutMix、随机透视、随机裁剪或独立翻转。
- 输入归一化与现有 `DetectionTrainer.preprocess_batch` 一致。

### 7.2 学生强视图

从教师弱视图张量复制得到，只应用不改变 bbox 坐标的张量增强：

- 随机亮度与对比度。
- 随机 RGB 通道增益或饱和度变化。
- 小概率灰度化。
- 小概率平均池化模糊。
- 轻量矩形擦除，限制最大遮挡面积。

第一版不在强视图上使用独立翻转、旋转、透视、缩放、裁剪或 Mosaic，因此教师框无需几何变换即可用于学生。强增强实现必须支持当前最低 PyTorch 版本，不能依赖 `torchvision.transforms.v2` 或强制安装 Albumentations。

## 8. 教师伪标签生成

每个 step 的目标分支按以下顺序执行：

1. `teacher = self.ema.ema`，保持 `eval()` 和 `torch.no_grad()`。
2. 在 autocast 环境中对弱视图推理。
3. 复用当前检测预测/NMS 逻辑，兼容普通 Detect 输出和 YOLO26 end-to-end 输出。
4. 按 `pseudo_conf` 和 `pseudo_iou` 过滤。
5. 删除宽或高小于 2 像素的框，以及非有限坐标/置信度。
6. 将 `xyxy` 转为当前输入尺寸下归一化 `xywh`。
7. 构建原生检测损失需要的 `img`、`cls`、`bboxes`、`batch_idx`。

精度优先规则：

- 一张目标图片没有任何高置信度伪框时，不把它作为空背景计算无监督损失，而是从本 step 的学生目标子 batch 中移除。
- 整个目标子 batch 都没有伪框时，本 step 退化为纯监督训练，`Lunsup=0`。
- 第一版使用硬伪标签，不修改原生检测损失以加入逐框置信度权重。
- 目标图片存在未被教师发现的真实物体时仍可能产生假负样本，这是第一版已知限制；先通过高质量源域初始化、warmup 和监控控制风险。

## 9. 学生单次前向与损失

DDP 下不要在一次 backward 前对包装后的学生模型分别执行两次 forward。实现采用一次学生前向：

```python
student_images = torch.cat((source_batch["img"], strong_target_images), dim=0)
preds = self.model(student_images)
source_preds, target_preds = split_predictions(preds, len(source_batch["img"]))

source_loss, source_items = student.loss(source_batch, source_preds)
target_loss, target_items = student.loss(pseudo_batch, target_preds)
loss = source_loss + unsup_lambda * target_loss
```

`split_predictions` 只沿 batch 维递归处理张量、list、tuple 和 dict，不改变检测头内部结构。这样普通 Detect、end-to-end 的 `one2many/one2one` 字典输出都走同一学生 forward，DDP 只观察到一次模型调用。

原生检测 loss 已按各子 batch 大小缩放；当 `Bs`、`Bu` 与 `unsup_ratio` 一致时，直接相加即可同时体现数据比例和 `unsup_weight`。不要再次用固定 1:1 平均覆盖用户配置的比例。

无监督权重使用线性 ramp：

```text
epoch < unsup_warmup_epochs:
    λ(t) = 0

warmup 结束后的前 5 个 epoch:
    λ(t) 从 0 线性增长到 unsup_weight

之后:
    λ(t) = unsup_weight
```

训练 loss 日志继续输出原检测 loss 名称，值为监督项与加权无监督项之和，避免破坏验证 loss 键。额外的伪标签统计作为独立训练指标记录。

## 10. 训练器所有权与文件改动

遵循现有 owner，不新增平行任务或大而全的 UDA 框架。

### 10.1 `ultralytics/cfg/default.yaml`

- 增加第 3 节配置项。
- 默认 `unsup=False`，确保现有用户零行为变化。

### 10.2 `ultralytics/engine/trainer.py`

- 将当前 forward/loss 小段提取为可覆盖的 `compute_loss(batch)` 方法。
- 默认实现保持当前 `compile` 和非 `compile` 行为逐行等价。
- `_do_train` 只调用 `compute_loss`，不直接加入 detect/UDA 条件分支。
- optimizer、EMA 更新、验证、checkpoint 和 OOM 重建仍由 BaseTrainer 负责。

### 10.3 `ultralytics/data/build.py`

- 增加最小 epoch-aware role sampler。
- 允许 `build_dataloader` 接收显式 sampler。
- 未传 sampler 时保持当前随机、连续和分布式 sampler 行为不变。

### 10.4 `ultralytics/models/yolo/detect/train.py`

UDA 行为由 `DetectionTrainer` 负责：

- 校验配置和任务组合。
- 计算监督/无监督子 batch 大小。
- 构建目标数据集、DataLoader 和 sampler。
- 在 epoch 开始时更新内部随机划分。
- 获取并预处理目标 batch。
- 执行 EMA 教师推理和伪标签过滤。
- 创建强增强视图。
- 单次学生前向、拆分预测并组合损失。
- 记录伪标签统计。
- OOM 自动缩 batch 重建 pipeline 时同步重建目标 loader。

第一版优先把短小的 detect 专用私有方法留在该 owner 中；只有出现第二个实际调用方时才抽到公共工具。

### 10.5 不修改的区域

- 不修改检测 head。
- 不修改 `v8DetectionLoss` 的匹配和损失公式。
- 不修改 `ModelEMA` 更新公式。
- 不扩展 `TASKS` 或 `MODES`。
- 不把功能塞入固定教师特征蒸馏的 `DistillationModel`。
- 不改变推理、验证、导出和 tracking API。

## 11. Epoch 生命周期

每个 epoch 的关键顺序：

```text
on_train_epoch_start
    ├─ 如果仍在 unsup warmup：使用完整监督 loader
    ├─ 内部模式：sampler.set_epoch(epoch)，生成互补角色索引
    ├─ 外部模式：设置目标域分布式 sampler epoch
    └─ 创建/重置目标 iterator

for source_batch in train_loader
    ├─ 获取 target_batch
    ├─ 教师弱视图推理
    ├─ 过滤并构造伪标签
    ├─ 生成目标强视图
    ├─ 学生单次联合前向
    ├─ Lsup + λ(t)Lunsup
    ├─ backward + optimizer step
    └─ 现有 ModelEMA.update(student)
```

warmup 期间不创建无意义的教师伪标签计算；内部模式使用完整源域监督 loader，保持与现有训练一致。进入无监督阶段时再切换为角色 sampler。

## 12. DDP、AMP、OOM 与断点恢复

### 12.1 DDP

- 全局角色划分由相同 `seed + epoch` 生成。
- sampler 再按 rank 切分，保证各 rank 样本数量可对齐。
- 学生只执行一次包装模型 forward。
- 教师是各 rank 本地 EMA；由于学生参数同步、EMA 更新次数相同，其权重应保持一致。
- 伪标签统计在展示前使用 `all_reduce` 汇总。

### 12.2 AMP

- 教师和学生都在现有 autocast 上下文运行。
- NMS、框合法性过滤和坐标归一化在需要时转 FP32，避免半精度边界问题。
- 教师始终 `no_grad()`，不保留激活图。

### 12.3 OOM 自动恢复

- `batch` 仍表示学生联合 batch，因此现有 batch 减半逻辑可以继续使用。
- `_build_train_pipeline()` 重建时必须同时重算 `Bs/Bu` 并重建目标 loader。
- batch 减小后若某个子 batch 变为 0，直接报错并提示降低/提高 `unsup_ratio` 或显式选择更大 batch，不静默改变比例。

### 12.4 Checkpoint 与 resume

- 当前 checkpoint 已保存学生对应的 EMA，继续作为恢复后的教师。
- resume 时允许更新设备和 batch 等现有白名单参数，但无监督关键配置应从 checkpoint 恢复，避免中途改变训练语义。
- 恢复到 epoch `e` 后使用 `seed + e` 重建当期角色划分。
- 不需要把每个 epoch 的索引数组写入 checkpoint。
- `best.pt` 继续保存 EMA 模型，部署侧无需感知 UDA 训练过程。

## 13. 监控指标

每个 epoch 至少记录：

```text
train/box_loss
train/cls_loss
train/dfl_loss
train/pseudo_boxes
train/pseudo_images
train/pseudo_accept_ratio
train/pseudo_mean_conf
train/unsup_weight
```

日志额外输出按类别累计的伪标签数量，优先使用一行紧凑摘要，不把每类都永久扩展为 CSV 列。

重点观察：

- `pseudo_accept_ratio` 长期接近 0：教师无法跨域工作，不能简单持续降低阈值。
- 某个类别占绝大多数伪标签：可能发生类别塌缩或源/目标类别分布差异。
- 平均置信度快速上升但目标 mAP 不升：可能是教师自我强化错误。
- 源域 mAP 大幅下降：`unsup_weight`、`unsup_ratio` 或强增强过大。

## 14. 推荐训练命令

### 14.1 外部目标域适应

```bash
yolo detect train \
    model=source_best.pt \
    data=source.yaml \
    unsup=True \
    target_data=/path/to/target/images/train \
    unsup_ratio=0.5 \
    unsup_weight=1.0 \
    unsup_warmup_epochs=2 \
    pseudo_conf=0.7 \
    batch=16 \
    epochs=50
```

### 14.2 源域内部噪声鲁棒自训练

```bash
yolo detect train \
    model=yolo26n.pt \
    data=noisy_source.yaml \
    unsup=True \
    unsup_ratio=0.25 \
    unsup_weight=1.0 \
    unsup_warmup_epochs=5 \
    pseudo_conf=0.7 \
    mosaic=0.0 \
    mixup=0.0 \
    cutmix=0.0 \
    batch=16 \
    epochs=100
```

内部模式即使用户没有显式关闭三种多图增强，训练器第一版也必须覆盖为 0 并输出一次原因明确的 warning。

## 15. 实施阶段

### 阶段 A：配置和零回归入口

- [ ] 增加配置项和参数校验。
- [ ] 提取 BaseTrainer 默认 `compute_loss`。
- [ ] 验证 `unsup=False` 时普通 detect 训练行为和 checkpoint 不变。
- [ ] 明确拒绝第一版不支持的组合。

### 阶段 B：数据角色与 DataLoader

- [ ] 实现单卡/DDP epoch-aware role sampler。
- [ ] 构建内部模式的监督/无监督数据集实例。
- [ ] 构建外部目标域 DataLoader。
- [ ] 实现按 `unsup_ratio` 计算 `Bs/Bu`。
- [ ] 处理 warmup、epoch 切换、iterator 循环和 OOM 重建。
- [ ] 内部模式阻止多图增强标签泄漏。

### 阶段 C：教师与伪标签

- [ ] 接入现有 EMA 教师。
- [ ] 兼容普通 Detect 和 end-to-end 输出。
- [ ] 完成 NMS、阈值、合法框和最小尺寸过滤。
- [ ] 构造原生检测损失 batch。
- [ ] 跳过没有可靠伪标签的目标图片。

### 阶段 D：强增强与联合损失

- [ ] 实现无几何变化的张量强增强。
- [ ] 实现递归预测 batch 拆分。
- [ ] 使用一次学生 forward 计算监督和无监督损失。
- [ ] 实现无监督权重 ramp。
- [ ] 增加伪标签训练指标。

### 阶段 E：恢复、验证和性能

- [ ] 验证 AMP、单卡和 DDP。
- [ ] 验证 last.pt resume 后划分、EMA 和 loss 连续。
- [ ] 验证 OOM 自动减 batch 能重建两个 loader。
- [ ] 在 A100 上测量吞吐、显存和目标域 mAP。
- [ ] 对完整 live diff 做一次冷审查，删除重复和无调用代码。

## 16. 聚焦验证矩阵

### 16.1 配置与回归

1. `unsup=False`，运行现有一轮 detect smoke training。
2. `target_data` 非空但 `unsup=False`，确认错误信息明确。
3. 非法 ratio、阈值、batch 和不支持任务组合均在训练前失败。

### 16.2 内部随机划分

1. 同一 epoch 的监督/无监督索引无交集。
2. 两者合集覆盖完整训练集。
3. 相同 seed/epoch 结果一致，不同 epoch 结果变化。
4. resume 到相同 epoch 后结果一致。
5. DDP 各 rank 合并后满足全局划分，允许分布式补齐产生的最小重复。
6. 无监督 batch 中原始标签不进入伪标签或损失。

### 16.3 教师学生训练

1. warmup 阶段不执行目标教师推理。
2. 全部目标预测低于阈值时退化为监督 loss，且 backward 正常。
3. 普通 Detect 和 YOLO26 end-to-end 模型均能生成伪标签。
4. AMP 下 loss 有限，教师参数无梯度。
5. optimizer step 后 EMA 参数发生预期变化。

### 16.4 A100 实机验证

至少运行以下实验：

| 实验                    | 目的                        |
| ----------------------- | --------------------------- |
| Source-only             | 建立源域和目标域基线        |
| 内部 `unsup_ratio=0.25` | 验证噪声鲁棒自训练          |
| 外部 `unsup_ratio=0.5`  | 验证真实跨域迁移            |
| resume 训练             | 验证 checkpoint 连续性      |
| 2-GPU DDP smoke         | 验证 sampler 和单次前向设计 |

## 17. 验收标准

功能验收：

- `unsup=False` 时无目标 loader、教师伪标签和新增开销。
- 两种无监督模式均可完成训练、验证、保存和恢复。
- 内部模式每个 epoch 的角色划分随机、互斥、完整且可复现。
- 目标分支从不使用原始目标标签或内部无监督样本的原标签。
- best/last checkpoint 可直接用于现有 predict、val 和 export。

质量验收：

- 外部目标域 mAP 高于 source-only 基线。
- 内部模式在人工注入或真实噪声数据上优于 source-only，且干净源域 mAP 不发生不可接受下降。
- 伪标签接受率、类别分布和平均置信度在训练后期稳定，不出现单类塌缩。
- 单卡和 DDP 的关键指标趋势一致。

工程验收：

- 变更集中在配置、训练 owner 和数据 sampler，不新增顶层 task 或平行训练框架。
- 普通 detect 训练的默认路径保持简单且通过现有检查。
- 不引入新的必选依赖。
- Ruff、格式化和聚焦训练验证通过。

## 18. 已知风险与后续增强

第一版暂不处理：

- 开放集或类别集合不一致的域适应。
- 教师漏检导致的目标域假负样本屏蔽。
- 按伪标签置信度进行逐框 loss 加权。
- 类别自适应阈值和目标域类别先验校正。
- 对抗式特征域判别器。
- segment、pose、obb 等任务。
- 内部划分模式的 partition-aware Mosaic。

只有在 Mean Teacher 基线通过验收且指标暴露出明确瓶颈后，再按顺序考虑：类别自适应阈值、教师不确定区域屏蔽、逐框置信度权重。不要在第一版同时加入域判别器或额外特征对齐损失。
