# ElevatorDetect 无监督域适应实施文档

## 1. 实施目标

本实现只服务于当前仓库的 `ElevatorDetect` 四层检测头，不把它抽象成普通 Detect UDA。训练目标同时包含：

- 按钮检测框与二分类：`floor`（类别 0）和 `other`（类别 1）。
- 楼层字符属性：Slot-1 的 5 类输出和 Slot-2 的 11 类输出。
- 按钮灯状态：二值输出和 `light_valid` 有效位。

开启 `unsup=True` 后使用固定高衰减的 `ModelEMA.ema` 作为教师、当前模型作为学生。没有设置 `target_data` 时，每个 epoch 都按 `seed + epoch` 将源数据重新随机划分为监督集合和无监督集合；无监督集合的原标签完全不参与损失或伪标签筛选。

`unsup=False` 时保留现有训练路径：不创建目标数据集、不运行教师推理，仍调用原来的模型 forward/loss。

## 2. 支持范围与限制

第一版要求：

- `task=detect`。
- 源数据 YAML 必须包含 `elevator: true`。
- 模型最后一层必须是 `ElevatorDetect`。
- 外部目标数据也必须是 ElevatorDetect YAML，且 `names` 与源数据完全一致。
- 显式整数 batch，且每个 rank 的 batch 至少为 2。
- `compile=False`、`rect=False`、`multi_scale=0`。
- 不与 `distill_model` 同时使用。

内部划分会禁用 `mosaic`、`mixup`、`cutmix` 和 `copy_paste`，避免增强逻辑绕过 sampler 后把无监督图片的原标签混入监督分支。外部目标域模式不拆分源数据，因此保留源数据原有增强。

## 3. 配置

```yaml
unsup: False
target_data: # 目标域 ElevatorDetect 数据 YAML；不设置时按 epoch 划分源数据
unsup_ratio: 0.25
unsup_weight: 0.25
unsup_warmup_epochs: 5
uda_teacher_decay: 0.999
pseudo_conf: 0.9
pseudo_iou: 0.7
pseudo_floor_conf: 0.9
pseudo_light_conf: 0.9
pseudo_max_boxes: 30
pseudo_max_floor_ratio: 0.98
```

| 配置                     | 含义                                                                   |
| ------------------------ | ---------------------------------------------------------------------- |
| `unsup_ratio`            | 每个 rank 的目标子 batch 比例；内部模式也用作每 epoch 的无监督图片比例 |
| `unsup_weight`           | 目标伪标签损失的最终权重                                               |
| `unsup_warmup_epochs`    | 只计算源监督损失的 epoch 数                                            |
| `uda_teacher_decay`      | UDA 教师的固定 EMA 衰减率                                              |
| `pseudo_conf`            | 教师检测置信度阈值                                                     |
| `pseudo_iou`             | 教师 class-agnostic NMS IoU 阈值                                       |
| `pseudo_floor_conf`      | 合法楼层组合的最低几何平均置信度                                       |
| `pseudo_light_conf`      | 灯亮/灭标签有效阈值，必须不小于 0.5                                    |
| `pseudo_max_boxes`       | 单张目标图片允许进入损失的最大伪框数                                   |
| `pseudo_max_floor_ratio` | 一个目标 batch 允许的最大 `floor` 类伪框占比                           |

`target_data` 非空而 `unsup=False` 会直接报错，避免用户误以为已执行无监督训练。

## 4. 数据路径

### 4.1 外部目标域

```text
data.train        -> 源监督 DataLoader
target_data.train -> 目标无监督 DataLoader
data.val          -> 原验证 DataLoader
```

源 DataLoader 定义 epoch 长度；目标 DataLoader 较短时循环。目标数据集使用无增强的 letterbox 弱视图，数据中的标签即使存在也会被忽略。

### 4.2 源数据内部划分

源训练集大小为 `N`，无监督比例为 `r`：

```text
Nu = round(N * r)
permutation = randperm(N, seed + epoch)
target_indices = permutation[:Nu]
source_indices = permutation[Nu:]
```

两个 `EpochSplitSampler` 使用同一全局 permutation，分别选择互补集合，然后再按 rank 分片。DDP 不会在各 rank 独立划分。`InfiniteDataLoader` 在 epoch 切换后重置，防止 persistent worker 预取上一 epoch 的索引。

监督和无监督数据集实例会校验 `im_files` 顺序完全一致。无监督实例采用 `mode=target`，不会开启训练增强，也不会把原标签传给损失。

## 5. Batch 与 warmup

每个 rank 的逻辑 batch 为 `B`：

```text
Bu = clamp(round(B * unsup_ratio), 1, B - 1)
Bs = B - Bu
```

源 loader 使用 `Bs`，目标 loader 使用 `Bu`。warmup 阶段只计算源监督损失，但仍保持 `Bs`，避免训练中途重建 loader。warmup 结束后 5 个 epoch 将目标损失权重线性提升到 `unsup_weight`：

```text
epoch < warmup: lambda = 0
之后第 1..5 个 epoch: lambda = unsup_weight * (1..5) / 5
之后: lambda = unsup_weight
```

## 6. ElevatorDetect 伪标签筛选

教师对目标弱视图输出 23 个通道：

```text
4 box + 2 class + 5 slot1 + 11 slot2 + 1 light
```

筛选严格沿用 ElevatorDetect 的推理语义：

1. 使用 `pseudo_conf`、`pseudo_iou` 和 class-agnostic NMS。
2. 删除非有限预测，以及宽或高小于 2 像素的框。
3. 对类别 0（`floor`），调用 `ElevatorAttributes.floor_scores`，只在仓库定义的 24 个合法楼层编码中选择最大值。
4. 合法组合得分为 `sqrt(slot1_prob * slot2_prob)`；低于 `pseudo_floor_conf` 时删除整个 floor 框，因为当前 loss 没有独立的 `floor_valid` 掩码。
5. 对类别 1（`other`），Slot-1 和 Slot-2 均写为 0，现有损失本来就只对类别 0 计算楼层属性。
6. 灯概率大于等于 0.5 时伪标签为亮，否则为灭。
7. 只有概率大于等于 `pseudo_light_conf` 或小于等于 `1 - pseudo_light_conf` 时 `light_valid=1`；中间区域保留检测框，但屏蔽灯损失。
8. 一张图片超过 `pseudo_max_boxes` 时拒绝整张图片，避免检测结果饱和到 NMS 上限后进入训练。
9. 当一个 batch 至少有 20 个伪框且 `floor` 占比超过 `pseudo_max_floor_ratio` 时拒绝整个目标 batch，阻断类别坍缩。
10. 将框转换成输入尺寸下的归一化 `xywh`，构造现有 ElevatorDataset/loss 使用的原生字段：`cls`、`bboxes`、`batch_idx`、`elevator=[slot1, slot2, light, light_valid]`。

一张图片没有任何合格框时，不把它当作纯背景加入目标损失；整个目标 batch 没有合格框时，该 step 自动退化为源监督训练。

## 7. 学生训练

目标强视图直接复制弱视图，只使用不改变框坐标的增强：亮度、对比度、饱和度、灰度化、均值模糊和轻量擦除。

学生只执行一次 forward：

```python
images = torch.cat((source_batch["img"], strong_target_images))
predictions = student(images)
source_predictions, target_predictions = split_predictions(predictions, len(source_batch["img"]))

source_loss = elevator_loss(source_predictions, source_batch)
target_loss = elevator_loss(target_predictions, pseudo_batch)
loss = source_loss + lambda * source_batch_size / accepted_target_images * target_loss
```

预测字典中的 `boxes`、`scores`、`feats`、`slot1`、`slot2`、`light` 都沿 batch 维递归拆分。两个分支都复用现有 `v8ElevatorDetectionLoss`，没有复制或简化 ElevatorDetect 的属性损失。

目标 loss 乘以 `source_batch_size / accepted_target_images`，避免某个 step 因筛选掉较多目标图片而仅由内部 batch 缩放改变域损失权重。

## 8. 教师更新、验证与监控

教师复用训练器已有 EMA，但在 UDA 模式下将其更新率固定为 `uda_teacher_decay=0.999`，不使用偏向快速跟随学生的早期 EMA ramp。学生 optimizer step 完成后自动更新教师，不创建第三份模型或额外 checkpoint 字段；验证与保存仍使用该教师。

验证仍使用原 `data.val` 和现有 ElevatorDetect 验证器。每次验证附加以下训练指标：

- `train/pseudo_boxes`
- `train/pseudo_images`
- `train/pseudo_accept_ratio`
- `train/pseudo_mean_conf`
- `train/pseudo_floor_boxes`
- `train/pseudo_light_valid_ratio`
- `train/pseudo_dense_rejected_images`
- `train/pseudo_class_guard_rejected_batches`
- `train/unsup_weight`

DDP 下先对计数求和，再由 rank 0 写入结果。

## 9. 文件改动

| 文件                                      | 改动                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `ultralytics/cfg/default.yaml`            | 增加 UDA 配置及默认值                                                  |
| `ultralytics/cfg/__init__.py`             | 将新配置加入集中类型校验                                               |
| `ultralytics/data/build.py`               | 增加 `EpochSplitSampler`，允许显式 sampler                             |
| `ultralytics/data/__init__.py`            | 导出 sampler                                                           |
| `ultralytics/engine/trainer.py`           | 提取默认 `compute_loss`，增加 epoch-aware sampler hook                 |
| `ultralytics/models/yolo/detect/train.py` | 实现 ElevatorDetect 数据、教师、伪标签、强增强和组合损失               |
| `scripts/train_elevator.py`               | 暴露 UDA 和全部伪标签筛选参数；预训练 UDA 默认低学习率及低 bias warmup |
| `ultralytics/utils/plotting.py`           | 支持奇数个 loss/metric 图表，避免 ElevatorDetect 结果绘图越界          |

## 10. 执行示例

外部目标域：

```bash
python scripts/train_elevator.py \
    --unsup \
    --target-data scripts/elevator-target.yaml \
    --batch 16 \
    --epochs 100
```

源数据每 epoch 随机划分：

```bash
python scripts/train_elevator.py \
    --unsup \
    --batch 16 \
    --unsup-ratio 0.25 \
    --epochs 100 \
    --patience 0 \
    --save-period 10
```

`--patience 0` 关闭 EarlyStopping；`--save-period 10` 保留周期 checkpoint，`last.pt` 和 `best.pt` 仍照常更新。

普通监督训练保持不变：

```bash
python scripts/train_elevator.py --batch 16 --epochs 100
```

## 11. 验收标准

1. `unsup=False` 的 forward/loss、EMA、验证、保存路径保持原行为。
2. 内部模式同一 epoch 的源/目标索引互斥且覆盖全集，不同 epoch 的划分可复现地变化。
3. 无监督分支不读取原标签参与 loss。
4. 伪标签只接受合法 ElevatorDetect 楼层编码，灯不确定时正确写入 `light_valid=0`。
5. 单图伪框过密或 batch 类别分布坍缩时能稳定回退到源监督。
6. 无伪框 batch 能稳定回退到源监督。
7. 单卡 A100 最小训练能够完成 forward、backward、固定衰减 EMA 更新、验证和 checkpoint 保存。
8. Ruff 格式与静态检查通过。

## 12. A100 实施验证记录

在 `A100-gpu003` 的 `button-uda` 分支完成了以下验证：

- 配置解析与修改文件的 Python 编译通过。
- `EpochSplitSampler` 在单卡、4 rank 和“样本数小于 rank 数”的 8 rank 场景下，源/目标集合互斥、覆盖全集，并且每个 shard 长度与 `__len__` 一致。
- `unsup=False` 的 4 张训练图、2 张验证图对照训练完成 forward、backward、验证与 checkpoint 保存。
- 内部划分 UDA 的 8 张训练图、2 张验证图训练完成；高阈值无伪框时正确回退到纯监督分支。
- 后续稳定性改进增加固定高衰减教师、单图伪框上限、类别分布保护、较低目标损失权重、更严格阈值，以及预训练 UDA 的低学习率与低 bias warmup 默认值。
- 使用真实 ElevatorDetect 数据完成默认筛选正路径、单图密集保护、类别分布保护、`unsup=False` 回归以及两卡 DDP 冒烟验证；训练、验证、保存和结果绘图均完成。
- Markdown 已通过仓库指定的 Prettier 3.8.5 检查，`git diff --check` 无空白错误。
