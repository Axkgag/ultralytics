# ElevatorDetect UDA 参数说明

## 1. UDA 主开关与数据划分

| 参数            | 默认值 | 作用                                                                                                    |
| --------------- | -----: | ------------------------------------------------------------------------------------------------------- |
| `--unsup`       |   关闭 | 开启 ElevatorDetect 教师–学生无监督训练                                                                 |
| `--target-data` | 未设置 | 指定无标签目标域数据 YAML                                                                               |
| `--unsup-ratio` | `0.25` | 每个 batch 中目标域图片的比例；未设置 `target_data` 时，也表示每个 epoch 从源训练集划为无监督数据的比例 |

数据模式分为两种：

- 设置 `target_data`：使用源域有标签数据和新场景无标签数据，执行跨场景无监督域适应。
- 未设置 `target_data`：每个 epoch 将源训练集重新随机划分为 75% 监督数据和 25% 无监督数据，更接近半监督抗噪训练。

内部划分按 `seed + epoch` 可复现地变化。监督集合与无监督集合互斥并覆盖当前训练集，无监督集合的原标签不会参与损失或伪标签筛选。

## 2. Teacher 与无监督损失

| 参数                    |  默认值 | 作用                                                  |
| ----------------------- | ------: | ----------------------------------------------------- |
| `--uda-teacher-decay`   | `0.999` | Teacher 的固定 EMA 衰减率                             |
| `--unsup-weight`        |  `0.25` | 目标伪标签损失的最终最大权重                          |
| `--unsup-warmup-epochs` |     `5` | 开始加入目标伪标签损失前，仅训练源监督损失的 epoch 数 |

默认无监督损失权重调度如下：

```text
epoch 1～5:  0
epoch 6:    0.05
epoch 7:    0.10
epoch 8:    0.15
epoch 9:    0.20
epoch 10+:  0.25
```

Teacher 和 Student 均从 `--weights` 指定的源域权重初始化。Student 通过反向传播更新，Teacher 通过固定衰减率 EMA 更新。训练验证与最终 checkpoint 使用 Teacher 权重。

## 3. 伪标签筛选参数

| 参数                       | 默认值 | 作用                                           |
| -------------------------- | -----: | ---------------------------------------------- |
| `--pseudo-conf`            |  `0.9` | Teacher 检测框的最低置信度                     |
| `--pseudo-iou`             |  `0.7` | class-agnostic NMS 的 IoU 阈值                 |
| `--pseudo-floor-conf`      |  `0.9` | Slot-1 和 Slot-2 合法楼层组合的最低置信度      |
| `--pseudo-light-conf`      |  `0.9` | 灯状态伪标签的有效置信度阈值                   |
| `--pseudo-max-boxes`       |   `30` | 单张目标图片允许进入损失的最大伪框数           |
| `--pseudo-max-floor-ratio` | `0.98` | 一个目标 batch 中允许的最大 `floor` 类伪框占比 |

### 3.1 灯状态筛选

默认 `pseudo_light_conf=0.9` 时：

```text
P(light) >= 0.9  → 点亮，light_valid=1，参与 light loss
P(light) <= 0.1  → 未点亮，light_valid=1，参与 light loss
0.1 < P < 0.9    → 状态不确定，light_valid=0，不计算 light loss
```

`pseudo_light_conf` 越高，被屏蔽的中间不确定区域越大，进入灯状态损失的伪标签越少。

### 3.2 异常伪标签保护

- 单张图片筛选后仍超过 `pseudo_max_boxes`：拒绝整张目标图片。
- 一个目标 batch 至少有 20 个伪框，且 `floor` 占比超过 `pseudo_max_floor_ratio`：拒绝整个目标 batch。
- 一张图片没有合格伪框：不将其作为纯背景加入目标损失。
- 整个目标 batch 没有合格伪框：当前 step 自动退化为纯源监督训练。
- 楼层属性只接受 ElevatorDetect 定义的合法楼层编码组合。

最近一次 `train_qwen_adv_unsup_150_v2` 训练显式使用了：

```yaml
pseudo_max_boxes: 35
```

其余主要 UDA 参数使用上述默认值。

## 4. 预训练 UDA 优化参数

| 参数               |          预训练 UDA 默认值 | 作用                                                      |
| ------------------ | -------------------------: | --------------------------------------------------------- |
| `--weights`        | 脚本默认权重或用户指定权重 | 初始化 Student 和 Teacher                                 |
| `--optimizer`      |                    `AdamW` | 预训练 UDA 使用的优化器                                   |
| `--lr0`            |                   `0.0001` | 初始学习率                                                |
| `--warmup-bias-lr` |                   `0.0001` | bias 参数的 warmup 学习率                                 |
| `--from-scratch`   |                       关闭 | 开启后不加载 `--weights`，并恢复非预训练 UDA 的优化默认值 |

UDA 应显式加载训练完成的源域模型：

```bash
--weights /path/to/source_best.pt
```

不建议从随机初始化开始 UDA，否则训练初期 Teacher 生成的伪标签质量较差，容易造成错误累积。

## 5. 训练控制与 checkpoint 保存

| 参数            | 默认值 | 建议值 | 作用                                                   |
| --------------- | -----: | -----: | ------------------------------------------------------ |
| `--epochs`      |  `100` |  `150` | 最大训练 epoch 数                                      |
| `--patience`    |   `50` |    `0` | EarlyStopping 等待轮数；`0` 表示完全关闭 EarlyStopping |
| `--save-period` |   `-1` |   `10` | 周期 checkpoint 保存间隔；`-1` 表示不额外保留周期权重  |

权重目录中：

- `last.pt`：持续更新，完整训练结束后对应最终模型。
- `best.pt`：按当前 fitness（检测 `mAP50-95`）选择的最佳模型。
- `epoch*.pt`：启用 `--save-period` 后保留的周期 checkpoint。

当前 Ultralytics 内部的周期 checkpoint 使用从 0 开始的 epoch 编号，因此 `--save-period 10` 生成的文件类似：

```text
epoch0.pt
epoch10.pt
epoch20.pt
...
epoch140.pt
last.pt
best.pt
```

当前 `best.pt` 只依据检测 `mAP50-95` 选择。若 UDA 明显改善楼层或灯状态、但降低高 IoU 框定位精度，相关模型可能只存在于周期权重或 `last.pt` 中。

## 6. 推荐训练命令

### 6.1 使用新场景无标签目标数据

```bash
/mnt/yihao/envs/ultralytics/bin/python scripts/train_elevator.py \
    --unsup \
    --target-data /path/to/elevator-target.yaml \
    --device 0,1 \
    --batch 64 \
    --epochs 150 \
    --patience 0 \
    --save-period 10 \
    --name train_qwen_adv_unsup_150_v3 \
    --weights /mnt/yihao/codes/buttonDet/ultralytics/runs/elevator/1010/train_qwen_adv_150/weights/best.pt
```

### 6.2 不设置目标数据，按 epoch 内部划分源数据

```bash
/mnt/yihao/envs/ultralytics/bin/python scripts/train_elevator.py \
    --unsup \
    --device 0,1 \
    --batch 64 \
    --epochs 150 \
    --patience 0 \
    --save-period 10 \
    --name train_qwen_adv_unsup_150_v3 \
    --weights /mnt/yihao/codes/buttonDet/ultralytics/runs/elevator/1010/train_qwen_adv_150/weights/best.pt
```

## 7. 训练和测试灯状态阈值的区别

以下两个参数用途不同：

- `--pseudo-light-conf 0.9`：UDA 训练时判断 Teacher 的灯状态伪标签是否可靠。
- `--light-threshold 0.3`：验证或推理时将最终模型的灯概率转换为亮/灭标签。

例如测试时使用：

```bash
/mnt/yihao/envs/ultralytics/bin/python scripts/val_elevator.py \
    --weights /path/to/model.pt \
    --light-threshold 0.3
```

降低测试阶段的 `light_threshold` 不会改变 UDA 训练时已经生成的伪标签。

## 8. 训练监控指标

UDA 训练的 `results.csv` 中包含以下额外指标：

| 指标                                        | 含义                                    |
| ------------------------------------------- | --------------------------------------- |
| `train/pseudo_boxes`                        | 当前 epoch 接受的伪框数量               |
| `train/pseudo_images`                       | 至少接受一个伪框的目标图片数量          |
| `train/pseudo_accept_ratio`                 | 接受伪标签的目标图片比例                |
| `train/pseudo_mean_conf`                    | 接受伪框的平均检测置信度                |
| `train/pseudo_floor_boxes`                  | 接受的 `floor` 类伪框数量               |
| `train/pseudo_light_valid_ratio`            | 灯状态有效伪标签占比                    |
| `train/pseudo_dense_rejected_images`        | 因单图伪框过多而被拒绝的图片数量        |
| `train/pseudo_class_guard_rejected_batches` | 因类别分布异常而被拒绝的目标 batch 数量 |
| `train/unsup_weight`                        | 当前 epoch 实际使用的无监督损失权重     |

判断 UDA 是否实际参与训练时，应同时确认：

- `train/unsup_weight > 0`。
- `train/pseudo_boxes > 0`。
- `train/pseudo_images > 0`。
- 周期权重或 `last.pt` 相对于初始化源模型发生了实际变化。
