# 电梯按钮检测、楼层识别与点亮状态判断改造计划

## 1. 目标与范围

基于 Ultralytics 8.4.138 的 `yolov8-p2.yaml`，保留 P2/P3/P4/P5 四个检测输出层，在共享 Backbone 和 FPN/PAN 后增加三个解耦分支：

1. 按钮检测分支：预测 bbox 以及 `floor/other` 两类按钮。
2. 楼层识别分支：识别 `-2、-1、G、1～21`。
3. 上下文灯光分支：预测每个按钮的点亮概率。

继续使用 Ultralytics 的 `detect` 任务，不新增顶层任务。楼层和灯光结果作为检测框属性，与检测候选共享尺度、空间位置和候选索引，不进行第二次 bbox 匹配。

## 2. 总体架构

```text
输入图像
   ↓
YOLOv8 Backbone
   ↓
P2-P5 FPN/PAN
   ↓
P2/4 ─┐
P3/8 ─┼─→ ElevatorDetect
P4/16 ┤       ├─ 检测分支：bbox + floor/other
P5/32 ┘       ├─ 楼层分支：Slot-1 + Slot-2
               └─ 灯光分支：local + context + contrast
   ↓
Decode + class-agnostic NMS
   ↓
bbox + button type + floor + floor confidence + light probability
```

现有四尺度输出改为：

```yaml
- [[18, 21, 24, 27], 1, ElevatorDetect, [nc]] # P2, P3, P4, P5
```

- 第 18 层：P2，stride 4。
- 第 21 层：P3，stride 8。
- 第 24 层：P4，stride 16。
- 第 27 层：P5，stride 32。

所有分支严格按照 `P2 → P3 → P4 → P5` 的顺序展平和拼接。

## 3. 按钮检测分支

检测类别固定为：

```yaml
names:
  0: floor
  1: other
```

- `floor`：带楼层标识的按钮，需要计算楼层识别损失。
- `other`：开门、关门、报警等其他按钮，只计算检测和灯光损失。

`ElevatorDetect` 继承现有 `Detect`，复用 `cv2` bbox/DFL、`cv3` 按钮分类，以及 stride、anchor point、bbox decode 和 bias 初始化。当前 YOLOv8 没有独立 objectness，按钮置信度由最高按钮类别概率表示。

## 4. 楼层识别分支

原 Tens/Ones Head 无法表达负号和字母，改为两位置字符 Head：

```text
Floor Recognition Branch
    ↓
Shared Floor Stem
    ├─ Slot-1 Head：5 logits
    └─ Slot-2 Head：11 logits
```

### 4.1 Slot-1 词表

```text
0: blank
1: -
2: G
3: 1
4: 2
```

### 4.2 Slot-2 词表

```text
0: blank
1: 0
2: 1
3: 2
4: 3
5: 4
6: 5
7: 6
8: 7
9: 8
10: 9
```

该词表是满足 `-2、-1、G、1～21` 的最小集合。

### 4.3 楼层编码

| 楼层 | Slot-1 | Slot-2 |
| --- | ---: | ---: |
| `-2` | `1 (-)` | `3 (2)` |
| `-1` | `1 (-)` | `2 (1)` |
| `G` | `2 (G)` | `0 (blank)` |
| `1` | `0 (blank)` | `2 (1)` |
| `2` | `0 (blank)` | `3 (2)` |
| `9` | `0 (blank)` | `10 (9)` |
| `10` | `3 (1)` | `1 (0)` |
| `17` | `3 (1)` | `8 (7)` |
| `20` | `4 (2)` | `1 (0)` |
| `21` | `4 (2)` | `2 (1)` |

不采用完整楼层 24 分类 Head，以保留字符共享能力并减少输出通道。

## 5. 上下文灯光分支

P2～P5 每个尺度分别计算：

```text
local = LocalConv(P_i)
context = ContextConv(AvgPool(P_i, kernel=15, stride=1, padding=7))
contrast = local - context

light_feature = Concat(local, context, contrast)
light_feature = Conv1x1(light_feature)
light_feature = Conv3x3(light_feature)
light_logit = Conv1x1(light_feature)
```

第一版不加入 P4 全局池化上下文、Self-Attention 或动态 ROIAlign，以控制四尺度版本计算量，并保持 ONNX、TensorRT、RKNN 兼容性。

## 6. Head 输出约定

训练阶段返回：

```python
{
    "boxes": boxes,          # B × (4 * reg_max) × N
    "scores": button_logits, # B × 2 × N
    "feats": features,
    "slot1": slot1_logits,   # B × 5 × N
    "slot2": slot2_logits,   # B × 11 × N
    "light": light_logits,   # B × 1 × N
}
```

原始推理输出通道数为 `4 + 2 + 5 + 11 + 1 = 23`：

```text
B × 23 × N

[x, y, w, h,
 floor_probability, other_probability,
 slot1_probs[5], slot2_probs[11], light_probability]
```

NMS 后每个目标仍有 23 个值：

```text
[x1, y1, x2, y2,
 button_confidence, button_class,
 slot1_probs[5], slot2_probs[11], light_probability]
```

## 7. 标签格式

继续使用 YOLO TXT，每行扩展为 9 列：

```text
button_cls cx cy width height slot1 slot2 light light_valid
```

示例：

```text
# -2 楼，点亮
0 0.50 0.40 0.08 0.06 1 3 1 1

# -1 楼，未点亮
0 0.50 0.45 0.08 0.06 1 2 0 1

# G 层，点亮
0 0.50 0.50 0.08 0.06 2 0 1 1

# 7 楼，未点亮
0 0.50 0.55 0.08 0.06 0 8 0 1

# 17 楼，点亮
0 0.50 0.60 0.08 0.06 3 8 1 1

# 21 楼，灯光状态不可靠
0 0.50 0.65 0.08 0.06 4 2 0 0

# other，不计算楼层识别损失
1 0.50 0.70 0.08 0.06 0 0 0 1
```

规则：

- `button_cls=0` 时必须提供合法 Slot-1/Slot-2 编码。
- `button_cls=1` 时使用 `0 0` 占位，楼层损失必须忽略。
- `light_valid=1` 时 `light` 只能为 0 或 1。
- `light_valid=0` 时忽略灯光损失。
- bbox 坐标归一化到 `[0,1]`。
- 如果未来存在“确定为 floor，但字符不可辨认”的标注，再增加 `floor_valid`；当前不预先增加该字段。

## 8. 数据增强与标签对齐

楼层和灯光属性必须与 bbox 一起经过 Mosaic、MixUp/CutMix、随机透视、裁剪、无效框删除和空间 Albumentations。

建议在增强内部暂时组合为：

```text
cls_ext = [button_cls, slot1, slot2, light, light_valid]
```

现有增强路径筛选或拼接 `cls` 时即可同步处理全部属性。最终 `Format` 阶段拆分为：

```python
batch["cls"]       # N × 1
batch["elevator"]  # N × 4: slot1, slot2, light, light_valid
```

空间 Albumentations 当前将 `cls` 强制恢复为 `N×1`，需要改为保留实际列数，防止属性丢失。

## 9. 损失函数

```text
Ldetect = λbox × Lbox + λcls × Lfloor_other + λdfl × Ldfl

Lattribute =
    λslot1 × CrossEntropy(slot1)
  + λslot2 × CrossEntropy(slot2)
  + λlight × BCEWithLogits(light)

Ltotal = Ldetect + Lattribute
```

计算流程：

1. 复用 TaskAlignedAssigner 完成检测正样本分配。
2. 获取 `fg_mask` 和 `target_gt_idx`。
3. 使用 `target_gt_idx` 将每个正候选映射到对应 GT 属性。
4. Slot-1/Slot-2 只在 `button_cls == floor` 的正样本上计算。
5. Light 在两类正样本上都可计算，但要求 `light_valid == 1`。
6. 每个辅助损失按自身有效正样本数归一化。
7. 没有有效属性标签时，零损失仍连接对应预测张量，避免 DDP unused-gradient 错误。

训练日志增加：

```text
box_loss
cls_loss
dfl_loss
slot1_loss
slot2_loss
light_loss
```

专用模型 YAML 中设置初始权重：

```yaml
slot1_gain: 1.0
slot2_gain: 1.0
light_gain: 1.0
```

后续根据损失量级、梯度和验证指标调节，不修改全局 `default.yaml`。

## 10. NMS 与推理后处理

NMS 只使用 bbox 和 `floor/other` 分数，Slot-1、Slot-2、Light 作为 extra 原样保留。

当前 detect 推理若传入 `nc=0`，会把新增 extra 误判为检测类别，因此预测和验证路径必须显式传入：

```text
nc = len(model.names) = 2
```

运行时使用 `agnostic_nms=True`，不修改通用 NMS 算法。

### 10.1 合法楼层解码

```python
slot1_tokens = ["", "-", "G", "1", "2"]
slot2_tokens = ["", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
```

不要直接对两个位置独立 argmax 后拼接，因为可能产生 `G5、-9、29` 等非法组合。应枚举合法集合 `{-2, -1, G, 1, 2, ..., 21}`，对每个合法楼层计算：

```text
Pchars(f) = sqrt(P(slot1=f.slot1) × P(slot2=f.slot2))
floor_score(f) = P(button=floor) × Pchars(f)
```

选择合法集合中 `floor_score` 最高的楼层。

### 10.2 灯光状态

保留原始 `light_probability`，业务状态建议为：

```text
light_probability >= 0.70 → on
light_probability <= 0.30 → off
其他                       → unknown
```

默认楼层匹配分数不乘灯光概率。只有查询明确要求“已点亮的某楼层按钮”时，才将灯光概率加入联合分数。

## 11. 验证指标

检测指标保持 Precision、Recall、mAP50 和 mAP50-95。属性指标只在预测类别与 GT 一致且 IoU ≥ 0.5 的匹配目标上计算。

楼层指标仅统计 GT 类别为 `floor`：

- Slot-1 accuracy。
- Slot-2 accuracy。
- Floor exact accuracy。
- Negative-floor accuracy（`-2/-1`）。
- Ground-floor accuracy（`G`）。
- Numeric-floor accuracy（`1～21`）。
- Target bbox success：楼层正确且 bbox IoU ≥ 0.5。

灯光指标统计所有 `light_valid=1` 的正确匹配目标：Accuracy、Precision、Recall、F1、ROC-AUC 和 Unknown 区间占比。

`other` 不进入楼层准确率分母，但参与检测 mAP 和灯光指标。训练集、验证集和测试集应按电梯型号或面板划分，避免同一面板跨集合造成数据泄漏。

## 12. 文件修改清单

仓库根目录：`/mnt/yihao/codes/buttonDet/ultralytics`

### 12.1 修改现有文件

1. `ultralytics/nn/modules/head.py`
   - 新增 `ElevatorDetect(Detect)`。
   - 保留 `cv2/cv3`，增加四尺度 Floor Stem、Slot Head 和 Light Head。
   - 扩展 `forward_head()`、`_inference()` 和 `fuse()`。

2. `ultralytics/nn/modules/__init__.py`
   - 导出 `ElevatorDetect`。

3. `ultralytics/nn/tasks.py`
   - 导入 `ElevatorDetect`。
   - 加入 `parse_model()` 的 Detect Head 集合。
   - 为该 Head 选择 `v8ElevatorDetectionLoss`。

4. `ultralytics/utils/loss.py`
   - 新增 `v8ElevatorDetectionLoss`。
   - 复用检测分配结果计算三个辅助损失。

5. `ultralytics/data/utils.py`
   - 标签校验支持扩展的 9 列格式。
   - 分别校验 bbox、Slot-1、Slot-2、light、light_valid。

6. `ultralytics/data/dataset.py`
   - 新增 `ElevatorDataset` 和 `ElevatorFormat`。
   - 管理复合标签和最终 `elevator` 张量。
   - 使用与普通 detect 数据不同的缓存标识。

7. `ultralytics/data/augment.py`
   - 空间 Albumentations 保留复合 `cls` 的实际列数。

8. `ultralytics/data/build.py`
   - 数据 YAML 含 `elevator: true` 时选择 `ElevatorDataset`。

9. `ultralytics/models/yolo/detect/predict.py`
   - NMS 显式传入真实 `nc`。
   - 解码 extra 通道并构造电梯属性结果。

10. `ultralytics/models/yolo/detect/val.py`
    - NMS 显式传入真实 `nc`。
    - 增加楼层、灯光和联合业务指标。

11. `ultralytics/engine/results.py`
    - 增加 `ElevatorAttributes` 和 `Results.elevator`。
    - 支持索引、`.cpu()`、`.numpy()`、`summary()`、JSON 和绘图输出。

### 12.2 新增配置文件

1. `ultralytics/cfg/models/v8/yolov8-elevator-p2.yaml`
   - 基于现有 `yolov8-p2.yaml`。
   - `nc: 2`，最后一层改为 `ElevatorDetect`。
   - 设置 `end2end: false` 和辅助损失权重。

2. `ultralytics/cfg/datasets/elevator-button.yaml`
   - 设置 `elevator: true`。
   - 设置 `names: {0: floor, 1: other}`。
   - 配置 train/val/test 路径。

### 12.3 第一版不修改

- `ultralytics/models/yolo/detect/train.py`。
- `ultralytics/utils/nms.py`。
- `ultralytics/engine/exporter.py`。
- `ultralytics/cfg/default.yaml`。
- 原始 `ultralytics/cfg/models/v8/yolov8-p2.yaml`。
- YOLO 顶层 `task_map`。

## 13. 初始化、训练与部署

- Backbone、FPN/PAN 和 bbox 分支从 YOLOv8-P2 权重加载。
- 如果预训练类别数不是 2，最终分类卷积重新初始化。
- Floor 和 Light 分支始终重新初始化。
- 从 `yolov8n-p2` 开始，优先裁剪和透视矫正电梯面板。
- 比较 1280、1600、1920 三档输入，不直接默认最高分辨率。
- 第一版使用轻量属性 Stem，不加入全局上下文和 Attention。

端侧导出建议：

```text
end2end=false
nms=false
```

NPU 输出原始 `B × 23 × N` 预测，CPU 完成 decode、class-agnostic NMS、合法楼层枚举和灯光状态判断。

## 14. 实施顺序与验收

1. 完成标签编码、校验和 Dataset，验证增强前后属性与 bbox 对齐。
2. 完成 `ElevatorDetect`，验证 P2～P5 输出形状和拼接顺序。
3. 完成专用损失，覆盖空目标、无 floor、无有效灯光标签等 batch。
4. 完成 NMS extra 保留和合法楼层解码。
5. 完成 Results API 和可视化。
6. 完成验证指标及按电梯型号划分的数据集。
7. 对 PyTorch、ONNX 和目标 RKNN/NPU 做端到端数值对齐。

最低验收条件：

- 普通 detect 模型训练、预测、导出行为不变。
- 四尺度 bbox、Slot-1、Slot-2、Light 候选数完全一致。
- Mosaic、裁剪和空间增强后属性没有错配。
- `other` 不产生楼层损失。
- `light_valid=0` 不产生灯光损失。
- NMS 前后每个保留框的 17 个 extra 通道索引一致。
- 解码结果只能属于 `{-2, -1, G, 1～21}`。
- ONNX/RKNN 输出布局固定为 `4 + 2 + 5 + 11 + 1 = 23`。
