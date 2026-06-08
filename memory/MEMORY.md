# Memory Index

## 算子优化经验
- [历史探索经验积累方案](kernel-opt-framework.md) — 算子分类体系、四层隔离存储模型、复用机制与防依赖策略
- [Pad 算子优化经验](kernel-opt-pad.md) — 多 kernel 分支、维度压缩、constant 模式特化、边界映射模板

- [Repeat 算子优化经验](kernel-opt-repeat.md) — transformation-memory 类算子、逐维度串行处理、constexpr 循环展开、多核分区策略
- [Rotarymul 算子优化经验](kernel-opt-rotarymul.md) — 2D Tiling、Uniform Grid Splitting、Broadcast stride 处理
- [Iou 算子优化经验](kernel-opt-iou.md) — 2D Tiling + Broadcast、Grid 限制为核数、multibuffer、输出转置语义
## 完整代码归档（Layer 4，Agent 默认不可读）
- [归档目录说明](archive/README.md) — 读取约束、归档规则、目录结构
- [Pad](archive/pad/pad_v1_20260522.py) — 1.68x，51/51；[R](archive/pad/pad_v1_20260522_report.md)/[S](archive/pad/pad_v1_20260522_summary.json)
- [Repeat](archive/repeat/repeat_v2_20260526.py) — 0.88x，49/49 pass
- [Rotarymul 算子最佳实现](archive/rotarymul/rotarymul_v5_20260604.py) — 几何平均加速比 1.06x，50/50 cases 通过；配套 [report](archive/rotarymul/rotarymul_v5_20260604_report.md) / [summary](archive/rotarymul/rotarymul_v5_20260604_summary.json)
- [Iou 算子最佳实现](archive/iou/iou_v2_20260604.py) — 几何平均加速比 2.26x，30/30 cases 通过；配套 [report](archive/iou/iou_v2_20260604_report.md) / [summary](archive/iou/iou_v2_20260604_summary.json)
