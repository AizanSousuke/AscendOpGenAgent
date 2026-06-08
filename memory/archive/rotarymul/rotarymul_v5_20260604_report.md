# RotaryMul Triton-Ascend 算子生成报告

## 基本信息

- **算子名称**: RotaryMul
- **架构**: ascend910b1
- **工作目录**: /home/whd/WorkSpace_Single/0603_rotarymul/memory_test4/triton_ascend_output/op_1_RotaryMul_20260604_1558_7655
- **任务来源**: /home/whd/project/0529/AscendOpGenAgent/benchmarks/NPUKernelBench/level2/1_RotaryMul.py

## 生成结果

- **Phase 3 迭代次数**: 1（iter_0 验证通过）
- **Phase 4 迭代次数**: 1（opt_iter_0 尝试 constexpr stride + multibuffer，性能略降，回退至基线）
- **最终版本来源**: Phase 3 基线代码

## 性能指标

- **目标加速比**: 0.8
- **是否达到目标**: 是
- **实际最佳加速比**: 1.0563
- **Shape 通过率**: 50/50 (100%)

## 性能数据

| 指标 | 数值 |
|------|------|
| 几何平均加速比 | 1.0563 |
| Framework 平均延迟 (ms) | 0.1022 |
| Implementation 平均延迟 (ms) | 0.0310 |

## Shape 明细

| Case | Shape | Status | Speedup |
|------|-------|--------|---------|
| 1 | [1,1,128,64] fp16 half | pass | 2.3200 |
| 2 | [1,1,256,64] fp16 half | pass | 1.9000 |
| 3 | [1,1,512,64] fp16 half | pass | 1.4146 |
| 4 | [1,1,1024,64] fp16 half | pass | 0.9831 |
| 5 | [1,1,2048,64] fp16 half | pass | 0.8571 |
| ... | ... | ... | ... |
| 50 | [1,8,32768,64] bf16 interleave | pass | (见 perf_result.json) |

全部 50 个 case 验证通过，无异常索引。

## 代码路径

- 最终代码: `1_RotaryMul_generated.py`

## 关键设计决策

1. **2D Tiling**: 每个 program 处理 TILE_S=16 个连续 S 位置，D 维用 tl.arange 向量化
2. **Uniform Grid Splitting**: 将 B*H*num_s_tiles 个 block 均匀分配到各 vector core
3. **Broadcast Stride 处理**: Host 侧显式计算 broadcast stride（broadcast 维 stride=0）
4. **FP32 精度计算**: fp16/bf16 在 kernel 内升精度到 fp32 计算，再转回原精度存储
5. **Half/Interleave 双模式支持**: 通过 IS_HALF tl.constexpr 在编译期分支

## 优化尝试

- **尝试**: 将 stride 参数声明为 tl.constexpr + multibuffer=True
- **结果**: 几何平均加速比从 1.0563 降至 1.0289，性能劣化
- **结论**: 回退至 Phase 3 基线代码
