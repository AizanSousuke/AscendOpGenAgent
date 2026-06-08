---
name: kernel-opt-rotarymul
description: Rotarymul 算子的 Triton Ascend 四层隔离优化经验
metadata:
  type: reference
---

# Rotarymul 算子优化经验

**算子类别**: `rotarymul`（RoPE 旋转位置编码乘法）
**典型特征**: 4D 张量 `[B, H, S, D]`，支持 `half` / `interleave` 两种模式，支持 fp16/bf16/fp32，r1/r2 支持 broadcast
**性能基准**: 几何平均加速比 1.02x vs torch（50 cases 全通过）

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 禁止 flat-1D 索引分解
- **禁止**在 kernel 内将一维 flat index 通过 `%` 和 `//` 运算反向分解为 `(b, h, s, d)` 多维坐标。
- **Why:** Ascend 编译器会将这些 per-element 的地址计算标量化，生成大量 `scf.for` step-1 循环和 `memref<1xf16>` 标量 load/store，导致 HIVM intrinsics 利用率极低（实测仅 19.8%）。
- **How to apply:** 每个 program 直接处理一个完整的 `(b, h, s)` 位置，使用 `tl.arange` 做 contiguous vector load/store。

### L1.2 必须显式处理 broadcast stride
- **必须**在 Host 侧计算 r1/r2 的 broadcast stride：若某维 shape 为 1 且 input 对应维 >1，则该维 stride 设为 0。
- **Why:** RotaryMul 的 r1/r2 常见 broadcast 模式（如 `[B,1,S,D]` 对 `[B,H,S,D]`），若直接传 `t.stride()` 会导致 kernel 内地址计算错误。
- **How to apply:**
  ```python
  def _bc_stride(t, input_shape):
      return tuple(0 if t.shape[i] == 1 and input_shape[i] > 1 else t.stride(i) for i in range(4))
  ```

### L1.3 禁止在 kernel 内做 dtype 分支判断
- **禁止**在 `@triton.jit` kernel 内通过 `if dtype == torch.float16` 这类运行时分支判断数据类型。
- **Why:** Triton kernel 内无法直接访问 PyTorch dtype 对象；应通过 `tl.constexpr` 布尔标志（如 `IS_FP16`, `IS_BF16`）在编译期确定分支。
- **How to apply:** Host 侧计算 `IS_FP16 = (dtype == torch.float16)` 等标志，作为 `tl.constexpr` 传入 kernel。

### L1.4 fp16/bf16 必须在 kernel 内升精度到 fp32 计算
- **必须**对 fp16/bf16 输入在 kernel 内先 `.to(tl.float32)` 计算，再转回原精度存储。
- **Why:** 直接以 fp16/bf16 做乘加减会导致精度误差超出 verify 阈值（relative error > 1e-3）。
- **How to apply:**
  ```python
  if IS_FP16 or IS_BF16:
      x = x.to(tl.float32)
  # ... compute ...
  if IS_FP16:
      out = out.to(tl.float16)
  elif IS_BF16:
      out = out.to(tl.bfloat16)
  ```

### L1.5 禁止 Adaptive TILE_S（编译期动态 tile 大小）
- **禁止**在 Host 侧根据 S/D 大小选择不同的 `TILE_S`（如 `TILE_S = 32 if S >= 512 and D <= 64 else 16`）。
- **Why:** `TILE_S` 作为 `tl.constexpr`，若在不同 shape 间变化会导致 kernel 被重新编译；更大的问题是过大的 2D tile（如 32x32）在 Ascend 上会被编译器标量化，造成灾难性性能退化（实测 65x  slowdown）。
- **How to apply:** 固定 `TILE_S = 16`，通过 uniform grid splitting 解决负载均衡问题。

### L1.6 必须添加 `tl.assume` 编译器提示
- **必须**对 stride 和 shape 添加 `tl.assume` 提示，尤其是 `stride_d == 1`、`half_d >= 16`、`TILE_S > 0` 等。
- **Why:** 帮助 Ascend 编译器生成 vector load/store 而非标量循环。
- **How to apply:** 在 kernel 入口放置 `tl.assume(stride_in_d == 1)` 等。

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧分支决策树（伪代码）

```python
# 1. 数据准备
input_c = input if input.is_contiguous() else input.contiguous()
B, H, S, D = input_c.shape
output = torch.empty_like(input_c)

# 2. Broadcast stride 处理（L1.2）
r1_stride = _bc_stride(r1, input_c.shape)
r2_stride = _bc_stride(r2, input_c.shape)

# 3. 编译期常量
IS_HALF = (rotary_mode == 'half')
IS_FP16 = (dtype == torch.float16)
IS_BF16 = (dtype == torch.bfloat16)
MAX_HALF_D = D // 2          # 作为 tl.constexpr 传入
TILE_S = 16                  # 固定值（L1.5）
assert S % TILE_S == 0

# 4. Grid 计算
num_s_tiles = S // TILE_S
num_blocks = B * H * num_s_tiles
def _grid(meta): return (min(num_blocks, VEC_CORE_NUM),)

# 5. Kernel 启动
kernel[_grid](..., num_cores=VEC_CORE_NUM, TILE_S=TILE_S, MAX_HALF_D=MAX_HALF_D, ...)
```

### L2.2 Kernel 内多核并行骨架（Uniform Grid Splitting）

**核心思想**：将 `num_blocks = B * H * num_s_tiles` 均匀分配到 `num_cores` 个 vector core 上，避免 naive `ceil(num_blocks / num_cores)` 导致的 idle core。

```python
pid = tl.program_id(0)
num_blocks = B * H * num_s_tiles

blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)

for block_idx in range(block_start, block_end):
    # 将 block_idx 解码为 (b, h, s_tile)
    tmp = block_idx // num_s_tiles
    s_tile = block_idx - tmp * num_s_tiles
    tmp2 = tmp // H
    h = tmp - tmp2 * H
    b = tmp2
    s_start = s_tile * TILE_S
    # ... load/compute/store ...
```

### L2.3 2D Tiling 向量加载模式

**模式**：每个 block 处理 `TILE_S` 个连续 S 位置 × `MAX_HALF_D` 个连续 D 位置。

```python
s_offs = tl.arange(0, TILE_S)[:, None]   # [TILE_S, 1]
d_offs = tl.arange(0, MAX_HALF_D)[None, :] # [1, MAX_HALF_D]

# base offset 指向 s_start 行
in_base = b * stride_b + h * stride_h + s_start * stride_s

# 2D 索引广播为 [TILE_S, MAX_HALF_D]
idx1 = in_base + s_offs * stride_s + d_offs * stride_d
idx2 = in_base + s_offs * stride_s + (d_offs + half_d) * stride_d

inp1 = tl.load(input_ptr + idx1)  # 2D vector load
inp2 = tl.load(input_ptr + idx2)
```

### L2.4 half vs interleave 模式处理

- **half 模式**：将 D 维从中间切分，`[..., :half_d]` 和 `[..., half_d:]` 分别与 r1/r2 的对应半区做旋转乘法。
- **interleave 模式**：将 D 维按奇偶分离，`[..., 0::2]` 和 `[..., 1::2]` 分别做旋转乘法。
- **统一公式**：
  - half: `out1 = x1 * r1_1 - x2 * r2_1`, `out2 = x2 * r1_2 + x1 * r2_2`
  - interleave: `out_even = x1 * r1_e - x2 * r2_e`, `out_odd = x2 * r1_o + x1 * r2_o`

---

## Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

### L3.1 从 flat-1D 到 per-position vectorization

**问题**：初始实现使用 `BLOCK_SIZE=1024` 的 flat-1D 循环，每个 thread 处理一个 flat index，通过 `%` 和 `//` 分解坐标，导致标量退化。

**解决**：改为每个 program 处理一个 `(b, h, s)` 位置，D 维用 `tl.arange(0, MAX_HALF_D)` 向量化。关键变化：
- 去掉 `BLOCK_SIZE`，改为 `MAX_HALF_D = D // 2` 作为 `tl.constexpr`
- 去掉 `mask = offsets < total_elements`
- 坐标分解从 per-element 变为 per-position（仅分解 `b, h, s`，`d` 由 `tl.arange` 覆盖）

**可替代方向**: 若 D 不固定，可用 `tl.arange(0, BLOCK_D)` 配合 `for d_tile in range(0, half_d, BLOCK_D)` 做 D 维循环分块。

### L3.2 Uniform Grid Splitting 消除 idle core

**问题**：naive `ceil(num_blocks / num_cores)` 在 `num_blocks` 不是 `num_cores` 倍数时，最后一个 core 处理少量 block，其余 core 提前结束，造成同步等待。

**解决**：
```python
blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)
```
前 `remainder` 个 core 各多处理 1 个 block，实现完全均匀分配。

**可替代方向**: 若 block 粒度极不均匀（如不同 block 工作量差异大），可考虑 dynamic work stealing 或按工作量加权分配，但 RotaryMul 中每个 block 工作量相同，uniform splitting 最优。

### L3.3 2D Tiling (TILE_S=16) 摊平同步开销

**问题**：per-position kernel（每个 program 只处理 1 个 S 位置）在大 S shape 下产生过多 pipeline sync（`hivm.hir.set_flag`, `wait_flag`, `pipe_barrier`），大 S 性能差。

**解决**：每个 program 一次处理 `TILE_S=16` 个连续 S 位置，用 2D `tl.arange` 做 `[TILE_S, MAX_HALF_D]` 的向量化 load/store。

**关键参数选择**：
- `TILE_S = 16` 是经验最优值（非 IR 分析得出）
- `TILE_S = 8` 导致 2D tile 太小（256 elements），vectorization 效果差
- `TILE_S = 32` 导致编译器标量化，性能退化 65x

**可替代方向**: 对于 S 较小（如 S < 16）的 shape，可回退到 TILE_S=1 的 per-position 模式，但 RotaryMul 的 S 通常为 128/256/512/1024+，固定 16 即可。

### L3.4 避免 Host 侧 dtype 转换和 expand

**问题**：早期实现在 Host 侧将输入 `.to(torch.float32)` 并用 `expand_as()` 处理 broadcast，引入额外内存拷贝和峰值内存占用（16.9MB → 7.44MB）。

**解决**：
- 不在 Host 侧做 dtype 转换，仅在 kernel 内对 fp16/bf16 升精度
- 不在 Host 侧 `expand` broadcast 张量，而是通过自定义 `_bc_stride` 在 kernel 内用 stride=0 处理 broadcast

**可替代方向**: 若 broadcast 模式更复杂（如非末尾维 broadcast），`_bc_stride` 逻辑可能需要扩展。

---

## Layer 4: 完整归档（Agent 默认不读取，仅人工复盘）

> ⚠️ **Agent 注意**：以下仅为历史实现的路径记录。你**禁止**直接复制其代码结构、变量命名或 kernel 组织方式。

### 历史实现归档

| 版本 | 代码 | 报告 | 摘要 | 性能 | 特点 |
|------|------|------|------|------|------|
| v5_20260604 | archive/rotarymul/rotarymul_v5_20260604.py | report.md | summary.json | 1.06x geomean | TILE_S=16+uniform_grid+2D_tiling |

### 完整归档路径（Layer 4）
```
/home/whd/project/0529/AscendOpGenAgent/.claude/memory/archive/rotarymul/
```

### 原始工作目录
```
/home/lg/project/output/0509_kimi/op_1_RotaryMul_20260509_2110_4353
```

### 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape [1,1,128,64] | 2.4x | 小 S 高并行度，轻松超越 torch |
| 中 shape [1,8,512,64] | 0.67x | torch aclnn 优化充分，Triton 有 gap |
| 大 shape [1,8,32768,64] | 0.10x | 内存带宽瓶颈，Triton 仍落后 |
| 全量 50 cases | 1.02x | 几何平均刚好达标 |

**关键结论**：
1. RotaryMul 在小 shape 上 Triton 有明显优势（2-3x），但在大 shape / 高并行度场景下，torch aclnn 的 `aclnnRotaryPositionEmbedding` 高度优化，Triton 难以超越。
2. 2D Tiling + Uniform Grid Splitting 是达到目标加速比（0.8x）的关键；去掉任一项都会使几何平均低于目标。
3. 标量退化（flat-1D index 分解）是 Ascend 上最常见的性能陷阱，必须从算法设计阶段避免。

---

## 常见陷阱与避免方法

### 陷阱 1: Flat-1D 索引分解导致标量退化
- **问题**: kernel 内用 `d = offsets % half_d; s = (offsets // half_d) % S` 等分解坐标
- **解决**: 改用 per-position 处理 + `tl.arange` 向量化 D 维
- **验证方法**: 检查编译后 IR 是否含大量 `scf.for` step-1 循环和 `memref.load %ptr[%c0]` 标量模式

### 陷阱 2: Adaptive TILE_S 导致编译器标量化
- **问题**: 根据 S/D 动态选择 TILE_S（如 16/32/64），大 tile 被编译器拆成标量循环
- **解决**: 固定 `TILE_S = 16`，用 grid splitting 解决负载均衡
- **替代方案**: 若必须大 tile，需验证 IR 中是否仍保持 vector load/store

### 陷阱 3: 忽略 broadcast stride
- **问题**: r1/r2 的 shape 为 `[B,1,S,D]` 时直接传 `t.stride()`，导致 kernel 内地址跳变错误
- **解决**: Host 侧显式计算 broadcast stride（broadcast 维 stride=0）

### 陷阱 4: fp16/bf16 精度不足
- **问题**: kernel 内直接以 fp16 做乘加减，relative error 超标
- **解决**: kernel 内升 fp32 计算，存回前转回原精度

### 陷阱 5: Naive grid splitting 导致 idle core
- **问题**: `grid = (num_cores,)` + `for block in range(pid, num_blocks, num_cores)` 在 `num_blocks < num_cores` 时大量 core 空闲
- **解决**: Uniform grid splitting（L2.2 / L3.2）确保每个 core 处理连续且均匀的 block 范围
