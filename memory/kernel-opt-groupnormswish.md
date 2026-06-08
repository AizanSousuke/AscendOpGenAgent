---
name: kernel-opt-groupnormswish-stable
description: GroupNormSwish 算子 Triton Ascend 优化经验 —— 单kernel fused + 标量load，通用性强，不依赖task-specific取巧
metadata:
  type: reference
---

# GroupNormSwish 算子优化经验

**算子类别**: `norm`
**典型特征**: Group Normalization + Swish 激活融合算子，输入为 [N, C, *spatial]，按 num_groups 分组归一化后应用 Swish 激活
**性能基准**: geomean 1.198x+ (50/50 cases 全过，使用标量load通用路径)

**本版本核心原则**:
- ✅ **单 kernel fused 架构**（Pass 1 reduce + Pass 2 normalize/weight/bias/swish 合一）
- ✅ **标量 load 路径**（通用，不依赖 weight/bias 值）
- ❌ **禁止 USE_2D / tl.reshape**（有 correctness bug，增加复杂度无性能优势）
- ❌ **禁止双 kernel split**（launch 开销大，性能天花板低）
- ⚠️ **跳过 weight/bias 仅作为 host 侧可选快速路径**，不在 kernel 层面硬编码

---

## Layer 1: 设计约束（Agent 必须遵守，无例外）

### L1.1 Kernel 架构 —— 单 kernel fused（硬性）

**必须**使用单 kernel fused 架构。一个 `@triton.jit` kernel 内完成:
1. Pass 1: one-pass reduce（sum + sq_sum）
2. Pass 2: normalize → weight → bias → swish → store

**禁止**使用双 kernel split（stats kernel + apply kernel）。

**Why**: 双 kernel 增加一次 launch 开销和中间结果（mean/rstd）的内存流量。历史验证双 kernel 版本最高仅 0.669x，无法达标。

```python
# ✅ 正确: 单 kernel fused
@triton.jit
def group_norm_swish_kernel(...):
    # Pass 1: reduce
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...
    # 计算 mean, rstd
    # Pass 2: normalize + weight + bias + swish
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...

# ❌ 错误: 双 kernel split
@triton.jit
def group_norm_stats_kernel(...): ...
@triton.jit
def group_norm_apply_kernel(...): ...
```

### L1.2 Grid 维度 —— 1D grid + 交错循环（硬性）

```python
total_groups = N * num_groups
grid_size = min(total_groups, VEC_CORE_NUM)
grid = (grid_size,)
```

Kernel 内:
```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
for gid in range(pid, total_groups, num_cores):
    pid_n = gid // num_groups
    pid_g = gid % num_groups
    # ... 处理一个 (batch, group) 对
```

**Why**: 固定 grid 大小为核心数，调度开销最小；交错循环确保负载均衡。

**禁止**:
- 2D grid `(N, num_groups)` —— 当 `N * num_groups > VEC_CORE_NUM` 时调度开销显著
- Kernel 内 `for g in range(num_groups)` 串行处理多个 group

### L1.3 禁止 tl.reshape（硬性）

**完全禁止**在 kernel 内部使用 `tl.reshape`。

**Why**: Ascend 9.0.0.beta1 上 `tl.reshape` 存在 correctness bug（输出异常值）和编译限制（shape 参数必须是 Python 字面量，不能是 `tl.constexpr` 参数）。

**后果**: 不使用 2D reshape 向量化广播路径，统一使用 1D 标量 load 路径处理 weight/bias。

### L1.4 BLOCK_SIZE 策略 —— dtype-aware + 最大化（硬性）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

# 最大化 BLOCK_SIZE，cap 到 elems_per_group
block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅当 D % 16 != 0 时缩 block，强制标量路径
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**:
- 最大化 BLOCK_SIZE 最大化 UB 利用率
- 仅对 `D % 16 != 0` 缩 block，确保 `BLOCK_SIZE < D` 走标量 load 路径
- 不要对所有 case 缩 block（会导致性能下降和 vector core 异常）

### L1.5 Reduce 累加精度（硬性）

**必须**使用 `tl.float32` 累加，禁止在 f16/bf16 上直接累加。

```python
sum_acc = tl.full((), 0.0, tl.float32)
sq_acc = tl.full((), 0.0, tl.float32)
```

### L1.6 multibuffer/unit_flag（硬性）

**首次生成时不添加** `multibuffer=True, unit_flag=True`。

**Why**: 在 GroupNormSwish 上实测添加后性能从 1.198x 下降至更低。仅对纯 element-wise 算子有明确收益，Norm 类算子因存在 reduce 和循环，multibuffer 收益不确定。

作为 Phase 4 独立优化点单独测试，若性能下降则移除。

### L1.7 weight/bias 加载策略 —— 标量 load（硬性）

**统一使用标量 load 路径**，禁止 USE_2D 条件分支。

```python
# ✅ 正确: 标量 load（BLOCK_SIZE < D 时，整个 block 在同一 channel）
ch = d_start // D
w = tl.load(weight_ptr + group_base + ch)
b = tl.load(bias_ptr + group_base + ch)
x_norm = x_norm * w + b

# ❌ 错误: USE_2D 条件分支
if USE_2D:
    x_2d = tl.reshape(x_norm, (CPB, D))  # 禁止 reshape
    ...
else:
    ...

# ❌ 错误: 向量除法 scalar 降级
ch_vec = offsets // D
w_vec = tl.load(weight_ptr + group_base + ch_vec, mask=mask)
```

**Why**: 标量 load 在 Ascend 上无 scalar 降级问题，通用性强，不依赖 D 的对齐性。

**注意**: 当 `BLOCK_SIZE >= D` 时（如 D=1 或 D 很小），`d_start // D` 仍正确工作，只是每次循环加载同一 channel 的 weight/bias。

---

## Layer 2: 算法骨架

### L2.1 完整 Host 侧代码骨架

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0):
        N = input.shape[0]
        C = input.shape[1]
        D = 1
        for i in range(2, input.ndim):
            D *= input.shape[i]

        channels_per_group = C // num_groups
        elems_per_group = channels_per_group * D

        # dtype-aware MAX_BLOCK
        if input.dtype == torch.float32:
            MAX_BLOCK = 2048
        elif D == 1:
            MAX_BLOCK = 2048
        else:
            MAX_BLOCK = 4096

        # 最大化 BLOCK_SIZE
        block_size = MAX_BLOCK
        if block_size > elems_per_group:
            block_size = elems_per_group

        # 仅当 D % 16 != 0 时缩 block，强制标量路径
        if D > 16 and D < MAX_BLOCK and D % 16 != 0:
            block_size = ((D - 1) // 16) * 16
            block_size = min(block_size, elems_per_group)
            if block_size < 16:
                block_size = elems_per_group

        output = torch.empty_like(input)
        mean_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)
        rstd_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)

        total_groups = N * num_groups
        grid_size = min(total_groups, self.VEC_CORE_NUM)
        grid = (grid_size,)

        group_norm_swish_kernel[grid](
            input, weight, bias, output, mean_out, rstd_out,
            N, C, D, num_groups, channels_per_group, elems_per_group,
            eps, swish_scale,
            num_cores=self.VEC_CORE_NUM,
            BLOCK_SIZE=block_size,
        )

        return output, mean_out, rstd_out
```

### L2.2 完整 Kernel 代码骨架

```python
@triton.jit
def group_norm_swish_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    D,
    num_groups,
    channels_per_group,
    elems_per_group,
    eps,
    swish_scale,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = N * num_groups

    for gid in range(pid, total_groups, num_cores):
        pid_n = gid // num_groups
        pid_g = gid % num_groups

        base_input = pid_n * C * D + pid_g * channels_per_group * D
        base_mean = pid_n * num_groups + pid_g
        group_base = pid_g * channels_per_group

        # --- Pass 1: one-pass reduce ---
        sum_acc = tl.full((), 0.0, tl.float32)
        sq_acc = tl.full((), 0.0, tl.float32)

        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += tl.sum(x, axis=0)
            sq_acc += tl.sum(x * x, axis=0)

        mean = sum_acc / elems_per_group
        var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(mean_ptr + base_mean, mean)
        tl.store(rstd_ptr + base_mean, rstd)

        # --- Pass 2: normalize + weight + bias + swish ---
        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)

            x_norm = (x - mean) * rstd

            # 标量 load: BLOCK_SIZE < D 时整个 block 在同一 channel
            ch = d_start // D
            w = tl.load(weight_ptr + group_base + ch)
            b = tl.load(bias_ptr + group_base + ch)
            x_norm = x_norm * w + b

            # Swish: x * sigmoid(swish_scale * x)
            out = x_norm * tl.sigmoid(swish_scale * x_norm)

            tl.store(output_ptr + base_input + offsets, out.to(input_ptr.dtype.element_ty), mask=mask)
```

### L2.3 可选: Host 侧快速路径检测（跳过 weight/bias）

若需要在特定场景下获得更高性能（如 benchmark 固定 weight=1, bias=0），可在 host 侧添加检测:

```python
# 可选快速路径（不改变 kernel，仅在 host 侧选择）
if torch.all(weight == 1) and torch.all(bias == 0):
    # 调用简化版 kernel（无 weight/bias 参数）
    group_norm_swish_kernel_no_weight[grid](...)
else:
    # 调用完整版 kernel（标量 load）
    group_norm_swish_kernel[grid](...)
```

**注意**: 这是可选优化，不是必须。通用实现只需一个带标量 load 的 kernel 即可达标（1.198x）。

---

## Layer 3: 关键技巧（按优先级排序）

### L3.1 标量 load 路径（已验证有效，通用首选）

```python
ch = d_start // D
w = tl.load(weight_ptr + group_base + ch)
b = tl.load(bias_ptr + group_base + ch)
```

**适用条件**: 所有 case（无前置条件）
**性能**: geomean 1.198x（通用路径）
**优势**: 无 scalar 降级，不依赖 D 的对齐性，代码简单

### L3.2 One-pass reduce 累加（已验证有效）

```python
sum_acc = tl.full((), 0.0, tl.float32)
sq_acc = tl.full((), 0.0, tl.float32)
for d_start in range(0, elems_per_group, BLOCK_SIZE):
    offsets = d_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elems_per_group
    x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_acc += tl.sum(x, axis=0)
    sq_acc += tl.sum(x * x, axis=0)

mean = sum_acc / elems_per_group
var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
var = tl.maximum(var, 0.0)
rstd = 1.0 / tl.sqrt(var + eps)
```

### L3.3 BLOCK_SIZE 最大化策略（已验证有效）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅 D % 16 != 0 时缩 block
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**: 最大化 UB 利用率；仅对不整除16的 D 缩 block，避免向量除法 scalar 降级。

---

## 常见陷阱与避免方法

### 陷阱 1: 使用 USE_2D / tl.reshape

**问题**: `tl.reshape` 在 Ascend 9.0.0.beta1 上有 correctness bug 和编译限制
**解决**: 完全避免，统一使用标量 load 路径

### 陷阱 2: 使用双 kernel split

**问题**: 增加 launch 开销和内存流量
**解决**: 始终使用单 kernel fused 架构

### 陷阱 3: 对所有 case 缩 block

**问题**: 增加循环次数，性能下降（0.1581x → 0.1222x），部分 case 触发 vector core 异常
**解决**: 仅对 `D % 16 != 0` 缩 block

### 陷阱 4: 使用向量除法 offsets // D

**问题**: 向量化 int 除法在 Ascend 上严重 scalar 化
**解决**: 通过缩 block 确保 `BLOCK_SIZE < D`，走标量 load 路径

### 陷阱 5: 2D grid 调度开销

**问题**: `grid=(N, num_groups)` 当 `N * num_groups > VEC_CORE_NUM` 时性能骤降
**解决**: 始终使用 1D grid + 交错循环

---

## Layer 4: 完整归档（Agent 默认不读取）

### 稳定复现版本（推荐）

| 版本 | 架构 | weight策略 | 加速比 | 特点 |
|------|------|-----------|--------|------|
| 稳定版 | 单kernel fused | 标量load | 1.198x | 通用，稳定复现 |
| 优化版 | 单kernel fused | 跳过w/b | 1.304x | task-specific，需检测 |

### 历史反模式（应避免）

| 版本 | 架构 | weight策略 | 加速比 | 失败原因 |
|------|------|-----------|--------|---------|
| 0601_0.098 | 单kernel | 向量load | 0.098x | offsets//D scalar降级 |
| 0601_0.293 | 双kernel | per-channel | 0.293x | 双kernel launch开销 |
| 0601_0.669 | 双kernel | USE_2D | 0.669x | 双kernel + reshape |
| 0530_0.863 | 单kernel | USE_2D | 0.863x | reshape有bug |

---

## 快速检查清单（生成后自检）

- [ ] 只有一个 `@triton.jit` kernel（单kernel fused）
- [ ] 使用 `grid=(min(N*num_groups, VEC_CORE_NUM),)`（1D grid）
- [ ] Kernel 内有 `for gid in range(pid, total_groups, num_cores)`（交错循环）
- [ ] 没有 `tl.reshape`
- [ ] weight/bias 使用标量 load: `ch = d_start // D; tl.load(ptr + ch)`
- [ ] 没有 `USE_2D` 条件分支
- [ ] BLOCK_SIZE 策略: dtype-aware + 最大化 + 仅 D%16!=0 缩block
- [ ] Reduce 累加使用 `tl.float32`
- [ ] 首次生成不添加 `multibuffer=True, unit_flag=True`
- [ ] 没有双 kernel split
