import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


@triton.jit
def rotary_mul_kernel(
    input_ptr, r1_ptr, r2_ptr, output_ptr,
    B, H, S, D,
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,
    stride_r1_b, stride_r1_h, stride_r1_s, stride_r1_d,
    stride_r2_b, stride_r2_h, stride_r2_s, stride_r2_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    num_cores: tl.constexpr,
    TILE_S: tl.constexpr,
    HALF_D: tl.constexpr,
    IS_HALF: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_s_tiles = S // TILE_S
    num_blocks = B * H * num_s_tiles

    blocks_per_core = num_blocks // num_cores
    remainder = num_blocks % num_cores
    block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
    block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)

    s_offs = tl.arange(0, TILE_S)[:, None]
    d_offs = tl.arange(0, HALF_D)[None, :]

    for block_idx in range(block_start, block_end):
        tmp = block_idx // num_s_tiles
        s_tile = block_idx - tmp * num_s_tiles
        tmp2 = tmp // H
        h = tmp - tmp2 * H
        b = tmp2
        s_start = s_tile * TILE_S

        in_base = b * stride_in_b + h * stride_in_h + s_start * stride_in_s
        out_base = b * stride_out_b + h * stride_out_h + s_start * stride_out_s

        r1_base = b * stride_r1_b + h * stride_r1_h + s_start * stride_r1_s
        r2_base = b * stride_r2_b + h * stride_r2_h + s_start * stride_r2_s

        if IS_HALF:
            idx1 = in_base + s_offs * stride_in_s + d_offs * stride_in_d
            idx2 = in_base + s_offs * stride_in_s + (d_offs + HALF_D) * stride_in_d

            x1 = tl.load(input_ptr + idx1)
            x2 = tl.load(input_ptr + idx2)

            r1_idx1 = r1_base + s_offs * stride_r1_s + d_offs * stride_r1_d
            r1_idx2 = r1_base + s_offs * stride_r1_s + (d_offs + HALF_D) * stride_r1_d
            r2_idx1 = r2_base + s_offs * stride_r2_s + d_offs * stride_r2_d
            r2_idx2 = r2_base + s_offs * stride_r2_s + (d_offs + HALF_D) * stride_r2_d

            r1_1 = tl.load(r1_ptr + r1_idx1)
            r1_2 = tl.load(r1_ptr + r1_idx2)
            r2_1 = tl.load(r2_ptr + r2_idx1)
            r2_2 = tl.load(r2_ptr + r2_idx2)

            if IS_FP16 or IS_BF16:
                x1 = x1.to(tl.float32)
                x2 = x2.to(tl.float32)
                r1_1 = r1_1.to(tl.float32)
                r1_2 = r1_2.to(tl.float32)
                r2_1 = r2_1.to(tl.float32)
                r2_2 = r2_2.to(tl.float32)

            out1 = x1 * r1_1 - x2 * r2_1
            out2 = x2 * r1_2 + x1 * r2_2

            if IS_FP16:
                out1 = out1.to(tl.float16)
                out2 = out2.to(tl.float16)
            elif IS_BF16:
                out1 = out1.to(tl.bfloat16)
                out2 = out2.to(tl.bfloat16)

            tl.store(output_ptr + out_base + s_offs * stride_out_s + d_offs * stride_out_d, out1)
            tl.store(output_ptr + out_base + s_offs * stride_out_s + (d_offs + HALF_D) * stride_out_d, out2)
        else:
            even_offs = d_offs * 2
            odd_offs = d_offs * 2 + 1

            idx_even = in_base + s_offs * stride_in_s + even_offs * stride_in_d
            idx_odd = in_base + s_offs * stride_in_s + odd_offs * stride_in_d

            x1 = tl.load(input_ptr + idx_even)
            x2 = tl.load(input_ptr + idx_odd)

            r1_even = tl.load(r1_ptr + r1_base + s_offs * stride_r1_s + even_offs * stride_r1_d)
            r1_odd = tl.load(r1_ptr + r1_base + s_offs * stride_r1_s + odd_offs * stride_r1_d)
            r2_even = tl.load(r2_ptr + r2_base + s_offs * stride_r2_s + even_offs * stride_r2_d)
            r2_odd = tl.load(r2_ptr + r2_base + s_offs * stride_r2_s + odd_offs * stride_r2_d)

            if IS_FP16 or IS_BF16:
                x1 = x1.to(tl.float32)
                x2 = x2.to(tl.float32)
                r1_even = r1_even.to(tl.float32)
                r1_odd = r1_odd.to(tl.float32)
                r2_even = r2_even.to(tl.float32)
                r2_odd = r2_odd.to(tl.float32)

            out_even = x1 * r1_even - x2 * r2_even
            out_odd = x2 * r1_odd + x1 * r2_odd

            if IS_FP16:
                out_even = out_even.to(tl.float16)
                out_odd = out_odd.to(tl.float16)
            elif IS_BF16:
                out_even = out_even.to(tl.bfloat16)
                out_odd = out_odd.to(tl.bfloat16)

            tl.store(output_ptr + out_base + s_offs * stride_out_s + even_offs * stride_out_d, out_even)
            tl.store(output_ptr + out_base + s_offs * stride_out_s + odd_offs * stride_out_d, out_odd)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        except Exception:
            self.VEC_CORE_NUM = 48

    def forward(self, input: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = 'half') -> torch.Tensor:
        input_c = input if input.is_contiguous() else input.contiguous()
        B, H, S, D = input_c.shape
        output = torch.empty_like(input_c)

        def _bc_stride(t, input_shape):
            return tuple(0 if t.shape[i] == 1 and input_shape[i] > 1 else t.stride(i) for i in range(4))

        r1_stride = _bc_stride(r1, input_c.shape)
        r2_stride = _bc_stride(r2, input_c.shape)

        IS_HALF = (rotary_mode == 'half')
        IS_FP16 = (input_c.dtype == torch.float16)
        IS_BF16 = (input_c.dtype == torch.bfloat16)
        TILE_S = 16

        assert S % TILE_S == 0, f"S={S} must be divisible by TILE_S={TILE_S}"

        num_s_tiles = S // TILE_S
        num_blocks = B * H * num_s_tiles
        grid_size = num_blocks if num_blocks < self.VEC_CORE_NUM else self.VEC_CORE_NUM
        grid = (grid_size,)

        HALF_D = D // 2
        rotary_mul_kernel[grid](
            input_c, r1, r2, output,
            B, H, S, D,
            input_c.stride(0), input_c.stride(1), input_c.stride(2), input_c.stride(3),
            r1_stride[0], r1_stride[1], r1_stride[2], r1_stride[3],
            r2_stride[0], r2_stride[1], r2_stride[2], r2_stride[3],
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            num_cores=self.VEC_CORE_NUM,
            TILE_S=TILE_S,
            HALF_D=HALF_D,
            IS_HALF=IS_HALF,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
        )
        return output
