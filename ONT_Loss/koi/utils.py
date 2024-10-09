import torch
from koi._runtime import ffi
from collections import namedtuple


Buffer = namedtuple("Buffer", "data ptr")


def void_ptr(x):
    """
    Return a void * for given Tensor `x`.
    """
    return ffi.cast("void *", 0 if x is None else x.data_ptr())


def empty(size, device, dtype=torch.float16):
    """
    Create an empty Tensor of size `size` on device `device`.
    """
    x = torch.empty(size, dtype=dtype, device=device)
    return Buffer(x, void_ptr(x))


def zeros(size, device, dtype=torch.float16):
    """
    Create an zeros Tensor of size `size` on device `device`.
    """
    x = torch.zeros(size, dtype=dtype, device=device)
    return Buffer(x, void_ptr(x))


def quantize_tensor(tensor, levels=256, dim=0, z=None):
    """
    Quantize a tensor to int8, returning the per-channel scales and the quantized tensor.

    If z is provided, the floating point range used for quantisation is clipped to
    z times the standard deviation from the mean for each channel.
    """
    fp_range = tensor.abs().amax(dim)

    if z is not None:
        fp_mean = tensor.mean(axis=0)
        fp_std = tensor.std(axis=0)
        fp_range_z = abs(fp_mean) + fp_std * z
        fp_range = torch.min(fp_range, fp_range_z)

    quant_scale = (levels / 2) / fp_range
    quant_max = (levels / 2) - 1
    tensor_quantized = (tensor * quant_scale).round().clip(-quant_max, quant_max)
    return quant_scale.float(), tensor_quantized.char()


def show_diff_result(
    label, ref_out, out_bfr, mean_limit, max_limit, assert_limits=False
):
    diff = torch.abs(out_bfr.to(torch.float32) - ref_out.to(torch.float32))
    diff_mean = diff.mean().item()
    diff_max = diff.max().item()
    is_good = ("❌", "🟢")[diff_mean <= mean_limit and diff_max <= max_limit]
    print(
        f"{is_good} Compare {label} to reference: diff mean {diff_mean}, max {diff_max}"
    )
    if assert_limits:
        assert diff_mean <= mean_limit
        assert diff_max <= max_limit
