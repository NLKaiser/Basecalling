import torch
from koi._runtime import lib
from itertools import groupby
from koi.utils import void_ptr, empty, quantize_tensor
from torch.nn import Module, ModuleList, Parameter, Sequential


class Permute(Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class LSTM(Module):
    def __init__(self, layer_size, reverse, quantized=False):
        super().__init__()
        if layer_size not in [96, 128] or (layer_size == 128 and not quantized):
            raise ValueError(
                f"Unsupported LSTM: layer_size {layer_size}, quantized {quantized}"
            )
        self.w_ih = Parameter(
            torch.empty(layer_size * 4, layer_size), requires_grad=False
        )
        self.w_hh = Parameter(
            torch.empty(layer_size * 4, layer_size), requires_grad=False
        )
        self.b_ih = Parameter(torch.empty(layer_size * 4), requires_grad=False)
        self.b_hh = Parameter(torch.empty(layer_size * 4), requires_grad=False)
        self.quantization_scale = None
        self.quantized = quantized
        self._rearranged = False
        self.direction = -1 if reverse else 1

    def _rearrange_weights_buffer(self):
        if not self._rearranged:
            self.w_ih = Parameter(self.w_ih.T.contiguous(), requires_grad=False)
            self.w_hh = Parameter(self.w_hh.T.contiguous(), requires_grad=False)
            self._rearranged = True

    def _quantize_weights(self):
        if self.quantized and self.quantization_scale is None:
            scale, w_hh = quantize_tensor(self.w_hh)
            self.quantization_scale = scale
            self.w_hh = Parameter(w_hh, requires_grad=False)

    def forward(self, input_buffer, out_buffer):
        self._rearrange_weights_buffer()
        self._quantize_weights()
        assert input_buffer.data.size(2) == self.w_ih.size(0)
        args = [
            input_buffer.data.size(0),
            input_buffer.data.size(1),
            input_buffer.data.size(2),
            self.direction,
            void_ptr(input_buffer.data @ self.w_ih),
            void_ptr(self.w_hh),
            void_ptr(self.b_ih),
            void_ptr(self.quantization_scale if self.quantized else None),
            out_buffer.ptr,
        ]
        res = lib.host_small_lstm(*args)
        if res != 0:
            raise ValueError(f"Small LSTM kernel execution failed: {res}, {args}")


class LSTMStack(Module):
    def __init__(
        self,
        directions,
        layer_size,
        batch_size,
        chunk_size,
        device="cuda",
        quantized=False,
    ):
        super().__init__()

        self.buffers = (
            empty((batch_size, chunk_size, layer_size), device),
            empty((batch_size, chunk_size, layer_size), device),
        )

        self.layers = ModuleList(
            [LSTM(layer_size, direction, quantized) for direction in directions]
        )

    def forward(self, data):
        if data.dtype != torch.float16:
            raise TypeError("Expected fp16 but received %s" % data.dtype)
        buff1, buff2 = self.buffers
        buff1.data[: data.shape[0], :, :] = data
        for layer in self.layers:
            layer(buff1, buff2)
            buff1, buff2 = buff2, buff1
        return buff1.data


def update_graph(model, batchsize=640, chunksize=720, quantize=True):
    """
    Replace a stack of PyTorch LSTMs with a koi LSTMStack.
    """
    for name, layers in groupby(model, lambda x: x.__class__.__name__):
        if name == "LSTM":
            features = next(layers).rnn.input_size

    if not quantize and features == 128:
        decode_only = True
    elif features not in [96, 128]:
        decode_only = True
    else:
        decode_only = False

    # The standard PyTorch Bonito CRF has a single Permute(2, 0, 1)
    # between the convolutions layers and the recurrent layers.
    #
    #  pytorch   NCT -> Permute(2, 0, 1) -> TNC
    #
    # KOI LSTMs expects NTC and beam_search expects NTC
    #
    #  koi lstm    NCT -> Permute(0, 2, 1) -> NTC -> Permute(1, 0, 2) -> TNC
    #  koi decode  NCT -> Permute(2, 0, 1) -> TNC -> Permute(1, 0, 2) -> NTC
    #  both        NCT -> Permute(0, 2, 1) -> NTC -> Identity()       -> NTC

    modules = []

    for name, layers in groupby(model, lambda x: x.__class__.__name__):
        if name == "LSTM":
            if decode_only:
                modules.extend(
                    [
                        Permute([2, 0, 1]),  # NCT -> TNC for PyTorch LSTMs
                        *layers,  #                PyTorch LSTMs
                        Permute([1, 0, 2]),  # TNC -> NTC for Koi decoding
                    ]
                )
            else:
                modules.extend(
                    [
                        Permute([0, 2, 1]),  # NCT -> NTC for Koi LSTMStack
                        LSTMStack(
                            [lstm.reverse for lstm in layers],
                            features,
                            batchsize,
                            chunksize,
                            quantized=quantize,
                        )
                        # Identity()       # Still NTC for Koi decoding
                    ]
                )
        elif name == "Permute":
            continue
        elif name == "LinearCRFEncoder":
            layer = next(layers)
            layer.expand_blanks = False
            modules.append(layer)
        else:
            modules.extend(layers)

    return Sequential(*modules)
