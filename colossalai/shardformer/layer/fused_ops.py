from typing import Optional

from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from colossalai.kernel.jit import bias_dropout_add_fused_train, bias_gelu_impl


class Bias_Dropout_Add(Module):
    def __init__(
        self,
        p: float = 0.5,
        bias: Optional[Parameter] = None,
        residual: Optional[Parameter] = None,
    ) -> None:
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.bias = bias
        self.residual = residual

    def forward(self, input_: Tensor) -> Tensor:
        return bias_dropout_add_fused_train(input_, self.bias, self.residual, self.p)


class Bias_Gelu(Module):
    def __init__(
        self,
        bias: Optional[Parameter] = None,
    ) -> None:
        super().__init__()

        self.bias = bias

    def forward(self, input_: Tensor) -> Tensor:
        return bias_gelu_impl(input_, self.bias)
