from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple_ints(x: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return tuple(int(v) for v in x)


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Union[int, Sequence[int]] = (128, 128),
    ) -> None:
        super().__init__()

        hidden = _as_tuple_ints(hidden_dims)
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            prev = h

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


if __name__ == "__main__":
    net = QNetwork(input_dim=5, n_actions=6)  
    print(net)
