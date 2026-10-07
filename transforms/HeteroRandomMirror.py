import random
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform, LinearTransformation

# Based on the version from PyTorch Geometric, but uses HeteroData
class RandomMirror(BaseTransform):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def forward(self, data: HeteroData) -> HeteroData:
        # reference for device/dtype:
        ref = data['image'].pos
        device, dtype = ref.device, ref.dtype

        matrix = torch.eye(3, device=device, dtype=dtype)

        if self.axis == 0:
            matrix[0, 0] = random.choice([1.0, -1.0])
        elif self.axis == 1:
            matrix[1, 1] = random.choice([1.0, -1.0])
        else:
            matrix[2, 2] = random.choice([1.0, -1.0])

        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis})'
