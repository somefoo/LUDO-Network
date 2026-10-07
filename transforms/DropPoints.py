from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
import torch

class DropPoints(BaseTransform):
    r"""Removes points from the :obj:`data` object with a random probability
    that is at most equal to :obj:`maximum_probability`.

    Args:
        maximum_probability (float): maximum probability of an element to be dropped.
    """
    def __init__(self, maximum_probability: float = 0.5):
        super().__init__()  # good practice with BaseTransform (nn.Module-like)
        if maximum_probability < 0. or maximum_probability > 1.:
            raise ValueError(
                f"Expected probability to be in range [0, 1], but got {maximum_probability}."
            )
        self.maximum_probability = maximum_probability

    def forward(self, data: HeteroData) -> HeteroData:
        if self.maximum_probability < 0.00001:
            return data

        # sample random value between 0 and maximum_probability
        random_value = torch.rand(1, device=data['image'].pos.device) * self.maximum_probability

        number_of_points = data['image'].pos.size(0)
        num_points_to_keep = max(1, int(number_of_points * (1 - float(random_value))))

        points_to_keep = torch.randperm(number_of_points, device=data['image'].pos.device)[:num_points_to_keep]

        data['image'].pos = data['image'].pos[points_to_keep]
        data['image'].x = data['image'].x[points_to_keep]
        data['image'].y = data['image'].y[points_to_keep]

        return data
