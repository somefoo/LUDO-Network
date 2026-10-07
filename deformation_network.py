import torch
import math
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # set random_start to False to guarantee deterministic output for
        # inference
        if self.training:
            idx = fps(pos, batch, ratio=self.ratio, random_start=True)
        else:
            idx = fps(pos, batch, ratio=self.ratio, random_start=False)

        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

# Unpooling
class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class Net(torch.nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        number_of_auxiliary: list[int],
        positional_encoding_min_inclusive: int,
        positional_encoding_max_inclusive: int,
        hidden_layers_in_mlp: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()

        print('TODO: the number of classes includes the auxiliary learning tasks, separate this!')

        output_size = number_of_classes + sum(number_of_auxiliary)

        self.number_of_classes = number_of_classes
        self.number_of_auxiliary = number_of_auxiliary
        self.positional_encoding_min_inclusive = positional_encoding_min_inclusive
        self.positional_encoding_max_inclusive = positional_encoding_max_inclusive
        self.positional_encoding_range = abs(positional_encoding_max_inclusive - positional_encoding_min_inclusive)

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 128, 512, 1024]))

        self.frequencies = math.pi * torch.pow(
            2,
            torch.arange(
                self.positional_encoding_min_inclusive,
                self.positional_encoding_max_inclusive,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.cfrequencies = 1j * self.frequencies

        node_count = 512

        layer_structure_1 = [1024 + (self.positional_encoding_range * 2 * 3)] + [node_count] * (hidden_layers_in_mlp // 2)
        layer_structure_2 = [node_count + 1024 + (self.positional_encoding_range * 2 * 3)] + [node_count] * (hidden_layers_in_mlp // 2 - 1) + [output_size]

        dropout_1 = [0.0] + [dropout] * (len(layer_structure_1) - 2)
        dropout_2 = [0.0] + [dropout] * (len(layer_structure_2) - 3) + [0.0]

        self.mlp_1 = MLP(layer_structure_1, dropout=dropout_1, norm="batch_norm", plain_last=False)
        self.mlp_2 = MLP(layer_structure_2, dropout=dropout_2, norm="batch_norm", plain_last=True)


    def forward(self, data, use_softmax_on_occupancy=True, allow_dropout_during_inference=False):
        image = data["image"]
        occupancy = data["occupancy"]

        # Used for monte carlo dropout
        if allow_dropout_during_inference:
            self.mlp_1.training = True
            self.mlp_2.training = True

        # Positional encoding
        pos = torch.view_as_real(torch.exp(self.cfrequencies * occupancy.pos.unsqueeze(-1))).reshape(-1, 2*3*self.positional_encoding_range)

        # image.x should be all ones
        sa0_out = (image.x, image.pos, image.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        x, _, _ = sa3_out
        # stretch out leading dimension by repeating each point cloud
        # fingerprint once for each occupancy query point from its scene
        if 'ptr' in occupancy:
            counts = occupancy.ptr[1:] - occupancy.ptr[:-1]
            if counts.numel() > 1 and not torch.all(counts == counts[0]):
                raise ValueError(
                    "Batched occupancy point counts differ across scenes, but the current "
                    "fingerprint expansion assumes equal counts. "
                    "Got counts per scene: "
                    f"{counts.detach().cpu().tolist()}"
                )
            f = torch.repeat_interleave(x, counts[0], dim=0)
        else:
            f = x.expand(occupancy.pos.shape[0], -1)
        # concatenate point cloud fingerprints with query points
        x = torch.cat((f, pos), dim=1)
        x = self.mlp_1(x)
        x = torch.cat((x, f, pos), dim=1)
        x = self.mlp_2(x)

        if use_softmax_on_occupancy:
            occupancy = x[:, :self.number_of_classes].log_softmax(dim=-1)
        else:
            occupancy = x[:, :self.number_of_classes]
        auxiliary = x[:, self.number_of_classes:]

        auxiliary_outputs = []
        for i in range(len(self.number_of_auxiliary)):
            auxiliary_outputs.append(auxiliary[:, :self.number_of_auxiliary[i]])
            auxiliary = auxiliary[:, self.number_of_auxiliary[i]:]

        return occupancy, *auxiliary_outputs
