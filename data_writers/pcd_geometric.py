import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData

# Often, objects may be larger than the volume [-1,1]^3, as the normalization is done only on the image point cloud.
# This variable increases the reconstruction volume heuristically, to better capture the object.
SCALE_FACTOR = 1.5

def float_to_packed_rgba(b):
    # Scale the float [0, 1] to [0, 255] and convert to int
    red = int(b * 255)
    green = 0
    blue = 0
    alpha = 255  # Fully opaque

    # Pack the 8-bit channels into a single 32-bit integer
    packed = (red << 24) | (green << 16) | (blue << 8) | alpha

    return packed

class PCDWriter:
    def __init__(self, device, num = 40000, positional_encoding = None, indicies_to_generate = [0]):
        if num > 200000:
            raise ValueError("num must be less than 200000")
        query_tensor = self.created_query_samples(num) * SCALE_FACTOR
        self.positional_encoding = positional_encoding
        self.indices_to_generate = indicies_to_generate

        query_data = HeteroData()
        query_data["occupancy"].pos = query_tensor

        self.query_data = query_data
        self.query_data.to(device)
        self.num = num
        self.device = device


        # A smaller cloud used to find the position of the object (performance optimization)
        self.small_num = 10000
        small_query_tensor = self.created_query_samples(self.small_num) * SCALE_FACTOR

        small_query_data = HeteroData()
        small_query_data["occupancy"].pos = small_query_tensor

        self.small_query_data = small_query_data
        self.small_query_data.to(device)


    def created_query_samples(self, num):
        return torch.rand(num, 3, dtype=torch.float32) * 2 - 1

    def _write_pcd(self, name, index, pcd_results, path='/tmp/pcds/'):
        """
        Write a point cloud to a .pcd file.

        Parameters:
        - name: The name of the file.
        - index: The index of the file.
        - pcd_results: The point cloud data.
        - path: The path to save the file.
        """

        # Prepare
        os.makedirs(path, exist_ok=True)
        # We write to a tmp file because writing is not atomic (can cause problems if copied while writing)
        random_number = random.randint(100, 999)
        tmp_file = f"/tmp/.out_{random_number}_tmp.pcd"

        number_of_points = pcd_results.shape[0]
        values_per_point = pcd_results.shape[1]

        if values_per_point > 8:
            raise ValueError("Each point can have at most 8 values, but got " + str(values_per_point))

        sizes_string = " ".join([str(4) for _ in range(values_per_point)])
        types_string = " ".join(["F" for _ in range(values_per_point)])
        count_string = " ".join(["1" for _ in range(values_per_point)])
        fields = " ".join([name for name in ["x", "y", "z", "rgb", "unc", "e1", "e2", "e3"][:values_per_point]])

        try:
            with open(tmp_file, "w") as f:
                f.write(f"# .PCD v0.7 - Point Cloud Data file format\n")
                f.write(f"VERSION 0.7\n")
                f.write(f"FIELDS {fields}\n")
                f.write(f"SIZE {sizes_string}\n")
                f.write(f"TYPE {types_string}\n")
                f.write(f"COUNT {count_string}\n")
                f.write(f"WIDTH {number_of_points}\n")
                f.write(f"HEIGHT 1\n")
                f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {number_of_points}\n")
                f.write(f"DATA ascii\n")
                for row in pcd_results:
                    f.write(" ".join([str(value) for value in row]) + "\n")

            os.rename(tmp_file, f"{path}/{name}_{str(index).zfill(5)}.pcd")
        except:
            print("Creating file: ", f"{path}/{name}_{str(index).zfill(5)}.pcd")


    def get_query_data(self, data_loader, index, small=False, add_noise=0.0, keep_fraction_of_points=1.0, silent=False):
        # Ensure index in indices_to_generate
        if index not in self.indices_to_generate:
            raise ValueError("Index not in indices_to_generate")

        data = data_loader.dataset[index]
        if not silent:
            print("Generating from: ", data["image"].path)
        fraction_of_points_in_image = int(data["image"].y.shape[0] * keep_fraction_of_points)

        number_of_points_in_image = int(data["image"].y.shape[0])
        self.query_data["image"].x = data["image"].x[0:fraction_of_points_in_image]
        self.query_data["image"].y = data["image"].y[0:fraction_of_points_in_image]
        self.query_data["image"].pos = data["image"].pos[0:fraction_of_points_in_image]
        if add_noise > 0.0:
            self.query_data["image"].pos += torch.randn_like(self.query_data["image"].pos) * add_noise
        self.query_data["image"].batch = torch.zeros(fraction_of_points_in_image, dtype=torch.int64)
        self.query_data["image"].ptr = torch.tensor([0, fraction_of_points_in_image], dtype=torch.int64)
        self.query_data.to(self.device)

        self.small_query_data["image"].x = data["image"].x
        self.small_query_data["image"].y = data["image"].y
        self.small_query_data["image"].pos = data["image"].pos
        self.small_query_data["image"].batch = torch.zeros(number_of_points_in_image, dtype=torch.int64)
        self.small_query_data["image"].ptr = torch.tensor([0, number_of_points_in_image], dtype=torch.int64)
        self.small_query_data.to(self.device)

        data_scale = data["image"].scale.item()
        data_offset = data["image"].mid_offset.cpu().numpy()[0]

        return self.query_data, self.small_query_data, data_scale, data_offset

    @torch.no_grad()
    def optimize_query_data(self, model, data_loader, index):
        model.eval()

        _, small_query_data, _, _ = self.get_query_data(data_loader, index, small=True, silent=True)

        results = model(small_query_data)[0].argmax(dim=1).int()

        # Remove all points with a value of 0
        small_query_data_inside_only = small_query_data['occupancy'].pos[results != 0]


        extreme_min = small_query_data_inside_only.min(dim=0).values * 1.10
        extreme_max = small_query_data_inside_only.max(dim=0).values * 1.10

        self.query_data["occupancy"].pos = torch.rand(self.num, 3, device=self.device) * (extreme_max - extreme_min) + extreme_min


    @torch.no_grad()
    def create_pcd(self, model, data_loader):
        model.eval()

        for index in self.indices_to_generate:
            time_start = time.perf_counter()

            self.optimize_query_data(model, data_loader, index)

            query_data, _, data_scale, data_offset = self.get_query_data(data_loader, index)

            results = model(query_data)[0].argmax(dim=1).int().cpu().numpy()

            # Append the position and the result
            pcd_cloud = np.concatenate((query_data["occupancy"].pos.cpu().numpy(), results[:, None]), axis=1)

            # Remove all points with a value of 0
            pcd_results = pcd_cloud[pcd_cloud[:,3] != 0]


            pcd_results = pcd_results.reshape(-1, 4)

            # Transform the points to the original space
            pcd_results[:, :3] = pcd_results[:, :3] / data_scale + data_offset

            self._write_pcd("out_occupancy", index, pcd_results)

            time_end = time.perf_counter()
            print(f"PCD Generation Time (s): {time_end - time_start}")

        return None


    def _normalize_quantile(self, data, lower_quantile = 0.0, upper_quantile = 1.0):
        average_value = np.mean(data)
        print(f"Global uncertainty average: {average_value}")

        lqr = np.quantile(data, lower_quantile)
        uqr = np.quantile(data, upper_quantile)

        print(f"Global uncertainty upper quantile: {uqr}")

        data = np.clip(data, lqr, uqr)
        data = (data - lqr) / (uqr - lqr)

        return data

    def _normalize_minmax(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    @torch.no_grad()
    def warmup(self, model, data_loader):
        model.eval()

        for index in self.indices_to_generate:
            self.optimize_query_data(model, data_loader, index)

            query_data, _, _, _ = self.get_query_data(data_loader, index, silent=True)

            _ = model(query_data)

    @torch.no_grad()
    def create_pcd_with_uncertainty_activation(self, model, data_loader):
        model.eval()

        for index in self.indices_to_generate:
            time_start = time.perf_counter()
            self.optimize_query_data(model, data_loader, index)

            query_data, _, data_scale, data_offset = self.get_query_data(data_loader, index)


            # Make sure to disable softmax on occupancy
            occ, _, _ = model(query_data, use_softmax_on_occupancy=False)

            #Entropy
            occ = F.softmax(occ, dim=1)
            relative_uncertainty = -torch.sum(occ * torch.log(occ + 1e-10), dim=1).cpu().numpy()
            relative_uncertainty = self._normalize_quantile(relative_uncertainty)



            # Get the class
            results = occ.argmax(dim=1).int().cpu().numpy()

            # Append the position and the, the estimated class and the uncertainty
            pcd_results = np.concatenate((query_data["occupancy"].pos.detach().cpu().numpy(), results[:, None], relative_uncertainty[:, None]), axis=1)

            pcd_results = pcd_results.reshape(-1, 5)

            # Transform the points to the original space
            pcd_results[:, :3] = pcd_results[:, :3] / data_scale + data_offset


            time_end = time.perf_counter()
            self._write_pcd("out_occupancy_with_uncertainty_activation", index, pcd_results)
            print(f"[Activation Uncertainty]: PCD Generation Time (s): {time_end - time_start}")

    @torch.no_grad()
    def create_pcd_with_uncertainty_monte_carlo(self, model, data_loader):
        model.eval()

        for index in self.indices_to_generate:
            time_start = time.perf_counter()

            self.optimize_query_data(model, data_loader, index)

            query_data, _, data_scale, data_offset = self.get_query_data(data_loader, index)

            model_count = 30

            # Monte Carlo Dropout 100 times
            occs = [
                occ
                for occ, _, _ in [
                    model(query_data, allow_dropout_during_inference=True, use_softmax_on_occupancy=False)
                    for _ in range(model_count)
                ]
            ]

            # Do a softmax to get "probabilities"
            occs = [F.softmax(occ, dim=1) for occ in occs]

            P = torch.stack(occs, dim=0)  # Shape: (model_count, number_of_points, number_of_classes)

            # Step 1: Average predictions across models
            P_hat = torch.mean(P, dim=0)  # Shape: (number_of_points, number_of_classes)

            # Step 2: Compute Predictive Entropy
            entropy = -torch.sum(P_hat * torch.log(P_hat + 1e-10), dim=1)  # Shape: (number_of_points,)

            # Use one sampled forward pass for class prediction and entropy for uncertainty.
            results = occs[0].argmax(dim=1).int().cpu().numpy()

            uncertainty = self._normalize_quantile(entropy.cpu().numpy())


            pcd_cloud = np.concatenate((query_data["occupancy"].pos.detach().cpu().numpy(), results[:, None], uncertainty[:, None]), axis=1)

            pcd_results = pcd_cloud.reshape(-1, 5)

            # Transform the points to the original space
            pcd_results[:, :3] = pcd_results[:, :3] / data_scale + data_offset


            time_end = time.perf_counter()
            self._write_pcd("out_occupancy_with_uncertainty_monte_carlo", index, pcd_results)
            print(f"[Monte Carlo Uncertainty]: PCD Generation Time (s): {time_end - time_start}")

        return None
