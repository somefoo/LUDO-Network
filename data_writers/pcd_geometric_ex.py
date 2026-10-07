import os
import random
import time

import numpy as np
import torch

from torch_geometric.nn import fps, radius

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
        number_of_points = query_tensor.shape[0]
        self.positional_encoding = positional_encoding
        self.indices_to_generate = indicies_to_generate

        query_data = HeteroData()
        query_data["occupancy"].x = None
        query_data["occupancy"].y = None
        query_data["occupancy"].pos = query_tensor
        query_data["occupancy"].batch = torch.zeros(number_of_points, dtype=torch.int64)
        query_data["occupancy"].ptr = torch.tensor([0, number_of_points], dtype=torch.int64)


        # Create list of 31 random color triplets (between 0 and 1)
        rng = np.random.default_rng(seed=3141592)
        self.possible_colors = rng.random((31, 3))
        # Make sure that the first color is black
        self.possible_colors[0] = np.array([0.0, 0.0, 0.0])

        self.query_data = HeteroData()
        if positional_encoding is not None:
            transformed = positional_encoding(query_data["occupancy"])
            self.query_data["occupancy_whole"].x = transformed.x
        else:
            transformed = query_data["occupancy"]
        self.query_data["occupancy_whole"].pos = transformed.pos
        self.query_data["occupancy_whole"].batch = transformed.batch
        self.query_data["occupancy_whole"].ptr = transformed.ptr
        self.query_data["occupancy"].ptr = transformed.ptr
        self.query_data.to(device)
        self.num = num
        self.device = device

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

        os.makedirs(path, exist_ok=True)
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

    @torch.no_grad()
    def create_pcd_with_patch_explainability(self, model, data_loader, modes = ["occupancy", "sdf"], radius_factor = 0.5):
        model.eval()

        for index in self.indices_to_generate:
            time_start = time.perf_counter()
            data = data_loader.dataset[index]
            data.to(self.device)

            number_of_points_in_image = data["image"].y.shape[0]

            if radius_factor < 0.0 or radius_factor > 1.0:
                print(f"Radius factor is invalid (< 0.0 or > 1.0): {radius_factor}")
                return exit(1)

            self.query_data["image"].x = data["image"].x
            self.query_data["image"].y = data["image"].y
            self.query_data["image"].pos = data["image"].pos
            self.query_data["image"].batch = torch.zeros(number_of_points_in_image, dtype=torch.int64)
            self.query_data["image"].ptr = torch.tensor([0, number_of_points_in_image], dtype=torch.int64)
            print("Generating from: ", data["image"].path)

            # Compute ratio, such that 100 points are selected
            ratio = 1.0

            self.query_data.to(self.device)
            idx = fps(self.query_data["image"].pos, self.query_data["image"].batch, ratio=ratio, random_start=False)
            # Compute the distance between the closest points from idx
            distance = torch.cdist(self.query_data["image"].pos[idx], self.query_data["image"].pos[idx], p=2)
            distance = distance.flatten().sort().values[len(idx):] # Sort and skip the first element (distance to itself)

            # Get the 10% percentile of the distances
            distance = torch.kthvalue(distance, int(radius_factor * distance.shape[0]), dim=0).values

            # Patch radius
            print(f"Patch radius: {distance.item()}")

            rad = radius(self.query_data["image"].pos,
                         self.query_data["image"].pos[idx],
                         distance, self.query_data["image"].batch,
                         self.query_data["image"].batch[idx],
                         max_num_neighbors=number_of_points_in_image)



            # find the positions of the point cloud points in the volume model
            position_indices = self.query_data["image"].pos / SCALE_FACTOR
            position_indices = (position_indices + 1) / 2 # convert to interval [0, 1]
            position_indices *= torch.tensor(self.num, device=self.device) # scale up resolution
            position_indices = torch.floor(position_indices).int() # group into buckets
            position_indices = position_indices.cpu().numpy()

            # compute the scale
            data_scale = data["image"].scale.item()
            data_offset = data["image"].mid_offset.cpu().numpy()[0]

            error_caused_by_index_to_skip = []
            reference_occ = None
            for index_to_skip in range(-1, idx.shape[0]):
                # Get all indices except for rad[1][rad[0] == 0]
                valid_indices = torch.arange(0, number_of_points_in_image, device=self.device)
                remove_indices = rad[1][rad[0] == index_to_skip]
                valid_indices = valid_indices[~torch.isin(valid_indices, remove_indices)]
                # Send to device

                self.query_data["image"].x = data["image"].x[valid_indices]
                self.query_data["image"].y = data["image"].y[valid_indices]
                self.query_data["image"].pos = data["image"].pos[valid_indices]
                self.query_data["image"].batch = torch.zeros(valid_indices.shape[0], dtype=torch.int64)
                self.query_data["image"].ptr = torch.tensor([0, valid_indices.shape[0]], dtype=torch.int64)
                self.query_data.to(self.device)


                query_size = 300*300
                number_of_points = self.query_data["occupancy_whole"].batch.shape[0]
                for i in range(0, number_of_points, query_size):
                    start = i
                    end = min(i + query_size, number_of_points)
                    self.query_data["occupancy"].ptr[0] = 0
                    self.query_data["occupancy"].ptr[1] = end - start
                    self.query_data["occupancy"].batch = self.query_data["occupancy_whole"].batch[start:end]
                    self.query_data["occupancy"].pos = self.query_data["occupancy_whole"].pos[start:end]
                    if self.positional_encoding is not None:
                        self.query_data["occupancy"].x = self.query_data["occupancy_whole"].x[start:end]


                    if "occupancy" in modes:
                        occ, _, _ = model(self.query_data)
                        results = occ.argmax(dim=1).int().cpu().numpy()

                        if index_to_skip == -1:
                            reference_occ = occ
                        else:
                            # Compute the difference between the reference and the current occupancy
                            diff = (occ.argmax(dim=1) != reference_occ.argmax(dim=1)).abs().sum()
                            results = diff.cpu().numpy()
                            error_caused_by_index_to_skip.append(results.mean())

            # Interpolate the error caused by the index to skip to the original points
            original_points = data["image"].pos
            error_points = data["image"].pos[idx]
            errors = torch.tensor(error_caused_by_index_to_skip, device=self.device)

            # Add dimension to original points and error points
            # Original: x,y,z,counter,error_accumulator
            original_points = torch.cat((original_points,
                                         torch.zeros_like(original_points[:, 0])[:, None],
                                         torch.zeros_like(original_points[:, 0])[:, None]
                                         ),
                                        dim=1)
            # Error: x,y,z,add_value, error
            error_points = torch.cat((error_points, torch.ones_like(error_points[:, 0])[:, None], errors[:, None]), dim=1)

            # Compute the distance between the original points and the error points
            for index_to_skip in range(0, idx.shape[0]):
                affected_point_indices = rad[1][rad[0] == index_to_skip]
                # Set all last values to error_points[index_to_skip][-1]
                original_points[affected_point_indices, -2:] += error_points[index_to_skip][-2:]

            # Average the overlapping points
            original_points[original_points[:, -2] != 0, -1] /= original_points[original_points[:, -2] != 0, -2]

            pcd_results = original_points.cpu().numpy()

            # Transform the points to the original space
            pcd_results[:, :3] = pcd_results[:, :3] / data_scale + data_offset

            self._write_pcd("ex_out", index, pcd_results)

            time_end = time.perf_counter()
            print(f"PCD Generation Time (s): {time_end - time_start}")



        return None
