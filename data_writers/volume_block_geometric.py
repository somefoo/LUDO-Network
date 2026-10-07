import os
import random
import time
from typing import Tuple

import numpy as np
import torch

from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt

from functools import partial

# Often, objects may be larger than the volume [-1,1]^3, as the normalization is done only on the image point cloud.
# This variable increases the reconstruction volume heuristically, to better capture the object.
SCALE_FACTOR = 1.3

class NRRDWriter:
    def __init__(self, num: Tuple[int,int,int], device, positional_encoding = None, indicies_to_generate = [0], create_only_one_randomly_from_indicies = False):
        query_tensor = self.create_indexed_volume_aabb(num, [-1,-1,-1], [1,1,1]) * SCALE_FACTOR
        number_of_points = query_tensor.shape[0]
        self.positional_encoding = positional_encoding
        # Keep misspelled argument name for backward compatibility with existing configs.
        self.indices_to_generate = indicies_to_generate
        self.create_only_one_randomly_from_indicies = create_only_one_randomly_from_indicies

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

    def create_indexed_volume_aabb(self, num, aabb_min, aabb_max):
        x = np.linspace(aabb_min[0], aabb_max[0], num=num[0], dtype=np.float32)
        y = np.linspace(aabb_min[1], aabb_max[1], num=num[1], dtype=np.float32)
        z = np.linspace(aabb_min[2], aabb_max[2], num=num[2], dtype=np.float32)

        space = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
        return torch.from_numpy(space)

    def display_averaged_image(self, results):
        # Average the results over last dimension
        results = np.mean(results, axis=2)
        # Normalize the results to lie within [0,1]
        results = (results - results.min()) / (results.max() - results.min())
        # Resize the results image to be 3 times larger using numpy
        results = np.repeat(np.repeat(results, 6, axis=0), 6, axis=1)


        # Open window to display the results as an image
        import cv2
        cv2.imshow("results", results)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()

    def create_depth_image(self, results):
        results = np.flip(results, axis=0)
        results = np.flip(results, axis=2)
        axis = 0
        images = []

        for axis in range(3):
            depth = (results>0.0).argmax(axis=axis)

            if axis == 0:
                # Find the value at each depth
                id_at_depth = results[depth, np.arange(depth.shape[0])[:,None], np.arange(depth.shape[1])]
            elif axis == 1:
                # Find the value at each depth
                id_at_depth = results[np.arange(depth.shape[0])[:,None], depth, np.arange(depth.shape[1])]
            else: # axis == 2:
                # Find the value at each depth
                id_at_depth = results[np.arange(depth.shape[0])[:,None], np.arange(depth.shape[1]), depth]

            # Normalize the results to lie within [0,1]
            normalized_depth = np.clip(1-depth**0.9 / depth.max(), 0.0, 1.0)

            # Create a color image from the depth image
            color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=float)
            color = self.possible_colors[id_at_depth % 31] * normalized_depth[:,:,None]

            # Compute screen space normals
            dx = np.gradient(depth, 1, axis=1)
            dy = np.gradient(depth, 1, axis=0)
            normals = np.stack([-dx, -dy, np.ones_like(dx)], axis=2)
            normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)

            list_of_lights = [
                    (np.array([0.0,0.0, -1.0]), np.array([1.0, 1.0, 1.0])),
                    ]

            # Light contribution from each light
            light_contribution = np.zeros_like(color)
            for l in list_of_lights:
                light_direction = l[0] / np.linalg.norm(l[0])
                light_color = l[1]
                dot = np.sum(normals * (-light_direction), axis=2)
                dot = np.clip(dot, 0.0, 1.0)
                light_contribution = light_contribution + dot[:,:,None] * light_color[None,None,:]



            # Combine the color and light contribution
            color = color * (light_contribution * (color > 0.0))

            # Make background white
            color[depth == 0] = 1.0

            # Increase saturation
            color = np.clip(color * 2.0, 0.0, 1.0)

            # Wanted size = 1000x1000

            # Rotate image by 90 degrees
            color = np.rot90(color, 3)

            # Append image to list of images
            images.append(color)

        # Combine all images in images list into a single image
        final_image = np.concatenate(images, axis=1)

        # Open window to display the results as an image
        return final_image

    @torch.no_grad()
    def create_nrrd(self, model, data_loader, modes = ["occupancy", "sdf"], fingerprints = None):
        model.eval()

        indices_to_generate = self.indices_to_generate
        if self.create_only_one_randomly_from_indicies:
            indices_to_generate = [random.choice(self.indices_to_generate)]

        for index in indices_to_generate:
            time_start = time.perf_counter()
            occ_results_accumulator = []
            sdf_results_accumulator = []
            data = data_loader.dataset[index]
            if fingerprints is not None:
                fingerprint = fingerprints[data["image"].unique_index]

            number_of_points_in_image = data["image"].y.shape[0]
            self.query_data["image"].x = data["image"].x
            self.query_data["image"].y = data["image"].y
            self.query_data["image"].pos = data["image"].pos
            self.query_data["image"].batch = torch.zeros(number_of_points_in_image, dtype=torch.int64)
            self.query_data["image"].ptr = torch.tensor([0, number_of_points_in_image], dtype=torch.int64)
            print("Generated from: ", data["image"].path)
            self.query_data.to(self.device)

            # find the positions of the point cloud points in the volume model
            position_indices = self.query_data["image"].pos / SCALE_FACTOR
            position_indices = (position_indices + 1) / 2 # convert to interval [0, 1]
            position_indices *= torch.tensor(self.num, device=self.device) # scale up resolution
            position_indices = torch.floor(position_indices).int() # group into buckets
            position_indices = position_indices.cpu().numpy()

            # compute the scale of each voxel
            voxel_scale = np.divide(2.0, self.num) / data["image"].scale.item() * SCALE_FACTOR
            # The space origin in a NRRD file defines the location
            # of the first voxel, we need to offset to define the center
            first_voxel_offset = voxel_scale * self.num / 2 - (voxel_scale/2)
            offset = data["image"].mid_offset.cpu().numpy()[0] - first_voxel_offset

            os.makedirs("/tmp/nrrds", exist_ok=True)
            output_file = f"/tmp/nrrds/out_{str(index).zfill(5)}.nrrd"
            output_image_file = f"/tmp/nrrds/out_{str(index).zfill(5)}.png"

            # Get a random number between 100000 and 999999
            random_number = random.randint(100000, 999999)

            tmp_file = f"/tmp/nrrds/.out_{random_number}_tmp.nrrd"

            with open(tmp_file, "w") as f:
                try:
                    f.write("NRRD0004\n")
                    f.write("# NRRD created by Anonymized - Occupancy Learning\n")
                    f.write("type: short\n")
                    f.write("dimension: 3\n")
                    f.write("space: left-posterior-superior\n")
                    f.write(f"sizes: {self.num[0]} {self.num[1]} {self.num[2]}\n")
                    # Swap X and Y to ensure we have the same coordinate system as the PCD file
                    f.write(f"space directions: (0,{voxel_scale[1]},0) ({voxel_scale[0]},0,0) (0,0,{voxel_scale[2]})\n")
                    f.write("kinds: domain domain domain\n")
                    f.write("endian: little\n")
                    f.write("encoding: raw\n")
                    f.write(f"space origin: ({offset[0]},{offset[1]},{offset[2]})\n\n")
                except:
                    print("Error writing header")

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


                    if fingerprints is not None:
                        model = partial(model, fingerprints=fingerprint)

                    if "occupancy" in modes:
                        results = model(self.query_data)[0].argmax(dim=1).int().cpu().numpy()
                        occ_results_accumulator.append(results * 1000)
                    if "sdf" in modes:
                        results = (model(self.query_data)[1] < 0).int().cpu().numpy()
                        sdf_results_accumulator.append(results * 1000)


                if "occupancy" in modes:
                    occ_results = np.concatenate(occ_results_accumulator, axis=0)
                    occ_results = occ_results.reshape(self.num)
                    # render the original point cloud in the volume model
                    for p in position_indices:
                        occ_results[p[2],p[0],p[1]] = 10000
                    self.occ_image = self.create_depth_image(occ_results)

                    # Create renders from the volume model
                if "sdf" in modes:
                    sdf_results = np.concatenate(sdf_results_accumulator, axis=0)
                    sdf_results = sdf_results.reshape(self.num)
                    # render the original point cloud in the volume model
                    for p in position_indices:
                        sdf_results[p[2],p[0],p[1]] = 10000
                    self.sdf_image = self.create_depth_image(sdf_results * 15)

                if "occupancy" in modes and "sdf" in modes:
                    self.image = np.concatenate([self.occ_image, self.sdf_image], axis=0)
                    # Default to occupancy
                    results = occ_results.flatten()
                elif "occupancy" in modes:
                    self.image = self.occ_image
                    results = occ_results.flatten()
                elif "sdf" in modes:
                    self.image = self.sdf_image
                    results = sdf_results.flatten()

                try:
                    results.astype(np.int16).tofile(f)
                except:
                    print("Error writing data")



            try:
                os.replace(tmp_file, output_file)
                print(f"NRRD written to {output_file}")
            except OSError as exc:
                print(f"Error finalizing NRRD file {output_file}: {exc}")

            time_end = time.perf_counter()
            print(f"NRRD Generation Time (s): {time_end - time_start}")

            # Write image to /tmp/nrrds/out.png using matplotlib
            plt.imsave(output_image_file, self.image)
            print(f"Preview PNG written to {output_image_file}")


        return self.image
