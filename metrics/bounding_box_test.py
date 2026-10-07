import os
import time
from typing import Tuple

import numpy as np
import torch

from torch_geometric.data import HeteroData

from functools import partial

# Often, objects may be larger than the volume [-1,1]^3, as the normalization is done only on the image point cloud.
# This variable increases the reconstruction volume heuristically, to better capture the object.
SCALE_FACTOR = 1.3

class BoundingBoxTest:
    def __init__(self, num: Tuple[int,int,int], device, positional_encoding = None):
        query_tensor = self.create_indexed_volume_aabb(num, [-1,-1,-1], [1,1,1]) * SCALE_FACTOR
        number_of_points = query_tensor.shape[0]
        self.positional_encoding = positional_encoding

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

    @torch.no_grad()
    def test_bounding_box(self, model, data_loader, modes = ["occupancy", "sdf"], fingerprints = None):
        time_start = time.perf_counter()
        model.eval()

        iou_list = []
        scale_list = []
        extra_volume_list = []
        distance_error_list = []

        for index in range(len(data_loader.dataset)):
            data = data_loader.dataset[index]

            occ_results_accumulator = []
            sdf_results_accumulator = []
            if fingerprints is not None:
                fingerprint = fingerprints[data["image"].unique_index]

            number_of_points_in_image = data["image"].y.shape[0]
            self.query_data["image"].x = data["image"].x
            self.query_data["image"].y = data["image"].y
            self.query_data["image"].pos = data["image"].pos
            self.query_data["image"].batch = torch.zeros(number_of_points_in_image, dtype=torch.int64)
            self.query_data["image"].ptr = torch.tensor([0, number_of_points_in_image], dtype=torch.int64)
            self.query_data["bounding_boxes_original_space"] = data["bounding_boxes_original_space"]
            print("Generated from: ", data["image"].path)
            self.query_data.to(self.device)

            # find the positions of the point cloud points in the volume model
            position_indices = self.query_data["image"].pos / SCALE_FACTOR
            position_indices = (position_indices + 1) / 2 # convert to interval [0, 1]
            position_indices *= torch.tensor(self.num, device=self.device) # scale up resolution
            position_indices = torch.floor(position_indices).int() # group into buckets
            position_indices = position_indices.cpu().numpy()

            # Track min/max positions for each semantic label bucket.
            mins = np.full((1000, 3), np.inf)
            maxs = np.full((1000, 3), -np.inf)

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
                    results = model(self.query_data)[0].argmax(dim=1).int()
                    occ_results_accumulator.append(results.cpu().numpy() * 1000)
                    # Find all unique non-zero values
                    unique = np.unique(results.cpu().numpy())

                    for value in unique:
                        if value == 0:
                            continue
                        positions = self.query_data["occupancy"].pos[results == value]
                        # Get x, y, z min and max
                        min_x = positions[:,0].min().item()
                        min_y = positions[:,1].min().item()
                        min_z = positions[:,2].min().item()
                        max_x = positions[:,0].max().item()
                        max_y = positions[:,1].max().item()
                        max_z = positions[:,2].max().item()


                        mins[value] = np.minimum(mins[value], [min_x, min_y, min_z])
                        maxs[value] = np.maximum(maxs[value], [max_x, max_y, max_z])


                if "sdf" in modes:
                    results = (model(self.query_data)[1] < 0).int()
                    sdf_results_accumulator.append(results.cpu().numpy() * 1000)



                # For all pos in batch, find the min and max for each label


            # Create .csv file /tmp/estimated.csv, in the format: label, min_x, min_y, min_z, max_x, max_y, max_z
            with open(f"/tmp/estimated_{str(index).zfill(4)}.csv", "w") as file:
                # IoU and scale lists


                local_iou_list = []
                local_scale_list = []
                local_extra_volume_list = []
                local_distance_error_list = []

                # Print all mins and maxs
                for i in range(0, 1000):
                    if np.isfinite(mins[i]).any():
                        bounding_boxes = self.query_data["bounding_boxes_original_space"]

                        # Get the bounding box where the label is i
                        min_bounding_box = bounding_boxes['min'][bounding_boxes['id'] == i]
                        max_bounding_box = bounding_boxes['max'][bounding_boxes['id'] == i]
                        min_bounding_box = min_bounding_box.cpu().numpy()[0]
                        max_bounding_box = max_bounding_box.cpu().numpy()[0]


                        # Scale and offset the min and max
                        scale = data["image"].scale.item()
                        offset = data["image"].mid_offset.cpu().numpy()[0]

                        estimated_min = (mins[i] / scale) + offset
                        estimated_max = (maxs[i] / scale) + offset
                        min_bounding_box = (min_bounding_box / scale) + offset
                        max_bounding_box = (max_bounding_box / scale) + offset

                        file.write(f"{i},{min_bounding_box[0]},{min_bounding_box[1]},{min_bounding_box[2]},{max_bounding_box[0]},{max_bounding_box[1]},{max_bounding_box[2]}\n")

                        def compute_iou(estimated, estimated_min, estimated_max, min_bounding_box, max_bounding_box):
                            # Compute IoU
                            intersection = np.maximum(0, np.minimum(estimated_max, max_bounding_box) - np.maximum(estimated_min, min_bounding_box))
                            intersection = intersection.prod()

                            union = (estimated_max - estimated_min).prod() + (max_bounding_box - min_bounding_box).prod() - intersection
                            union = max(union, 1e-6)
                            iou = intersection / union
                            return iou


                        def compute_scale():
                            # Compute how much the estimated bounding box needs to be scaled up to enclose the ground truth bounding box
                            scale = 1.0
                            center_position = (estimated_min + estimated_max) / 2
                            center_position_reference = (min_bounding_box + max_bounding_box) / 2
                            distance_error = np.linalg.norm(center_position - center_position_reference)
                            while True:
                                # Scale the estimated min and max (around the center of the bounding box), then move it back to the original position
                                scaled_est_min = (estimated_min - center_position) * scale + center_position
                                scaled_est_max = (estimated_max - center_position) * scale + center_position
                                # Check if the estimated bounding box encloses the ground truth bounding box
                                if np.all(scaled_est_min <= min_bounding_box) and np.all(scaled_est_max >= max_bounding_box):
                                    # Compute how much more volume the estimated bounding box has compared to the ground truth bounding box
                                    volume_reference = ((estimated_max - estimated_min) / 10).prod()
                                    volume_scaled = ((scaled_est_max - scaled_est_min) / 10).prod()

                                    extra_volume = volume_scaled - volume_reference #As units^3 (most liekely decimeters^3)

                                    return scale, extra_volume, distance_error
                                scale += 0.05
                                if scale > 20:
                                    print(f"Label {i}: Scale: {scale} is insane")
                                    exit()


                        iou = compute_iou(estimated_min, estimated_min, estimated_max, min_bounding_box, max_bounding_box)
                        scale, extra_volume, distance_error = compute_scale()
                        local_iou_list.append((i, iou))
                        local_scale_list.append((i, scale))
                        local_extra_volume_list.append((i, extra_volume))
                        local_distance_error_list.append((i, distance_error))

                        # Print local distance error and index
                        print(f"GREPME {index} Label {i}: Distance Error: {distance_error}")


                # Print average iou and scale
                print(f"Average IoU for {index}: {np.mean([iou for _, iou in local_iou_list])}")
                print(f"Average Scale for {index}: {np.mean([scale for _, scale in local_scale_list])}")
                print(f"Average Extra Volume for {index}: {np.mean([extra_volume for _, extra_volume in local_extra_volume_list])}")
                print(f"Average Distance Error for {index}: {np.mean([distance_error for _, distance_error in local_distance_error_list])}")

                iou_list.append(local_iou_list)
                scale_list.append(local_scale_list)
                extra_volume_list.append(local_extra_volume_list)
                distance_error_list.append(local_distance_error_list)


        time_end = time.perf_counter()
        print(f"Bounding Box Test Time (s): {time_end - time_start}")


        # Aggregate per-label statistics (mean/std/median/min/max and missing labels).
        # For each index, collect all IoU and scale values.
        label_dict_iou = {}
        for index in range(len(iou_list)):
            for label, iou in iou_list[index]:
                if label not in label_dict_iou:
                    label_dict_iou[label] = []
                label_dict_iou[label].append(iou)


        label_dict_scale = {}
        for index in range(len(scale_list)):
            for label, scale in scale_list[index]:
                if label not in label_dict_scale:
                    label_dict_scale[label] = []
                label_dict_scale[label].append(scale)

        label_dict_extra_volume = {}
        for index in range(len(extra_volume_list)):
            for label, extra_volume in extra_volume_list[index]:
                if label not in label_dict_extra_volume:
                    label_dict_extra_volume[label] = []
                label_dict_extra_volume[label].append(extra_volume)

        label_dict_distance_error = {}
        for index in range(len(distance_error_list)):
            for label, distance_error in distance_error_list[index]:
                if label not in label_dict_distance_error:
                    label_dict_distance_error[label] = []
                label_dict_distance_error[label].append(distance_error)

        # Now we have all IoU and scale values for each label
        # Lets calculate the statistics

        per_label_statistics = []

        for label in label_dict_iou:
            ious = label_dict_iou[label]
            scales = label_dict_scale[label]
            extra_volumes = label_dict_extra_volume[label]
            print(f"Label {label}:")
            print(f" Average IoU: {np.mean(ious)}")
            print(f" Standard Deviation IoU: {np.std(ious)}")
            print(f" Median IoU: {np.median(ious)}")
            print(f" Min IoU: {np.min(ious)}")
            print(f" Max IoU: {np.max(ious)}")

            print(f" Average Scale: {np.mean(scales)}")
            print(f" Standard Deviation Scale: {np.std(scales)}")
            print(f" Median Scale: {np.median(scales)}")
            print(f" Min Scale: {np.min(scales)}")
            print(f" Max Scale: {np.max(scales)}")
            print(f" Average Extra Volume: {np.mean(extra_volumes)}")
            print(f" Standard Deviation Extra Volume: {np.std(extra_volumes)}")
            print(f" Median Extra Volume: {np.median(extra_volumes)}")
            print(f" Min Extra Volume: {np.min(extra_volumes)}")
            print(f" Max Extra Volume: {np.max(extra_volumes)}")

            print(f" Worst IoU Index (warn, may be wrong): {np.argmin(ious)}")
            print(f" Worst Scale Index (warn, may be wrong): {np.argmin(scales)}")

            # Create dict
            per_label_values = {
                "average_iou": np.mean(ious),
                "std_iou": np.std(ious),
                "median_iou": np.median(ious),
                "min_iou": np.min(ious),
                "max_iou": np.max(ious),
                "average_scale": np.mean(scales),
                "std_scale": np.std(scales),
                "median_scale": np.median(scales),
                "min_scale": np.min(scales),
                "max_scale": np.max(scales),
                "average_extra_volume": np.mean(extra_volumes),
                "std_extra_volume": np.std(extra_volumes),
                "median_extra_volume": np.median(extra_volumes),
                "min_extra_volume": np.min(extra_volumes),
                "max_extra_volume": np.max(extra_volumes),
                "average_distance_error": np.mean(label_dict_distance_error[label]),
                "std_distance_error": np.std(label_dict_distance_error[label]),
                "median_distance_error": np.median(label_dict_distance_error[label]),
                "min_distance_error": np.min(label_dict_distance_error[label]),
                "max_distance_error": np.max(label_dict_distance_error[label]),
            }

            per_label_statistics.append(per_label_values)

        all_ious = []
        all_scales = []
        all_extra_volumes = []
        all_distance_errors = []
        for label in label_dict_iou:
            all_ious.extend(label_dict_iou[label])
            all_scales.extend(label_dict_scale[label])
            all_extra_volumes.extend(label_dict_extra_volume[label])
            all_distance_errors.extend(label_dict_distance_error[label])

        print(f"Average IoU for all: {np.mean(all_ious)}")
        print(f"Standard Deviation IoU for all: {np.std(all_ious)}")
        print(f"Median IoU for all: {np.median(all_ious)}")
        print(f"Min IoU for all: {np.min(all_ious)}")
        print(f"Max IoU for all: {np.max(all_ious)}")
        print(f"Average Scale for all: {np.mean(all_scales)}")
        print(f"Standard Deviation Scale for all: {np.std(all_scales)}")
        print(f"Median Scale for all: {np.median(all_scales)}")
        print(f"Min Scale for all: {np.min(all_scales)}")
        print(f"Max Scale for all: {np.max(all_scales)}")
        print(f"Average Extra Volume for all: {np.mean(all_extra_volumes)}")
        print(f"Standard Deviation Extra Volume for all: {np.std(all_extra_volumes)}")
        print(f"Median Extra Volume for all: {np.median(all_extra_volumes)}")
        print(f"Min Extra Volume for all: {np.min(all_extra_volumes)}")
        print(f"Max Extra Volume for all: {np.max(all_extra_volumes)}")

        all_statistics = {
            "average_iou": np.mean(all_ious),
            "std_iou": np.std(all_ious),
            "median_iou": np.median(all_ious),
            "min_iou": np.min(all_ious),
            "max_iou": np.max(all_ious),
            "average_scale": np.mean(all_scales),
            "std_scale": np.std(all_scales),
            "median_scale": np.median(all_scales),
            "min_scale": np.min(all_scales),
            "max_scale": np.max(all_scales),
            "average_extra_volume": np.mean(all_extra_volumes),
            "std_extra_volume": np.std(all_extra_volumes),
            "median_extra_volume": np.median(all_extra_volumes),
            "min_extra_volume": np.min(all_extra_volumes),
            "max_extra_volume": np.max(all_extra_volumes),
            "average_distance_error": np.mean(all_distance_errors),
            "std_distance_error": np.std(all_distance_errors),
            "median_distance_error": np.median(all_distance_errors),
            "min_distance_error": np.min(all_distance_errors),
            "max_distance_error": np.max(all_distance_errors),
        }


        # Check if there are labels which have less values
        print("TODO: Check if there are labels which have less values")

        # Never print using scientific notation
        np.set_printoptions(suppress=True)

        # Create file /tmp/scale_estimate.csv
        # It should contain the estimated scale for each label in the format: label, average_scale, std_scale, median_scale, min_scale, max_scale
        # It also has a header: ValueNumber,Mean,Std, Median, Min, Max
        with open(f"/tmp/scale_estimate.csv", "w") as file:
            file.write("ValueNumber, Mean, Std, Median, Min, Max\n")
            for label in label_dict_scale:
                scales = label_dict_scale[label]
                mean = np.mean(scales)
                std = np.std(scales)
                median = np.median(scales)
                min_value = np.min(scales)
                max_value = np.max(scales)
                file.write(f"{label},{mean},{std},{median},{min_value},{max_value}\n")

        # Same for IoU
        with open(f"/tmp/iou_estimate.csv", "w") as file:
            file.write("ValueNumber, Mean, Std, Median, Min, Max\n")
            for label in label_dict_iou:
                ious = label_dict_iou[label]
                mean = np.mean(ious)
                std = np.std(ious)
                median = np.median(ious)
                min_value = np.min(ious)
                max_value = np.max(ious)
                file.write(f"{label},{mean},{std},{median},{min_value},{max_value}\n")

        # Same for Extra Volume
        with open(f"/tmp/extra_volume_estimate.csv", "w") as file:
            file.write("ValueNumber, Mean, Std, Median, Min, Max\n")
            for label in label_dict_extra_volume:
                extra_volumes = label_dict_extra_volume[label]
                mean = np.mean(extra_volumes)
                std = np.std(extra_volumes)
                median = np.median(extra_volumes)
                min_value = np.min(extra_volumes)
                max_value = np.max(extra_volumes)
                file.write(f"{label},{mean},{std},{median},{min_value},{max_value}\n")

        # Same for Distance Error
        with open(f"/tmp/distance_error_estimate.csv", "w") as file:
            file.write("ValueNumber, Mean, Std, Median, Min, Max\n")
            for label in label_dict_distance_error:
                distance_errors = label_dict_distance_error[label]
                mean = np.mean(distance_errors)
                std = np.std(distance_errors)
                median = np.median(distance_errors)
                min_value = np.min(distance_errors)
                max_value = np.max(distance_errors)
                file.write(f"{label},{mean},{std},{median},{min_value},{max_value}\n")




        return iou_list, scale_list, per_label_statistics, all_statistics
