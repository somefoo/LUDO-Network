import glob
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, HeteroData
from .read_pcd import read_pcd_array


class CustomDataset(InMemoryDataset):
    """ A custom class to load point clouds """

    @staticmethod
    def _print_cache_banner(usage: str, path: str) -> None:
        banner = "=" * 72
        print(banner)
        print(" USING PROCESSED DATASET CACHE ")
        print(f" Split : {usage}")
        print(f" File  : {path}")
        print(" Raw files were not reprocessed for this dataset load.")
        print(banner)

    def __init__(
        self,
        root,
        usage="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        positional_encoding=None,
        force_reprocess=False,
    ):
        self.positional_encoding = positional_encoding
        processed_paths = [Path(root) / "processed" / name for name in self.processed_file_names]
        cache_available_before_load = all(path.exists() for path in processed_paths) and not force_reprocess
        super().__init__(
            str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reprocess,
        )

        usage_to_path = {
            "train": self.processed_paths[0],
            "validation": self.processed_paths[1],
            "test": self.processed_paths[2],
        }
        try:
            path = usage_to_path[usage]
        except KeyError as exc:
            raise ValueError("Invalid usage. (valid: train, validation, test)") from exc

        # Load pre-processed dataset if available
        if cache_available_before_load:
            self._print_cache_banner(usage, path)
        self.data, self.slices = torch.load(path, weights_only=False)
        # Compute number of classes in dataset
        self.class_values = torch.unique(self._data["occupancy"].y)

    @property
    def raw_file_names(self) -> list:
        """ If this raw file exists, the data is available """
        return ['00000_occupancy.pcd']

    @property
    def processed_file_names(self) -> list:
        """ If all of these files exist, the data is processed """
        return ['training.pt', 'validation.pt', 'test.pt']

    @property
    def num_classes(self) -> int:
        """ Returns the number of unique class id's in the dataset """
        return self.class_values.shape[0]

    def process_files(self, is_test: bool) -> None:
        """ Processes and stores files """
        # Find all files in test or training data folders
        dataset_dir = Path(self.root)
        if is_test:
            occupancy_data_list = sorted(glob.glob(str(dataset_dir / "raw_test/*occupancy.npz")))
            occupancy_data_list += sorted(glob.glob(str(dataset_dir / "raw_test/*occupancy.pcd")))
            image_data_list = sorted(glob.glob(str(dataset_dir / "raw_test/*image.pcd")))
            bounding_boxes_data_list = sorted(glob.glob(str(dataset_dir / "raw_test/*bounding_boxes.csv")))
        else:
            occupancy_data_list = sorted(glob.glob(str(dataset_dir / "raw_train/*occupancy.pcd")))
            image_data_list = sorted(glob.glob(str(dataset_dir / "raw_train/*image.pcd")))
            bounding_boxes_data_list = sorted(glob.glob(str(dataset_dir / "raw_train/*bounding_boxes.csv")))

        # Sanity check
        if len(occupancy_data_list) != len(image_data_list) or len(occupancy_data_list) != len(bounding_boxes_data_list):
            raise ValueError(
                "Error, number of occupancy files "
                f"({len(occupancy_data_list)}) and image files ({len(image_data_list)}) do not match. "
                f"Or number of bounding box files ({len(bounding_boxes_data_list)}) do not match."
            )

        data_list = list(zip(occupancy_data_list, image_data_list, bounding_boxes_data_list))

        # Prepare data
        occupancy_list = []
        distance_field_list = []
        image_list = []
        bounding_box_list = []
        for pair in data_list:
            # If pair[0] is a .npz file, use numpy to load it
            # Else use the custom read_pcd_array function
            if pair[0].endswith('.npz'):
                # Test data is a .npz file
                occupancy_data = torch.from_numpy(np.load(pair[0])['occ'])
            else:
                occupancy_data = read_pcd_array(pair[0])

            # Image data is always a .pcd file
            image_data = read_pcd_array(pair[1])

            # First three components are x, y, z
            occupancy_position = occupancy_data[:, :3].type(torch.float)

            # Fourth component is the label
            occupancy_label = occupancy_data[:, 3].type(torch.long)

            # Next three components are the vector which points to the closest surface
            sign = torch.clamp(occupancy_label, 0, 1) * -2 + 1
            distance_field_direction = occupancy_data[:, 4:7].type(torch.float)
            distance_field_distance = distance_field_direction.norm(dim=1) * sign

            if image_data.shape[0] == 0:
                raise ValueError(f"Image data is empty for {pair[1]}, perhaps no objects are visible to the camera?")
            # First three components are x, y, z
            image_position = image_data[:, :3]
            # Fourth component is the label
            image_label = image_data[:, 3].type(torch.long)

            # x stores optional per-point features. We currently fill image.x with ones.
            occupancy = Data(pos=occupancy_position, x=None, y=occupancy_label, path=pair[0])
            # Store distance-field direction in `pos` so geometric transforms apply automatically.
            distance_field = Data(pos=distance_field_direction, distance=distance_field_distance)
            image = Data(pos=image_position, x=torch.ones_like(image_position), y=image_label, path=pair[1])

            occupancy_list.append(occupancy)
            distance_field_list.append(distance_field)
            image_list.append(image)

            # Bounding box data
            # The bounding box is stored in the format [name_string, label, x_min, y_min, z_min, x_max, y_max, z_max]
            with open(pair[2], 'r') as f:
                bounding_box = f.readlines()
                bounding_box = [list(map(float, line.strip().split(',')[1:])) for line in bounding_box]
                bounding_box = torch.tensor(bounding_box)
                bounding_box = Data(id=bounding_box[:, 0], min=bounding_box[:, 1:4], max=bounding_box[:, 4:7])
                bounding_box_list.append(bounding_box)

        if self.pre_filter is not None:
            # Remove points if a filter was given
            occupancy_list = [data for data in occupancy_list if self.pre_filter(data)]
            image_list = [data for data in image_list if self.pre_filter(data)]
            distance_field_list = [data for data in distance_field_list if self.pre_filter(data)]
            bounding_box_list = [data for data in bounding_box_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            # Scale both the image and the occupancy point clouds
            # such that the image is centered at the origin and has a diameter of 1
            transformed = [self.pre_transform(*pair) for pair in zip(image_list, occupancy_list, distance_field_list, bounding_box_list)]
            image_list = [p[0] for p in transformed]
            occupancy_list = [p[1] for p in transformed]
            distance_field_list = [p[2] for p in transformed]
            bounding_box_list = [p[3] for p in transformed]
            # Do not transform the bounding box data, as it needs to stay AABB

        if self.positional_encoding is not None:
            # Optional pre-processing for older pipelines that expect encoded occupancy inputs.
            occupancy_list = [self.positional_encoding(p) for p in occupancy_list]

        # Move everything into a hetero data object
        hetero_list = []
        unique_index = 0
        for img, occ, df, aabb in zip(image_list, occupancy_list, distance_field_list, bounding_box_list):
            hetero_data = HeteroData()
            hetero_data["occupancy"].x = occ.x
            hetero_data["occupancy"].y = occ.y
            hetero_data["occupancy"].pos = occ.pos
            hetero_data["occupancy"].scale = occ.scale
            hetero_data["occupancy"].mid_offset = occ.mid_offset
            hetero_data["occupancy"].path = occ.path

            hetero_data["image"].x = img.x
            hetero_data["image"].y = img.y
            hetero_data["image"].pos = img.pos
            hetero_data["image"].scale = img.scale
            hetero_data["image"].mid_offset = img.mid_offset
            hetero_data["image"].path = img.path
            hetero_data["image"].unique_index = unique_index

            hetero_data["distance_field"].pos = df.pos
            hetero_data["distance_field"].distance = df.distance

            hetero_data["bounding_boxes_original_space"].id = aabb.id
            hetero_data["bounding_boxes_original_space"].min = aabb.min
            hetero_data["bounding_boxes_original_space"].max = aabb.max

            unique_index += 1

            hetero_list.append(hetero_data)

        if hetero_list == []:
            return

        # Store data as a pre-processed dataset
        if is_test:
            torch.save(self.collate(hetero_list), self.processed_paths[2])
        else:
            if len(hetero_list) < 2:
                raise ValueError(
                    "Need at least 2 training samples in raw_train to create training/validation splits, "
                    f"but found {len(hetero_list)}."
                )
            # Split into training and validation set
            training_size = int(0.90 * len(hetero_list))
            hetero_training_list = hetero_list[:training_size]
            hetero_validation_list = hetero_list[training_size:]

            torch.save(self.collate(hetero_training_list), self.processed_paths[0])
            torch.save(self.collate(hetero_validation_list), self.processed_paths[1])

    def process(self):
        # Process and store training/validation and test data
        self.process_files(is_test=True)
        self.process_files(is_test=False)
