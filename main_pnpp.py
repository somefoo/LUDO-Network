import sys
from pathlib import Path
import time

from utilities.config_loader import load_config
from deformation_network import Net

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from metrics.basic_metrics import compute_and_log_accuracy, compute_and_log_intersection_over_union
from metrics.bounding_box_test import BoundingBoxTest

import wandb

from transforms.NormalizeScale import NormalizeScale
from transforms.DropPoints import DropPoints
from data_loaders.data_loader_geometric import CustomDataset
from data_writers.volume_block_geometric import NRRDWriter
from data_writers.pcd_geometric import PCDWriter
from data_writers.pcd_geometric_ex import PCDWriter as PCDWriterEx
from transforms.HeteroRandomRotate import RandomRotate
from transforms.HeteroRandomMirror import RandomMirror
from torch_geometric.transforms import Compose


def create_model(config, device):
    return Net(
        config['number_of_classes'],
        [1, 3],
        config['positional_encoding_min_inclusive'],
        config['positional_encoding_max_inclusive'],
        config['hidden_layers_in_mlp'],
        config['dropout'],
        device,
    ).to(device)


def create_augmentation_transform(config):
    augmentations = []

    if config['augment_maximum_drop_point_probability'] > 0:
        print("Augmenting point drop")
        augmentations.append(DropPoints(config['augment_maximum_drop_point_probability']))
    if config['augment_rotation']:
        print("Augmenting rotation")
        augmentation_rotation_degrees = config['augment_rotation_degrees']
        augmentations.extend([
            RandomRotate(augmentation_rotation_degrees, axis=0),
            RandomRotate(augmentation_rotation_degrees, axis=1),
            RandomRotate(augmentation_rotation_degrees, axis=2),
        ])
    if config['augment_mirror']:
        print("Augmenting mirror")
        augmentations.extend([RandomMirror(axis=0), RandomMirror(axis=1), RandomMirror(axis=2)])

    return Compose(augmentations)


def get_lr(optimizer):
    """ Returns the current learning rate. """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(loader, model, optimizer, scheduler, config, device):
    """ Train the model on the given dataset. """
    model.train()

    # Standard training loop
    total_loss = correct_nodes = total_nodes = batch_count = 0
    total_occupancy_loss = total_distance_loss = total_direction_loss = 0
    max_dist = config["clamp_distance"]
    for data in loader:
        batch_count += 1
        data = data.to(device)
        optimizer.zero_grad()
        occ, distance, direction = model(data)
        distance = distance.view(-1)

        occupancy_loss = F.nll_loss(occ, data["occupancy"].y)
        distance = torch.clamp(distance, -max_dist, max_dist)
        target_distance = torch.clamp(data["distance_field"].distance, -max_dist, max_dist)
        distance_loss = F.l1_loss(distance, target_distance) * 100
        direction_loss = -F.cosine_similarity(direction, data["distance_field"].pos, dim=1).mean()

        if config['loss_type'] == "occ":
            loss = occupancy_loss
        elif config['loss_type'] == "sdf":
            loss = distance_loss
        elif config['loss_type'] == "occ+sdf":
            loss = occupancy_loss + distance_loss
        elif config['loss_type'] == "occ+sdf+cos":
            loss = occupancy_loss + distance_loss + direction_loss
        else:
            raise ValueError(f"Invalid loss type {config['loss_type']}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_occupancy_loss += occupancy_loss.item()
        total_distance_loss += distance_loss.item()
        total_direction_loss += direction_loss.item()
        correct_nodes += occ.argmax(dim=1).eq(data["occupancy"].y).sum().item()
        total_nodes += data["occupancy"].num_nodes

    training_info = {
        "Train Batch Loss": total_loss / batch_count,
        "Train Batch Occupancy Loss": total_occupancy_loss / batch_count,
        "Train Batch Distance Loss": total_distance_loss / batch_count,
        "Train Batch Direction Loss": total_direction_loss / batch_count,
        "Train Batch Accuracy": correct_nodes / total_nodes,
        "Learning Rate": get_lr(optimizer)
    }

    wandb.log(training_info, commit=False)
    print(", ".join([
        f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in training_info.items()
    ]))

    if config['learning_rate_decay']:
        scheduler.step()


def initialize_runtime(config):
    wandb_mode = None if config['run_type'] == 'train' else 'disabled'
    run = wandb.init(
        entity="semantic-scene-completion",
        project="deformable-reconstruction",
        group=config['scene'],
        config=config,
        mode=wandb_mode,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config, device)
    return run, device, model


def validate_paths(config):
    dataset_path = config['dataset_prefix_path'] / config['scene']
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Error, could not find dataset at {dataset_path}")

    if config['log_model_path'].exists():
        if not config['log_model_path'].is_dir():
            raise FileExistsError(f"File exists at model store path {config['log_model_path']}, but is not a directory")
    else:
        config['log_model_path'].mkdir(parents=True)

    return dataset_path


def build_data_loaders(dataset_path, config, pre_transform, augmentation):
    train_dataset = CustomDataset(
        dataset_path,
        "train",
        transform=augmentation,
        pre_transform=pre_transform,
        force_reprocess=config['force_reprocess'],
    )
    # Intentionally uses the same augmentation as training for validation.
    validation_dataset = CustomDataset(dataset_path, "validation", transform=augmentation, pre_transform=pre_transform)
    test_dataset = CustomDataset(dataset_path, "test", pre_transform=pre_transform)

    assert test_dataset.transform is None

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=6)
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)
    return train_loader, validation_loader, test_loader


def run_single_mode(model, dataset_path, config, device, pre_transform):
    start_time = time.time()
    single_dataset = CustomDataset(dataset_path, "test", pre_transform=pre_transform)
    single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False, num_workers=6)

    pcd_writer = PCDWriter(device=device, num=config['occupancy_pcd_resolution'], indicies_to_generate=[0])
    pcd_writer_ex = PCDWriterEx(device=device, num=config['explainability_pcd_resolution'], indicies_to_generate=[0])

    model.load_state_dict(torch.load(config['log_model_path'] / "model.pth", weights_only=False))

    pcd_writer.warmup(model, single_loader)
    pcd_writer.create_pcd_with_uncertainty_activation(model, single_loader)
    pcd_writer.create_pcd_with_uncertainty_monte_carlo(model, single_loader)
    if 'explainability_radius_factor' not in config:
        pcd_writer_ex.create_pcd_with_patch_explainability(model, single_loader)
    else:
        pcd_writer_ex.create_pcd_with_patch_explainability(
            model,
            single_loader,
            radius_factor=config['explainability_radius_factor'],
        )

    print("Inference time: ", time.time() - start_time)


def run_nrrd_mode(model, test_loader, config, device):
    nrrd_range = config['nrrd_range']
    nrrd_indices = [i for i in range(nrrd_range[0], nrrd_range[1] + 1)]

    nrrd_writer = NRRDWriter(config['nrrd_resolution'], device=device, indicies_to_generate=nrrd_indices)
    model.load_state_dict(torch.load(config['log_model_path'] / "model.pth", weights_only=False))
    nrrd_writer.create_nrrd(model, test_loader)


def run_test_mode(model, test_loader, config, device):
    nrrd_writer = NRRDWriter(config['nrrd_resolution_test'], device=device)
    print("TODO: Introduce new config for bounding box test resolution")
    bounding_box_tester = BoundingBoxTest(config['nrrd_resolution_test'], device=device)

    model.load_state_dict(torch.load(config['log_model_path'] / "model.pth"))

    compute_and_log_intersection_over_union(model, test_loader, device, config['loss_type'])

    nrrd_writer.create_nrrd(model, test_loader)
    bounding_box_tester.test_bounding_box(model, test_loader)


def run_train_mode(model, train_loader, validation_loader, test_loader, config, device):
    nrrd_writer = NRRDWriter(config['nrrd_resolution_train'], device=device)

    wandb.watch(model)
    wandb.save(f"./{config['log_model_path']}/*.pth", policy="end")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    lambdaf = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaf)

    for epoch in range(0, config["epochs"]):
        print(f"{epoch = } ")
        train(train_loader, model, optimizer, scheduler, config, device)
        if (epoch + 1) % config["log_accuracy_period"] == 0:
            compute_and_log_accuracy(model, validation_loader, device)
            img = nrrd_writer.create_nrrd(model, test_loader)
            wandb.log({"Visual from NRRD": wandb.Image(img, mode='RGB')}, commit=False)
        if (epoch + 1) % config["log_iou_period"] == 0:
            compute_and_log_intersection_over_union(model, test_loader, device, config['loss_type'])
        if (epoch + 1) % config["log_model_period"] == 0:
            backup_model_path = Path(f"{config['log_model_path'] / 'model'}_{epoch:0>6d}.backup.pth")
            torch.save(model.state_dict(), backup_model_path)
        wandb.log({}, commit=True)

    torch.save(model.state_dict(), config['log_model_path'] / 'model.pth')


def run_by_mode(config, model, device, dataset_path, pre_transform):
    if config['run_type'] == "single":
        run_single_mode(model, dataset_path, config, device, pre_transform)
        return

    augmentation = create_augmentation_transform(config)
    train_loader, validation_loader, test_loader = build_data_loaders(
        dataset_path,
        config,
        pre_transform,
        augmentation,
    )

    run_handlers = {
        "nrrd": lambda: run_nrrd_mode(model, test_loader, config, device),
        "test": lambda: run_test_mode(model, test_loader, config, device),
        "train": lambda: run_train_mode(model, train_loader, validation_loader, test_loader, config, device),
    }
    try:
        run_handlers[config['run_type']]()
    except KeyError as exc:
        raise ValueError("Error, run_type must be either 'train' or 'test' or 'nrrd' or 'single'") from exc


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: expected parameter <configuration file>")

    config = load_config(sys.argv[1])

    run, device, model = initialize_runtime(config)
    try:
        dataset_path = validate_paths(config)
        pre_transform = NormalizeScale()
        run_by_mode(config, model, device, dataset_path, pre_transform)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
