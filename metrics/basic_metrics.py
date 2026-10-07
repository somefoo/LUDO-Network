import torch
import wandb
from torchmetrics import JaccardIndex

@torch.no_grad()
def compute_and_log_accuracy(model, loader, device):
    """ Computes and logs the accuracy of the model on the given dataset. """
    model.eval()
    correct_nodes = total_nodes = 0
    for data in loader:
        data.to(device)
        correct_nodes += model(data)[0].argmax(dim=1).eq(data["occupancy"].y).sum().item()
        total_nodes += data["occupancy"].num_nodes

    print(f"Validation Accuracy: {correct_nodes/total_nodes:.4}")
    wandb.log({"Validation Batch Accuracy": correct_nodes / total_nodes}, commit = False)


@torch.no_grad()
def compute_and_log_intersection_over_union(model, loader, device, loss_type, print_for_each_scene=False):
    """ Computes and logs mIoU and binary IoU """

    jaccard_multi = JaccardIndex(
        task='multiclass',
        num_classes=loader.dataset.num_classes,
        average=None,
    ).to(device)
    jaccard_binary = JaccardIndex(task='binary').to(device)

    jaccard_multi_list = []
    jaccard_binary_list = []

    model.eval()

    for data in loader:
        data.to(device)
        # Compute prediction
        if loss_type == "sdf":
            prediction = (model(data)[1] < 0).long()
        else:
            prediction = model(data)[0].argmax(dim=1)


        jaccard_multi_list.append(jaccard_multi(prediction, data["occupancy"].y))


        # Transform the prediction and y into a binary mask
        prediction_binary = (prediction > 0).long()
        y_binary = (data["occupancy"].y > 0).long()

        jaccard_binary_list.append(jaccard_binary(prediction_binary, y_binary))

        if print_for_each_scene:
            object_segment_iou = jaccard_multi_list[-1][1:]
            print("IoU per object segment class of: ", data['image'].path)
            print("    ", object_segment_iou)
            print("Binary IoU of: ", data['image'].path)
            print("    ", round(jaccard_binary_list[-1].item(), 2))

    iou_per_class = torch.stack(jaccard_multi_list).mean(dim=0)
    object_segment_iou_per_class = iou_per_class[1:]
    iou_mean = object_segment_iou_per_class.mean(0)
    iou_binary = torch.stack(jaccard_binary_list).mean()

    # Print and log
    print(f"IoU per object segment class: {object_segment_iou_per_class}")
    print(f"mIoU mean      : {iou_mean.item():.4f}")
    print(f"Binary IoU     : {iou_binary.item():.4f}")

    wandb.log({
        "Test mIoU": iou_mean.item(),
        "Test Binary IoU": iou_binary.item(),
    }, commit=False)

    return {"mIoU": iou_mean.item(), "Binary IoU": iou_binary.item()}
