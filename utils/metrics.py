import torch
import numpy as np

def compute_accuracy(predictions, labels):
    """
    Compute the accuracy of predictions against the ground truth labels.
    The input can be soft label or hard label.

    Args:
        predictions (torch.Tensor): Predictions with shape (num_classes, depth, height, width).
        labels (torch.Tensor): Ground truth labels with shape (num_classes, depth, height, width).

    Returns:
        float: Accuracy as a ratio of correctly predicted voxels to total voxels.
    """
    
    # Check input dimensions
    if predictions.ndim != 4 or labels.ndim != 4:
        raise ValueError("Both predictions and labels must have 4 dimensions: (num_classes, depth, height, width).")
    
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    num_classes, depth, height, width = predictions.shape
    total_voxels = depth * height * width
    
    # Convert predictions and labels to the class with the highest score
    predicted_classes = torch.argmax(predictions, dim=0)
    true_classes = torch.argmax(labels, dim=0)
    
    # Calculate accuracy
    accuracy = (predicted_classes == true_classes).sum().item() / total_voxels
    return accuracy


def compute_mean_iou(predictions, labels, ignore_background=True):
    """
    Compute the Mean Intersection over Union (mIoU) for predictions against the ground truth labels.

    Args:
        predictions (torch.Tensor): Predictions with shape (num_classes, depth, height, width).
        labels (torch.Tensor): Ground truth labels with shape (num_classes, depth, height, width).

    Returns:
        float: Mean IoU across all classes.
    """
    
    # Check input dimensions
    if predictions.ndim != 4 or labels.ndim != 4:
        raise ValueError("Both predictions and labels must have 4 dimensions: (num_classes, depth, height, width).")
    
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    num_classes = predictions.shape[0]

    # Convert predictions and labels to class indices
    predicted_classes = torch.argmax(predictions, dim=0)  # shape: (depth, height, width)
    true_classes = torch.argmax(labels, dim=0)            # shape: (depth, height, width)

    # Initialize IoU for each class
    iou_per_class = torch.zeros(num_classes, device=predictions.device)

    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue

        # Create boolean masks for predicted and true classes
        true_mask = (true_classes == cls)
        pred_mask = (predicted_classes == cls)

        # Calculate intersection and union
        intersection = (true_mask & pred_mask).sum().item()
        union = true_mask.sum().item() + pred_mask.sum().item() - intersection

        if union > 0:
            iou_per_class[cls] = intersection / union

    # Calculate mean IoU
    if ignore_background:
        iou_per_class = iou_per_class[1:]  # Exclude background class
    mean_iou = iou_per_class.mean().item()  # Convert tensor to scalar

    return mean_iou

def compute_mean_dice(predictions, labels, ignore_background=True):
    """
    Compute the Mean Dice Coefficient for predictions against the ground truth labels.

    Args:
        predictions (torch.Tensor): Predictions with shape (num_classes, depth, height, width).
        labels (torch.Tensor): Ground truth labels with shape (num_classes, depth, height, width).

    Returns:
        float: Mean Dice Coefficient across all classes.
    """
    
    # Check input dimensions
    if predictions.ndim != 4 or labels.ndim != 4:
        raise ValueError("Both predictions and labels must have 4 dimensions: (num_classes, depth, height, width).")
    
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    num_classes = predictions.shape[0]

    # Convert predictions and labels to class indices
    predicted_classes = torch.argmax(predictions, dim=0)  # shape: (depth, height, width)
    true_classes = torch.argmax(labels, dim=0)            # shape: (depth, height, width)

    # Initialize Dice for each class
    dice_per_class = torch.zeros(num_classes, device=predictions.device)

    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue
        # Create boolean masks for predicted and true classes
        true_mask = (true_classes == cls)
        pred_mask = (predicted_classes == cls)

        # Calculate intersection and sum of areas
        intersection = (true_mask & pred_mask).sum().item()
        total_area = true_mask.sum().item() + pred_mask.sum().item()

        # Calculate Dice coefficient for the class
        if total_area > 0:
            dice_per_class[cls] = 2 * intersection / total_area

    # Calculate mean Dice coefficient
    if ignore_background:
        dice_per_class = dice_per_class[1:]
    mean_dice = dice_per_class.mean().item()  # Convert tensor to scalar

    return mean_dice

def compute_mean_mcc(predictions, labels, ignore_background=True):
    """
    Compute the Mean Matthews Correlation Coefficient (MCC) for predictions against the ground truth labels.

    Args:
        predictions (torch.Tensor): Predictions with shape (num_classes, depth, height, width).
        labels (torch.Tensor): Ground truth labels with shape (num_classes, depth, height, width).

    Returns:
        float: Mean MCC across all classes.
    """
    
    # Check input dimensions
    if predictions.ndim != 4 or labels.ndim != 4:
        raise ValueError("Both predictions and labels must have 4 dimensions: (num_classes, depth, height, width).")
    
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    num_classes = predictions.shape[0]

    # Convert predictions and labels to class indices
    predicted_classes = torch.argmax(predictions, dim=0)  # shape: (depth, height, width)
    true_classes = torch.argmax(labels, dim=0)            # shape: (depth, height, width)

    # Initialize MCC for each class
    mcc_per_class = torch.zeros(num_classes, device=predictions.device)

    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue
        # Create boolean masks for predicted and true classes
        true_mask = (true_classes == cls)
        pred_mask = (predicted_classes == cls)

        # Calculate true positives, false positives, true negatives, and false negatives
        tp = torch.sum(true_mask & pred_mask).float()
        tn = torch.sum(~true_mask & ~pred_mask).float()
        fp = torch.sum(~true_mask & pred_mask).float()
        fn = torch.sum(true_mask & ~pred_mask).float()

        # Calculate MCC for the current class
        numerator = (tp * tn) - (fp * fn)
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            mcc_per_class[cls] = 0
        else:
            mcc_per_class[cls] = numerator / denominator

    # Compute mean MCC across all classes
    if ignore_background:
        mcc_per_class = mcc_per_class[1:]
    mean_mcc = torch.mean(mcc_per_class).item()

    return mean_mcc


if __name__ == '__main__':
    a = torch.randint(0,2,(3,128,128,128))
    b = torch.randint(0,2,(3,128,128,128))
    print(compute_accuracy(a,b))
    print(compute_mean_iou(a,b))
    print(compute_mean_dice(a,b))
    print(compute_mean_mcc(a,b))

