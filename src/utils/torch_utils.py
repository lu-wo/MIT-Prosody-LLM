import torch
import torch.nn as nn


def masked_loss(labels, predictions, mask, loss_fn=nn.MSELoss(reduction="none")):
    """
    Compute the masked loss for given labels, predictions and mask.

    :param labels: Tensor containing the ground truth labels
    :param predictions: Tensor containing the predicted labels
    :param mask: Tensor containing the mask to apply on the loss
    :param loss_function: PyTorch loss function to compute the loss (default: nn.MSELoss(reduction="none"))

    :return: Masked loss
    """
    # Compute the element-wise loss
    loss = loss_fn(labels, predictions)

    # Apply the mask to the loss
    masked_loss = loss * mask

    # Compute the mean of the masked loss
    masked_loss_mean = torch.sum(masked_loss) / torch.sum(mask)

    return masked_loss_mean
