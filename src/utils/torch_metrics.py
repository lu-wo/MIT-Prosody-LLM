import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError


class MaskedMeanSquaredError(MeanSquaredError):
    r"""Computes masked mean squared error (MSE) given a mask:

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N w_i(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions, and :math:`w_i` is the mask weight for each sample.

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mask = torch.tensor([1, 1, 0, 1], dtype=torch.float)
        >>> masked_mean_squared_error = MaskedMeanSquaredError()
        >>> masked_mean_squared_error(preds, target, mask)
        tensor(0.6667)
    """

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        """Update state with predictions, targets, and mask.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: Mask to apply on the loss
        """
        # Compute the element-wise squared error
        squared_error = torch.square(preds - target)

        # Apply the mask to the squared error
        masked_squared_error = squared_error * mask

        # Update the state
        self.sum_squared_error += torch.sum(masked_squared_error)
        self.total += torch.sum(mask)


class MaskedAccuracy(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        # process_group: Any = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            # process_group=process_group,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        preds = torch.argmax(preds, dim=-1)
        correct = (preds == target) * mask.bool()
        self.correct += torch.sum(correct)
        self.total += torch.sum(mask)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
