import torch


def compute_iou(output: torch.Tensor, truths: torch.Tensor) -> float:
    output = output.detach()
    truths = truths.detach()

    ## EXERCISE #####################################################################
    #
    # Implement the IoU metric that is used by the benchmark to grade your results.
    #
    # `output` is a tensor of dimensions [Batch, Classes, Height, Width]
    # `truths` is a tensor of dimensions [Batch, Height, Width]
    #
    # Tip: Peform a sanity check that tests your implementation on a user-defined
    #      tensor for which you know what the output should be.
    #
    #################################################################################
    pred = torch.argmax(output, dim=1)

    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (pred & truths).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | truths).sum((1, 2))  # Will be zzero if both are 0

    iou = intersection / union  # We smooth our devision to avoid 0/0

    #################################################################################

    return iou.mean().cpu().numpy()