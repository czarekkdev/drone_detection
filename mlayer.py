"""Multi-layer input stem: the model receives a normal RGB image and internally
stacks [R, G, B, gray, inverted gray] before the first convolution.
Doing it inside the model keeps augmentation identical for both variants and
exports cleanly to ONNX / Hailo (it is just a few multiply-adds)."""
import copy
import torch
import torch.nn as nn


class MultiLayerStem(nn.Module):
    def __init__(self, conv_block):
        super().__init__()
        # copy ultralytics bookkeeping attributes so the graph still works
        for a in ("f", "i", "type", "np"):
            if hasattr(conv_block, a):
                setattr(self, a, getattr(conv_block, a))
        self.block = copy.deepcopy(conv_block)
        old = self.block.conv
        new = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding,
                        old.dilation, old.groups, old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight  # start identical to the RGB model
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.block.conv = new
        self.register_buffer("w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x):  # x: RGB in [0,1]
        gray = (x * self.w).sum(1, keepdim=True)
        return self.block(torch.cat([x, gray, 1.0 - gray], 1))


def make_multilayer(det_model):
    """Replace first layer of an ultralytics DetectionModel in place."""
    if not isinstance(det_model.model[0], MultiLayerStem):
        det_model.model[0] = MultiLayerStem(det_model.model[0])
    return det_model
