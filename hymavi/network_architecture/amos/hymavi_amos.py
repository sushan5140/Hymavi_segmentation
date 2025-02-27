from torch import nn
from typing import Tuple, Union
from hymavi.network_architecture.neural_network import SegmentationNetwork
from hymavi.network_architecture.hymavi_amos.config import config as cf


from hymavi.network_architecture.hymavi_amos.model.hymavi_unet import HymaviFullModel
class HymaviNet(SegmentationNetwork):
    def __init__(
            self,
            out_channels: int =cf.num_classes,
            pos_embed: str = "perceptron",
            dropout_rate: float = 0.0,
            depths=None,
            conv_op=nn.Conv3d,
            do_ds=cf.is_do_ds,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.model = HymaviFullModel()


    def forward(self, x, is_train=False):
        if is_train:
            out, feat = self.model(x, is_train=is_train)

            return out, feat
        else:
            out = self.model(x, is_train=is_train)

            return out