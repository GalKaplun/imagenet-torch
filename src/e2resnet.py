import torch
import copy
from e2cnn import nn as enn
from e2cnn import gspaces


def conv1x1(in_type, out_type, stride=1):
    """1x1 convolution without padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=0,
                      bias=False)


def conv3x3(in_type, out_type, stride=1):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=1,
                      bias=False)


# TODO: Change this to be implemented as an EquivariantModule
class BasicBlock(torch.nn.Module):
    def __init__(self, in_type, out_type, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_type, out_type, stride)
        self.conv2 = conv3x3(out_type, out_type)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.relu2 = enn.ReLU(out_type, inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = torch.nn.Sequential(
                conv1x1(in_type, out_type, stride=stride),
                enn.InnerBatchNorm(out_type)
            )

    def forward(self, x):
        y = x
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu2(x+y)

# TODO: Change this to be implemented as an EquivariantModule
class E2_ResNetFPN_8(torch.nn.Module):
    """
    E2 equivariant ResNet+FPN,
    rotation invariant output resolutions are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, block_dims = [16, 24, 32], nbr_rotations=4):
        super().__init__()
        self.ds_ratio_coarse = 8
        initial_dim = block_dims[0]

        block = BasicBlock
        self.r2_act = gspaces.Rot2dOnR2(N=nbr_rotations)
        self.triv_in_type = enn.FieldType(self.r2_act,
                                          [self.r2_act.trivial_repr])
        dim_reduction = 2
        self.in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])
        # dummy variable used to track input types to each block
        self._in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])

        reg_repr_blocks = [
            enn.FieldType(self.r2_act,
                          (bd // dim_reduction)*[self.r2_act.regular_repr])
            for bd in block_dims
        ]
        b3_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[2]*[self.r2_act.trivial_repr])


        # Networks
        self.conv1 = enn.R2Conv(self.triv_in_type,
            self.in_type, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)

        self.layer1 = self._make_layer(block, reg_repr_blocks[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, reg_repr_blocks[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, reg_repr_blocks[2], stride=2)  # 1/8


        # 3. FPN upsample
        self.layer3_outconv = conv1x1(reg_repr_blocks[2], reg_repr_blocks[2])
        self.layer3triv = enn.R2Conv(reg_repr_blocks[2], b3_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                pass  # TODO: deltaorth initiation?

    def _make_layer(self, block, out_type, stride=1):
        layer1 = block(self._in_type, out_type, stride=stride)
        layer2 = block(out_type, out_type, stride=1)
        layers = (layer1, layer2)

        self._in_type = out_type
        return torch.nn.Sequential(*layers)

    def make_new_layers(self):
        result = self
        result.conv1 = copy.deepcopy(result.conv1)
        return result

    def forward(self, x):
        x = enn.GeometricTensor(x, self.triv_in_type)

        # ResNet Backbone
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        # rotation invariarize
        x3_inv = self.layer3triv(x3_out)

        return x3_inv.tensor

# TODO: Change this to be implemented as an EquivariantModule
class E2_ResNetFPN_8_2(torch.nn.Module):
    """
    E2 equivariant ResNet+FPN,
    rotation invariant output resolutions are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, block_dims = [16, 24, 32], nbr_rotations=4):
        super().__init__()
        self.ds_ratio_coarse = 8
        self.ds_ratio_fine = 2
        initial_dim = block_dims[0]

        block = BasicBlock
        self.r2_act = gspaces.Rot2dOnR2(N=nbr_rotations)
        self.triv_in_type = enn.FieldType(self.r2_act,
                                          [self.r2_act.trivial_repr])
        dim_reduction = 2
        self.in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])
        # dummy variable used to track input types to each block
        self._in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])

        reg_repr_blocks = [
            enn.FieldType(self.r2_act,
                          (bd // dim_reduction)*[self.r2_act.regular_repr])
            for bd in block_dims
        ]
        b1_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[0]*[self.r2_act.trivial_repr])
        b3_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[2]*[self.r2_act.trivial_repr])


        # Networks
        self.conv1 = enn.R2Conv(self.triv_in_type,
            self.in_type, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)

        self.layer1 = self._make_layer(block, reg_repr_blocks[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, reg_repr_blocks[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, reg_repr_blocks[2], stride=2)  # 1/8


        # 3. FPN upsample
        self.layer3_outconv = conv1x1(reg_repr_blocks[2], reg_repr_blocks[2])
        self.layer3triv = enn.R2Conv(reg_repr_blocks[2], b3_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.up3to2 = enn.R2Upsampling(reg_repr_blocks[2], 2, align_corners=True)
        self.layer2_outconv = conv1x1(reg_repr_blocks[1], reg_repr_blocks[2])
        self.layer2_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[2]),
            enn.InnerBatchNorm(reg_repr_blocks[2]),
            enn.ReLU(reg_repr_blocks[2]),
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[1]),
        )
        self.up2to1 = enn.R2Upsampling(reg_repr_blocks[1], 2, align_corners=True)
        self.layer1_outconv = conv1x1(reg_repr_blocks[0], reg_repr_blocks[1])
        self.layer1_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[1], reg_repr_blocks[1]),
            enn.InnerBatchNorm(reg_repr_blocks[1]),
            enn.ReLU(reg_repr_blocks[1]),
            conv3x3(reg_repr_blocks[1], b1_triv_repr),
        )

        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                pass  # TODO: deltaorth initiation?

    def _make_layer(self, block, out_type, stride=1):
        layer1 = block(self._in_type, out_type, stride=stride)
        layer2 = block(out_type, out_type, stride=1)
        layers = (layer1, layer2)

        self._in_type = out_type
        return torch.nn.Sequential(*layers)

    def make_new_layers(self):
        result = self
        result.conv1 = copy.deepcopy(result.conv1)
        return result

    def forward(self, x):
        x = enn.GeometricTensor(x, self.triv_in_type)

        # ResNet Backbone
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = self.up3to2(x3_out)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = self.up2to1(x2_out)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # rotation invariarize
        x3_inv = self.layer3triv(x3_out)
        x1_inv = x1_out  # trivial repr by design

        return x3_inv.tensor, x1_inv.tensor

# TODO: Change this to be implemented as an EquivariantModule
class E2_ResNetFPN_2(torch.nn.Module):
    """
    E2 equivariant ResNet+FPN,
    rotation invariant output resolutions are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, block_dims = [16, 24, 32], nbr_rotations=4):
        super().__init__()
        self.ds_ratio_coarse = 2
        initial_dim = block_dims[0]

        block = BasicBlock
        self.r2_act = gspaces.Rot2dOnR2(N=nbr_rotations)
        self.triv_in_type = enn.FieldType(self.r2_act,
                                          [self.r2_act.trivial_repr])
        dim_reduction = 2
        self.in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])
        # dummy variable used to track input types to each block
        self._in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])

        reg_repr_blocks = [
            enn.FieldType(self.r2_act,
                          (bd // dim_reduction)*[self.r2_act.regular_repr])
            for bd in block_dims
        ]
        b1_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[0]*[self.r2_act.trivial_repr])
        b3_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[2]*[self.r2_act.trivial_repr])


        # Networks
        self.conv1 = enn.R2Conv(self.triv_in_type,
            self.in_type, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)

        self.layer1 = self._make_layer(block, reg_repr_blocks[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, reg_repr_blocks[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, reg_repr_blocks[2], stride=2)  # 1/8


        # 3. FPN upsample
        self.layer3_outconv = conv1x1(reg_repr_blocks[2], reg_repr_blocks[2])
        self.layer3triv = enn.R2Conv(reg_repr_blocks[2], b3_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.up3to2 = enn.R2Upsampling(reg_repr_blocks[2], 2, align_corners=True)
        self.layer2_outconv = conv1x1(reg_repr_blocks[1], reg_repr_blocks[2])
        self.layer2_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[2]),
            enn.InnerBatchNorm(reg_repr_blocks[2]),
            enn.ReLU(reg_repr_blocks[2]),
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[1]),
        )
        self.up2to1 = enn.R2Upsampling(reg_repr_blocks[1], 2, align_corners=True)
        self.layer1_outconv = conv1x1(reg_repr_blocks[0], reg_repr_blocks[1])
        self.layer1_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[1], reg_repr_blocks[1]),
            enn.InnerBatchNorm(reg_repr_blocks[1]),
            enn.ReLU(reg_repr_blocks[1]),
            conv3x3(reg_repr_blocks[1], b1_triv_repr),
        )

        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                pass  # TODO: deltaorth initiation?

    def _make_layer(self, block, out_type, stride=1):
        layer1 = block(self._in_type, out_type, stride=stride)
        layer2 = block(out_type, out_type, stride=1)
        layers = (layer1, layer2)

        self._in_type = out_type
        return torch.nn.Sequential(*layers)

    def make_new_layers(self):
        result = self
        result.conv1 = copy.deepcopy(result.conv1)
        return result

    def forward(self, x):
        x = enn.GeometricTensor(x, self.triv_in_type)

        # ResNet Backbone
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = self.up3to2(x3_out)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = self.up2to1(x2_out)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # rotation invariarize
        x3_inv = self.layer3triv(x3_out)
        x1_inv = x1_out  # trivial repr by design

        return x1_inv.tensor


class E2_ResNetFPN_new(torch.nn.Module):
    """
    E2 equivariant ResNet+FPN,
    rotation invariant output resolutions are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, block_dims, nbr_rotations=8, e2_same_nbr_filters=False):
        super().__init__()
        # Config
        block = BasicBlock
        # initial_dim should be same as block_dims[0] (TODO: why is it written like this?)
        # These should also ideally be divisible by nbr_rotations.
        initial_dim = block_dims[0]
        # nbr_rotations = 1  # e.g. 8 for C8-symmetry

        self.r2_act = gspaces.Rot2dOnR2(N=nbr_rotations)
        self.triv_in_type = enn.FieldType(self.r2_act,
                                          [self.r2_act.trivial_repr])
        if e2_same_nbr_filters:
            dim_reduction = nbr_rotations
        else:
            dim_reduction = 1
        self.in_type = enn.FieldType(self.r2_act, 
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])
        # dummy variable used to track input types to each block
        self._in_type = enn.FieldType(self.r2_act, 
                                     (initial_dim // dim_reduction)*[self.r2_act.regular_repr])

        reg_repr_blocks = [
            enn.FieldType(self.r2_act,
                          (bd // dim_reduction)*[self.r2_act.regular_repr])
            for bd in block_dims
        ]
        b1_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[0]*[self.r2_act.trivial_repr])
        b3_triv_repr = enn.FieldType(self.r2_act,
                                    block_dims[2]*[self.r2_act.trivial_repr])


        # Networks
        self.conv1 = enn.R2Conv(self.triv_in_type,
            self.in_type, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)

        self.layer1 = self._make_layer(block, reg_repr_blocks[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, reg_repr_blocks[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, reg_repr_blocks[2], stride=2)  # 1/8


        # 3. FPN upsample
        self.layer3_outconv = conv1x1(reg_repr_blocks[2], reg_repr_blocks[2])
        self.layer3triv = enn.R2Conv(reg_repr_blocks[2], b3_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.up3to2 = enn.R2Upsampling(reg_repr_blocks[2], 2, align_corners=True)
        self.layer2_outconv = conv1x1(reg_repr_blocks[1], reg_repr_blocks[2])
        self.layer2_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[2]),
            enn.InnerBatchNorm(reg_repr_blocks[2]),
            enn.ReLU(reg_repr_blocks[2]),
            conv3x3(reg_repr_blocks[2], reg_repr_blocks[1]),
        )
        self.up2to1 = enn.R2Upsampling(reg_repr_blocks[1], 2, align_corners=True)
        self.layer1_outconv = conv1x1(reg_repr_blocks[0], reg_repr_blocks[1])
        self.layer1_outconv2 = torch.nn.Sequential(
            conv3x3(reg_repr_blocks[1], reg_repr_blocks[1]),
            enn.InnerBatchNorm(reg_repr_blocks[1]),
            enn.ReLU(reg_repr_blocks[1]),
            conv3x3(reg_repr_blocks[1], b1_triv_repr),
        )

        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                pass  # TODO: deltaorth initiation?

    def _make_layer(self, block, out_type, stride=1):
        layer1 = block(self._in_type, out_type, stride=stride)
        layer2 = block(out_type, out_type, stride=1)
        layers = (layer1, layer2)

        self._in_type = out_type
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.triv_in_type)

        # ResNet Backbone
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = self.up3to2(x3_out)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = self.up2to1(x2_out)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # rotation invariarize
        x3_inv = self.layer3triv(x3_out)
        x1_inv = x1_out  # trivial repr by design

        return x3_inv.tensor



if __name__ == "__main__":
	net = E2_ResNetFPN_8_2()
	inp = torch.zeros((10, 1, 256, 256), dtype=torch.float32)
	d = 'cuda:1'
	out, _ = net.to(d)(inp.to(d))
	print(out.shape)
