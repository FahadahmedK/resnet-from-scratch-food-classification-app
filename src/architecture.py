from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, outplanes, kernel_size=3, strides=1, use_1x1conv=False):
        super().__init__()

        self.conv1 = nn.LazyConv2d(
            out_channels=outplanes,
            kernel_size=kernel_size,
            padding=1,
            stride=strides,
        )
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(
            out_channels=outplanes,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(
                out_channels=outplanes, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        print(X.shape, Y.shape)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, arch_config, input_channels=3, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                ),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )

        def SuperBlock(num_res_blocks, outplanes, first_block=False):
            blk = []
            for j in range(num_res_blocks):
                if j == 0 and not first_block:
                    blk.append(
                        ResidualBlock(outplanes=outplanes, use_1x1conv=True, strides=2)
                    )
                else:
                    blk.append(ResidualBlock(outplanes=outplanes))

            return nn.Sequential(*blk)

        for i, super_block_config in enumerate(arch_config):
            self.net.add_module(
                f"super_block {i+2}",
                SuperBlock(*super_block_config, first_block=(i == 0)),
            )

        self.net.add_module(
            "last",
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(num_classes)
            ),
        )


class ResNet18(ResNet):
    def __init__(self, input_channels=3, num_classes=2):
        super().__init__(
            ((2, 64), (2, 128), (2, 256), (2, 512)),
            input_channels=input_channels,
            num_classes=num_classes,
        )

    def forward(self, X):
        return self.net(X)
