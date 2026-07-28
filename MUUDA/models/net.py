import torch
import torch.nn as nn

class DSAN(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN, self).__init__()
        self.n_outputs = 192 + 96
        self.feature_layers = DCRN(n_band, patch_size, num_class)
        self.fc = nn.Linear(self.n_outputs, num_class)


    def forward(self,x):
        features, band_weights = self.feature_layers(x)
        outputs = self.fc(features)

        return features, band_weights ,outputs


class SWL(nn.Module):
    def __init__(self, num_channels):
        super(SWL, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels // 8, out_channels=num_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.size()
        x_avg, x_max = self.avgpool(x), self.maxpool(x)
        avg_weights, max_weights = self.channel_attention(x_avg), self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights


class DCRN(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes, alpha=0.1):
        super(DCRN, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        self.n_classes = n_classes

        # 类别级光谱波段学习模块
        self.swl = SWL(input_channels)

        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()

        # Convolution Layer 2
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)

        self.inter_size = 192+96

        # attention
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        # Average pooling
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, labels=None):
        B, C, H, W = x.shape

        # 类别级光谱波段学习
        band_weights = self.swl(x)  # [B, C, 1, 1]

        # 应用波段权重
        x_weighted = x + x * band_weights  # 增强重要波段

        # 重塑为3D格式
        x_3d = x_weighted.unsqueeze(1)  # [B, 1, C, H, W]

        # Convolution layer 1
        x1 = self.conv1(x_3d)
        x1 = self.activation1(self.bn1(x1))

        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))

        x2 = self.conv5(x_3d)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2
        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        # attention map
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        # 训练时返回波段权重用于计算类别一致性损失
        return x, band_weights.squeeze()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

