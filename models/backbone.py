import torch
import torch.nn as nn
import math

# --- Building Blocks from the original repo ---
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# --- The Custom Lightweight ResNet Feature Extractor ---
class CustomLightweightResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(CustomLightweightResNet, self).__init__()
        
        # MODIFICATION 1: Change input channels from 3 to 6
        # The original repo used a 3x3 kernel here, not 7x7 like standard ResNet
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Keep only layer1 and layer2, as per the original repo implementation
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # layers 3 and 4 are defined in the original code but unused in their forward pass.
        # We omit them entirely here for clarity.

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MODIFICATION 2: Removed self.fc layer and internal interpolate/NPR methods
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input shape [Batch*SeqLen, 6, H, W] (Our 6-channel input)
        Returns:
            torch.Tensor: Feature vectors. Shape depends on block type.
                          For BasicBlock (resnet18/34 style): [B*S, 128]
                          For Bottleneck (resnet50 style): [B*S, 512]
        """
        # Original repo logic:
        # x = self.conv1(NPR*2.0/3.0) -> They scaled their internal NPR
        # We just pass our pre-calculated 6-channel input directly.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # layer3 and layer4 are skipped, making it lightweight

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# --- Helper function to get the model ---
def get_lightweight_backbone(architecture='resnet50'):
    """
    Returns the custom lightweight backbone based on the original repo's structure.
    NOTE: 'resnet50' here refers to using Bottleneck blocks, but truncated at layer 2.
    """
    if architecture == 'resnet18':
        # Uses BasicBlock, layers=[2, 2, 2, 2] (only first two used)
        # Output feature dim: 128
        model = CustomLightweightResNet(BasicBlock, [2, 2, 2, 2])
        feature_dim = 128
    elif architecture == 'resnet50':
        # Uses Bottleneck, layers=[3, 4, 6, 3] (only first two used)
        # Output feature dim: 128 * 4 = 512
        model = CustomLightweightResNet(Bottleneck, [3, 4, 6, 3])
        feature_dim = 512
    else:
        raise ValueError("Unsupported architecture")
    return model, feature_dim


# --- Small verification block ---
if __name__ == '__main__':
    # Test with 'resnet50' style blocks (Bottleneck)
    cnn, feature_dim = get_lightweight_backbone('resnet50')
    print(f"Backbone created. Feature dimension: {feature_dim}")

    # Test with a dummy batch of 2 frames, 6 channels each
    dummy_input = torch.randn(2, 6, 256, 256)
    output = cnn(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output is [2, 512] because Bottleneck layer2 outputs 128*expansion(4) = 512
    assert output.shape == (2, 512)
    print("✅ CustomLightweightResNet module verification passed!")