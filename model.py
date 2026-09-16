import torch
import torch.nn as nn

class SuperResolutionNet(nn.Module):
    """Simple but effective super-resolution network"""
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        
        # Input: 1 channel, 128x128
        # Output: 1 channel, 256x256
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Upsampling (2x)
        self.upsample = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(2)  # 2x upsampling
        )
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.conv4(x)
        return x

if __name__ == "__main__":
    model = SuperResolutionNet()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([1, 1, 256, 256])")
    print("\n✓ Model works!")