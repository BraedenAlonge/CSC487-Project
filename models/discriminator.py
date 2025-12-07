import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, use_sigmoid=True, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.use_sigmoid = use_sigmoid
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        
        # Build convolutional layers
        conv_layers = []
        
        # Conv2d: 64×64×3 → 32×32×64
        conv_layers.append(
            nn.Conv2d(nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        )
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            conv_layers.append(nn.Dropout2d(dropout))
        
        # Conv2d: 32×32×64 → 16×16×128
        conv_layers.append(
            nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        )
        conv_layers.append(nn.BatchNorm2d(ndf * 2))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            conv_layers.append(nn.Dropout2d(dropout))
        
        # Conv2d: 16×16×128 → 8×8×256
        conv_layers.append(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        )
        conv_layers.append(nn.BatchNorm2d(ndf * 4))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            conv_layers.append(nn.Dropout2d(dropout))
        
        # Conv2d: 8×8×256 → 4×4×512
        conv_layers.append(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        )
        conv_layers.append(nn.BatchNorm2d(ndf * 8))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            conv_layers.append(nn.Dropout2d(dropout))
        
        self.main = nn.Sequential(*conv_layers)
        
        # Flatten → Linear: 4×4×512 → 1
        fc_layers = [
            nn.Flatten(),
        ]
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(ndf * 8 * 4 * 4, 1))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, input):
        x = self.main(input)
        x = self.fc(x)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        return x

def test_discriminator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Discriminator (BCE mode)...")
    netD = Discriminator(nc=3, ndf=64, use_sigmoid=True, kernel_size=4, 
                        stride=2, padding=1, dropout=0.0).to(device)
    
    # Create random images
    real_images = torch.randn(4, 3, 64, 64, device=device)
    
    # Discriminate
    with torch.no_grad():
        output = netD(real_images)
    
    print(f"Discriminator output shape: {output.shape}")
    print(f"Expected shape: (4, 1)")
    print(f"Min value: {output.min().item():.3f}, Max value: {output.max().item():.3f}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    assert 0.0 <= output.min().item() <= 1.0, "Output should be in [0, 1] range (Sigmoid)"
    assert 0.0 <= output.max().item() <= 1.0, "Output should be in [0, 1] range (Sigmoid)"
    print("Discriminator test passed!")


if __name__ == "__main__":
    test_discriminator()