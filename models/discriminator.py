import torch
import torch.nn as nn
from .attention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, kernel_size=4, stride=2, padding=1, dropout=0.0, 
                 use_spectral_norm=False, attention=False, attention_layer=32):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm
        self.attention = attention
        self.attention_layer = attention_layer
        
        # Helper function to optionally apply spectral normalization
        def maybe_spectral_norm(module):
            if use_spectral_norm:
                return nn.utils.spectral_norm(module)
            return module
        
        # Conv2d: 64×64×3 → 32×32×64
        self.conv1 = maybe_spectral_norm(nn.Conv2d(nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Attention at 32×32 if enabled
        if attention and attention_layer == 32:
            self.attention_32 = SelfAttention(ndf)
        else:
            self.attention_32 = None
        
        # Conv2d: 32×32×64 → 16×16×128
        self.conv2 = maybe_spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        if not use_spectral_norm:
            self.bn2 = nn.BatchNorm2d(ndf * 2)
        else:
            self.bn2 = None
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Attention at 16×16 if enabled
        if attention and attention_layer == 16:
            self.attention_16 = SelfAttention(ndf * 2)
        else:
            self.attention_16 = None
        
        # Conv2d: 16×16×128 → 8×8×256
        self.conv3 = maybe_spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        if not use_spectral_norm:
            self.bn3 = nn.BatchNorm2d(ndf * 4)
        else:
            self.bn3 = None
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        # Conv2d: 8×8×256 → 4×4×512
        self.conv4 = maybe_spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        if not use_spectral_norm:
            self.bn4 = nn.BatchNorm2d(ndf * 8)
        else:
            self.bn4 = None
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        # Flatten → Linear: 4×4×512 → 1
        self.flatten = nn.Flatten()
        self.fc = maybe_spectral_norm(nn.Linear(ndf * 8 * 4 * 4, 1))
    
    def forward(self, input):
        # 64×64×3 → 32×32×64
        x = self.conv1(input)
        x = self.relu1(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        
        # Attention at 32×32
        if self.attention_32 is not None:
            x = self.attention_32(x)
        
        # 32×32×64 → 16×16×128
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        
        # Attention at 16×16
        if self.attention_16 is not None:
            x = self.attention_16(x)
        
        # 16×16×128 → 8×8×256
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x = self.relu3(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        
        # 8×8×256 → 4×4×512
        x = self.conv4(x)
        if self.bn4 is not None:
            x = self.bn4(x)
        x = self.relu4(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        
        # Flatten and final linear
        x = self.flatten(x)
        if self.dropout > 0:
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        
        return torch.sigmoid(x)

def test_discriminator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Discriminator (BCE mode)...")
    netD = Discriminator(nc=3, ndf=64, kernel_size=4, 
                        stride=2, padding=1, dropout=0.0, use_spectral_norm=False,
                        attention=False, attention_layer=32).to(device)
    
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