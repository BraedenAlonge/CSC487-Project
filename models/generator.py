import torch
import torch.nn as nn
from .attention import SelfAttention

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=512, nc=3, kernel_size=4, stride=2, padding=1, dropout=0.0,
                 attention=False, attention_layer=32):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.attention = attention
        self.attention_layer = attention_layer
        
        # Linear layer: 100 -> 4×4×512
        linear_layers = [
            nn.Linear(nz, 4 * 4 * ngf),
            nn.BatchNorm1d(4 * 4 * ngf),
            nn.ReLU(True)
        ]
        if dropout > 0:
            linear_layers.append(nn.Dropout(dropout))
        self.linear = nn.Sequential(*linear_layers)
        
        # TransposeConv2d: 4×4×512 → 8×8×256
        self.conv1 = nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf // 2)
        self.relu1 = nn.ReLU(True)
        
        # TransposeConv2d: 8×8×256 → 16×16×128
        self.conv2 = nn.ConvTranspose2d(ngf // 2, ngf // 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf // 4)
        self.relu2 = nn.ReLU(True)
        
        # Attention at 16×16 if enabled
        if attention and attention_layer == 16:
            self.attention_16 = SelfAttention(ngf // 4)
        else:
            self.attention_16 = None
        
        # TransposeConv2d: 16×16×128 → 32×32×64
        self.conv3 = nn.ConvTranspose2d(ngf // 4, ngf // 8, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf // 8)
        self.relu3 = nn.ReLU(True)
        
        # Attention at 32×32 if enabled
        if attention and attention_layer == 32:
            self.attention_32 = SelfAttention(ngf // 8)
        else:
            self.attention_32 = None
        
        # TransposeConv2d: 32×32×64 → 64×64×3
        self.conv4 = nn.ConvTranspose2d(ngf // 8, nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        x = self.linear(input)
        x = x.view(-1, self.ngf, 4, 4)  # Reshape to (batch, 512, 4, 4)
        
        # 4×4×512 → 8×8×256
        x = self.conv1(x)
        x = self.bn1(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        x = self.relu1(x)
        
        # 8×8×256 → 16×16×128
        x = self.conv2(x)
        x = self.bn2(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        x = self.relu2(x)
        
        # Attention at 16×16
        if self.attention_16 is not None:
            x = self.attention_16(x)
        
        # 16×16×128 → 32×32×64
        x = self.conv3(x)
        x = self.bn3(x)
        if self.dropout > 0:
            x = nn.functional.dropout2d(x, p=self.dropout, training=self.training)
        x = self.relu3(x)
        
        # Attention at 32×32
        if self.attention_32 is not None:
            x = self.attention_32(x)
        
        # 32×32×64 → 64×64×3
        x = self.conv4(x)
        x = self.tanh(x)
        
        return x

def test_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = Generator(nz=100, ngf=512, nc=3, kernel_size=4, stride=2, padding=1, dropout=0.0,
                     attention=False, attention_layer=32).to(device)
    
    # Create random noise
    noise = torch.randn(4, 100, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = netG(noise)
    
    print(f"Generator output shape: {fake_images.shape}")
    print(f"Expected shape: (4, 3, 64, 64)")
    print(f"Min value: {fake_images.min().item():.3f}, Max value: {fake_images.max().item():.3f}")
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    
    assert fake_images.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {fake_images.shape}"
    assert -1.0 <= fake_images.min().item() <= 1.0, "Output should be in [-1, 1] range"
    assert -1.0 <= fake_images.max().item() <= 1.0, "Output should be in [-1, 1] range"
    print("Generator test passed!")


if __name__ == "__main__":
    test_generator()