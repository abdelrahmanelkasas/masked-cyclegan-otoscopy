"""CycleGAN ag mimarileri: ResNet tabanli Generator + PatchGAN Discriminator."""
import torch, torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c),
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """9 residual block'lu ResNet generator (256x256 icin standart)."""
    def __init__(self, in_ch=3, out_ch=3, n_res=9, ngf=64):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7),
             nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):                       # downsample
            m += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        m += [ResBlock(c) for _ in range(n_res)]
        for _ in range(2):                       # upsample
            m += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1),
                  nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        m += [nn.ReflectionPad2d(3), nn.Conv2d(c, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*m)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """70x70 PatchGAN."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def blk(i, o, s=2, norm=True):
            l = [nn.Conv2d(i, o, 4, s, 1)]
            if norm: l += [nn.InstanceNorm2d(o)]
            l += [nn.LeakyReLU(0.2, True)]
            return l
        self.model = nn.Sequential(
            *blk(in_ch, ndf, norm=False), *blk(ndf, ndf * 2),
            *blk(ndf * 2, ndf * 4), *blk(ndf * 4, ndf * 8, s=1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )
    def forward(self, x):
        return self.model(x)

def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    return net
