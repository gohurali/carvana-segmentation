import torch
import torch.nn.functional as F
import sys
"""
Customizable UNet Implementation with residual connections

Papers:
Kaiming et al ResNets:
https://arxiv.org/pdf/1512.03385v1.pdf
Minaee et al. Image Segmentation for DL
https://arxiv.org/pdf/2001.05566.pdf
Ronneberger et al. U-Net:
https://arxiv.org/pdf/1505.04597.pdf
Long et al. Fully Convolutional Networks for Semantic Segmentation
https://arxiv.org/pdf/1411.4038.pdf
"""

class UNet(torch.nn.Module):

    class DownSampleBlock(torch.nn.Module):
        def __init__(self, i_c=3, o_c=32, ks=3, s=1):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=i_c,
                    out_channels=o_c,
                    kernel_size=ks,
                    stride=s,
                    padding=ks//2,
                    bias=False
                ),
                torch.nn.BatchNorm2d(num_features=o_c),
                torch.nn.ReLU()
            )

            self.downsample = None
            if(i_c != o_c or s != 1):
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=i_c,
                        out_channels=o_c,
                        kernel_size=1,
                        stride=s,
                        padding=1//2
                    ),
                    torch.nn.BatchNorm2d(num_features=o_c),
                    torch.nn.ReLU()
                )

        def forward(self, x):
            identity = x
            if(self.downsample is not None):
                identity = self.downsample(identity)
            return self.block(x) + identity

    class UpSampleBlock(torch.nn.Module):
        def __init__(self, i_c=3, o_c=32, ks=3, s=1, p=None, op=0):
            super().__init__()

            if(p is None):
                p = ks//2

            self.block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=i_c,
                    out_channels=o_c,
                    kernel_size=ks,
                    stride=s,
                    padding=p,
                    output_padding=ks//2
                ),
                torch.nn.BatchNorm2d(num_features=o_c),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            return self.block(x)
    
    class ReduceDim(torch.nn.Module):
        def __init__(self, i_c=3, o_c=32, ks=3, s=1, p=None, op=0):
            super().__init__()
            
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=i_c,
                    out_channels=o_c,
                    kernel_size=3,
                    stride=1,
                    padding=3//2
                ),
                torch.nn.BatchNorm2d(num_features=o_c),
                torch.nn.ReLU()
            )
        def forward(self,x):
            return self.block(x)

    def __init__(self):
        super().__init__()
        
        # STK Mean & STD
        # self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        # self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])
        
        # ImageNet Mean & STD
        self.input_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.input_std = torch.Tensor([0.229, 0.224, 0.225])
        
        # self.input_mean = torch.Tensor([0.0,0.0,0.0])
        # self.input_std = torch.Tensor([1.0,1.0,1.0]) * 255.0
        
        # 5 -> downsample by a factor of 32
        self.hourglass_size = 5
        in_channels=3
        out_channels=1
        
        c = 8
        c2 = c
        for i in range(self.hourglass_size):    
            if i == 0:
                self.add_module(f'ds{i}',self.DownSampleBlock(i_c=in_channels, o_c=c, ks=3, s=2))
            else:
                self.add_module(f'ds{i}',self.DownSampleBlock(i_c=c, o_c=c2, ks=3, s=2))
            c = c2
            c2 = c2*2
        
        d = c2 // 2
        for i in range(self.hourglass_size):
            self.add_module(f'us{i}',self.UpSampleBlock(i_c=c2//2, o_c=d//2, ks=3, s=2,p=0))
            if(i < self.hourglass_size-1):
                self.add_module(f'rd{i}',self.ReduceDim(i_c=d, o_c=d//2, ks=3, s=2,p=0))
            else:
                self.add_module(f'rd{i}',self.ReduceDim(i_c=d-1,o_c=d//2))
            c2 //= 2
            d //= 2
        
        self.root = torch.nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1, stride=1, padding=0)
    
    def forward(self,x):
        x = (
                (x - self.input_mean[None, :, None, None].to(x.device)) / 
                self.input_std[None, :, None, None].to(x.device)
            )
        skip_c = []
        for i in range(self.hourglass_size):
            layer = self._modules[f'ds{i}']
            skip_c.append(x)
            x = layer(x)
        
        for i in range(self.hourglass_size):
            layer = self._modules[f'us{i}']
            rd_layer = self._modules[f'rd{i}']
            x = layer(x)
            x = x[...,:skip_c[~i].shape[2],:skip_c[~i].shape[3]]
            x = torch.cat([x,skip_c[~i]],dim=1)
            x = rd_layer(x)
        
        logits = self.root(x)
        return logits


def main():
    pass


if __name__ == '__main__':
    main()