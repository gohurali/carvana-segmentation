import torch
import torch.nn.functional as F

"""
UNet with Residual connections
"""

class FCN(torch.nn.Module):

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
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(p=0.2)
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
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.block(x)

    def __init__(self):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(num_features=3)

        c2 = 64
        # [B,3,96,128] -> [B,8,48,64]
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=c2,
            kernel_size=7,
            stride=2,
            padding=7//2
        )
        # [B,64,48,64] -> [B,64,24,32]
        self.mp1 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=3//2
        )

        # [B,64,24,32] -> [B,64,12,16]
        self.ds_1 = self.DownSampleBlock(i_c=c2, o_c=c2, ks=3, s=2)
        self.ds_2 = self.DownSampleBlock(
            i_c=c2, o_c=c2, ks=3, s=2)  # [B,64,12,16] -> [B,64,6,8]
        self.ds_3 = self.DownSampleBlock(
            i_c=c2, o_c=c2, ks=3, s=2)  # [B,64,6,8] -> [B,64,3,4]

        self.ds_4 = self.DownSampleBlock(
            i_c=c2, o_c=c2, ks=3, s=1)  # [B,64,3,4] -> [B,64,3,4]

        self.us_1 = self.UpSampleBlock(i_c=c2*2, o_c=c2, ks=3, s=2, p=0)
        self.us_2 = self.UpSampleBlock(i_c=c2*2, o_c=c2, ks=3, s=2, p=0)
        self.us_3 = self.UpSampleBlock(i_c=c2*2, o_c=c2, ks=3, s=2, p=0)

        self.us_4 = self.UpSampleBlock(i_c=c2*2, o_c=c2, ks=3, s=2, p=0)
        self.us_5 = self.UpSampleBlock(i_c=c2*2, o_c=c2, ks=3, s=2, p=0)

        self.final = torch.nn.Conv2d(
            in_channels=c2, out_channels=1, kernel_size=1, stride=1, padding=1//2)
        # self.do1 = torch.nn.Dropout2d(p=0.2)

    def forward(self, x):

        x = self.bn0(x)
        B, C, H0, W0 = x.shape
        x = F.relu(self.conv1(x))
        mp1 = self.mp1(x)

        ds1 = self.ds_1(mp1)
        ds2 = self.ds_2(ds1)
        ds3 = self.ds_3(ds2)

        ds4 = self.ds_4(ds3)

        skip_c1 = torch.cat([ds3, ds4], 1)
        us1 = self.us_1(skip_c1)
        us1 = us1[..., :ds2.size(2), :ds2.size(3)]

        skip_c2 = torch.cat([us1, ds2], 1)
        us2 = self.us_2(skip_c2)
        us2 = us2[..., :ds1.size(2), :ds1.size(3)]

        skip_c3 = torch.cat([us2, ds1], 1)
        us3 = self.us_3(skip_c3)
        us3 = us3[..., :mp1.size(2), :mp1.size(3)]  # (24,32)

        skip_c4 = torch.cat([us3, mp1], 1)
        us4 = self.us_4(skip_c4)
        us4 = us4[..., :x.size(2), :x.size(3)]

        skip_c5 = torch.cat([us4, x], 1)
        us5 = self.us_5(skip_c5)
        us5 = us5[..., :H0, :W0]

        logits = self.final(us5)

        return logits

if __name__ == '__main__':
    model = FCN()
    print(model)