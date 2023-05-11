import torch
import torch.nn.functional as F
from torch import nn
import math

def featureL2Norm(feature):
        epsilon = 1e-6
            #        print(feature.size())
                #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)


class CrossGeoAtt(nn.Module):
    def __init__(self):
        super(CrossGeoAtt, self).__init__()
        en_out_channels=1024


        self.ReLU=nn.ReLU()
        self.linear_e = nn.Linear(en_out_channels,en_out_channels,bias = False)
        self.linear_e2 = nn.Linear(en_out_channels,en_out_channels,bias = False)

        # Use a 1x1 convolution to increase the number of channels
        self.conv = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)

        # Use a 2x2 max pooling layer to decrease the spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Use a bilinear upsampling layer to increase the spatial dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Use a 1x1 convolution to reduce the number of channels
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)


    def forward(self, hrn_features_a,hrn_features_b):
        hrn_features_a = self.conv(hrn_features_a)
        hrn_features_b = self.conv(hrn_features_b)

        hrn_features_a = self.pool(hrn_features_a)
        hrn_features_b = self.pool(hrn_features_b)

        b,c,h,w= hrn_features_a.size()
        feature_a = hrn_features_a.view(b,c,h*w)
        feature_b = hrn_features_b.view(b,c,h*w).transpose(1,2)
        feature_mul = torch.bmm(self.linear_e(feature_b),feature_a)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        self.corr_ab = featureL2Norm(self.ReLU(correlation_tensor))

        feature_a2 = hrn_features_a.view(b,c,h*w).transpose(1,2)# size [b,c,h*w]
        feature_b2 = hrn_features_b.view(b,c,h*w)
        feature_mul2 = torch.bmm(self.linear_e2(feature_a2),feature_b2)
        correlation_tensor2 = feature_mul2.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        self.corr_ba = featureL2Norm(self.ReLU(correlation_tensor2))
        en_features_a=torch.bmm(self.corr_ab.view(b,h*w,h*w),feature_a2)
        en_features_b=torch.bmm(self.corr_ba.view(b,h*w,h*w),feature_b)

        en_features_a = en_features_a.view(b,c,h,w)
        en_features_b = en_features_b.view(b,c,h,w)

        en_features_a = self.upsample(en_features_a)
        en_features_b = self.upsample(en_features_b)

        en_features_a = self.conv2(en_features_a)
        en_features_b = self.conv2(en_features_b)

        return en_features_a,en_features_b
