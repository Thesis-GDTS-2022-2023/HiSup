from einops import rearrange
from timm.models.layers import DropPath
import timm
import torch
import torch.nn.functional as F
from torch import nn





class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class LWMCrossAttention(nn.Module):
    def __init__(self,
                 dim=16,
                 num_heads=8,
                 window_size=16,
                 qkv_bias=False
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.eps = 1e-6
        self.ws = window_size

        self.q = Conv(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = Conv(dim, dim*2, kernel_size=1, bias=qkv_bias)
        self.proj = ConvBN(dim, dim, kernel_size=1)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps))
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps))
        return x

    def l2_norm(self, x):
        try:
            return torch.einsum("bhcn, bhn->bhcn", x, 1 / torch.norm(x, p=2, dim=-2))
        except Exception as e:
            # print("x shape:", x.shape)
            # print("norm_x shape:", (1 / torch.norm(x, p=2, dim=-2)).shape)
            raise e
            exit()

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        _, _, H, W = x1.shape
        x1 = self.pad(x1, self.ws)
        x2 = self.pad(x2, self.ws)
        # print(x1.shape, x2.shape)


        B, C, Hp, Wp = x1.shape
        hh, ww = Hp//self.ws, Wp//self.ws

        q = self.q(x1)
        kv = self.kv(x2)
        # print(q.shape, kv.shape)

        q, k, v = rearrange(q, 'b (q h d) (hh ws1) (ww ws2) -> q (b hh ww) h d (ws1 ws2)',
                            b=B, h=self.num_heads, d=C//self.num_heads, q=1, ws1=self.ws, ws2=self.ws), \
                  *rearrange(kv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h d (ws1 ws2)',
                             b=B, h=self.num_heads, d=C//self.num_heads, qkv=2, ws1=self.ws, ws2=self.ws)
        q = q[0]
        # print(q.shape, k.shape, v.shape)

        k = self.l2_norm(k)
        q = self.l2_norm(q).permute(0, 1, 3, 2)
        tailor_sum = 1 / (self.ws * self.ws + torch.einsum("bhnc, bhc->bhn", q, torch.sum(k, dim=-1) + self.eps))
        attn = torch.einsum('bhmn, bhcn->bhmc', k, v)
        attn = torch.einsum("bhnm, bhmc->bhcn", q, attn)
        v = torch.einsum("bhcn->bhc", v).unsqueeze(-1)
        v = v.expand(B*hh*ww, self.num_heads, C//self.num_heads,  self.ws * self.ws)
        attn = attn + v
        attn = torch.einsum("bhcn, bhn->bhcn", attn, tailor_sum)
        attn = rearrange(attn, '(b hh ww) h d (ws1 ws2) -> b (h d) (hh ws1) (ww ws2)',
                         b=B, h=self.num_heads, d=C // self.num_heads, ws1=self.ws, ws2=self.ws,
                         hh=Hp // self.ws, ww=Wp // self.ws)
        attn = attn[:, :, :H, :W]

        return attn

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class Mlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNAct(in_features, hidden_features, kernel_size=1)
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
                                 norm_layer(hidden_features),
                                 act_layer())
        self.fc3 = ConvBN(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)

        return x

class CrossGeoBlock(nn.Module):
    def __init__(self, dim=16, num_heads=8,  mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ws = window_size
        self.attn = LWMCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                          window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x1,x2):
        x1 = x1 + self.drop_path(self.attn(self.norm1(x1),self.norm1(x2)))
        x1 = x1 + self.drop_path(self.mlp(x1))

        x2 = x2 + self.drop_path(self.attn(self.norm1(x2),self.norm1(x1)))
        x2 = x2 + self.drop_path(self.mlp(x2))
        return x1,x2