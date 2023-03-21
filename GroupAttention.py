import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

Norm = nn.LayerNorm


class AnyAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def apply_pos(self, tensor, pos):
        if pos is None:
            return tensor
        elif len(tensor.shape) != len(pos.shape):
            tensor = rearrange(tensor, "b n (g c) -> b n g c", g=self.num_heads)
            tensor = tensor + pos
            tensor = rearrange(tensor, "b n g c -> b n (g c)")
        else:
            tensor = tensor + pos

        return tensor

    def get_qkv(self, q, k, v, qpos, kpos):
        q = self.apply_pos(q, qpos)
        k = self.apply_pos(k, kpos)
        v = self.apply_pos(v, None)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None, qpos=None, kpos=None):
        q, k, v = self.get_qkv(q, k, v, qpos, kpos)

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # attn matrix calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        attn *= self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        return out


class MultiScalePooling(nn.Module):
    def __init__(self, scales):
        super(MultiScalePooling, self).__init__()
        self.msp_braches = []
        for idx, scale in enumerate(scales):
            this_pool_branch = nn.AdaptiveAvgPool2d((scale, scale))
            self.msp_braches.append(this_pool_branch)

    def forward(self, feature):
        """
        Args:
            feature: [B, C, H, W]
        Returns:
            output: [B, C, N=sum(scales)]
        """
        bs, ch, _, _ = feature.size()
        pool_feas = []
        for idx, layer in enumerate(self.msp_braches):
            this_pool_fea = layer(feature).view(bs, ch, -1)
            pool_feas.append(this_pool_fea)

        output = torch.cat(pool_feas, dim=2)
        return output


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(nn.Module):
    def __init__(self, cfg, in_dim):
        super(GroupAttention, self).__init__()

        self.num_head = cfg.MODEL.GROUP_ATTENTION.NUM_HEADS
        # in_dim = cfg.MODEL.GROUP_ATTENTION.CHANNEL[indx]
        drop_rate = cfg.MODEL.GROUP_ATTENTION.DROP_RATE
        msp_scales = cfg.MODEL.GROUP_ATTENTION.MSP_SCALES

        self.attention = AnyAttention(in_dim, self.num_head)
        self.drop_path = DropPath(drop_prob=drop_rate)
        self.msp_block = MultiScalePooling(msp_scales)
        self.ffn = MLP(in_dim, hidden_features=in_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not torch.sum(m.weight.data == 0).item() == m.num_features:
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, feas, ds=False, scale=0.5):
        bs, _, h, w = feas.shape
        if ds:
            feas_scale = nn.AdaptiveMaxPool2d((int(h*scale), int(w*scale)))(feas)
            # feas_scale = F.interpolate(feas, scale_factor=scale, mode="bilinear")
            query_feas = rearrange(feas_scale, "b c h w -> (b h w) c").unsqueeze(0)
        else:
            query_feas = rearrange(feas, "b c h w -> (b h w) c").unsqueeze(0)
        pooled_feas = self.msp_block(feas)
        pooled_feas = rearrange(pooled_feas, "b c n -> (b n) c").unsqueeze(0)
        attn_out = self.attention(q=query_feas, k=pooled_feas, v=pooled_feas)
        if ds:
            attn_out = rearrange(attn_out.squeeze(), "(b h w) c -> b c h w", b=bs, h=int(h*scale))
            # _, _, o_h, o_w = attn_out.shape
            attn_out = F.interpolate(attn_out, scale_factor=1/scale, mode="nearest")
            # attn_out = F.interpolate(attn_out, (o_h/scale, o_w/scale), mode="nearest")
        else:
            attn_out = rearrange(attn_out.squeeze(), "(b h w) c -> b c h w", b=bs, h=h)
        feas = feas + self.drop_path(attn_out)
        feas = rearrange(feas, "b c h w -> (b h w) c", b=bs, h=h).unsqueeze(0)
        if self.ffn is not None:
            feas = feas + self.drop_path(self.ffn(feas))
        feas = rearrange(feas.squeeze(), "(b h w) c -> b c h w", b=bs, h=h)
        return feas
