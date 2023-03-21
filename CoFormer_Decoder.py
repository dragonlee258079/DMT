import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from einops import rearrange

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, get_norm
from timm.models.layers import trunc_normal_

from GroupAttention import GroupAttention


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


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
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

        self.mlp = nn.Conv1d(dim, dim, 1, groups=dim)

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
        q = rearrange(q, "n b (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "n b (g c) -> b n g c", g=self.num_heads)
        v = v.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # attn matrix calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        attn *= self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        # out = rearrange(out, "b q g c -> b q (g c)")
        # out = self.proj1(out)

        cta_att = torch.einsum("b q g c, b p g d -> b q p c d", self.proj1(out), self.proj2(out))
        b, n1, n2, c1, c2 = cta_att.shape
        # extract non diagonal
        cta_att = cta_att.flatten(1, 2)[:, :-1, :, :].view(b, n1-1, n2+1, c1, c2)[:, :, 1:, :, :].\
            flatten(1, 2).view(b, n1, n2-1, c1, c2)

        cta_att = cta_att.permute(0, 1, 3, 4, 2)
        cta_att = cta_att.flatten(-2).mean(-1)

        cta_att = self.mlp(-1*cta_att.transpose(1, 2)).transpose(1, 2)
        cta_att = torch.sigmoid(cta_att)

        out = out * cta_att.unsqueeze(dim=2)

        out = out.mean(dim=2)

        return out.transpose(0, 1)


class Trans_Feat2Tokes(nn.Module):
    def __init__(
            self,
            in_channel,
            num_heads,
            feedforward_dim,
            dropout=0.1,
    ):
        super(Trans_Feat2Tokes, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_channel, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn = AnyAttention(
            dim=in_channel, num_heads=num_heads
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_channel, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, in_channel)

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.norm3 = nn.LayerNorm(in_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            tgt,
            memory,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            q=tgt2, qpos=query_pos,
            k=memory, kpos=pos,
            v=memory
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class Trans_Tokes2Tokes(nn.Module):
    def __init__(
            self,
            in_channel,
            num_heads,
            feedforward_dim,
            dropout=0.1,
    ):
        super(Trans_Tokes2Tokes, self).__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channel, num_heads=num_heads, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # ignore the `SimpleReasoning` because we only output one token
        # self.reason = SimpleReasoning(num_parts, dim)

        self.linear1 = nn.Linear(in_channel, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, in_channel)

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)

        self.activation = nn.GELU()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            tgt,
            memory,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(memory, pos),
            memory
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Parse(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        feed_forward=512,
        drop_path=0.1,
    ):
        super(Parse, self).__init__()

        self.trans_feat2gr = Trans_Feat2Tokes(
            in_channel=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feed_forward,
            dropout=drop_path,
        )
        self.trans_gr2com = Trans_Tokes2Tokes(
            in_channel=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feed_forward,
            dropout=drop_path,
        )
        self.trans_com2gr = Trans_Tokes2Tokes(
            in_channel=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feed_forward,
            dropout=drop_path,
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        group_tgt,
        com_tgt,
        x,
        group_tokens: Optional[Tensor] = None,
        com_tokens: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        group_tgt = self.trans_feat2gr(
            tgt=group_tgt,
            memory=x,
            pos=pos,
            query_pos=group_tokens
        )

        co_tgt, bg_tgt = group_tgt.split(1, dim=0)
        co_tokens, _ = group_tokens.split(1, dim=0)

        co_tgt = co_tgt.transpose(0, 1)
        co_tokens = co_tokens.transpose(0, 1)
        com_tgt = self.trans_gr2com(
            tgt=com_tgt,
            memory=co_tgt,
            pos=co_tokens,
            query_pos=com_tokens
        )

        co_tgt = self.trans_com2gr(
            tgt=co_tgt,
            memory=com_tgt,
            pos=com_tokens,
            query_pos=co_tokens
        )

        return {
            'co_tgt': co_tgt.transpose(0, 1),
            # 'noco_tgt': noco_tgt,
            'bg_tgt': bg_tgt,
            'com_tgt': com_tgt
        }


class Decoder_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, start=False):
        super(Decoder_Conv, self).__init__()

        self.lateral_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", out_channel)
        ) if not start else None

        if not start:
            in_channel = out_channel

        self.output_conv = Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm("BN", out_channel),
            activation=F.relu,
        )

        if self.lateral_conv is not None:
            weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:
            cur_fpn = self.lateral_conv(enc_fea)
            dec_fea = cur_fpn + F.interpolate(dec_fea, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            dec_fea = self.output_conv(dec_fea)
        else:
            dec_fea = self.output_conv(enc_fea)
        return dec_fea


class Sal_Pred_with_Tok(nn.Module):
    def __init__(self, channel):
        super(Sal_Pred_with_Tok, self).__init__()

        self.linear1 = nn.Linear(channel, channel)
        self.linear2 = nn.Linear(channel, channel)

        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)

        self.scale = channel ** (-0.5)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, tok, fea):
        tok = self.norm1(tok)
        tok = self.linear2(F.relu(self.linear1(tok)))
        if tok.shape[1] == 1:
            tok = tok.expand(-1, fea.shape[0], -1)
        preds = torch.einsum("qbc,bchw->bqhw", self.norm2(tok), fea) * self.scale

        return preds


class Prototype_Refinement(nn.Module):
    def __init__(self, channel):
        super(Prototype_Refinement, self).__init__()

        self.gr_sal_pred = Sal_Pred_with_Tok(channel)
        # self.com_sal_pred_1 = Sal_Pred_with_Tok(channel)
        self.com_sal_pred_2 = Sal_Pred_with_Tok(channel)
        self.com_sal_pred_3 = Sal_Pred_with_Tok(channel)

        # self.fuse_noco = Conv2d(
        #     channel * 2,
        #     channel,
        #     kernel_size=1,
        #     bias=False,
        #     norm=get_norm("BN", channel),
        #     activation=F.relu
        # )
        # weight_init.c2_xavier_fill(self.fuse_noco)

        self.fuse_co = Conv2d(
            channel * 2,
            channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", channel),
            activation=F.relu
        )
        weight_init.c2_xavier_fill(self.fuse_co)

        self.fuse_bg = Conv2d(
            channel * 2,
            channel,
            kernel_size=1,
            bias=False,
            norm=get_norm("BN", channel),
            activation=F.relu
        )
        weight_init.c2_xavier_fill(self.fuse_bg)

    def _generate_proto(self, pred):
        bs, _, _, _ = pred.shape
        pred = torch.sigmoid(pred)
        min_pred = torch.min(pred.view(bs, -1), dim=1)[0].view(bs, 1, 1, 1)
        max_pred = torch.max(pred.view(bs, -1), dim=1)[0].view(bs, 1, 1, 1)
        norm_pred = (pred - min_pred) / (max_pred - min_pred)

        return norm_pred

    def pred_sigmoid_normalization(self, pred):
        bs, _, _, _ = pred.shape
        pred = torch.sigmoid(pred)
        min_pred = torch.min(pred.view(bs, -1), dim=1)[0].view(bs, 1, 1, 1)
        max_pred = torch.max(pred.view(bs, -1), dim=1)[0].view(bs, 1, 1, 1)
        norm_pred = (pred - min_pred) / (max_pred - min_pred)
        return norm_pred

    def Weighted_GAP(self, fea, mask):
        b, c, h, w = fea.shape
        fea = fea * mask
        ratio = h * w / (torch.sum(mask.view(b, 1, -1), dim=2) + 1e-8)
        vec = ratio * torch.mean(fea.view(b, c, -1), dim=2)
        return vec.view(b, c, 1, 1).repeat(1, 1, h, w)

    def forward(self, gr_tokens, com_tokens, fea):
        # saliency prediction using group tokens
        N = gr_tokens.shape[1]

        grs_tgt = torch.cat([gr_tokens, com_tokens.repeat(1, N, 1)], dim=0)
        gr_preds = self.gr_sal_pred(grs_tgt, fea)
        co_pred, bg_pred, com_pred = gr_preds.split(1, dim=1)

        # saliency prediction using com tokens
        # com_pred = self.com_sal_pred_1(com_tokens, fea)

        # generate prototypes
        co_pred_norm = self.pred_sigmoid_normalization(co_pred)
        # noco_pred_norm = self.pred_sigmoid_normalization(noco_pred)
        bg_pred_norm = self.pred_sigmoid_normalization(bg_pred)

        co_proto = self.Weighted_GAP(fea, co_pred_norm)
        # noco_proto = self.Weighted_GAP(fea, noco_pred_norm)
        bg_proto = self.Weighted_GAP(fea, bg_pred_norm)

        # refine the feature with prototypes, the order is noco, co, bg
        # fea_noco = self.fuse_noco(torch.cat([fea, noco_proto], dim=1))
        fea_co = self.fuse_co(torch.cat([fea, co_proto], dim=1))
        fea_bg = self.fuse_bg(torch.cat([fea_co, bg_proto], dim=1))

        co_tokens, _ = gr_tokens.split(1, dim=0)
        co_pred_1 = self.com_sal_pred_2(co_tokens, fea_co)
        co_pred_2 = self.com_sal_pred_3(co_tokens, fea_bg)

        return {
            "co_pred": co_pred,
            # "noco_pred": noco_pred,
            "bg_pred": bg_pred,
            "com_pred": com_pred,
            "co_pred_1": co_pred_1,
            "co_pred_2": co_pred_2
        }, fea_bg


class CoFormer_Decoder(nn.Module):
    def __init__(self, cfg):
        super(CoFormer_Decoder, self).__init__()

        # collating the encoding features' names and channels
        enc_name = cfg.MODEL.ENCODER.NAME
        ga_name = cfg.MODEL.GROUP_ATTENTION.NAME
        self.fea_names = enc_name + ga_name

        enc_channel = cfg.MODEL.ENCODER.CHANNEL
        ga_channel = cfg.MODEL.GROUP_ATTENTION.CHANNEL
        fea_channels = enc_channel + ga_channel

        # args for parse module
        hidden_dim = cfg.MODEL.COFORMER_DECODER.HIDDEN_DIM
        num_heads = cfg.MODEL.COFORMER_DECODER.NUM_HEADS
        feedforward_dim = cfg.MODEL.COFORMER_DECODER.FEEDFORWARD_DIM
        drop_path = cfg.MODEL.COFORMER_DECODER.DROP_PATH
        drop_path_ratios = torch.linspace(0, drop_path, len(fea_channels)-1)

        # position embedding
        self.pe_layer = PositionEmbeddingSine(hidden_dim//2, normalize=True)

        self.group_pos = nn.Embedding(2, hidden_dim)
        self.com_pos = nn.Embedding(1, hidden_dim)

        self.input_proj = Conv2d(
            fea_channels[-1],
            hidden_dim,
            kernel_size=1
        )
        weight_init.c2_xavier_fill(self.input_proj)

        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.bias, 1.0)

        for idx, channel in enumerate(fea_channels):
            # if idx != 0:
            dropout = drop_path_ratios[-idx]
            parse = Parse(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feed_forward=feedforward_dim,
                drop_path=dropout
            )
            self.add_module("parse_{}".format(idx + 1), parse)

            dec_conv = Decoder_Conv(
                in_channel=channel,
                out_channel=hidden_dim,
                start=idx == len(fea_channels) - 1
            )
            self.add_module("decoder_{}".format(idx + 1), dec_conv)

            if idx != 0 and idx != len(fea_channels) - 1:
                group_att = GroupAttention(cfg, hidden_dim)
                self.add_module("group_att_{}".format(idx + 1), group_att)

            if idx != 0:
                proto_refine = Prototype_Refinement(hidden_dim)
                self.add_module("proto_refine_{}".format(idx + 1), proto_refine)

        self.gr_sal_preds = Sal_Pred_with_Tok(hidden_dim)
        # self.com_sal_pred = Sal_Pred_with_Tok(hidden_dim)

    def forward(self, features: Dict):
        x = features[self.fea_names[-1]]
        x = self.input_proj(x)

        fea_pos = self.pe_layer(x)
        group_pos = self.group_pos.weight
        com_pos = self.com_pos.weight

        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        fea_pos = fea_pos.flatten(2).permute(2, 0, 1)
        group_pos = group_pos.unsqueeze(1).repeat(1, b, 1)
        com_pos = com_pos.unsqueeze(1)

        gr_tgt = torch.zeros_like(group_pos)
        com_tgt = torch.zeros_like(com_pos)

        x = self.norm(x)

        stage_co_preds = []
        # stage_noco_preds = []
        stage_bg_preds = []
        stage_com_preds = []
        stage_co_preds_1 = []
        stage_co_preds_2 = []

        dec_fea = None
        fea_nums = len(self.fea_names)
        for idx, fea_name in enumerate(self.fea_names[::-1]):
            enc_fea = features[fea_name]
            decoder_layer = getattr(self, "decoder_{}".format(fea_nums-idx))
            dec_fea = decoder_layer(
                enc_fea=enc_fea,
                dec_fea=dec_fea
            )

            if idx != fea_nums - 1 and idx != 0:
                group_att = getattr(self, "group_att_{}".format(fea_nums-idx))
                if dec_fea.shape[2] > 16:
                    dec_fea = group_att(dec_fea, ds=True, scale=16./dec_fea.shape[2])
                else:
                    dec_fea = group_att(dec_fea)

            parser_layer = getattr(self, "parse_{}".format(fea_nums-idx))
            tgts = parser_layer(
                group_tgt=gr_tgt,
                com_tgt=com_tgt,
                x=x,
                group_tokens=group_pos,
                com_tokens=com_pos,
                pos=fea_pos
            )
            co_tgt, bg_tgt = \
                tgts['co_tgt'], tgts['bg_tgt']
            com_tgt = tgts['com_tgt']

            gr_tgt = torch.cat([co_tgt, bg_tgt], dim=0)

            if idx != fea_nums - 1:
                proto_refine_layer = getattr(self, "proto_refine_{}".format(fea_nums-idx))
                preds, dec_fea = proto_refine_layer(
                    gr_tokens=gr_tgt,
                    com_tokens=com_tgt,
                    fea=dec_fea
                )

                stage_co_preds.append(preds["co_pred"])
                # stage_noco_preds.append(preds["noco_pred"])
                stage_bg_preds.append(preds["bg_pred"])
                stage_com_preds.append(preds["com_pred"])
                stage_co_preds_1.append(preds["co_pred_1"])
                stage_co_preds_2.append(preds["co_pred_2"])

        N = gr_tgt.shape[1]
        grs_tgt = torch.cat([gr_tgt, com_tgt.repeat(1, N, 1)], dim=0)
        gr_preds = self.gr_sal_preds(grs_tgt, dec_fea)
        co_pred, bg_pred, com_pred = gr_preds.split(1, dim=1)

        # com_pred = self.com_sal_pred(com_tgt, dec_fea)

        return {
            "stage_co_preds": stage_co_preds,
            # "stage_noco_preds": stage_noco_preds,
            "stage_bg_preds": stage_bg_preds,
            "stage_com_preds": stage_com_preds,
            "stage_co_preds_1": stage_co_preds_1,
            "stage_co_preds_2": stage_co_preds_2,
            "co_pred": co_pred,
            # "noco_pred": noco_pred,
            "bg_pred": bg_pred,
            "com_pred": com_pred
        }
