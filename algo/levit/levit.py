# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

import torch
import itertools
import torch.nn as nn
import utils

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model

import mask_utils

specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'LeViT_128': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'LeViT_192': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'LeViT_256': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'LeViT_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

__all__ = [specification.keys()]


@register_model
def LeViT_128S(num_classes=1000, distillation=True,
               pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_128(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              need_weight=False, mask_map=None, svd_type=None):
    return model_factory(**specification['LeViT_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         need_weight=need_weight, mask_map=mask_map, svd_type=svd_type)


@register_model
def LeViT_192(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              need_weight=False, mask_map=None, svd_type=None):
    return model_factory(**specification['LeViT_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         need_weight=need_weight, mask_map=mask_map, svd_type=svd_type)


@register_model
def LeViT_256(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              need_weight=False, mask_map=None, svd_type=None):
    return model_factory(**specification['LeViT_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         need_weight=need_weight, mask_map=mask_map, svd_type=svd_type)


@register_model
def LeViT_384(num_classes=1000, distillation=False,
              pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop
        self.recon_loss = 0

    def forward(self, x):
        if self.training and self.drop > 0:
            val = x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            val = x + self.m(x)
        # print(type(self.m))
        if isinstance(self.m, Attention):
            self.recon_loss = self.m.recon_loss
        return val


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14,
                 need_weight=False,
                 attn_mask=None,
                 svd_type=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))
        self.need_weight = need_weight
        self.attn_mask = attn_mask
        self.svd_type = svd_type
        self.recon_loss = 0

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        if need_weight:
             self.register_buffer('attention_sum', torch.zeros([num_heads, N, N]))
             self.register_buffer('num_attention', torch.zeros([1]))
             
        if self.svd_type == "mix_head_fc_qk":
            hidden = num_heads // 2 if not num_heads % 2 else num_heads // 2 + 1
            if num_heads == 3:
                hidden = 2
            elif num_heads == 5:
                hidden = 3
            elif num_heads == 6:
                hidden = 4
            # print("hidden: ", hidden)
            self.encoder_q = nn.Linear(num_heads, hidden)
            self.decoder_q = nn.Linear(hidden, num_heads)
            self.encoder_k = nn.Linear(num_heads, hidden)
            self.decoder_k = nn.Linear(hidden, num_heads)
        else:
            print('no need to build encoder/decoder!')
        
        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        # print("Attn: ", x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.svd_type == 'mix_head_fc_qk':

            q_shape = q.shape
            k_shape = k.shape

            q = q.contiguous().view(q.shape[0], q.shape[1], -1) # b x h x (nd)
            k = k.contiguous().view(k.shape[0], k.shape[1], -1) # b x h x (nd)

            q = q.permute(0, 2, 1)
            k = k.permute(0, 2, 1)

            pre_q = q
            pre_k = k
            q = self.decoder_q(self.encoder_q(q))
            k = self.decoder_k(self.encoder_k(k))

            self.recon_loss = torch.dist(q, pre_q) + torch.dist(k, pre_k)
            # print('reconstruction loss for q & k: {}'.format(self.recon_loss))

            del pre_q
            del pre_k

            q = q.permute(0, 2, 1)
            k = k.permute(0, 2, 1)

            q = q.view(q_shape)
            k = k.view(k_shape)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(x.device)  # to gpu
            attn_mask = attn_mask.repeat(B, 1, 1, 1)  # expand to num of bs
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(attn_mask, float('-inf'))
           
        attn = attn.softmax(dim=-1)
        if self.need_weight:
            self.num_attention += attn.size(0)  # (bs, h, nt, nt)
            self.attention_sum += attn.sum(0)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7,
                 need_weight=False,
                 attn_mask=None,
                 svd_type=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)
        self.svd_type = svd_type
        self.recon_loss = 0

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        self.need_weight = need_weight
        self.attn_mask = attn_mask
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))
        if need_weight:
             self.register_buffer('attention_sum', torch.zeros([num_heads, N_, N]))
             self.register_buffer('num_attention', torch.zeros([1]))

        if self.svd_type == "mix_head_fc_qk":
            hidden = num_heads // 2 if not num_heads % 2 else num_heads // 2 + 1
            # print("hidden: ", hidden)
            # if num_heads == 8:
            #     hidden = 3
            # elif num_heads == 16:
            #     hidden = 10
            self.encoder_q = nn.Linear(num_heads, hidden)
            self.decoder_q = nn.Linear(hidden, num_heads)
            self.encoder_k = nn.Linear(num_heads, hidden)
            self.decoder_k = nn.Linear(hidden, num_heads)
        else:
            print('no need to build encoder/decoder!')
        
        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        # print("Subsample Attn: ", x.shape)
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC

        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        if self.svd_type == 'mix_head_fc_qk':
            # pass
            q_shape = q.shape
            k_shape = k.shape

            q = q.contiguous().view(q.shape[0], q.shape[1], -1) # b x h x (nd)
            k = k.contiguous().view(k.shape[0], k.shape[1], -1) # b x h x (nd)

            q = q.permute(0, 2, 1)
            k = k.permute(0, 2, 1)

            pre_q = q
            pre_k = k
            # print("svd - q k shape: ", q.shape, k.shape)
            q = self.decoder_q(self.encoder_q(q))
            k = self.decoder_k(self.encoder_k(k))

            self.recon_loss = torch.dist(q, pre_q) + torch.dist(k, pre_k)
            # print('reconstruction loss for q & k: {}'.format(self.recon_loss))

            del pre_q
            del pre_k

            q = q.permute(0, 2, 1)
            k = k.permute(0, 2, 1)

            q = q.view(q_shape)
            k = k.view(k_shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)

        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(x.device)  # to gpu
            attn_mask = attn_mask.repeat(B, 1, 1, 1)  # expand to num of bs
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        if self.need_weight:
            # print(self.attention_sum.shape, attn.shape)
            self.num_attention += attn.size(0)  # (bs, h, nt, nt)
            self.attention_sum += attn.sum(0)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0,
                 need_weight=False,
                 svd_type=None,
                 mask_map=None):

        super().__init__()
        global FLOPS_COUNTER

        print('need_weight: ', need_weight)
        print('mask_map: ', mask_map)

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        # mask_path = {0: mask_path_1, 1: mask_path_2, 2: mask_path_3}
        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            mask_path = mask_map["depth" + str(i)]
            if mask_path:
                attn_mask = self._generate_patterns(mask_path, img_size, img_size // resolution,
                                                    nh, dpth, ed)
                # print("Atten depth: ", i)
                # print("Atten mask shape: ", len(attn_mask), len(attn_mask[0]), len(attn_mask[0][0]), len(attn_mask[0][0][0]), len(attn_mask[0][0][0][0]))
            else:
                attn_mask = [None] * dpth
            for j in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                        need_weight=need_weight,
                        attn_mask=attn_mask[j],
                        svd_type=svd_type
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                subsample_mask_path = mask_map["subsample_mask" + str(i)]
                if subsample_mask_path:
                # print ("num of heads in LeViT: ", i, nh)
                    attn_mask = self._generate_patterns(subsample_mask_path, img_size, img_size // resolution,
                                                    do[2], 1, ed)
                    # print("Subsample atten mask shape: ", len(attn_mask[0]), len(attn_mask[0][0]), len(attn_mask[0][0][0]), len(attn_mask[0][0][0][0]))
                else:
                    attn_mask = [None] * dpth
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_,
                        need_weight=need_weight,
                        attn_mask=attn_mask[0],
                        svd_type=False))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))

        self.blocks = torch.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0
        


    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        # print("After patch_embed: ", x.shape)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)

        recon_loss = 0
        # cnt = 0
        for i in range(len(self.blocks)):
            # print(type(self.blocks[i]))
            module = self.blocks[i]
            # if (self.blocks[i].recon_loss != 0):
            #     cnt +=1
            recon_loss += self.blocks[i].recon_loss
        # print("cnt: ", cnt)
        return x, recon_loss


    def _generate_patterns(self, mask_path,
                            input_size, patch_size,
                            heads, layers,
                            width,
                            ):
        with torch.no_grad():
            # num_layers * [1, num_heads, num_tokens, num_tokens]
            pattern_mask = mask_utils.mask_read_files(
                train_size_thw=(1,input_size,input_size),
                patch_size=patch_size,
                num_heads=heads, num_layers=layers,
                filepath=mask_path,
                )

            # Print patterns.
            if utils.is_main_process():
                mask_utils.print_pattern_ratio(pattern_mask)
                self.reduced_Gflops = mask_utils.cal_reduced_Gflops(
                                    pattern_mask, L=layers, C=width)
        return pattern_mask



def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, need_weight=False, svd_type=None, mask_map=None):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation,
        need_weight=need_weight,
        svd_type=svd_type,
        mask_map=mask_map,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    if fuse:
        utils.replace_batchnorm(model)

    return model


if __name__ == '__main__':
    for name in specification:
        net = globals()[name](fuse=True, pretrained=True)
        net.eval()
        net(torch.randn(4, 3, 224, 224))
        print(name,
              net.FLOPS, 'FLOPs',
              sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
