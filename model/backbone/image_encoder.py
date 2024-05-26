from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .utils import Transformer, LayerNorm, QuickGELU, Transformer_modify, Bottleneck, AttentionPool2d


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224,
                 patch_size=16,
                 width=768,
                 layers=12,
                 heads=12,
                 output_dim=512,
                 out_indices=[3, 5, 7, 11],
                 prior_indices=[1],
                 replace_indices=[2, 3, 4],
                 pretrained=None,
                 get_embeddings=False,
                 use_prompt=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('input_resolution:', input_resolution)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        self.output_dim = output_dim
        self.layers = layers
        self.out_indices = out_indices
        self.prior_indices = prior_indices
        self.replace_indices = replace_indices
        self.use_prompt = use_prompt

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # cls token
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # position embedding of (patch embedding + cls token)
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads)

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width
        self.patch_size = patch_size

        if len(self.replace_indices) != 0:
            self.text_cls_proj = nn.Linear(self.output_dim, width)  # make the dimension of text embedding align to the dimension of cls token in the middle transformer blocks
        else:
            self.text_cls_proj = None

        if self.use_prompt:
            self.prompt = {}
            for i in range(self.layers):
                self.prompt[str(i * 2)] = nn.Parameter(
                    torch.nn.init.normal_(torch.zeros(width, 2), mean=0.0, std=1.0))
                self.prompt[str(i * 2 + 1)] = nn.Parameter(torch.zeros(2, width))
            self.prompt = nn.ParameterDict(self.prompt)


    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {} #new model_ori

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)   upsample the positional_embedding for larger input
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]  # only interpolate the pos embeddings of image patch embeddings
                    if self.patch_size == 16:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    elif self.patch_size == 32:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 7, 7, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    else:
                        assert AttributeError('Patch Size should be 14, 16 or 32')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    # def forward(self, x: torch.Tensor, text_cls, text_cls_proj, prompt, x_m):
    def forward(self, x: torch.Tensor, text_cls, x_m):
        x = self.conv1(x)
        B, C, H, W = x.shape  # C = width; H,W = grid,grid    grid = vision size//patch_size
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B,C,grid*grid   grid*grid: number of patch embeddings
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        ################ for large input, cls pos embedding is not interpolated ###################
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        ##########################################################################################
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND  BND -> NBD
        if self.text_cls_proj is not None:
            t = text_cls.contiguous()
            t = t.reshape(t.shape[0], t.shape[1], -1).permute(2, 0, 1)  # BDN -> NBD
            t = self.text_cls_proj(t)

        if not self.training:
            x_mp = torch.zeros_like(x_m)
        else:
            x_mp = x_m.contiguous()
        for i in range(x_mp.size(0)):
            ignore_pix = torch.where(x_mp[i] == 255)
            x_mp[i][ignore_pix[0], ignore_pix[1]] = 0
        x_mp = x_mp.float().unsqueeze(1)
        x_mp = F.interpolate(x_mp, size=(H, W), mode='bilinear', align_corners=True)

        features = []
        prior_mask_list = []
        outs = []
        cosine_eps = 1e-7
        for i, blk in enumerate(self.transformer.resblocks):
            if self.text_cls_proj is not None:
                if i in self.replace_indices:
                    x[0, :, :] = t
            if self.use_prompt:
                x_add = x @ self.prompt[str(i*2)]
                x_add = x_add @ self.prompt[str(i*2+1)]
                x = blk(x)
                x = x + x_add
            else:
                x = blk(x)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())  # contiguous:deep copy
            if len(self.prior_indices) > 0:
                if i in self.prior_indices:
                    xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    if i != (self.layers - 1):
                        aux_feat = Weighted_GAP(xp, x_mp)
                    else:
                        aux_feat = t
                    bs_, c_, h_, w_ = xp.size()[:]
                    # image feature
                    tmp_xp = xp.contiguous()
                    tmp_xp = tmp_xp.reshape(bs_, c_, -1)
                    tmp_xp_norm = torch.norm(tmp_xp, 2, 1, True)
                    # aux_feat (prototype or text feature)
                    tmp_aux_feat = aux_feat.contiguous()
                    tmp_aux_feat = tmp_aux_feat.reshape(bs_, c_, -1).permute(0, 2, 1)
                    tmp_aux_feat_norm = torch.norm(tmp_aux_feat, 2, 2, True)

                    similarity = torch.bmm(tmp_aux_feat, tmp_xp)/(torch.bmm(tmp_aux_feat_norm, tmp_xp_norm) + cosine_eps)
                    similarity = similarity.max(1)[0].reshape(bs_, h_*w_)
                    similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                    corr_query_mask = similarity.reshape(bs_, 1, h_, w_)
                    prior_mask_list.append(corr_query_mask)

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            global_embedding = x[:, 0]
            global_embedding = self.ln_post(global_embedding)
            global_embedding = global_embedding @ self.proj
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                features.append(visual_embedding)

            outs.append(features)
            outs.append(prior_mask_list)

            global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)  # clip original
            outs.append(global_embedding)

        return outs
