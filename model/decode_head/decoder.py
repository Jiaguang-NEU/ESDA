import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

from .utils import transformer


class Transformer(nn.Module):
    def __init__(self, width=768, layers=2, heads=16, embed_dim=512, classes=2):
        super(Transformer, self).__init__()
        self.layers = layers
        self.classes = classes

        fea_dim = width
        reduce_dim = 256

        self.text_embedding_proj = nn.Linear(embed_dim, width)
        self.text_img_concat_proj = nn.Linear(embed_dim + embed_dim, width)

        # feature interaction block
        self.VL_interaction = []
        self.SSE = []
        self.IEE = []
        for i in range(self.layers):
            self.VL_interaction.append(transformer(width, 1, heads))
            self.IEE.append(transformer(width, 1, heads))
            if i != self.layers-1:
                self.SSE.append(transformer(width, 1, heads))
        self.VL_interaction = nn.ModuleList(self.VL_interaction)
        self.SSE = nn.ModuleList(self.SSE)
        self.IEE = nn.ModuleList(self.IEE)

        # -----------------  conv cls + bilinear interpolate ----------------------
        self.res2 = nn.Sequential(
            nn.Conv2d(fea_dim + 1, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim // 2, reduce_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim // 4, classes, kernel_size=1)
        )

    def forward(self, img_feat, prior_mask_list, text_feat_plus, text_cls, img_cls):
        out_list = prior_mask_list
        bs, ch, h, w = img_feat.shape
        x = img_feat.contiguous()

        x = x.reshape(bs, ch, -1)
        x = x.permute(2, 0, 1)  # BDN --> NBD

        text_cls = text_cls.contiguous().squeeze(-1).permute(2, 0, 1)  # [1, B, D]
        img_cls = img_cls.contiguous().unsqueeze(0)
        text_img_concat = torch.cat([text_cls, img_cls], dim=-1)
        new_img_cls = self.text_img_concat_proj(text_img_concat)  # [1, B, D]

        t = self.text_embedding_proj(text_feat_plus)
        text_cls = self.text_embedding_proj(text_cls)
        t = t.permute(1, 0, 2)
        x = torch.cat([text_cls, t, new_img_cls, x], dim=0)

        for i in range(self.layers):
            x = self.VL_interaction[i](x)
            iee = x[-h * w:]
            iee = self.IEE[i](iee)
            sse = x[: -h * w]
            if i != self.layers-1:
                sse = self.SSE[i](sse)
            x = torch.cat([iee, sse], dim=0)

        x = x[-h*w:].permute(1, 2, 0).reshape(bs, ch, h, w).contiguous()  # NBD --> BCHW
        if len(prior_mask_list) != 0:
            x = torch.cat([x, prior_mask_list[-1]], dim=1)
        x = self.res2(x)
        out = self.cls(x)
        return out, out_list


