import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

from model import CLIPTextEncoder, CLIPVisionTransformer, Transformer


class ESDA(nn.Module):
    def __init__(self, args):
        super(ESDA, self).__init__()
        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.text_encoder = CLIPTextEncoder(context_length=args.context_length,
                                            vocab_size=args.vocab_size,
                                            transformer_width=args.transformer_width,
                                            transformer_heads=args.transformer_heads,
                                            transformer_layers=args.transformer_layers,
                                            embed_dim=args.embed_dim,
                                            pretrained=args.pretrained_path)
        self.image_encoder = CLIPVisionTransformer(input_resolution=args.train_h,
                                                       patch_size=args.patch_size,
                                                       width=args.width,
                                                       layers=args.layers,
                                                       heads=args.heads,
                                                       output_dim=args.output_dim,
                                                       out_indices=args.out_indices,
                                                       prior_indices=args.prior_indices,
                                                       replace_indices=args.replace_indices,
                                                       pretrained=args.pretrained_path,
                                                       get_embeddings=args.get_embeddings,
                                                       use_prompt=args.use_prompt)
        self.text_encoder.init_weights()
        self.image_encoder.init_weights()
        self.decoder = Transformer(width=args.width,
                                   layers=2,
                                   heads=args.heads,
                                   embed_dim=args.embed_dim,
                                   classes=args.classes,
                                   )

    def forward(self, x, x_m, img_cls):
        bs, c, h, w = x.size()

        # ---------------- text feature ------------------------
        with torch.no_grad():
            text = CLIPTextEncoder.tokenize(img_cls).to('cuda')
            text_cls, text_feat_plus = self.text_encoder(text)
            text_cls = text_cls.unsqueeze(-1).unsqueeze(-1)
            text_feat_plus = text_feat_plus

        # ---------------- image feature ------------------------
        img_feat_plus = self.image_encoder(x, text_cls, x_m)
        img_feat = img_feat_plus[0][0]
        prior_mask_list = img_feat_plus[1]
        img_cls = img_feat_plus[2]

        # ---------------- decoder ------------------------
        out, out_list = self.decoder(img_feat, prior_mask_list, text_feat_plus, text_cls, img_cls)

        # ---------------- output part ------------------------
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, x_m.long())
            aux_loss1 = torch.zeros_like(main_loss).cuda()
            aux_loss2 = torch.zeros_like(main_loss).cuda()
            if len(out_list) > 0:
                for idx_k in range(len(out_list)):
                    inner_out = out_list[idx_k]
                    inner_out = torch.cat([inner_out, 1.0-inner_out], dim=1)
                    inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                    if idx_k == len(out_list)-1:
                        aux_loss1 = aux_loss1 + self.criterion(inner_out, x_m.long())
                    else:
                        aux_loss2 = aux_loss2 + self.criterion(inner_out, x_m.long())
                if len(out_list) > 1:
                    aux_loss2 = aux_loss2 / (len(out_list)-1)
            return out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return out










