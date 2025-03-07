import torch.nn as nn
import numpy as np
from geomloss import SamplesLoss
from CvT_ST.function import calc_content_loss, calc_style_loss, normal, normal_W
from einops import rearrange
from itertools import repeat
from collections.abc import Iterable

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Content_ConvEmbed(nn.Module):
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor of shape [B, L, C] where L = H * W.
            H: Height of the feature map.
            W: Width of the feature map.
        Returns:
            x: Output tensor of shape [B, L, embed_dim] with spatial resolution preserved.
        """
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        # Apply convolution
        x = self.proj(x)
        B, C_new, H_new, W_new = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') # Rearrange back to [B, L, embed_dim]
        return x

class main_ST(nn.Module):
    def __init__(self, SC_encoder, transformer, CNN_decoder, CNN_encoder):

        super().__init__()
        enc_layers = list(CNN_encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.SC_encoder = SC_encoder
        self.transformer = transformer
        self.decoder = CNN_decoder

        self.sinkhorn_loss_fn = SamplesLoss("sinkhorn", debias=True)

        self.convpos = Content_ConvEmbed(
                 patch_size=3,
                 in_chans=512,
                 embed_dim=512,
                 stride=1,
                 padding=1,
                 norm_layer=None)

    def flatten_features(self, feat_list):
        new_feats = []
        for feat in feat_list:
            if feat.dim() == 4:
                B, C, H, W = feat.shape
                new_feats.append(feat.view(B, C, H * W).permute(0, 2, 1))
            else:
                new_feats.append(feat)
        return new_feats

    def decode(self, encoding, h, w, target_hw):
        B, N, C= encoding.shape          
        hs = encoding.permute(0, 2, 1)
        hs = hs.view(B, C, h, w)
        image = self.decoder(hs, target_hw) 
        return image 
        
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]               

    def forward(self, samples_c, samples_s, W_sim = False, separate = False, test = False):

        content_image = samples_c
        style_image = samples_s

        _, _, content_H, content_W = samples_c.shape
        _, _, style_H, style_W = samples_s.shape

        C_style, C_content, C_style_inter, C_content_inter, h_c, w_c  = self.SC_encoder(content_image, test=('content' if test else None))
        S_style, S_content, S_style_inter, S_content_inter, h_s, w_s = self.SC_encoder(style_image, test=('style' if test else None))
        
        embedd_C_content = self.convpos(C_content, h_c, w_c)
        embedd_S_content = self.convpos(S_content, h_s, w_s)

        hs = self.transformer(embedd_C_content, S_style ,h_c, w_c, h_s, w_s)[0]
        if test:
            return self.decode(hs, h_c, w_c, (content_H, content_W))
        I_hs = self.transformer(embedd_S_content, C_style, h_s, w_s, h_c, w_c)[0]

        Stylized = self.decode(hs, h_c, w_c, (content_H, content_W))
        I_Stylized = self.decode(I_hs, h_s, w_s, (style_H, style_W))       

        content_feats = self.encode_with_intermediate(content_image)
        style_feats = self.encode_with_intermediate(style_image)
        Stylized_feats = self.encode_with_intermediate(Stylized)
        I_Stylized_feats = self.encode_with_intermediate(I_Stylized)

        loss_c = calc_content_loss(normal(Stylized_feats[-1]), normal(content_feats[-1]))+calc_content_loss(normal(Stylized_feats[-2]), normal(content_feats[-2]))
        loss_s = calc_style_loss(Stylized_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += calc_style_loss(Stylized_feats[i], style_feats[i])

        I_loss_c = calc_content_loss(normal(I_Stylized_feats[-1]), normal(style_feats[-1]))+calc_content_loss(normal(I_Stylized_feats[-2]), normal(style_feats[-2]))
        I_loss_s = calc_style_loss(I_Stylized_feats[0], content_feats[0])
        for i in range(1, 5):
            I_loss_s += calc_style_loss(I_Stylized_feats[i], content_feats[i]) 

        content_loss = loss_c + I_loss_c
        style_loss = loss_s + I_loss_s

        Icc = self.decode(self.transformer(embedd_C_content, C_style, h_c, w_c, h_c, w_c)[0], h_c, w_c, (content_H, content_W))
        Iss = self.decode(self.transformer(embedd_S_content, S_style, h_s, w_s, h_s, w_s)[0], h_s, w_s, (style_H, style_W))
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)

        loss_lambda1 = calc_content_loss(Icc,content_image)+calc_content_loss(Iss,style_image)
        loss_lambda2 = calc_content_loss(Icc_feats[0], content_feats[0])+calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += calc_content_loss(Icc_feats[i], content_feats[i])+calc_content_loss(Iss_feats[i], style_feats[i])

        if W_sim:
            W_loss = 0.0
            # flatten intermidiate features
            C_content_inter_flat = self.flatten_features(C_content_inter)
            C_style_inter_flat   = self.flatten_features(C_style_inter)
            S_content_inter_flat = self.flatten_features(S_content_inter)
            S_style_inter_flat   = self.flatten_features(S_style_inter)
    
            num_layers = len(C_content_inter_flat) 
    
         # loop for each layer
            for layer in range(num_layers):
                layer_loss = 0.0
                B = C_content_inter_flat[layer].shape[0]

                for b in range(B):
                    C_content_b = normal_W(C_content_inter_flat[layer][b:b+1])
                    C_style_b   = normal_W(C_style_inter_flat[layer][b:b+1])
                    S_content_b = normal_W(S_content_inter_flat[layer][b:b+1])
                    S_style_b   = normal_W(S_style_inter_flat[layer][b:b+1])

                    S_loss_c = self.sinkhorn_loss_fn(C_content_b, C_style_b).mean()
                    S_loss_s = self.sinkhorn_loss_fn(S_content_b, S_style_b).mean()
            
                    layer_loss += (S_loss_c + S_loss_s)
                # take mean as a layer loss
                layer_loss = layer_loss / (B * 2)

                W_loss += layer_loss
        else:
            W_loss = 0

        return Stylized, I_Stylized, content_loss, style_loss, loss_lambda1, loss_lambda2, W_loss                