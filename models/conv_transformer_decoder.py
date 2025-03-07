import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops.layers.torch import Rearrange
from einops import rearrange
from CvT_ST.models.cvt_block import Attention, Mlp
from timm.layers import DropPath
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CrossAttention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False,
                 **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        # Build convolutional projections for query (from content)
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q, 'linear' if method=='avg' else method
        )
        # Build convolutional projections for key and value (from style)
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))
        return proj

    def forward_conv(self, x, h, w, proj):
        """
        A helper to apply a convolutional projection.
        Args:
          x: Input tensor of shape [B, L, dim_in]
          h, w: spatial dimensions such that L = h * w.
          proj: The projection module to apply.
        Returns:
          The projected tokens, shape [B, L, dim_out]
        """
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if proj is not None:
            x = proj(x)  # [B, dim_in, new_h, new_w]
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x

    def forward(self, query, h_q, w_q, key_value, h_kv, w_kv):
        """
        Args:
            query: Tensor from content branch, shape [B, L_q, dim_in] (used for query)
            key_value: Tensor from style branch, shape [B, L_kv, dim_in] (used for key and value)
            h_q, w_q: spatial dimensions for query tokens (L_q = h_q * w_q)
            h_kv, w_kv: spatial dimensions for key/value tokens (L_kv = h_kv * w_kv)
        Returns:
            x: Output tensor of shape [B, L_q, dim_out] after cross-attention.
        """
        #apply convolution and projection
        q = self.forward_conv(query, h_q, w_q, self.conv_proj_q)
        k = self.forward_conv(key_value, h_kv, w_kv, self.conv_proj_k)
        v = self.forward_conv(key_value, h_kv, w_kv, self.conv_proj_v)

        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # Compute attention scores.
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values.
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1,
                 mlp_ratio=4.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(d_model) # Pre-norm for self-attention
        self.self_attn = Attention(dim_in=d_model, dim_out=d_model, num_heads=nhead,
                                     qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(d_model) # Pre-norm for cross-attention.
        self.cross_attn = CrossAttention(dim_in = d_model, dim_out = d_model, num_heads = nhead)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(d_model) # Pre-norm for the MLP.
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=hidden_dim, out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, content, style, h_c, w_c, h_s, w_s):
        """
        Args:
            content: Tensor from the content branch, shape [B, T, d_model]. Used as query.
            style: Tensor from the style branch, shape [B, T, d_model]. Used as key and value.
            h, w: Spatial dimensions so that T = h * w.
        Returns:
            output: Tensor of shape [B, T, d_model].
        """
        # 1. Self-Attention on content tokens.
        x = content
        x_res = x
        x = self.norm1(x)
        self_attn_out = self.self_attn(x, h_c, w_c)  
        x = x_res + self.drop_path1(self_attn_out)

        # 2. Cross-Attention: content as query, style as key and value.
        x_res = x
        x = self.norm2(x) 
        cross_attn_out = self.cross_attn(x, h_c, w_c, style, h_s, w_s)
        x = x_res + self.drop_path2(cross_attn_out)

        # 3. MLP Block.
        x_res = x
        x = self.norm3(x)
        mlp_out = self.mlp(x)
        x = x_res + self.drop_path3(mlp_out)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers = 3, norm=None, return_intermediate=False):
        """
        Args:
            decoder_layer (nn.Module): An instance of your decoder layer block (e.g., TransformerDecoderLayerBlock).
            num_layers (int): Number of decoder layers to stack.
            norm (nn.Module, optional): Normalization layer to apply at the end.
            return_intermediate (bool, optional): If True, return a tensor stacking the intermediate outputs.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, content, style, h_c, w_c, h_s, w_s):
        """
        Args:
            content: Tensor from content branch, shape [B, T, d_model] (query).
            style: Tensor from style branch, shape [B, T, d_model] (key and value).
            h_c, w_c: Spatial dimensions for content tokens (such that T = h_c * w_c).
            h_s, w_s: Spatial dimensions for style tokens.
            (Optional masks are provided if needed.)
        Returns:
            output: Tensor of shape [B, T, d_model] after processing through the decoder.
                     If return_intermediate is True, returns a stacked tensor of all intermediate outputs.
        """
        # content = content.permute(1, 0, 2)
        # style = style.permute(1, 0, 2)
        output = content  # initial query is the content encoding
        intermediates = []
        for layer in self.layers:
            # Each decoder layer fuses content and style representations.
            output = layer(output, style, h_c, w_c, h_s, w_s)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediates.append(self.norm(output))
                else:
                    intermediates.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediates)  # shape: [num_layers, B, T, d_model]
        return output.unsqueeze(0)

# Example usage:
# if __name__ == "__main__":
#     B = 2
#     h_c, w_c = 14, 14
#     h_s, w_s = 14, 14
#     T_c = h_c * w_c
#     T_s = h_s * w_s
#     d_model = 512
#     nhead = 4

#     # Create dummy content and style encodings.
#     content = torch.randn(B, T_c, d_model)
#     style = torch.randn(B, T_s, d_model)
#     decoder_layer = TransformerDecoderLayerBlock(d_model=d_model, nhead=nhead, dropout=0.1, mlp_ratio=4.0)
#     decoder = TransformerDecoder(decoder_layer, num_layers=3, norm=nn.LayerNorm(d_model), return_intermediate=True)

#     # Forward pass through the decoder.
#     output= decoder(content, style, h_c, w_c, h_s, w_s)
#     print("Output shape:", output.shape)  # Expected: [B, T_c, d_model]
#     for i in output:
#         print(i.shape)
