import torch
import torch.nn as nn
from einops import rearrange

from models.cvt_block import VisionTransformer

common_spec = {
    "NUM_STAGES": 2,
    "PATCH_SIZE": [7, 3],
    "PATCH_STRIDE": [2, 2],
    "PATCH_PADDING": [3, 1],
    "DIM_EMBED": [64, 128],
    "DEPTH": [1, 2],
    "NUM_HEADS": [1, 4],
    "MLP_RATIO": [4, 4],
    "QKV_BIAS": [True, True],
    "DROP_RATE": [0.0, 0.0],
    "ATTN_DROP_RATE": [0.0, 0.0],
    "DROP_PATH_RATE": [0.0, 0.0],
    "CLS_TOKEN": [False, False],
    "QKV_PROJ_METHOD": ["dw_bn", "dw_bn"],
    "KERNEL_QKV": [3, 3],
    "PADDING_Q": [1, 1],
    "PADDING_KV": [1, 1],
    "STRIDE_KV": [2, 2],
    "STRIDE_Q": [1, 1]
}

branch_spec = {
    "NUM_STAGES": 1,
    "PATCH_SIZE": [3],
    "PATCH_STRIDE": [2],
    "PATCH_PADDING": [1],
    "DIM_EMBED": [512],
    "DEPTH": [3],
    "NUM_HEADS": [8],
    "MLP_RATIO": [4],
    "QKV_BIAS": [True],
    "DROP_RATE": [0.0],
    "ATTN_DROP_RATE": [0.0],
    "DROP_PATH_RATE": [0.1],
    "CLS_TOKEN": [False],
    "QKV_PROJ_METHOD": ["dw_bn"],
    "KERNEL_QKV": [3],
    "PADDING_Q": [1],
    "PADDING_KV": [1],
    "STRIDE_KV": [2],
    "STRIDE_Q": [1]
}

# Helper functions
def get_intermediates_from_stage(stage, x):
    x = stage.patch_embed(x)# Apply patch embedding.
    B, C, H, W = x.size()
    x = rearrange(x, 'b c h w -> b (h w) c')
    
    cls_tokens = None
    if stage.cls_token is not None:
        cls_tokens = stage.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    
    x = stage.pos_drop(x)
    
    intermediates = []
    # Process each block and collect its output.
    for blk in stage.blocks:
        x = blk(x, H, W)
        intermediates.append(x)
    
    if stage.cls_token is not None:
        cls_tokens, x = torch.split(x, [1, H * W], 1)
    return intermediates, H, W

def get_final_feat_from_stage(stage, x):
    x = stage.patch_embed(x) # Apply patch embedding.
    B, C, H, W = x.size()
    x = rearrange(x, 'b c h w -> b (h w) c')
    
    if stage.cls_token is not None:
        cls_tokens = stage.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    x = stage.pos_drop(x)

    for blk in stage.blocks:
        x = blk(x, H, W)
    
    if stage.cls_token is not None:
        cls_tokens, x = torch.split(x, [1, H * W], 1)
    
    return x, H, W

class BranchedStyleContentEncoder(nn.Module):
    def __init__(self,
                 in_chans=3,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 common_spec=common_spec,   # Spec for common part
                 branch_spec=branch_spec,   # Spec for branch part 
            ): 
        super().__init__()
        
        # ----------------------------
        # Build the Common Part
        # ----------------------------
        self.num_common_stages = common_spec['NUM_STAGES']
        current_in_chans = in_chans
        self.common_stages = nn.ModuleList()
        for i in range(self.num_common_stages):
            kwargs = {
                'patch_size': common_spec['PATCH_SIZE'][i],
                'patch_stride': common_spec['PATCH_STRIDE'][i],
                'patch_padding': common_spec['PATCH_PADDING'][i],
                'embed_dim': common_spec['DIM_EMBED'][i],
                'depth': common_spec['DEPTH'][i],
                'num_heads': common_spec['NUM_HEADS'][i],
                'mlp_ratio': common_spec['MLP_RATIO'][i],
                'qkv_bias': common_spec['QKV_BIAS'][i],
                'drop_rate': common_spec['DROP_RATE'][i],
                'attn_drop_rate': common_spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': common_spec['DROP_PATH_RATE'][i],
                'with_cls_token': common_spec['CLS_TOKEN'][i],
                'method': common_spec['QKV_PROJ_METHOD'][i],
                'kernel_size': common_spec['KERNEL_QKV'][i],
                'padding_q': common_spec['PADDING_Q'][i],
                'padding_kv': common_spec['PADDING_KV'][i],
                'stride_kv': common_spec['STRIDE_KV'][i],
                'stride_q': common_spec['STRIDE_Q'][i],
            }
            stage = VisionTransformer(
                in_chans=current_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.common_stages.append(stage)
            # Update input channels for next stage
            current_in_chans = common_spec['DIM_EMBED'][i]
        
        # ----------------------------
        # Build the Branch Parts
        # ----------------------------
        self.num_branch_stages = branch_spec['NUM_STAGES']
        self.style_branch = nn.ModuleList()
        self.content_branch = nn.ModuleList()
        for i in range(self.num_branch_stages):
            kwargs = {
                'patch_size': branch_spec['PATCH_SIZE'][i],
                'patch_stride': branch_spec['PATCH_STRIDE'][i],
                'patch_padding': branch_spec['PATCH_PADDING'][i],
                'embed_dim': branch_spec['DIM_EMBED'][i],
                'depth': branch_spec['DEPTH'][i],
                'num_heads': branch_spec['NUM_HEADS'][i],
                'mlp_ratio': branch_spec['MLP_RATIO'][i],
                'qkv_bias': branch_spec['QKV_BIAS'][i],
                'drop_rate': branch_spec['DROP_RATE'][i],
                'attn_drop_rate': branch_spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': branch_spec['DROP_PATH_RATE'][i],
                'with_cls_token': branch_spec['CLS_TOKEN'][i],
                'method': branch_spec['QKV_PROJ_METHOD'][i],
                'kernel_size': branch_spec['KERNEL_QKV'][i],
                'padding_q': branch_spec['PADDING_Q'][i],
                'padding_kv': branch_spec['PADDING_KV'][i],
                'stride_kv': branch_spec['STRIDE_KV'][i],
                'stride_q': branch_spec['STRIDE_Q'][i],
            }
            # Create separate stages for style and content branches
            stage_style = VisionTransformer(
                in_chans=current_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            stage_content = VisionTransformer(
                in_chans=current_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.style_branch.append(stage_style)
            self.content_branch.append(stage_content)
            # Update input channels for next branch stage
            current_in_chans = branch_spec['DIM_EMBED'][i]

    def forward_common(self, x):
        # Forward pass through common part stages.
        for stage in self.common_stages:
            x, _ = stage(x)
        return x
    
    def forward_branch(self, x, return_intermediates=False):
        # Process x through style and content branches independently.
        if return_intermediates:
            style_feats = []
            content_feats = []
            style_feat = x
            for stage in self.style_branch:
                inters, _, _ = get_intermediates_from_stage(stage, style_feat)
                style_feats.extend(inters) 
                style_feat = inters[-1]  # update input for next stage
            content_feats = []
            content_feat = x
            for stage in self.content_branch:
                inters, H, W = get_intermediates_from_stage(stage, content_feat)
                content_feats.extend(inters)
                content_feat = inters[-1]
            return style_feats, content_feats, H, W
        else:
            style_feat = x
            for stage in self.style_branch:
                s_feat, _, _ = get_final_feat_from_stage(stage, style_feat)
                style_feat = s_feat
            content_feat = x
            for stage in self.content_branch:
                c_feat, H, W = get_final_feat_from_stage(stage, content_feat)
                content_feat = c_feat
            return style_feat, content_feat, H, W
        
    def forward(self, x, test = None):
        common_feat = self.forward_common(x)

        # For test
        if test == 'content':
            content_feat = common_feat
            for stage in self.content_branch:
                content_feat, H, W = get_final_feat_from_stage(stage, content_feat)
            return None, content_feat, None, None, H, W
        if test == 'style':
             style_feat = common_feat
             for stage in self.style_branch:
                 style_feat, _, _ = get_final_feat_from_stage(stage, style_feat)
             return style_feat, None, None, None, _, _
        
        #For training
        style_intermediates, content_intermediates, H, W= self.forward_branch(common_feat, return_intermediates=True)
        style_encoding = style_intermediates[-1]
        content_encoding = content_intermediates[-1]
        return style_encoding, content_encoding, style_intermediates, content_intermediates, int(H), int(W)