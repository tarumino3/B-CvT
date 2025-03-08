import torch.nn as nn

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal_W(feat, eps=1e-5):
    feat_mean = feat.mean(dim=1, keepdim=True)
    feat_std = feat.std(dim=1, keepdim=True) + eps  # shape: [B, 1, C]  
    return (feat - feat_mean) / feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 

def calc_content_loss(input, target):
    mse = nn.MSELoss()
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return mse(input, target)

def calc_style_loss(input, target):
    mse = nn.MSELoss()
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse(input_mean, target_mean) + \
            mse(input_std, target_std)