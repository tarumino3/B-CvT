import os
import glob
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

from models.Branched_encoder import BranchedStyleContentEncoder
from models.conv_transformer_decoder import TransformerDecoder, TransformerDecoderLayerBlock
from models.CNNs import CNNDecoder, vgg
from models.main_ST import main_ST

def get_args():
    parser = argparse.ArgumentParser(description="Style Transfer Model Testing Script")
    parser.add_argument('--content', type=str, required=True,
                        help="Path to a content image or a directory of content images.")
    parser.add_argument('--style', type=str, required=True,
                        help="Path to a style image or a directory of style images.")
    parser.add_argument('--results_dir', type=str, default='./out/test',
                        help="Directory to save output images.")
    parser.add_argument('--network_ckpt', type=str, default='./checkpoints/main_30000itr.pth',
                        help="Path to the style transfer network checkpoint.")
    parser.add_argument('--vgg_weights', type=str, default='./checkpoints/vgg_normalised.pth',
                        help="Path to the VGG weights for the CNN encoder.")
    return parser.parse_args()

def load_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    return img

def process_pair(content_img, style_img, network, device):
    with torch.inference_mode():
        out, *_ = network(content_img, style_img, test = True)
    return out

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build network modules.
    decoder_layer_instance = TransformerDecoderLayerBlock()
    SC_encoder = BranchedStyleContentEncoder()
    transformerDecoder = TransformerDecoder(decoder_layer=decoder_layer_instance)
    CNN_decoder = CNNDecoder()
    CNN_encoder = vgg
    CNN_encoder.load_state_dict(torch.load(args.vgg_weights, weights_only=True))
    CNN_encoder.eval()

    network = main_ST(SC_encoder, transformerDecoder, CNN_decoder, CNN_encoder)
    network.to(device)
    state_dict = torch.load(args.network_ckpt, weights_only=True)
    network.load_state_dict(state_dict=state_dict)
    network.eval()

    # Determine whether the provided paths are directories or single images.
    content_paths = sorted(glob.glob(os.path.join(args.content, "*"))) if os.path.isdir(args.content) else [args.content]
    style_paths = sorted(glob.glob(os.path.join(args.style, "*"))) if os.path.isdir(args.style) else [args.style]

    # Create the results directory if it doesn't exist.
    os.makedirs(args.results_dir, exist_ok=True)
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    # Process each combination of content and style images.
    with torch.no_grad():
        for c_path in content_paths:
            for s_path in style_paths:
                print(f"Processing: Content: {c_path}  |  Style: {s_path}")
                c_img = load_image(c_path, transform).unsqueeze(0).to(device)
                s_img = load_image(s_path, transform).unsqueeze(0).to(device)
                out = process_pair(c_img, s_img, network, device)
                c_base = os.path.splitext(os.path.basename(c_path))[0]
                s_base = os.path.splitext(os.path.basename(s_path))[0]
                out_filename = f"{c_base}_{s_base}.jpg"
                out_path = os.path.join(args.results_dir, out_filename)
                save_image(out, out_path)
                print(f"Saved output image: {out_path}")

if __name__ == '__main__':
    main()