import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
from torchvision.utils import save_image
import argparse

from models.main_ST import main_ST
from models.Branched_encoder import BranchedStyleContentEncoder
from models.conv_transformer_decoder import TransformerDecoder, TransformerDecoderLayerBlock
from models.CNNs import CNNDecoder, vgg

def get_args():
    parser = argparse.ArgumentParser(description="Style Transfer Training Script with Argument Parsing")
    # Paths
    parser.add_argument('--content_folder', type=str, default="./data/content", help="Path to content folder")
    parser.add_argument('--style_folder', type=str, default="./data/style", help="Path to style folder")
    parser.add_argument('--save_folder', type=str, default="./out/train", help="Folder to save output images")
    parser.add_argument('--checkpoint_folder', type=str, default="./checkpoints/train", help="Folder to save checkpoints")
    parser.add_argument('--vgg_weights_path', type=str, default="./checkpoints/vgg_normalised.pth", help="Path to VGG weights")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--iterations', type=int, default=30000, help="Number of training iterations")
    parser.add_argument('--save_interval', type=int, default=500, help="Interval for saving images")
    parser.add_argument('--base_lr', type=float, default=0.0001, help="Base learning rate")
    parser.add_argument('--content_w', type=float, default=3.0, help="Content loss weight")
    parser.add_argument('--style_w', type=float, default=5.0, help="Style loss weight")
    parser.add_argument('--lamda1_w', type=float, default=30, help="Identity loss weight 1")
    parser.add_argument('--lamda2_w', type=float, default=3.0, help="Identity loss weight 2")
    parser.add_argument('--Wsim_w', type=float, default=0.03, help="Similarity loss weight")
    
    # Logging and checkpoints
    parser.add_argument('--log_interval', type=int, default=50, help="Interval for logging training metrics")
    parser.add_argument('--checkpoints_save_interval', type=int, default=10000, help="Interval for saving checkpoints")
   
    return parser.parse_args()

def safe_loader(path):
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        print(f"Error path: {path}, Error: {e}")
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = safe_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            print(f"skip: {path}")
            return self.__getitem__((index + 1) % len(self))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.checkpoint_folder, exist_ok=True)
    
    # Set up the GradScaler for AMP training
    scaler = torch.cuda.amp.GradScaler()
    
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    content_dataset = SafeImageFolder(args.content_folder, transform=transform)
    style_dataset   = SafeImageFolder(args.style_folder, transform=transform)
    content_loader  = torch.utils.data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    style_loader    = torch.utils.data.DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    #Initialize models
    decoder_layer_instance = TransformerDecoderLayerBlock()
    SC_encoder = BranchedStyleContentEncoder()
    transformerDecoder = TransformerDecoder(decoder_layer=decoder_layer_instance)
    CNN_decoder = CNNDecoder()
    CNN_encoder = vgg
    CNN_encoder.load_state_dict(torch.load(args.vgg_weights_path, weights_only=True))
    network = main_ST(SC_encoder, transformerDecoder, CNN_decoder, CNN_encoder)
    network.to(DEVICE)
    network.train()

    # Optimizer setup
    optimizer = optim.Adam(
        list(SC_encoder.parameters()) +
        list(transformerDecoder.parameters()) +
        list(CNN_decoder.parameters()),
        lr=args.base_lr
    )

    # Training loop variables
    iteration_count = 0
    running_loss = 0.0
    content_iter = iter(content_loader)
    style_iter   = iter(style_loader)
    
    for step in range(args.iterations):
        iteration_count += 1
    
        try:
            content_img, _ = next(content_iter)
        except StopIteration:
            content_iter = iter(content_loader)
            content_img, _ = next(content_iter)
        try:
            style_img, _ = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_img, _ = next(style_iter)
    
        content_img = content_img.to(DEVICE)
        style_img = style_img.to(DEVICE)
    
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=str(DEVICE)):
            out, c_loss, s_loss, l_identity1, l_identity2, sim_loss = network(content_img, style_img)
    
            loss = (args.content_w * c_loss +
                    args.style_w * s_loss +
                    args.lamda1_w * l_identity1 +
                    args.lamda2_w * l_identity2 +
                    args.Wsim_w * sim_loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    
        if (step + 1) % args.log_interval == 0:
            grad_norm = torch.norm(next(network.SC_encoder.parameters()).grad.detach())
            print(f"Iteration {iteration_count}: Loss = {loss.item():.4f}, "
                  f"Content Loss = {c_loss.item():.4f}, Style Loss = {s_loss.item():.4f}, "
                  f"Identity1 = {l_identity1.item():.4f}, Identity2 = {l_identity2.item():.4f}, "
                  f"Sim Loss = {sim_loss.item():.4f},"
                  f"Grad Norm = {grad_norm:.4f}")

        if (step + 1) % args.save_interval == 0:
            output_name = '{:s}/{:s}_{:s}.jpg'.format(
                args.save_folder, str(step), "train"
            )
            combined_out = torch.cat((content_img, out), 0)
            combined_out = torch.cat((style_img, combined_out), 0)
            save_image(combined_out, output_name)
    
        if (step + 1) % args.checkpoints_save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_folder, f"main_ST_{step}.pth")
            optimizer_path = os.path.join(args.checkpoint_folder, f"Optimizer_{step}.pth")
            torch.save(network.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"Saved checkpoint at iteration {step + 1}")

    print(f"Training complete.")