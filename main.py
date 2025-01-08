from train import train
from models import VAE
from dataset import get_dataloaders
import os

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dims = [16, 32, 64, 128]
    batch_size = 128
    epochs = 10
    lr = 1e-3

    output_dir = "output"
    model_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size)

    for latent_dim in latent_dims:
        print(f"Training with latent_dim={latent_dim}")
        model = VAE(latent_dim)
        train(model, train_loader, test_loader, device,
              epochs, lr, output_dir, model_dir, latent_dim)
