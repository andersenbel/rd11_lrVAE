import torch
from torch import optim
from models import VAE
from dataset import get_dataloaders
from utils import loss_function, visualize_reconstruction, plot_loss
import os


def train(model, train_loader, test_loader, device, epochs, lr, output_dir, model_dir, latent_dim):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model.to(device)
    beta_values = [0.5, 1.0, 2.0]
    for beta in beta_values:
        train_losses = []
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x)
                loss = loss_function(recon_x, x, mu, logvar, beta=beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader.dataset)
            train_losses.append(avg_loss)
            print(f"Latent Dim: {latent_dim}, Beta: {
                  beta}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            visualize_reconstruction(
                model, test_loader, device, output_dir, epoch, latent_dim)

            # Save model checkpoint
            torch.save(model.state_dict(), os.path.join(
                model_dir, f"vae_latent_{latent_dim}_beta_{beta}_epoch_{epoch}.pth"))

        # Save training loss plot
        plot_loss(train_losses, output_dir, latent_dim, beta)
