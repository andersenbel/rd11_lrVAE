import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss


def save_plot(fig, path):
    fig.savefig(path)
    plt.close(fig)


def visualize_reconstruction(model, dataloader, device, output_dir, epoch, latent_dim):
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            recon_x, _, _ = model(x)
            x = x.cpu().numpy().transpose(0, 2, 3, 1)
            recon_x = recon_x.cpu().numpy().transpose(0, 2, 3, 1)

            fig, axes = plt.subplots(2, 10, figsize=(15, 4))
            for i in range(10):
                axes[0, i].imshow((x[i] + 1) / 2)
                axes[0, i].axis('off')
                axes[1, i].imshow((recon_x[i] + 1) / 2)
                axes[1, i].axis('off')

            output_path = os.path.join(output_dir, f"reconstruction_latent_{
                                       latent_dim}_epoch_{epoch}.png")
            save_plot(fig, output_path)
            break


def plot_loss(train_losses, output_dir, latent_dim, beta):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title(f"Training Loss (Latent Dim={latent_dim}, Beta={beta})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    path = os.path.join(output_dir, f"loss_latent_{
                        latent_dim}_beta_{beta}.png")
    plt.savefig(path)
    plt.close()
