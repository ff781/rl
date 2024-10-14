import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from env import create_rssm_env, DMCEnv
from nets import Cat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a simple autoencoder loss function
def autoencoder_loss(reconstructed, original):
    return nn.functional.mse_loss(reconstructed, original)

def train_autoencoder(rssm_env, optimizer, dataloader, num_epochs=100):
    device = next(rssm_env.parameters()).device
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.ion()  # Turn on interactive mode

    def update_plot(epoch):
        for ax in axes.flatten():
            ax.clear()

        # Get a batch of test images
        test_loader = DataLoader(
            datasets.MNIST(root='./data', train=False, download=True, transform=transform),
            batch_size=5, shuffle=True
        )
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)

        actions = torch.randn(10, 1, rssm_env.action_space.shape[0], device=device)
        simulated_batch = rssm_env.simulate(actions=actions, observations=test_images, steps=10)
        reconstructed_obs = simulated_batch['obs'][:-1]

        for i in range(5):
            axes[0, i].imshow(test_images[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')

            axes[1, i].imshow(reconstructed_obs[i, 0, 0].cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title('Reconstructed')

        plt.suptitle(f'Epoch {epoch+1}/{num_epochs}')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.size(0)
            seq_len = 10  # Assuming we want to process sequences of 10 frames

            # Reshape data to (seq_len, batch_size, channels, height, width)
            data = data.view(batch_size // seq_len, seq_len, *data.shape[1:]).permute(1, 0, 2, 3, 4)

            # Generate random actions (not used for reconstruction, but needed for RSSM)
            actions = torch.randn(seq_len, batch_size // seq_len, rssm_env.action_space.shape[0], device=device)

            # Simulate RSSM with mock observations
            simulated_batch = rssm_env.simulate(actions=actions, observations=data, steps=seq_len)

            # Extract original observations and reconstructed observations
            original_obs = data.flatten(end_dim=1)
            reconstructed_obs = simulated_batch['obs'][:-1].flatten(end_dim=1)

            # Calculate loss
            loss = autoencoder_loss(reconstructed_obs, original_obs)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Update the plot every epoch
        update_plot(epoch)

    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    # Set up MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=320, shuffle=True, num_workers=2)

    # Create a mock DMC environment for shape purposes
    dmc_env = DMCEnv("walker_walk", obs_type='img', img_size=64)

    # Create RSSM environment with parameters suitable for MNIST
    rssm_env = create_rssm_env(
        dmc_env,
        hidden_size=256,
        state_size=32,
        base_depth=32
    )

    # Move RSSM env to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rssm_env = rssm_env.to(device)

    # Create optimizer for the entire RSSM env (including encoder and decoder)
    optimizer = optim.Adam(rssm_env.parameters(), lr=1e-3)

    # Train the autoencoder and visualize progress
    train_autoencoder(rssm_env, optimizer, dataloader)