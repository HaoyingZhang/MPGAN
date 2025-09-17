import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
import numpy as np
import numba as nb
import stumpy


def znormalized_euclidian_distance(x, y):
    """Compute the Z-normalized Euclidian distance between two time series x and y."""
    return np.sqrt(np.sum(((x - np.mean(x)) / np.std(x) - (y - np.mean(y)) / np.std(y)) ** 2))


@nb.njit
def nb_znormalized_euclidian_distance(x, y):
    """Compute the Z-normalized Euclidian distance between two time series x and y using Numba."""
    return np.sqrt(np.sum(((x - np.mean(x)) / np.std(x) - (y - np.mean(y)) / np.std(y)) ** 2))


# The function to compute the distances by sliding window
def precompute_distances(x, m, n_mp, distance_function):
    distance_cache = np.zeros((n_mp, n_mp), dtype=np.float64)
    for i in range(n_mp):
        for j in range(i + 1, n_mp):
            dist_ij = distance_function(x[i:i + m], x[j:j + m])
            distance_cache[i, j] = dist_ij
            distance_cache[j, i] = dist_ij
    return distance_cache

# The Matrix Profile objective function
def objective_function(x_list, real_list, m, distance_function, coeff_dist=1, coeff_identity=1):
    '''
    x : List of variables
    mp: List of (n-m+1, 2) with mp[:, 0] MPD, and mp[:, 1] MPI
    distance_function: The distance function to calculate distances between subsequences
    coeff_dist, coeff_identity : put weights for distance constraints or index constraints 
    '''
    loss = 0
    for x, real_x in zip(x_list, real_list):
        mp = stumpy.stump(real_x, m)
        n_mp = len(mp)

        distance_cache = precompute_distances(x, m, n_mp, distance_function)

        # Compute distance_loss using vectorized operations
        distance_indices = mp[:, 1].astype(int)
        distance_losses = mp[:, 0] - distance_cache[np.arange(n_mp), distance_indices]
        distance_loss = np.sum(distance_losses ** 2)

        # Compute identity_loss using vectorized operations
        identity_diff = np.maximum(0.0, np.expand_dims(distance_cache[np.arange(n_mp), distance_indices], axis=1) - distance_cache)
        np.fill_diagonal(identity_diff, 0)  # Exclude self-comparisons
        identity_loss = np.sum(identity_diff)
        # print(f"distance_loss: {distance_loss}, identity_loss: {identity_loss}")
        loss += coeff_dist*distance_loss + coeff_identity*identity_loss
    return loss

def objective_function_mp(x_list, real_list, m, distance_function):
    loss = 0
    for x, real_x in zip(x_list, real_list):
        mp_real = stumpy.stump(real_x, m)
        mp_x = stumpy.stump(x, m)
        distance_loss = distance_function(np.array(mp_real[:, 0]), np.array(mp_x[:, 0]))
        indices_loss = np.mean([indice_x!=indice_real for indice_x, indice_real in zip(mp_x[:, 1], mp_real[:, 1])])
        loss += distance_loss + indices_loss
        
    return loss

### --- GENERATOR --- ###
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        out = F.relu(self.fc1(z))
        out, _ = self.lstm(out)
        return self.fc2(out)

### --- DISCRIMINATOR --- ###
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = DPLSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # use last timestep

### --- TRAINING LOOP --- ###
def train_dp_gan(train_loader, epsilon, delta, device='cuda'):
    from opacus import PrivacyEngine

    _, _, D = next(iter(train_loader))[0].shape
    # Used for noise input to Generator
    latent_dim = 16
    hidden_dim = 64

    # Models
    G = Generator(latent_dim, hidden_dim, D).to(device)
    D_net = Discriminator(D, hidden_dim).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-3)
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=1e-3)

    # Privacy engine for discriminator
    privacy_engine = PrivacyEngine()
    D_net, optimizer_D, train_loader = privacy_engine.make_private_with_epsilon(
        module=D_net,
        optimizer=optimizer_D,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=50,
        max_grad_norm=1.0,
    )

    criterion = nn.BCELoss(reduction="none")

    for epoch in range(10):
        # Train Discriminator
        for real_batch, _ in train_loader:
            if real_batch.shape[0] == 0:
                continue

            real_batch = real_batch.to(device)
            B, T, D = real_batch.shape

            z = torch.randn(B, T, latent_dim).to(device)
            with torch.no_grad():
                fake = G(z)

            # Use the discriminator to predict real or fake
            d_real = D_net(real_batch.float())
            d_fake = D_net(fake)

            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            loss_real = criterion(d_real, real_labels)
            loss_fake = criterion(d_fake, fake_labels)
            d_loss = (loss_real + loss_fake).mean()

            optimizer_D.zero_grad()

            try:
                d_loss.backward()
            except IndexError:
                for m in D_net.modules():
                    if hasattr(m, "activations"):
                        m.activations = []
                raise RuntimeError(
                    "Opacus activation tracking failed — ensure no D_net calls before backward."
                )

            optimizer_D.step()

        # Train Generator (must not touch Opacus)
        for _ in range(len(train_loader)):
            z = torch.randn(B, T, latent_dim).to(device)
            fake_data = G(z)

            with torch.no_grad():
                d_output = D_net(fake_data.detach())

            # Re-attach graph to fake_data
            g_loss_input = d_output + 0.0 * fake_data.sum()
            g_loss = -torch.mean(torch.log(g_loss_input + 1e-8))

            # Compute utility loss using Matrix Profile 
            real_sample = real_batch.detach().squeeze(-1).cpu().numpy().astype(np.float64)
            fake_sample = fake_data.detach().squeeze(-1).cpu().numpy().astype(np.float64)
            
            # MP loss
            mp_loss = objective_function_mp(fake_sample, real_sample, 10, znormalized_euclidian_distance)
            mp_loss = float(mp_loss)
            mp_loss = torch.tensor(mp_loss, dtype=torch.float32, device=device)
            
            g_loss += mp_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch+1}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    return G, D_net
