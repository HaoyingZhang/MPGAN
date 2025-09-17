import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from dp_gans import train_rns_dp_gan, rns_encode, rns_decode, train_dp_gan
from utils_matrix_profile import compute_matrix_profile_distance
from utils_mia import run_mia_attack
import numpy as np
import matplotlib.pyplot as plt

def normalize(time_series : np.ndarray) -> np.ndarray:
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())

# 1. Load synthetic data
torch.manual_seed(42)
# B: batch size (number of users), 
# T: time step (length of each time series), 
# D: dimension (Number of variables per time step)
X_full = torch.randint(0, 256, (2000, 24, 1))  
moduli = [5, 7, 11]
epsilon = 1.0
delta = 1e-5

# 2. Split: 50% train, 25% member test, 25% non-member
n = len(X_full)
train_set, member_test, non_member_test = random_split(X_full, [n//2, n//4, n//4])
# Convert Subset -> Tensor
train_tensor = torch.stack([X_full[i] for i in train_set.indices])
train_dataset = TensorDataset(train_tensor, torch.zeros(len(train_tensor)))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 3. Train RNS-DP-GAN
# G, _ = train_rns_dp_gan(train_loader, moduli, epsilon, delta, device='cpu')
G, _ = train_dp_gan(train_loader, epsilon, delta, device='cpu')

# 4. Generate samples
G.eval()
B, T, D = next(iter(train_loader))[0].shape
latent_dim = 16
with torch.no_grad():
    z = torch.randn(len(member_test), T, latent_dim).to('cpu')
    fake_data = G(z).cpu()

# 5. Evaluate Utility: Matrix Profile
real_sample = member_test[:200]
fake_sample = fake_data[:200]
utility_score = compute_matrix_profile_distance(real_sample, fake_sample)
print(f"✅ Matrix Profile Distance (lower is better): {utility_score:.4f}")

# 6. TODO : Evaluate Privacy: Membership Inference Attack
# mia_score = run_mia_attack(member_test, non_member_test, fake_data)
# print(f"🔒 MIA Attack Accuracy (higher = less private): {mia_score:.4f}")

# Plot the first feature (D=1 assumed) or slice index 0
plt.figure(figsize=(10, 12))
for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.plot(fake_sample[i, :, 0], label=f"Fake Sample {i}")
    plt.plot(normalize(real_sample[i, :, 0]), label=f"Real Sample {i}")
    plt.legend()
    plt.tight_layout()

plt.suptitle("Generated Time Series (First Feature)", fontsize=16, y=1.02)
plt.show()
