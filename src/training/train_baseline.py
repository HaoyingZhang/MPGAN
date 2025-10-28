# GLOBAL IMPORTS
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"
import stumpy
import sys, time
import torch.autograd as autograd
import pandas as pd

# LOCAL IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # Add root directory to path
from src.training.objectives import objective_function_unified

### --- WGAN-GP LOOP --- ###
def train_wgan_gp(train_loader, G, D_net, device='cuda', checkpoint_path="tmp.pth", epoch=10,
                  mp_window_size=10, k_violation=1.0, alpha=0.05, objective_func=None,
                  time_limit=30, lambda_gp=10.0, critic_iters=5):
    os.makedirs(checkpoint_path, exist_ok=True)
    G, D_net = G.to(device), D_net.to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=1e-4, betas=(0.5, 0.9))

    best_g_loss = float('inf')
    D_loss, G_loss = [], []
    start_time = time.time()

    def compute_gradient_penalty(D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones_like(d_interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    for ep in range(epoch):
        if time.time() - start_time >= time_limit:
            print(f"Stopping early at epoch {ep} due to time limit ({time_limit}s).")
            break

        epoch_d_losses, epoch_g_losses = [], []

        for real_batch, _ in train_loader:
            if real_batch.shape[0] == 0:
                continue
            real_batch = real_batch.to(device)
            B, n, _ = real_batch.shape
            m = mp_window_size
            mp_input_list, real_series_list = [], []

            for ts in real_batch.squeeze(-1).cpu().numpy().astype(np.float64):
                mp = stumpy.stump(ts, m=mp_window_size)
                if mp.shape[0] == 0: continue
                mp_clean = np.stack([np.nan_to_num(mp[:, 0]), np.nan_to_num(mp[:, 1])], axis=1)
                mp_clean = np.asarray(mp_clean, dtype=np.float32)
                mp_input_list.append(torch.tensor(mp_clean, dtype=torch.float32))
                real_series_list.append(torch.tensor(ts, dtype=torch.float32))

            if not mp_input_list: continue

            mp_input_batch = torch.stack(mp_input_list).to(device)
            real_batch = torch.stack(real_series_list).unsqueeze(-1).to(device)

            # --- Train Critic (Discriminator) ---
            for _ in range(critic_iters):
                with torch.no_grad():
                    fake_batch = G(mp_input_batch).detach()
                d_real = D_net(real_batch)
                d_fake = D_net(fake_batch)
                gp = compute_gradient_penalty(D_net, real_batch, fake_batch)
                d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            # --- Train Generator ---
            fake_batch = G(mp_input_batch)
            d_output = D_net(fake_batch)
            g_adv_loss = -d_output.mean()

            fake_series_list = [fb.squeeze(-1) for fb in fake_batch]

            obj_func_name = objective_func.__name__
            if obj_func_name == "objective_function_pytorch":
                mp_loss = objective_function_pytorch(
                    x_list=fake_series_list,
                    mp_list=mp_input_list,
                    m=m,
                    coeff_dist=1.0,
                    coeff_identity=2.0,
                    k = k_violation,
                    device=device
                )
            elif obj_func_name == "objective_function_exponential_pytorch":
                mp_loss = objective_function_exponential_pytorch(
                    x_list=fake_series_list,
                    mp_list=mp_input_list,
                    m=m,
                    coeff_dist=1.0,
                    coeff_identity=2.0,
                    device=device,
                    alpha=alpha
                )

            g_loss_total = g_adv_loss + mp_loss
            optimizer_G.zero_grad()
            g_loss_total.backward()
            optimizer_G.step()

            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss_total.item())

        if epoch_g_losses and epoch_d_losses:
            G_loss.append(np.mean(epoch_g_losses))
            D_loss.append(np.mean(epoch_d_losses))
            print(f"Epoch {ep+1}: D_loss={D_loss[-1]:.4f}, G_loss={G_loss[-1]:.4f}")

            if G_loss[-1] < best_g_loss:
                best_g_loss = G_loss[-1]
                torch.save(G.state_dict(), os.path.join(checkpoint_path, "best_model.pth"))
                print(f"✅ Saved new best generator at epoch {ep+1} with G_loss={best_g_loss:.4f}")

    return G, D_net, D_loss, G_loss

def to_conv1d_shape(x, device):
    # Move to device first
    x = x.to(device)
    # Collapse any singleton dims at the end or at dim=1
    # E.g. turns (B,n,1), (B,1,n), (B,1,n,1), etc → (B,n)
    x = x.squeeze(-1).squeeze(1)
    # Now insert the channel dimension at dim=1:
    # (B,n) → (B,1,n)
    return x.unsqueeze(1)

### --- TRAINING LOOP --- ###
def train_gan(
    train_loader,
    G,
    D_net,
    device='cuda',
    checkpoint_path="tmp.pth",
    epoch=10,
    mp_window_size=10,
    k_violation=1.00,
    alpha=0.5,
    activ_func="relu",
    time_limit=30,
    d_model="lstm",
    latent = False,
    pi_mp=0.05,
    pi_adv=1,
    coeff_dist = 1.0,
    coeff_identity = 1.0,
    lr_G=2e-4,             
    lr_D=2e-4
):
    os.makedirs(checkpoint_path, exist_ok=True)

    # Move models once
    G      = G.to(device)
    D_net  = D_net.to(device)

    optimizer_G = torch.optim.Adam(G.parameters(),     lr=lr_G)
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=lr_D)
    criterion   = nn.BCELoss()

    best_g_loss = float('inf')
    D_loss, G_loss, ADV_G_loss, MP_loss, TS_loss = [], [], [], [], []

    start_time = time.time()
    for ep in range(epoch):
        if time.time() - start_time >= time_limit:
            print(f"⏱️  Stopping early at epoch {ep} (time limit).")
            break

        epoch_d, epoch_g, epoch_adv_g, epoch_mp = [], [], [], []

        for mp_input_batch, time_series_batch in train_loader: # real_batch.shape = [10, 200, 1]
            # print(mp_input_batch.shape)
            # print(time_series_batch.shape)
            # Skip empty batches
            if mp_input_batch.size(0) == 0:
                continue

            mp_input_batch = mp_input_batch.to(device)
            if latent :
                z = torch.randn(mp_input_batch.size(0), 64, device=device) if G.z_dim else None
                with torch.no_grad():
                    fake = G(mp_input_batch, z=z)
            else:
                with torch.no_grad():
                    fake = G(mp_input_batch)

            # -------- Discriminator train ---------#
            if d_model == "pulse2pulse":
                fake = to_conv1d_shape(fake, device)
                time_series_batch = to_conv1d_shape(time_series_batch, device)
            else:
                fake = fake.unsqueeze(-1)
                fake = fake.to(device)
                time_series_batch = time_series_batch.unsqueeze(-1)
                time_series_batch = time_series_batch.to(device)   

            d_real = D_net(time_series_batch)
            d_fake = D_net(fake)

            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            loss_real = criterion(d_real, real_labels)
            loss_fake = criterion(d_fake, fake_labels)
            d_loss = loss_real + loss_fake

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            epoch_d.append(d_loss.item())

            # --- Generator Train ---#
            optimizer_G.zero_grad()

            if latent:
                z = torch.randn(mp_input_batch.size(0), 64, device=device) if G.z_dim else None
                fake_for_g = G(mp_input_batch, z=z)     # fresh graph
            else:
                fake_for_g = G(mp_input_batch)

            if d_model == "pulse2pulse":
                fake_g_in = to_conv1d_shape(fake_for_g, device)  # for D forward
            else:
                fake_g_in = fake_for_g.unsqueeze(-1)             # [B, n, 1]

            d_fake_for_g = D_net(fake_g_in)
            g_adv_loss = criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            ts_loss = ((fake_g_in - time_series_batch)**2).mean()
            TS_loss.append(ts_loss.item())
            epoch_adv_g.append(g_adv_loss.item())
            # TODO: add the distance with the real ts 
            
            # MP loss
            fake_series_list = [fb[0] if d_model=="pulse2pulse" else fb.squeeze(-1) 
                                for fb in fake_for_g]
            mp_loss = objective_function_unified(
                x_list=fake_series_list,
                mp_list=mp_input_batch,
                m=mp_window_size,
                coeff_dist=coeff_dist,
                coeff_identity=coeff_identity,
                k=k_violation,
                device=device,
                alpha=alpha,
                identity_activation=activ_func
            )
            epoch_mp.append(mp_loss.item())

            g_loss = (pi_adv* g_adv_loss + pi_mp * mp_loss + 10 * ts_loss)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            epoch_g.append(g_loss.item())
        
        D_loss.append(np.mean(epoch_d))
        G_loss.append(np.mean(epoch_g))
        ADV_G_loss.append(np.mean(epoch_adv_g))
        MP_loss.append(np.mean(epoch_mp))
        print(f"Epoch {ep+1}: adv={ADV_G_loss[-1]:.4f}, mp={MP_loss[-1]:.4f}, "
      f"G={G_loss[-1]:.4f}, D={D_loss[-1]:.4f}")

        # checkpoint best G
        if G_loss[-1] < best_g_loss:
            best_g_loss = G_loss[-1]
            torch.save(G.state_dict(), os.path.join(checkpoint_path, "best_model.pth"))
            print(f"✅ Saved best G (epoch {ep+1}, G_loss={best_g_loss:.4f})")
        loss_data = {"g_adv_loss":ADV_G_loss, "mp_loss":MP_loss, "G_loss":G_loss, "D_loss": D_loss}
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(os.path.join(checkpoint_path,"loss.csv"), index=False)
    return G, D_net, D_loss, G_loss, g_adv_loss, mp_loss
