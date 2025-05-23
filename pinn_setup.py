import torch
import torch.nn as nn
import numpy as np
from typing import List

from utils.constants import *
from config.settings import *
import os

class PINN(nn.Module):
    def __init__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                 ne: np.ndarray, Te: np.ndarray, layers: List[int],
                 use_pde: bool = True):
        super(PINN, self).__init__()

        # Training data
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.ne = torch.tensor(ne, dtype=torch.float32)
        self.Te = torch.tensor(Te, dtype=torch.float32)
        
        self.layers = layers
        self.use_pde = use_pde

        # Normalized diffusion coefficients
        self.diff_norms = DIFF_NORMS

        # Define networks
        self.net_ne = self.build_network(layers)
        self.net_Te = self.build_network(layers)

        if use_pde:
            self.net_phi = self.build_network(layers)
            self.net_v3 = self.build_network(layers)
            self.net_v4 = self.build_network(layers)
            self.loss_history = {
                'ne': [], 'Te': [], 'ne_pde': [], 'Te_pde': []
            }
        else:
            self.loss_history = {'ne': [], 'Te': []}

        # Placeholder for optimizers
        self.optimizers = {}

        # Setup device and move everything
        self.setup_device()

    def build_network(self, layers: List[int]):
        """Build a fully connected neural network with Tanh activation."""
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                net.append(nn.Tanh())
        return nn.Sequential(*net)

    def setup_device(self):
        """Setup device configuration and move data/networks."""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Move models
        self.net_ne.to(self.device)
        self.net_Te.to(self.device)
        if self.use_pde:
            self.net_phi.to(self.device)
            self.net_v3.to(self.device)
            self.net_v4.to(self.device)

        # Move tensors
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.t = self.t.to(self.device)
        self.ne = self.ne.to(self.device)
        self.Te = self.Te.to(self.device)

    def forward(self, X: torch.Tensor, net: nn.Sequential) -> torch.Tensor:
        """Forward pass through given network."""
        return net(X)

    def net_plasma(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Compute plasma variables and PDE terms."""
        # Concatenate inputs
        X = torch.cat([x, y, t], dim=1)

        # NNs predict log values from (already normalized) inputs,
        # preserving positivity in predictions
        log_ne = self.forward(X, self.net_ne)
        log_Te = self.forward(X, self.net_Te)
        ne = torch.exp(log_ne)
        Te = torch.exp(log_Te)

        # Original
        # # Core outputs
        # ne = self.forward(X, self.net_ne)
        # Te = self.forward(X, self.net_Te)

        # SCRATCH
        # # NNs predict normalized values from (already normalized) inputs
        # ne_norm = self.forward(X, self.net_ne)
        # Te_norm = self.forward(X, self.net_Te)
        # # Unnormalize predictions for output and use in PDE losses
        # ne = ne_norm * (4e19 - 4e18) + 4e18
        # Te = Te_norm * (80 - 8) + 8
        
        if not self.use_pde:
            return ne, Te, None, None, None, None, None
        
        # Additional PDE networks
        phi = self.forward(X, self.net_phi)
        v3 = self.forward(X, self.net_v3)
        v4 = self.forward(X, self.net_v4)
        
        # Derivatives
        ne_t = torch.autograd.grad(
            ne, t, grad_outputs=torch.ones_like(ne), create_graph=True
        )[0]
        ne_x = torch.autograd.grad(
            ne, x, grad_outputs=torch.ones_like(ne), create_graph=True
        )[0]
        ne_y = torch.autograd.grad(
            ne, y, grad_outputs=torch.ones_like(ne), create_graph=True
        )[0]
        phi_x = torch.autograd.grad(
            phi, x, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
        phi_y = torch.autograd.grad(
            phi, y, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
        Te_t = torch.autograd.grad(
            Te, t, grad_outputs=torch.ones_like(Te), create_graph=True
        )[0]
        Te_x = torch.autograd.grad(
            Te, x, grad_outputs=torch.ones_like(Te), create_graph=True
        )[0]
        Te_y = torch.autograd.grad(
            Te, y, grad_outputs=torch.ones_like(Te), create_graph=True
        )[0]
        
        # Physics
        B = (0.22 + 0.68) / (0.68 + 0.22 + MINOR_RADIUS * x)
        pe = ne * Te
        pe_y = torch.autograd.grad(
            pe, y, grad_outputs=torch.ones_like(pe), create_graph=True
        )[0]
        jp = ne * ((TAU_T ** 0.5) * v4 - v3)
        
        # # Log-transformed variables
        # log_ne = torch.log(ne)
        # log_Te = torch.log(Te)
        
        # Higher-order derivatives
        D_log_ne, D_log_Te = self.compute_high_order_derivs(log_ne, log_Te, x, y)
        
        # Source terms
        S_n, S_Ee = self.compute_source_terms(x, ne, Te)
        
        # PDE residuals
        f_ne, f_Te = self.compute_residuals(
            ne, Te, ne_t, ne_x, ne_y, phi_x, phi_y,
            Te_t, Te_x, Te_y, B, pe_y, jp, pe,
            D_log_ne, D_log_Te, S_n, S_Ee
        )
        
        return ne, Te, phi, v3, v4, f_ne, f_Te
        
    def compute_high_order_derivs(self, log_ne, log_Te, x, y):
        """Compute derivatives up to 4th order for diffusion."""
    
        def nth_derivative(f, var, n):
            """Recursively compute n-th derivative of f with respect to
            var."""
            for _ in range(n):
                grads = torch.autograd.grad(
                    f, var, grad_outputs=torch.ones_like(f),
                    create_graph=True, retain_graph=True
                )[0]
                f = grads
            return f
    
        # Compute 4th derivatives
        log_ne_xxxx = nth_derivative(log_ne, x, 4)
        log_ne_yyyy = nth_derivative(log_ne, y, 4)
        log_Te_xxxx = nth_derivative(log_Te, x, 4)
        log_Te_yyyy = nth_derivative(log_Te, y, 4)
    
        # Diffusion terms (with normalization)
        Dx_log_ne = -((50. / self.diff_norms['DiffX_norm']) ** 2) * log_ne_xxxx
        Dy_log_ne = -((50. / self.diff_norms['DiffY_norm']) ** 2) * log_ne_yyyy
        D_log_ne = Dx_log_ne + Dy_log_ne
    
        Dx_log_Te = -((50. / self.diff_norms['DiffX_norm']) ** 2) * log_Te_xxxx
        Dy_log_Te = -((50. / self.diff_norms['DiffY_norm']) ** 2) * log_Te_yyyy
        D_log_Te = Dx_log_Te + Dy_log_Te
    
        return D_log_ne, D_log_Te
    
    def compute_source_terms(self, x, ne, Te):
        """Compute source terms with conditions."""
        # Source terms before applying conditions
        S_n = N_SRC_A * torch.exp(
            -((x - X_SRC) ** 2) / (2. * SIG_SRC ** 2)
        )
        S_Ee = ENER_SRC_A * torch.exp(
            -((x - X_SRC) ** 2) / (2. * SIG_SRC ** 2)
        )
    
        # Flatten to shape (batch,) for condition checks
        S_n0 = S_n[:, 0] if S_n.ndim > 1 else S_n
        S_Ee0 = S_Ee[:, 0] if S_Ee.ndim > 1 else S_Ee
        x0 = x[:, 0] if x.ndim > 1 else x
        ne_0 = ne[:, 0] if ne.ndim > 1 else ne
        Te_0 = Te[:, 0] if Te.ndim > 1 else Te
    
        # Cond 1: S_n, S_Ee > 0.01 → keep; else → 0.001
        S_n = torch.where(S_n0 > 0.01, S_n0, torch.full_like(S_n0, 0.001))
        S_Ee = torch.where(S_Ee0 > 0.01, S_Ee0, torch.full_like(S_Ee0, 0.001))
    
        # Cond 2: x > X_SRC → keep; else → 0.5
        S_n = torch.where(x0 > X_SRC, S_n, torch.full_like(S_n, 0.5))
        S_Ee = torch.where(x0 > X_SRC, S_Ee, torch.full_like(S_Ee, 0.5))
    
        # Cond 4: ne > 5 → 0; else → keep
        S_n = torch.where(ne_0 > 5.0, torch.zeros_like(S_n), S_n)
    
        # Cond 4: Te > 1 → 0; else → keep
        S_Ee = torch.where(Te_0 > 1.0, torch.zeros_like(S_Ee), S_Ee)
    
        return S_n, S_Ee
    
    def compute_residuals(self, ne, Te, ne_t, ne_x, ne_y, phi_x, phi_y,
                          Te_t, Te_x, Te_y, B, pe_y, jp, pe, D_log_ne, D_log_Te,
                          S_n, S_Ee):
        """Compute PDE residuals."""
        f_ne = ne_t + (1. / B) * (phi_y * ne_x - phi_x * ne_y) - (
            -EPS_R * (ne * phi_y - ALPHA_D * pe_y) + S_n + ne * D_log_ne
        )
    
        f_Te = Te_t + (1. / B) * (phi_y * Te_x - phi_x * Te_y) - Te * (
            (5. * EPS_R * ALPHA_D * Te_y) / 3. +
            (2. / 3.) * (
                -EPS_R * (phi_y - ALPHA_D * pe_y / ne) +
                (1. / ne) * (
                    0.71 * EPS_V * 0.0 +  # May replace `0.0` with a term later
                    ETA * jp * jp / (Te * MASS_RATIO)
                )
            ) + (2. / (3. * pe)) * S_Ee + D_log_Te
        )
    
        return f_ne, f_Te
    
    def train_step(self, loss_fn, optimizer_ne, optimizer_Te,
                   optimizer_f=None):
        """Perform one training step."""
        # === 1. Sample a batch (without replacement) ===
        idx_batch = np.random.choice(
            len(self.x), SAMPLE_BATCH_SIZE, replace=False
        )
        x_b = self.x[idx_batch].clone().detach().requires_grad_(True)
        y_b = self.y[idx_batch].clone().detach().requires_grad_(True)
        t_b = self.t[idx_batch].clone().detach().requires_grad_(True)
        ne_target = self.ne[idx_batch]
        Te_target = self.Te[idx_batch]
    
        # === 2. Zero gradients ===
        optimizer_ne.zero_grad()
        optimizer_Te.zero_grad()
        if self.use_pde:
            optimizer_f.zero_grad()
    
        # === 3. Forward pass and compute losses ===
        ne_pred, Te_pred, _, _, _, f_ne, f_Te = self.net_plasma(
            x_b, y_b, t_b
        )

        # Evaluate data-fit losses on log values
        loss_ne = loss_fn(torch.log(ne_pred), torch.log(ne_target))
        loss_Te = loss_fn(torch.log(Te_pred), torch.log(Te_target))

        # SCRATCH
        # # Evaluate data-fit losses on normalized values
        # ne_pred_norm = (ne_pred - 4e18) / (4e19 - 4e18)
        # ne_target_norm = (ne_target - 4e18) / (4e19 - 4e18)
        # Te_pred_norm = (Te_pred - 8) / (80 - 8)
        # Te_target_norm = (Te_target - 8) / (80 - 8)
        # loss_ne = loss_fn(ne_pred_norm, ne_target_norm)
        # loss_Te = loss_fn(Te_pred_norm, Te_target_norm)

        # Original
        # loss_ne = loss_fn(ne_pred, ne_target)
        # loss_Te = loss_fn(Te_pred, Te_target)
    
        if self.use_pde:
            ne_pde_loss = loss_fn(f_ne, torch.zeros_like(f_ne))
            Te_pde_loss = loss_fn(f_Te, torch.zeros_like(f_Te))
            total_loss = loss_ne + loss_Te + ne_pde_loss + Te_pde_loss
        else:
            total_loss = loss_ne + loss_Te
    
        # === 4. Backward and optimizer step ===
        total_loss.backward()
        optimizer_ne.step()
        optimizer_Te.step()
        if self.use_pde:
            optimizer_f.step()
    
        # === 5. Record and return loss values ===
        self.loss_history['ne'].append(loss_ne.item())
        self.loss_history['Te'].append(loss_Te.item())
    
        if self.use_pde:
            self.loss_history['ne_pde'].append(ne_pde_loss.item())
            self.loss_history['Te_pde'].append(Te_pde_loss.item())
            return {
                'ne': loss_ne.item(), 'Te': loss_Te.item(),
                'ne_pde': ne_pde_loss.item(), 'Te_pde': Te_pde_loss.item(),
                'total': total_loss.item()
            }
        else:
            return {
                'ne': loss_ne.item(), 'Te': loss_Te.item(),
                'total': total_loss.item()
            }
    
    def setup_optimizers(self, lr=1e-3):
        """Setup Adam optimizers for networks."""
        # Optimizers for supervised data fits
        self.optimizer_ne = torch.optim.Adam(self.net_ne.parameters(), lr=lr)
        self.optimizer_Te = torch.optim.Adam(self.net_Te.parameters(), lr=lr)
    
        if self.use_pde:
            self.optimizer_f = torch.optim.Adam(
                list(self.net_phi.parameters()) +
                list(self.net_v3.parameters()) +
                list(self.net_v4.parameters()),
                lr=lr
            )
    
    def xavier_init(self, layer):
        """Apply Xavier initialization to a Linear layer."""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    @torch.no_grad()
    def predict(self, x_star: np.ndarray, y_star: np.ndarray,
                t_star: np.ndarray) -> dict:
        """Make predictions using trained model."""
        self.eval()
    
        # Convert inputs to torch tensors on device
        x = torch.tensor(x_star, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_star, dtype=torch.float32).to(self.device)
        t = torch.tensor(t_star, dtype=torch.float32).to(self.device)
    
        # Predict: Below is copied from net_plasma to avoid computing gradients
        # and clashing with no_grad decorator
        # Concatenate inputs
        X = torch.cat([x, y, t], dim=1)

        # NNs predict log values from (already normalized) inputs,
        # preserving positivity in predictions
        log_ne = self.forward(X, self.net_ne)
        log_Te = self.forward(X, self.net_Te)
        ne = torch.exp(log_ne)
        Te = torch.exp(log_Te)

        # Original
        # ne, Te, phi, v3, v4, _, _ = self.net_plasma(
        #     x.requires_grad_(), y.requires_grad_(), t.requires_grad_()
        # )

        if self.use_pde:
            phi = self.forward(X, self.net_phi)
            v3 = self.forward(X, self.net_v3)
            v4 = self.forward(X, self.net_v4)
        # ----------
    
        result = {'ne': ne.cpu().numpy(), 'Te': Te.cpu().numpy()}
        if self.use_pde:
            result.update({
                'phi': phi.cpu().numpy(), 'v3': v3.cpu().numpy(),
                'v4': v4.cpu().numpy()
            })
        return result
    
    def save(self, save_path: str):
        """Save model and metadata."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, 'model.pth'))
    
        # Save metadata
        metadata = {
            'layers': self.layers,
            'use_pde': self.use_pde,
            'diff_norms': self.diff_norms,
            'loss_history': self.loss_history
        }
        np.savez(os.path.join(save_path, 'metadata.npz'), **metadata)
    
    def load(self, load_path: str):
        """Load model and metadata."""
        # Load model weights
        self.load_state_dict(
            torch.load(
                os.path.join(load_path, 'model.pth'),
                map_location=self.device
            )
        )
    
        # Load metadata
        metadata = dict(
            np.load(
                os.path.join(load_path, 'metadata.npz'), allow_pickle=True
            )
        )
    
        if isinstance(metadata['layers'], np.ndarray):
            if metadata['layers'].dtype == np.dtype('O'):
                self.layers = metadata['layers'].item()
            else:
                self.layers = metadata['layers'].tolist()
        else:
            self.layers = metadata['layers']
    
        self.use_pde = bool(
            metadata['use_pde'].item()
            if isinstance(metadata['use_pde'], np.ndarray)
            else metadata['use_pde']
        )
        if isinstance(metadata['diff_norms'], np.ndarray):
            self.diff_norms = metadata['diff_norms'].item()
        else:
            self.diff_norms = metadata['diff_norms']
        
        if isinstance(metadata['loss_history'], np.ndarray):
            self.loss_history = metadata['loss_history'].item()
        else:
            self.loss_history = metadata['loss_history']
