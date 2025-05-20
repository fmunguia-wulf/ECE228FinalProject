import torch
import torch.nn as nn
import numpy as np
from typing import List

from utils.constants import * 
from config.settings import * 
from scipy.optimize import minimize
import os

class PINN(nn.Module):
    def __init__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                 v1: np.ndarray, v5: np.ndarray, layers: List[int],
                 use_pde: bool = True):
        super(PINN, self).__init__()

        # Raw data
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.v1 = torch.tensor(v1, dtype=torch.float32)
        self.v5 = torch.tensor(v5, dtype=torch.float32)
        self.layers = layers
        self.use_pde = use_pde

        # Normalization constants (assume global or class constant)
        self.diff_norms = DIFF_NORMS

        # Compute bounds for normalization
        X = torch.cat([self.x, self.y, self.t], dim=1)
        self.lb = X.min(dim=0).values
        self.ub = X.max(dim=0).values

        # Define networks
        self.net_v1 = self.build_network(layers)
        self.net_v5 = self.build_network(layers)

        if use_pde:
            self.net_v2 = self.build_network(layers)
            self.net_v3 = self.build_network(layers)
            self.net_v4 = self.build_network(layers)
            self.loss_history = {'v1': [], 'v5': [], 'f1': [], 'f5': []}
        else:
            self.loss_history = {'v1': [], 'v5': []}

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
        """Setup PyTorch device configuration and move data/networks."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move models
        self.net_v1.to(self.device)
        self.net_v5.to(self.device)
        if self.use_pde:
            self.net_v2.to(self.device)
            self.net_v3.to(self.device)
            self.net_v4.to(self.device)

        # Move tensors
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.t = self.t.to(self.device)
        self.v1 = self.v1.to(self.device)
        self.v5 = self.v5.to(self.device)
        self.lb = self.lb.to(self.device)
        self.ub = self.ub.to(self.device)

    def forward(self, X: torch.Tensor, net: nn.Sequential) -> torch.Tensor:
        """Apply input normalization and forward pass through given network."""
        X_d = torch.maximum(self.ub - self.lb, torch.tensor(1e-6, device=self.device))
        H = 2.0 * (X - self.lb) / X_d - 1.0
        return net(H)

    def net_plasma(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Compute plasma variables and PDE terms (PyTorch version)."""
        # Concatenate inputs
        X = torch.cat([x, y, t], dim=1)
        
        # Core outputs
        v1 = self.forward(X, self.net_v1)
        v5 = self.forward(X, self.net_v5)
        
        if not self.use_pde:
            return v1, v5, None, None, None, None, None
        
        # Additional PDE networks
        v2 = self.forward(X, self.net_v2)
        v3 = self.forward(X, self.net_v3)
        v4 = self.forward(X, self.net_v4)
        
        # Derivatives
        v1_t = torch.autograd.grad(v1, t, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v1_x = torch.autograd.grad(v1, x, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v1_y = torch.autograd.grad(v1, y, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v2_x = torch.autograd.grad(v2, x, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        v2_y = torch.autograd.grad(v2, y, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        v5_t = torch.autograd.grad(v5, t, grad_outputs=torch.ones_like(v5), create_graph=True)[0]
        v5_x = torch.autograd.grad(v5, x, grad_outputs=torch.ones_like(v5), create_graph=True)[0]
        v5_y = torch.autograd.grad(v5, y, grad_outputs=torch.ones_like(v5), create_graph=True)[0]
        
        # Physics
        B = (0.22 + 0.68) / (0.68 + 0.22 + MINOR_RADIUS * x)
        pe = v1 * v5
        pe_y = torch.autograd.grad(pe, y, grad_outputs=torch.ones_like(pe), create_graph=True)[0]
        jp = v1 * ((TAU_T ** 0.5) * v4 - v3)
        
        # Log-transformed variables
        lnn = torch.log(v1)
        lnTe = torch.log(v5)
        
        # Higher-order derivatives
        D_lnn, D_lnTe = self.compute_high_order_derivs(lnn, lnTe, x, y)
        
        # Source terms
        S_n, S_Ee = self.compute_source_terms(x, v1, v5)
        
        # PDE residuals
        f_v1, f_v5 = self.compute_residuals(
        v1, v5, v1_t, v1_x, v1_y, v2_x, v2_y,
        v5_t, v5_x, v5_y, B, pe_y, jp, pe,
        D_lnn, D_lnTe, S_n, S_Ee
        )
        
        return v1, v5, v2, v3, v4, f_v1, f_v5
        
    def compute_high_order_derivs(self, lnn, lnTe, x, y):
        """Compute derivatives up to 4th order for diffusion in PyTorch."""
    
        def nth_derivative(f, var, n):
            """Recursively compute n-th derivative of f with respect to var."""
            for _ in range(n):
                grads = torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f),
                                            create_graph=True, retain_graph=True)[0]
                f = grads
            return f
    
        # Compute 4th derivatives
        lnn_xxxx = nth_derivative(lnn, x, 4)
        lnn_yyyy = nth_derivative(lnn, y, 4)
        lnTe_xxxx = nth_derivative(lnTe, x, 4)
        lnTe_yyyy = nth_derivative(lnTe, y, 4)
    
        # Diffusion terms (with normalization)
        Dx_lnn = -((50. / self.diff_norms['DiffX_norm']) ** 2) * lnn_xxxx
        Dy_lnn = -((50. / self.diff_norms['DiffY_norm']) ** 2) * lnn_yyyy
        D_lnn = Dx_lnn + Dy_lnn
    
        Dx_lnTe = -((50. / self.diff_norms['DiffX_norm']) ** 2) * lnTe_xxxx
        Dy_lnTe = -((50. / self.diff_norms['DiffY_norm']) ** 2) * lnTe_yyyy
        D_lnTe = Dx_lnTe + Dy_lnTe
    
        return D_lnn, D_lnTe
    
    def compute_source_terms(self, x, v1, v5):
        """Compute source terms with conditions."""
        # Source terms before applying conditions
        S_n = N_SRC_A * torch.exp(-((x - X_SRC) ** 2) / (2. * SIG_SRC ** 2))
        S_Ee = ENER_SRC_A * torch.exp(-((x - X_SRC) ** 2) / (2. * SIG_SRC ** 2))
    
        # Flatten to shape (batch,) for condition checks
        S_n0 = S_n[:, 0] if S_n.ndim > 1 else S_n
        S_Ee0 = S_Ee[:, 0] if S_Ee.ndim > 1 else S_Ee
        x0 = x[:, 0] if x.ndim > 1 else x
        v1_0 = v1[:, 0] if v1.ndim > 1 else v1
        v5_0 = v5[:, 0] if v5.ndim > 1 else v5
    
        # Cond 1: S_n, S_Ee > 0.01 → keep; else → 0.001
        S_n = torch.where(S_n0 > 0.01, S_n0, torch.full_like(S_n0, 0.001))
        S_Ee = torch.where(S_Ee0 > 0.01, S_Ee0, torch.full_like(S_Ee0, 0.001))
    
        # Cond 2: x > X_SRC → keep; else → 0.5
        S_n = torch.where(x0 > X_SRC, S_n, torch.full_like(S_n, 0.5))
        S_Ee = torch.where(x0 > X_SRC, S_Ee, torch.full_like(S_Ee, 0.5))
    
        # Cond 4: v1 > 5 → 0; else → keep
        S_n = torch.where(v1_0 > 5.0, torch.zeros_like(S_n), S_n)
    
        # Cond 4: v5 > 1 → 0; else → keep
        S_Ee = torch.where(v5_0 > 1.0, torch.zeros_like(S_Ee), S_Ee)
    
        return S_n, S_Ee
    
    def compute_residuals(self, v1, v5, v1_t, v1_x, v1_y, v2_x, v2_y,
                          v5_t, v5_x, v5_y, B, pe_y, jp, pe, D_lnn, D_lnTe, S_n, S_Ee):
        """Compute PDE residuals (PyTorch version)."""
    
        f_v1 = v1_t + (1. / B) * (v2_y * v1_x - v2_x * v1_y) - (
            -EPS_R * (v1 * v2_y - ALPHA_D * pe_y) + S_n + v1 * D_lnn
        )
    
        f_v5 = v5_t + (1. / B) * (v2_y * v5_x - v2_x * v5_y) - v5 * (
            (5. * EPS_R * ALPHA_D * v5_y) / 3. +
            (2. / 3.) * (
                -EPS_R * (v2_y - ALPHA_D * pe_y / v1) +
                (1. / v1) * (
                    0.71 * EPS_V * 0.0 +  # You may replace `0.0` with a term later
                    ETA * jp * jp / (v5 * MASS_RATIO)
                )
            ) + (2. / (3. * pe)) * S_Ee + D_lnTe
        )
    
        return f_v1, f_v5
    
    def train_step(self, loss_fn, optimizer_v1, optimizer_v5, optimizer_f=None):
        """Perform one training step using PyTorch."""
        # === 1. Sample a batch ===
        idx_batch = np.random.choice(len(self.x), SAMPLE_BATCH_SIZE, replace=False)
        x_b = self.x[idx_batch].clone().detach().requires_grad_(True)
        y_b = self.y[idx_batch].clone().detach().requires_grad_(True)
        t_b = self.t[idx_batch].clone().detach().requires_grad_(True)
        v1_target = self.v1[idx_batch]
        v5_target = self.v5[idx_batch]
    
        # === 2. Zero gradients ===
        optimizer_v1.zero_grad()
        optimizer_v5.zero_grad()
        if self.use_pde:
            optimizer_f.zero_grad()
    
        # === 3. Forward pass and compute losses ===
        v1_pred, v5_pred, v2, v3, v4, f_v1, f_v5 = self.net_plasma(x_b, y_b, t_b)
    
        loss_v1 = loss_fn(v1_pred, v1_target)
        loss_v5 = loss_fn(v5_pred, v5_target)
    
        if self.use_pde:
            loss_f1 = loss_fn(f_v1, torch.zeros_like(f_v1))
            loss_f5 = loss_fn(f_v5, torch.zeros_like(f_v5))
            total_loss = loss_v1 + loss_v5 + loss_f1 + loss_f5
        else:
            total_loss = loss_v1 + loss_v5
    
        # === 4. Backward and optimizer step ===
        total_loss.backward()
        optimizer_v1.step()
        optimizer_v5.step()
        if self.use_pde:
            optimizer_f.step()
    
        # === 5. Record and return loss values ===
        self.loss_history['v1'].append(loss_v1.item())
        self.loss_history['v5'].append(loss_v5.item())
    
        if self.use_pde:
            self.loss_history['f1'].append(loss_f1.item())
            self.loss_history['f5'].append(loss_f5.item())
            return {
                'v1': loss_v1.item(), 'v5': loss_v5.item(),
                'f1': loss_f1.item(), 'f5': loss_f5.item(),
                'total': total_loss.item()
            }
        else:
            return {
                'v1': loss_v1.item(), 'v5': loss_v5.item(),
                'total': total_loss.item()
            }
    
    def setup_optimizers(self, lr=1e-3):
        """Setup Adam optimizers for PyTorch networks."""
        # Optimizers for supervised data fits
        self.optimizer_v1 = torch.optim.Adam(self.net_v1.parameters(), lr=lr)
        self.optimizer_v5 = torch.optim.Adam(self.net_v5.parameters(), lr=lr)
    
        if self.use_pde:
            self.optimizer_f = torch.optim.Adam(
                list(self.net_v2.parameters()) +
                list(self.net_v3.parameters()) +
                list(self.net_v4.parameters()),
                lr=lr
            )
    def xavier_init(self, layer):
        """Apply Xavier initialization to a PyTorch Linear layer."""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    @torch.no_grad()
    def predict(self, x_star: np.ndarray, y_star: np.ndarray, t_star: np.ndarray) -> dict:
        """Make predictions using trained PyTorch model."""
        self.eval()
    
        # Convert inputs to torch tensors on device
        x = torch.tensor(x_star, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_star, dtype=torch.float32).to(self.device)
        t = torch.tensor(t_star, dtype=torch.float32).to(self.device)
    
        v1, v5, v2, v3, v4, _, _ = self.net_plasma(x.requires_grad_(), y.requires_grad_(), t.requires_grad_())
    
        result = {'v1': v1.cpu().numpy(), 'v5': v5.cpu().numpy()}
        if self.use_pde:
            result.update({
                'v2': v2.cpu().numpy(), 'v3': v3.cpu().numpy(), 'v4': v4.cpu().numpy()
            })
        return result
    
    def save(self, save_path: str):
        """Save PyTorch model and metadata."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, 'model.pth'))
    
        # Save metadata
        metadata = {
            'layers': self.layers,
            'use_pde': self.use_pde,
            'lb': self.lb.cpu().numpy(),
            'ub': self.ub.cpu().numpy(),
            'diff_norms': self.diff_norms,
            'loss_history': self.loss_history
        }
        np.savez(os.path.join(save_path, 'metadata.npz'), **metadata)
    
    def load(self, load_path: str):
        """Load PyTorch model and metadata."""
        # Load model weights
        self.load_state_dict(torch.load(os.path.join(load_path, 'model.pth'), map_location=self.device))
    
        # Load metadata
        metadata = dict(np.load(os.path.join(load_path, 'metadata.npz'), allow_pickle=True))
    
        if isinstance(metadata['layers'], np.ndarray):
            if metadata['layers'].dtype == np.dtype('O'):
                self.layers = metadata['layers'].item()
            else:
                self.layers = metadata['layers'].tolist()
        else:
            self.layers = metadata['layers']
    
        self.use_pde = bool(metadata['use_pde'].item() if isinstance(metadata['use_pde'], np.ndarray) else metadata['use_pde'])
        self.lb = torch.tensor(metadata['lb'], dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(metadata['ub'], dtype=torch.float32).to(self.device)
        self.diff_norms = metadata['diff_norms'].item() if isinstance(metadata['diff_norms'], np.ndarray) else metadata['diff_norms']
        self.loss_history = metadata['loss_history'].item() if isinstance(metadata['loss_history'], np.ndarray) else metadata['loss_history']

