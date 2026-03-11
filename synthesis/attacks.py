# synthesis/attacks.py
import torch
import torch.nn as nn

class PGDAttacker:
    def __init__(self, system, controller, lyapunov, config):
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.bounds = config.state_bounds  # Tensor shape [x_dim, 2]
        self.steps = config.pgd_steps      # e.g., 10 to 50
        self.alpha = config.pgd_step_size  # e.g., 0.01
        self.kappa = config.kappa          # e.g., 0.1
        self.penalty_weight = 10.0         # lambda parameter

    def find_counter_examples(self, batch_size, rho):
        """
        Thực thi thuật toán PGD.
        Returns: Tensor x_adv kích thước (batch_size, x_dim)
        """
        x_dim = self.bounds.shape[0]
        min_bounds = self.bounds[:, 0]  # Shape: [x_dim]
        max_bounds = self.bounds[:, 1]  # Shape: [x_dim]

        # 1. Initialization: uniform random sample within physical bounds
        # x shape: (batch_size, x_dim)
        x = torch.rand(batch_size, x_dim) * (max_bounds - min_bounds) + min_bounds
        x = x.detach()

        # 2. Attack Loop
        for k in range(self.steps):
            x.requires_grad_(True)

            # Forward pass
            u = self.controller(x)                   # (batch_size, u_dim)
            x_next = self.system.step(x, u)          # (batch_size, x_dim)

            v_current = self.lyapunov(x)             # (batch_size,) or (batch_size, 1)
            v_next = self.lyapunov(x_next)           # (batch_size,) or (batch_size, 1)

            # Flatten in case lyapunov returns (batch_size, 1)
            v_current = v_current.squeeze(-1)        # (batch_size,)
            v_next = v_next.squeeze(-1)              # (batch_size,)

            # Calculate Lyapunov decrease violation
            # F_x > 0  ⟹  the decrease condition V(x') ≤ (1-κ)V(x) is violated
            F_x = v_next - (1.0 - self.kappa) * v_current  # (batch_size,)

            # Penalty: push samples back inside the ρ-sublevel set during search
            # ReLU(V(x) - ρ) is 0 when already inside, grows linearly outside
            penalty = self.penalty_weight * torch.relu(v_current - rho)  # (batch_size,)

            # Attack objective: maximise violation, penalise out-of-set samples
            # We take the mean so the gradient scale is batch-size independent
            attack_loss = (F_x - penalty).mean()

            # Backward pass w.r.t. input x
            attack_loss.backward()

            with torch.no_grad():
                grad_x = x.grad  # (batch_size, x_dim)

                # Gradient Ascent Step (FGSM-style sign update for stability)
                x_new = x + self.alpha * grad_x.sign()

                # Project back into physical bounds B
                x = torch.clamp(x_new, min=min_bounds, max=max_bounds)

            # Detach to free the computational graph and prevent memory leaks
            x = x.detach()

        return x  # (batch_size, x_dim)  — adversarial counter-example candidates