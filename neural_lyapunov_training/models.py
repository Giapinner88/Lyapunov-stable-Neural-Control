import torch
import torch.nn as nn

# ==========================================
# 1. NEURAL CONTROLLER
# ==========================================
class NeuralController(nn.Module):
    def __init__(
        self,
        nx: int,
        nu: int,
        hidden_sizes: list = [64, 64],
        u_bound: float = 6.0,
        state_limits: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.u_bound = u_bound
        self.nx = nx
        
        if state_limits is None:
            state_limits = torch.ones(nx)
        state_limits_tensor = torch.as_tensor(state_limits, dtype=torch.float32)
        if state_limits_tensor.numel() != nx:
            raise ValueError(f"state_limits must have {nx} elements, got {state_limits_tensor.numel()}")
        self.register_buffer("state_limits", state_limits_tensor.view(nx))
        self.register_buffer("origin", torch.zeros(nx), persistent=False)
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, bias=True))
            layers.append(nn.Tanh())  
            in_size = h
            
        layers.append(nn.Linear(in_size, nu, bias=True))
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / self.state_limits

        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        u_raw = self.net(x_norm)
        u_origin = self.net(zero_norm).squeeze(0)
        unclipped_output = (u_raw - u_origin) * self.u_bound

        # THAY THẾ CHO: torch.clamp(unclipped_output, -self.u_bound, self.u_bound)
        u_bound_tensor = torch.tensor(self.u_bound, device=x.device)
        f1 = torch.nn.functional.relu(unclipped_output - (-u_bound_tensor)) + (-u_bound_tensor)
        u_out = -(torch.nn.functional.relu(u_bound_tensor - f1) - u_bound_tensor)
        return u_out

    def load_state_dict(self, state_dict, strict: bool = True):
        # Keep checkpoint biases when present; only backfill missing bias tensors.
        filtered = dict(state_dict)
        current = self.state_dict()
        for k, v in current.items():
            if k.endswith(".bias") and k not in filtered:
                filtered[k] = v
        return super().load_state_dict(filtered, strict=strict)


# ==========================================
# 2. NEURAL LYAPUNOV FUNCTION
# ==========================================
class NeuralLyapunov(nn.Module):
    def __init__(
        self,
        nx: int,
        hidden_sizes: list = [64, 64],
        eps: float = 0.01,
        phi_dim: int = 1,
        absolute_output: bool = True,
        state_limits: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps phải > 0 để đảm bảo V(x) dương xác định nghiêm ngặt")
        if phi_dim <= 0:
            raise ValueError("phi_dim must be >= 1")
        
        if state_limits is None:
            state_limits = torch.ones(nx)
        state_limits_tensor = torch.as_tensor(state_limits, dtype=torch.float32)
        if state_limits_tensor.numel() != nx:
            raise ValueError(f"state_limits must have {nx} elements, got {state_limits_tensor.numel()}")
        self.register_buffer("state_limits", state_limits_tensor.view(nx))
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, bias=True))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, phi_dim, bias=True))
        self.phi_V = nn.Sequential(*layers)
        self.absolute_output = bool(absolute_output)
        
        # FIX: Khởi tạo R lớn hơn (từ 0.1 → 0.5) để P = R^T*R kích thước hợp lý
        self.R = nn.Parameter(torch.randn(nx, nx) * 0.1)
        self.eps = eps
        
        self.register_buffer("eye", torch.eye(nx))
        self.register_buffer("origin", torch.zeros(nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        phi_0 = self.phi_V(zero_norm)

        x_norm = x / self.state_limits
        phi_x = self.phi_V(x_norm)
        
        # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
        diff = phi_x - phi_0
        if self.absolute_output:
            term1 = torch.sum(
                torch.nn.functional.relu(diff) + torch.nn.functional.relu(-diff),
                dim=1,
                keepdim=True,
            )
        else:
            term1 = torch.sum(diff, dim=1, keepdim=True)
        
        P = self.eps * self.eye + torch.matmul(self.R.T, self.R)
        Px = torch.matmul(x, P)
        term2 = torch.sum(Px * x, dim=1, keepdim=True)
        
        V = term1 + term2
        return V

    def quadratic_matrix(self) -> torch.Tensor:
        return self.eps * self.eye + torch.matmul(self.R.T, self.R)

    def algebraic_decrease(self, x_next: torch.Tensor, x: torch.Tensor, alpha_lyap: float) -> torch.Tensor:
        scale = 1.0 - alpha_lyap

        # 1) Quadratic part with cross-term-free formulation.
        P = self.quadratic_matrix()
        q_next = torch.sum(torch.matmul(x_next, P) * x_next, dim=1, keepdim=True)
        q_curr = torch.sum(torch.matmul(x, P) * x, dim=1, keepdim=True)
        quad_decrease = q_next - scale * q_curr

        # 2) Neural part of Lyapunov value.
        v_curr = self.forward(x)
        v_next = self.forward(x_next)
        v_nn_curr = v_curr - q_curr
        v_nn_next = v_next - q_next
        nn_decrease = v_nn_next - scale * v_nn_curr

        return quad_decrease + nn_decrease

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward compatibility: old checkpoints stored origin as shape [1, nx].
        # Keep checkpoint biases when present; only backfill if missing.
        state_dict = dict(state_dict)
        out_key = "phi_V.-1.weight"
        if out_key in state_dict:
            target = self.phi_V[-1].weight
            source = state_dict[out_key]
            if source.shape != target.shape:
                if source.ndim == 2 and target.ndim == 2 and source.shape[1] == target.shape[1]:
                    # Legacy checkpoints may have multi-dimensional phi output. Collapse by mean.
                    state_dict = state_dict.copy()
                    state_dict[out_key] = source.mean(dim=0, keepdim=True).to(dtype=target.dtype)
                else:
                    raise RuntimeError(
                        f"Incompatible shape for {out_key}: checkpoint {tuple(source.shape)} vs model {tuple(target.shape)}"
                    )
        origin_key = "origin"
        if origin_key in state_dict:
            origin_tensor = state_dict[origin_key]
            if origin_tensor.ndim == 2 and origin_tensor.shape[0] == 1:
                state_dict = state_dict.copy()
                state_dict[origin_key] = origin_tensor.squeeze(0)
        current = self.state_dict()
        for k, v in current.items():
            if k.endswith(".bias") and k not in state_dict:
                state_dict[k] = v
        return super().load_state_dict(state_dict, strict=strict)