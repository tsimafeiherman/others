from dataclasses import dataclass

@dataclass
class ModelConfig():
    
    d_model: int = 768
    layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    mlp_mult: int = 4
    n_ctx: int = 1024
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5
    debug: bool = True
    vocab_size: int = 50768
    theta: int = 10000
    
    n_experts: int = 8
    k_experts: int = 2
    capacity_factor: float = 1.2
    use_load_balancing: bool = True
    load_balancing_weight: float = 0.02
    
@dataclass
class TrainConfig():
    
    epochs: int = 10
    save_model_dir: str = "models_checkpoints"
    save_logs_dir: str = "logs"
    lr: float = 1e-2