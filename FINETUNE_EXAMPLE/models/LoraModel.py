import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

### 2. Create a LoRA-wrapped linear layer:

# ```python
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super(LoRALinear, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Keep the original layer
        self.linear = linear_layer
        
        # Add LoRA layer
        self.lora = LoRALayer(
            self.in_features, 
            self.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

### 3. Implement a function to replace linear layers with LoRA layers:

# ```python
def replace_with_lora(model, target_modules=None, rank=4, alpha=1, exclude_parents=None):
    """
    Replace linear layers in the model with LoRA layers.
    target_modules: List of module names to replace. If None, replace all Linear layers.
    """
    if target_modules is None:
        target_modules = ['Linear']
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            if exclude_parents is not None and name in exclude_parents:
                continue
            # Recursive call for nested modules
            replace_with_lora(module, target_modules, rank, alpha, exclude_parents)

        if isinstance(module, nn.Linear) and (
            target_modules is None or 
            any(t.lower() in name.lower() for t in target_modules)
        ):
            # Replace the linear layer with LoRA layer
            setattr(
                model, 
                name, 
                LoRALinear(module, rank=rank, alpha=alpha)
            )