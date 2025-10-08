import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainInvariantModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_domains):
        super(DomainInvariantModel, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        task_output = self.task_classifier(features)
        domain_output = self.domain_classifier(features)
        return task_output, domain_output
