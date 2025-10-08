import torch
import torch.nn.functional as F
import sys

def cosine_similarity_loss(feat, target, mask, eps=1e-8, stride=-1, verbose=0):
    """
    Compute the cosine similarity loss for masked tokens.
    
    Args:
        output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, feature_size).
        mask (torch.Tensor): The binary mask tensor of shape (batch_size, sequence_length) with 1 indicating a masked token.
        eps (float): A small value to avoid division by zero.
        
    Returns:
        torch.Tensor: The cosine similarity loss.
    """
    # batch_size, sequence_length, feature_size = output.shape
    target_norm = target / (target.norm(dim=-1, keepdim=True) + eps)
    # Normalize the output along the feature dimension
    output_norm = feat / (feat.norm(dim=-1, keepdim=True) + eps)
    
    bsz, tsz = mask.shape

    # Create a matrix of cosine similarities
    sim_matrix = torch.exp(torch.bmm(output_norm, target_norm.transpose(1, 2)) / 0.1)
    if stride == -1: # full_version
        sim_matrix = sim_matrix / sim_matrix.sum(-1).unsqueeze(-1).expand(bsz, tsz, tsz)
    else:
        target_v = torch.stack([torch.sum(torch.tensor([F.one_hot(torch.tensor(j), tsz).tolist() for j in range(i % stride, tsz, stride)]), 0) for i in range(tsz)])\
                    .unsqueeze(0) \
                    .expand(bsz, tsz, tsz) \
                    .to(device=sim_matrix.device)
        if verbose == 1:
            print(target_v[0, :, :])
        sim_matrix = sim_matrix / (sim_matrix * target_v).sum(-1).unsqueeze(-1).expand(bsz, tsz, tsz)

    sim_matrix = -torch.log(sim_matrix)

    
    # Create the resulting matrices
    resulting_matrices = torch.zeros(bsz, tsz, tsz)

    # Set the diagonal elements based on the mask
    for i in range(bsz):
        resulting_matrices[i].diagonal().copy_(mask[i, :])
    
    # Compute the mean cosine similarity of the masked tokens against the others
    loss = (sim_matrix[resulting_matrices.bool()]).mean()
    
    return loss


def cosine_similarity_loss_topk(feat, target, mask, eps=1e-8, stride=-1, verbose=0, topk=3):
    """
    Compute the cosine similarity loss for masked tokens.
    
    Args:
        output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, feature_size).
        mask (torch.Tensor): The binary mask tensor of shape (batch_size, sequence_length) with 1 indicating a masked token.
        eps (float): A small value to avoid division by zero.
        
    Returns:
        torch.Tensor: The cosine similarity loss.
    """
    # batch_size, sequence_length, feature_size = output.shape
    target_norm = target / (target.norm(dim=-1, keepdim=True) + eps)
    # Normalize the output along the feature dimension
    output_norm = feat / (feat.norm(dim=-1, keepdim=True) + eps)
    
    bsz, tsz = mask.shape

    # Create a matrix of cosine similarities
    sim_matrix = torch.exp(torch.bmm(output_norm, target_norm.transpose(1, 2)) / 0.1)
    if stride == -1: # full_version
        sim_matrix = sim_matrix / sim_matrix.sum(-1).unsqueeze(-1).expand(bsz, tsz, tsz)
    else:
        target_v = torch.stack([torch.sum(torch.tensor([F.one_hot(torch.tensor(j), tsz).tolist() for j in range(i % stride, i + 1, stride)][-topk:]), 0) for i in range(tsz)])\
                    .unsqueeze(0) \
                    .expand(bsz, tsz, tsz) \
                    .to(device=sim_matrix.device)
        if verbose == 1:
            print(target_v[0, :, :])
        sim_matrix = sim_matrix / (sim_matrix * target_v).sum(-1).unsqueeze(-1).expand(bsz, tsz, tsz)

    sim_matrix = -torch.log(sim_matrix)

    
    # Create the resulting matrices
    resulting_matrices = torch.zeros(bsz, tsz, tsz)

    # Set the diagonal elements based on the mask
    for i in range(bsz):
        resulting_matrices[i].diagonal().copy_(mask[i, :])
    
    # Compute the mean cosine similarity of the masked tokens against the others
    loss = (sim_matrix[resulting_matrices.bool()]).mean()
    
    return loss
