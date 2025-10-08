import numpy as np
import torch

def generate_span_mask(batch_size, seq_len, max_gram=3, p=0.2, goal_num_predict=15, device='cpu'):
    """
    Generates span masks for a batch of sequences.

    Args:
        batch_size (int): Number of sequences in the batch.
        seq_len (int): Length of each sequence.
        max_gram (int): Maximum span length to mask (n-gram).
        p (float): Probability parameter for geometric distribution to sample span lengths.
        goal_num_predict (int): Target number of tokens to mask per sequence.
        device (str or torch.device): Device to place the mask tensor.

    Returns:
        torch.Tensor: Mask tensor of shape (batch_size, seq_len), where 1 indicates masked positions and 0 unmasked.
    """
    # Define possible span lengths from 1 to max_gram
    ngrams = np.arange(1, max_gram + 1)
    
    # Compute probabilities for each span length based on a geometric distribution
    pvals = p * np.power(1 - p, ngrams - 1)
    pvals = pvals / pvals.sum()
    
    # Initialize the mask tensor with zeros
    masks = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        masked_pos = set()
        attempts = 0  # To prevent infinite loops in edge cases
        max_attempts = goal_num_predict * 10  # Arbitrary large number
        
        while len(masked_pos) < goal_num_predict and attempts < max_attempts:
            attempts += 1
            
            # Sample a span length based on the defined probabilities
            span_length = np.random.choice(ngrams, p=pvals)
            span_length = min(span_length, goal_num_predict - len(masked_pos))
            if span_length <= 0:
                break  # No more tokens to mask
            
            # Randomly select a start position ensuring the span fits within the sequence
            start = np.random.randint(0, max(seq_len - span_length + 1, 1))
            
            # Check if the span overlaps with already masked positions
            overlap = any(i in masked_pos for i in range(start, start + span_length))
            if overlap:
                continue  # Skip overlapping spans
            
            # Add the span positions to the masked positions
            for i in range(start, start + span_length):
                if len(masked_pos) >= goal_num_predict:
                    break
                masked_pos.add(i)
        
        # Convert the set of masked positions to a list and set them in the mask tensor
        mask_indices = list(masked_pos)
        masks[b, mask_indices] = 1.0
    
    return masks
