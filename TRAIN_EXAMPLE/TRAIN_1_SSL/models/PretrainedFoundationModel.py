from .GumbelVectorQuantizer import Quantizer
from .Backbone import *

import torch
import torch.nn as nn


class FwBlock(nn.Module):
    def __init__(self, input_size, output_size, n_layers=2, hidden_params=[144]):
        super(FwBlock, self).__init__()
        
        # Ensure hidden_params has the correct number of elements
        if len(hidden_params) != n_layers - 1:
            raise ValueError("Length of hidden_params should be n_layers-1")
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_params[0]))
        
        # Hidden layers
        for i in range(1, n_layers - 1):
            layers.append(nn.Linear(hidden_params[i - 1], hidden_params[i]))
        
        # Output layer
        layers.append(nn.Linear(hidden_params[-1], output_size))
        
        # Store all layers in ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Apply each layer with a ReLU activation, except for the last layer
        for layer in self.layers[:-1]:
            x = gelu(layer(x))
        # No activation for the last layer
        x = self.layers[-1](x)
        return x
    
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
class WearablePretrainedEncoder(nn.Module):
    _pfm_version = { # mainly refers to the special setting of the transformer
        "pfm5":  [( 2, 4, 144), ( 3, 4, 64)],
        "pfm10": [( 4, 4, 256), ( 6, 4, 96)],
        "pfm18": [( 6, 6, 256), (12, 4, 96)],
        "pfm34": [(10, 8, 256), (24, 4, 128)],
        "pfm50": [(18, 8, 256), (32, 4, 128)]
    }
    def __init__(self,  seq_len, 
                        feature_size, 
                        hidden_dim, 
                        use_quantizer=True, 
                        codebook_num_groups=2,
                        codebook_num_vars=64,
                        codebook_vq_dim=64,
                        quantizer_training_temp= (2.0, 0.5, 0.999995),
                        use_trainable_mask=True,
                        share_layers_t1=False, # if True and use the "MLM" style training method -> LIMU-BERT Transformer Block
                        share_layers_t2=False,  # 
                        version = "pfm5"
                        ):
        super(WearablePretrainedEncoder, self).__init__()
        self.input_shape = (seq_len, feature_size) # tsz, fsz
        self.hidden_dim = hidden_dim
        
        self.use_quantizer = use_quantizer
        self.use_trainable_mask = use_trainable_mask

        # 1- global feature projector to feature_size
        self.global_feature_projector = FwBlock(feature_size, self.hidden_dim, hidden_params=[144,])
        self.norm1 = nn.LayerNorm(self.hidden_dim, eps=1e-12, elementwise_affine=True) 

        # 2-0-1 Quantizer [used in the pretraining only]
        self.codebook_num_groups = codebook_num_groups
        self.codebook_num_vars = codebook_num_vars
        self.codebook_vq_dim = codebook_vq_dim
        self.quantizer_training_temp = quantizer_training_temp
        assert (self.codebook_vq_dim // self.codebook_num_groups * self.codebook_num_groups) == self.codebook_vq_dim, "[ERR] Invalid quantizer parameters. Check your @vq_dim and @num_groups."
        
        self.quantizer = Quantizer(self.hidden_dim, 
                                    num_groups=self.codebook_num_groups, 
                                    num_vars=self.codebook_num_vars, 
                                    vq_dim=self.codebook_vq_dim, 
                                    tau_range=self.quantizer_training_temp)
        self.quantizer2hiddenDim = FwBlock(self.codebook_vq_dim, self.hidden_dim, hidden_params=[64, ])
        
        # 2-0-2 Mask Token
        # prepare for the masked training.
        if self.use_trainable_mask:
            self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        else:
            self.mask_token = torch.zeros(1, 1, self.hidden_dim)

        
        
        assert version in self._pfm_version
        self.version = version
        # 2-1 contextual feature extractor [contrastive module]
        t1_layers, t1_n_heads, t1_hidden_ff = self._pfm_version[version][0]
        self.contextual_feature_encoder_1 = Transformer(feature_num=self.hidden_dim,
                                                        hidden=self.hidden_dim,
                                                        hidden_ff=t1_hidden_ff,
                                                        n_layers=t1_layers,
                                                        n_heads=t1_n_heads,
                                                        seq_len=seq_len,
                                                        emb_norm=True, 
                                                        embed= nn.Identity(), # positional embeddings [skipped]
                                                        share_layer=share_layers_t1)
        
        # 2-2 contextual feature extractor [MLM module]
        t2_layers, t2_n_heads, t2_hidden_ff = self._pfm_version[version][1]
        self.contextual_feature_encoder_2 = Transformer(feature_num=self.hidden_dim,
                                                        hidden=self.hidden_dim,
                                                        hidden_ff=t2_hidden_ff,
                                                        n_layers=t2_layers,
                                                        n_heads=t2_n_heads,
                                                        seq_len=seq_len,
                                                        emb_norm=True, 
                                                        embed= nn.Linear(self.hidden_dim, self.hidden_dim), share_layer=share_layers_t2)
        
    def forward(self, input_seq, masked_pos=None):
        # record some parameters in use
        forward_info = {
            "masked_pos" : masked_pos,
        }

        bsz, tsz, fsz = input_seq.shape
        # input_seq: bsz, tsz(seq_len), fsz(feature_size)
        seq = self.global_feature_projector(input_seq.reshape(-1, fsz))
        seq = self.norm1(seq).reshape(bsz, tsz, -1)
        forward_info["input_embedding"] = seq

        if self.use_quantizer:
            quantized_vector, quantized_ids, quantized_info = self.quantizer(seq)
            forward_info.update(quantized_info)
            forward_info["_quantized_vector"] = quantized_vector                            # original vector with shape (bsz, tsz, vq_dim)
            forward_info["quantized_vector"] = self.quantizer2hiddenDim(quantized_vector)   # projected vector with shape (bsz, tsz, hidden_dim)
            forward_info["quantized_ids"]    = quantized_ids                                # ids with shape (bsz, tsz, num_groups) \in range(num_vars)
        
        # masking process
        if masked_pos is not None:
            # Ensure masked_pos is of shape (bsz, tsz)
            assert masked_pos.dim() == 2, "masked_pos should have shape (bsz, tsz)"
            batch_size, seq_len = masked_pos.shape

            # Expand mask_token to match batch size and sequence length
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # (bsz, tsz, hidden_dim)

            # Create a mask tensor
            mask = masked_pos.unsqueeze(-1).type_as(seq)  # (bsz, tsz, 1)

            # Apply masking: replace embeddings where mask == 1 with mask_token
            seq = seq * (1 - mask) + mask_tokens * mask

        context_s = self.contextual_feature_encoder_1(seq)

        forward_info["context_encoded"] = context_s     # (bsz, tsz, hidden_dim)

        seq = self.contextual_feature_encoder_2(context_s)
        
        return seq, forward_info
