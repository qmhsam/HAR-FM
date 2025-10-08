import torch.nn as nn


def _calculate_transposed_conv_output_size(input_size, kernel_size, stride, padding, output_padding):
    """
    Calculates the output size after a single transposed convolutional layer.
    
    Parameters:
    - input_size: The input size to the transposed convolutional layer.
    - kernel_size: The kernel size of the transposed convolutional layer.
    - stride: The stride of the transposed convolutional layer.
    - padding: The padding of the transposed convolutional layer.
    - output_padding: The output padding of the transposed convolutional layer.
    
    Returns:
    - The output size after the transposed convolutional layer.
    """
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    return output_size

class AutoEncoder(nn.Module):
    """
    An autoencoder model that learns to compress and reconstruct input data.
    """
    def __init__(self, feature_extractor, output_channels, output_size, 
                 feat_dim=96, latent_channels=6, 
                 decoder_channels_l1=64, 
                 decoder_channels_l2=32
                 ):
        """
        Initializes the AutoEncoder model.

        Args:
            feature_extractor (nn.Module): A pre-defined feature extraction module (encoder) that processes the input data.
            output_channels (int): The number of output channels for the final convolutional layer of the decoder.
            output_size (int): The size of the final output vector after decoding.
            latent_channels (int, optional): The number of channels in the latent space representation. Defaults to 6.
            feat_dim (int, optional): The feature dimension of the input data that the autoencoder will process. Defaults to 96.
            decoder_channels_l1 and l2 (int, optional): The configuration of the latent transposed convolutional layers. Defaults to 64 and 32.
        """
        super(AutoEncoder, self).__init__()
        self.feature_extractor  = feature_extractor
        self.latent_channels    = latent_channels
        assert (feat_dim // latent_channels) * latent_channels == feat_dim
        encoded_input_size      = feat_dim // latent_channels

        # two layers in use
        decoded_feat_size   = _calculate_transposed_conv_output_size(encoded_input_size, 3, 2, 1, 1)
        decoded_feat_size   = _calculate_transposed_conv_output_size(decoded_feat_size, 3, 2, 1, 1)

        # Define the decoder part
        self.decoder    = nn.Sequential(
            nn.ConvTranspose1d(self.latent_channels, 
                               decoder_channels_l1, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1), # assuming the 
            nn.BatchNorm1d(decoder_channels_l1),
            nn.ReLU(),
            nn.ConvTranspose1d(decoder_channels_l1, 
                               decoder_channels_l2,  
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.BatchNorm1d(decoder_channels_l2),
            nn.ReLU(),
            nn.ConvTranspose1d(decoder_channels_l2, 
                               output_channels,      
                               kernel_size=3, 
                               stride=1, 
                               padding=1),
            nn.Linear(decoded_feat_size, output_size)
        )

    def forward(self, x):
        # Encoder part
        encoded = self.feature_extractor(x)

        # Decoder part
        decoded = self.decoder(encoded.view(x.size(0), self.latent_channels, -1))
        return encoded, decoded