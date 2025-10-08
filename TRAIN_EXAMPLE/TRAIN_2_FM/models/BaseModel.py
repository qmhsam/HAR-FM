# This file implement the base model that allows customization for each dataset.

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNN_AE(nn.Module):
#     def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
#         super(CNN_AE, self).__init__()
#         self.backbone = backbone
#         self.n_channels = n_channels
#         self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(32),
#                                          nn.ReLU())
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#         self.dropout = nn.Dropout(0.35)
#         self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(64),
#                                          nn.ReLU())
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#         self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(out_channels),
#                                          nn.ReLU())
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#         self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv1 = nn.Sequential(nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(64),
#                                      nn.ReLU())
#         if n_channels == 9: # ucihar
#             self.lin1 = nn.Linear(33, 34)
#         elif n_channels == 6: # hhar
#             self.lin1 = nn.Identity()
#         elif n_channels == 3: # shar
#             self.lin1 = nn.Linear(39, 40)
#         self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(32),
#                                      nn.ReLU())
#         self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(n_channels),
#                                      nn.ReLU())
#         if n_channels == 9: # ucihar
#             self.lin2 = nn.Linear(127, 128)
#             self.out_dim = 18 * out_channels
#         elif n_channels == 6:  # hhar
#             self.lin2 = nn.Linear(99, 100)
#             self.out_dim = 15 * out_channels
#         elif n_channels == 3: # shar
#             self.out_dim = 21 * out_channels
#         if backbone == False:
#             self.classifier = nn.Linear(self.out_dim, n_classes)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x, indice1 = self.pool1(self.e_conv1(x))
#         x = self.dropout(x)
#         x, indice2 = self.pool2(self.e_conv2(x))
#         x_encoded, indice3 = self.pool3(self.e_conv3(x))
#         x = self.d_conv1(self.unpool1(x_encoded, indice3))
#         x = self.lin1(x)
#         x = self.d_conv2(self.unpool2(x, indice2))
#         x = self.d_conv3(self.unpool1(x, indice1))
#         if self.n_channels == 9: # ucihar
#             x_decoded = self.lin2(x)
#         elif self.n_channels == 6 : # hhar
#             x_decoded =self.lin2(x)
#         elif self.n_channels == 3: # shar
#             x_decoded = x
#         x_decoded = x_decoded.permute(0, 2, 1)
#         x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
#         if self.backbone:
#             return x_decoded, x_encoded
#         else:
#             out = self.classifier(x_encoded)
#             return out, x_decoded


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        """
        Initializes a CNN Block consisting of a convolutional layer, batch normalization, 
        ReLU activation, and an optional max pooling layer.
        
        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - kernel_size: Size of the convolutional kernel (default: 3).
        - stride: Stride of the convolution (default: 1).
        - padding: Padding added to both sides of the input (default: 1).
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the CNN Block.
        
        Parameters:
        - x: Input tensor (batch_size, in_channels, sequence_length).
        
        Returns:
        - Output tensor after applying convolution, batch normalization, ReLU, and pooling.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if x.size(-1) <= 20:
            x = nn.Identity()(x) # placeholder.
        else:
            x = self.pool(x)
        return x



class CNN_AE_Encoder(nn.Module):
    def __init__(self, conv_params):
        """
        Initializes the ConvFeatureExtractor module using CNN blocks.
        
        Parameters:
        - conv_params: List of tuples, each containing parameters for a CNN block.
          Each tuple should have the following format: (in_channels, out_channels, kernel_size, stride, padding).
        """
        super(CNN_AE_Encoder, self).__init__()
        
        self.cnn_blocks = nn.ModuleList()
        
        self.dropout = nn.Dropout(0.35)
        # Create CNN blocks based on provided parameters
        for params in conv_params:
            block = CNNBlock(bias=False, *params)
            self.cnn_blocks.append(block)


    def _get_output_size(self, input_size):
        """
        Calculates the size of the flattened encoded feature vector.
        
        Parameters:
        - input_size: The input size (sequence length) to the feature extractor.
        
        Returns:
        - The size of the flattened encoded feature vector.
        """
        current_size = input_size
        for block in self.cnn_blocks:
            # Calculate output size after convolution
            current_size = _calculate_output_size(current_size, block.conv.kernel_size[0], block.conv.stride[0], block.conv.padding[0])
            
            # Determine if pooling should be applied
            if current_size > 20:
                # Apply pooling only if the current size is larger than or equal to kernel size
                current_size //= 2
        
        # Flatten the output of the last CNN block
        output_size = current_size * self.cnn_blocks[-1].conv.out_channels
        return output_size
    
    

    def forward(self, x):
        """
        Forward pass through the convolutional feature extractor.
        
        Parameters:
        - x: Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
        - Output tensor after passing through the CNN blocks.
        """
        
        for i, block in enumerate(self.cnn_blocks):
            x = block(x)
            if i == 0:
                x = self.dropout(x)
        return x

def _calculate_output_size(input_size, kernel_size, stride, padding):
    """
    Calculates the output size after a single convolutional layer.
    
    Parameters:
    - input_size: The input size to the convolutional layer.
    - kernel_size: The kernel size of the convolutional layer.
    - stride: The stride of the convolutional layer.
    - padding: The padding of the convolutional layer.
    
    Returns:
    - The output size after the convolutional layer.
    """
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    return output_size

class ConvFeatureExtractor(nn.Module):
    def __init__(self, conv_params):
        """
        Initializes the ConvFeatureExtractor module using CNN blocks.
        
        Parameters:
        - conv_params: List of tuples, each containing parameters for a CNN block.
          Each tuple should have the following format: (in_channels, out_channels, kernel_size, stride, padding).
        """
        super(ConvFeatureExtractor, self).__init__()
        
        self.cnn_blocks = nn.ModuleList()
        
        # Create CNN blocks based on provided parameters
        for params in conv_params:
            block = CNNBlock(*params)
            self.cnn_blocks.append(block)


    
    def _get_output_size(self, input_size):
        """
        Calculates the size of the flattened encoded feature vector.
        
        Parameters:
        - input_size: The input size (sequence length) to the feature extractor.
        
        Returns:
        - The size of the flattened encoded feature vector.
        """
        current_size = input_size
        for block in self.cnn_blocks:
            # Calculate output size after convolution
            current_size = _calculate_output_size(current_size, block.conv.kernel_size[0], block.conv.stride[0], block.conv.padding[0])
            
            # Determine if pooling should be applied
            # if current_size >= block.pool.kernel_size[0]:
            if current_size > 20:
                # Apply pooling only if the current size is larger than or equal to kernel size
                current_size //= 2
        
        # Flatten the output of the last CNN block
        output_size = current_size * self.cnn_blocks[-1].conv.out_channels
        return output_size
    

    def forward(self, x):
        """
        Forward pass through the convolutional feature extractor.
        
        Parameters:
        - x: Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
        - Output tensor after passing through the CNN blocks.
        """
        
        for block in self.cnn_blocks:
            x = block(x)
        return x


class FeatureProjector(nn.Module):
    def __init__(self, feature_extractor, input_size=6, output_size=96):
        """
        Initializes the FeatureProjector module.
        
        Parameters:
        - feature_extractor: An instance of the feature extractor (e.g., ConvFeatureExtractor) 
          that processes the input data to extract features.
        - input_size: The size of the input feature vector (default: 6), which is used to 
          determine the output size of the feature extractor.
        - output_size: The desired size of the projected feature vector (default: 96).
        """
        super(FeatureProjector, self).__init__()
        
        # Store the feature extractor and input size
        self.feature_extractor = feature_extractor
        self.input_size = input_size
        self.output_size = output_size
        
        # Calculate the size of the features output by the feature extractor
        self.feature_size = self.feature_extractor._get_output_size(input_size)
        
        # Define a linear layer that projects the extracted features to the desired output size
        self.projection = nn.Linear(self.feature_size, self.output_size)
        
        # Define a layer normalization layer that normalizes the output of the linear layer
        self.layer_norm = nn.LayerNorm(self.output_size, elementwise_affine=False)
    
    def forward(self, x):
        """
        Forward pass through the FeatureProjector module.
        
        Parameters:
        - x: Input tensor, expected to be of shape (batch_size, in_channels, sequence_length).
        
        Returns:
        - Output tensor after feature extraction, linear projection, and normalization.
        """
        # Extract features using the feature extractor and reshape the output
        features = self.feature_extractor(x).view(-1, self.feature_size)
        
        # Apply the linear projection layer followed by layer normalization
        projected_features = self.layer_norm(self.projection(features))
        
        return projected_features


class CustomClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        """
        Initializes the CustomClassifier module.

        Parameters:
        - feature_extractor: Instance of the ConvFeatureExtractor module.
        - num_classes: Number of output classes for the final classification layer.
        """
        super(CustomClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        
        # Define the linear layers
        output_size = self.feature_extractor._get_output_size(1000)  # Adjust input size as needed
        self.fc1 = nn.Linear(feature_extractor.cnn_blocks[-1].conv.out_channels * output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the custom classifier.
        
        Parameters:
        - x: Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
        - Output tensor after passing through the feature extractor and the linear layers.
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
##########################################
class ClassProjector(nn.Module):
    def __init__(self, feature_size=96, output_size=8):
        super(ClassProjector, self).__init__()
        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        
    def forward(self, feat):
        output = self.fc1(feat)
        output = F.relu(output)
        output = self.fc2(output)

        return F.log_softmax(output, 1)
    
class Classifier(nn.Module):
    def __init__(self, feature, classifier):
        super(Classifier, self).__init__()
        self.feature    = feature
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.feature(x).view(x.size(0), -1))

class ClassAvgPoolProjector(nn.Module):
    def __init__(self, feature_size=96, 
                 output_size=8, min_sequence_length=5, random_seed=65536):
        super(ClassAvgPoolProjector, self).__init__()
        self.feature_size = feature_size
        self.output_size =output_size
        self.linear = nn.Linear(feature_size, output_size)
        self.min_sequence_length = min_sequence_length
        self._generator = torch.Generator()
        self._generator.manual_seed(random_seed)
        
    def forward(self, feat, rand_view=False):
        if self.training and rand_view:
            bsz, tsz, _ = feat.shape
            rng = torch.randint(0,  tsz - self.min_sequence_length, 
                                    (bsz,), 
                                    generator=self._generator)
            
            output = torch.stack([feat[i, rng[i]:, :].mean(dim=0) for i in range(bsz)])
        else:
            output = F.avg_pool1d(feat.transpose(1, 2), 2).mean(dim=2)
        # print(output.shape)
        output_ = self.linear(output)
        return F.log_softmax(output_, 1)
    


# prepare functions
def grabClassifier(conv_params, input_size, output_size):
    # with projector
    ft_o    = ConvFeatureExtractor(conv_params) # feature
    ft_p    = FeatureProjector(ft_o, input_size=input_size) # feature projector
    model   = Classifier(ft_p, ClassProjector(output_size=output_size))
    return model

def grabClassifier_pure(conv_params, input_size, output_size):
    # w/o projector
    ft_o    = ConvFeatureExtractor(conv_params) # feature
    model   = Classifier(ft_o, ClassProjector(feature_size=ft_o._get_output_size(input_size=input_size), output_size=output_size))
    return model

class LinearFeatureExtractor2(nn.Module):
    def __init__(self, input_channels, input_size, output_size, hidden_layer=144, norm=True):
        super(LinearFeatureExtractor2, self).__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_channels * input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_size)
        )
        if norm:
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.feat(x.reshape(x.size(0), -1)))
    
def grab_classifier_fwBlock(input_channels, input_size, output_size, seq_len=1):
    ft_o    = LinearFeatureExtractor2(input_channels, input_size, 96)
    model   = Classifier(ft_o, ClassProjector(feature_size = 96 * seq_len, output_size=output_size))
    return model
    
def grab_classifier_fwBlock_pure(input_channels, input_size, output_size, seq_len=1):
    ft_o    = LinearFeatureExtractor2(input_channels, input_size, 96, norm=False)
    model   = Classifier(ft_o, ClassProjector(feature_size = 96  * seq_len, output_size=output_size))
    return model