import math
import torch 
from torch import nn
import torch.nn.functional as F 

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * 
            -(torch.log(torch.tensor(10000.0))/ embed_size)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding.detach()
    
# Example usage
torch.manual_seed(100)
embed_size = 4  # Smaller embedding size for better visualization
max_len = 10   # Maximum length of your sequence

positional_encoder = PositionalEncoding(embed_size, max_len)
print('Positional vectors:')
print('shape of positional vectors :{}'.format(positional_encoder.encoding.shape))

print(positional_encoder.encoding)

# Assuming you have an embedded vector in a batch
embedded_vector = torch.randn(2, max_len, embed_size)  # Batch size of 2, sequence length of 10


# Applying positional encoding to the embedded vector
output_vector = positional_encoder(embedded_vector)

# Displaying the input and output vectors for comparison
print("Input Vector:")
print('shape of input vector : {}'.format(embedded_vector.size()))

print(embedded_vector)  # Displaying the first sequence in the batch

print("\nOutput Vector with Positional Encoding:")
print(output_vector)    # Displaying the first sequence in the batch after positional encoding


