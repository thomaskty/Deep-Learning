from torch import nn 
import torch 

# d_model - number of expected features in the input 
# nhead - number of heads in multihead attention 
# dim_feedforward - dimension of feed forward network in transformers 
# dropout - 
# activation - {'relu','gelu'}
# layer_norm_eps - the eps value in layer normalization 
# batch_first - input and output has the form (batch,seq,feature)
# norm_first - layer norm is done prior to attention and feedforward operations 
# bias - if false linear and layernorm will not learn additive bias 


encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
src = torch.rand(32, 10, 512)
out = encoder_layer(src)
print('applying a single encoder layer ')
print(src.shape)
print(out.shape)


print('\nApplying an encoder layer multiple time ->')
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
print(src.shape)
print(out.shape)

# note : input and output has same dimension

