import torch 
import torch.nn as nn 

class BERTEmbeddings(nn.Module):
    def __init__(self,vocab_size,n_segments,max_len,embed_dim,dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size,embed_dim)
        self.seg_embed = nn.Embedding(n_segments,embed_dim)
        self.pos_embed = nn.Embedding(max_len,embed_dim)
        
        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self,seq,seg):
        embed_val = self.tok_embed(seq)+self.seg_embed(seg)+self.pos_embed(self.pos_inp)
        return embed_val


        






if __name__=='__main__':
    # parameters of bert
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1
