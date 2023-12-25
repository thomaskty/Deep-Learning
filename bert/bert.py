import torch
import torch.nn as nn
torch.manual_seed(100)
import pprint 

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

class BERTEmbedding(nn.Module):
    def __init__(self,vocab_size,n_segments,max_len,embed_dim,dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.position_encoder= PositionalEncoding(embed_dim, max_len=max_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, seg):
        print('actual sequence dimension = {}'.format(seq.shape))
        print('after tok_embed = {}'.format(self.tok_embed(seq).shape))

        token_embeddings = self.tok_embed(seq).squeeze()
        print('after applying squeeze  = {}'.format(token_embeddings.shape))

        segment_embeddings = self.seg_embed(seg).squeeze()
        position_embeddings = self.position_encoder(token_embeddings)

        print('segment embedding shape = {}'.format(segment_embeddings.shape))
        print('position embeddings shape = {}'.format(position_embeddings.shape))

        combined_embedding = token_embeddings+segment_embeddings+position_embeddings
        print('comgined embedding shape = {}'.format(combined_embedding.shape))

        final_embedding = self.drop(combined_embedding)
        print('after dropout = {}'.format(final_embedding.shape))

        return final_embedding


class BERT(nn.Module):
    def __init__(self,vocab_size,n_segments,max_len,embed_dim,n_layers,attn_heads,dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4,batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out


if __name__ == "__main__":
    
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1
    BATCH_SIZE = 32

    parameters = {
        'VOCAB_SIZE':VOCAB_SIZE,
        'N_SEGMENTS':N_SEGMENTS,
        'MAX_LENGTH':MAX_LEN,
        'EMBED_DIM':EMBED_DIM,
        'N_LAYERS':N_LAYERS,
        'ATTN_HEADS':ATTN_HEADS,
        'DROPOUT':DROPOUT,
        'BATCH_SIZE':BATCH_SIZE
    }
    pprint.pprint(parameters)

    # creating a tensor input ( sample seq contains the indexes of tokens) 
    # sample seg represents the segment id (1,2,3) : ['cls'],[sentence a ],[setence b]
    sample_seq = torch.randint(MAX_LEN, size = (BATCH_SIZE,MAX_LEN))
    sample_seg = torch.randint(N_SEGMENTS,size=[BATCH_SIZE,MAX_LEN])

    bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)

    # final output shape will be same as input tensor (batch,token,features)
    out = bert(sample_seq, sample_seg)
