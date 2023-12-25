from transformers import BertConfig,BertModel,BertTokenizer
from transformers import PreTrainedTokenizer
from pprint import pprint 


bert_config = BertConfig()

# initializing the bert model with random weights 
bert_model = BertModel(bert_config)

# you can use automodal and autotokenizer to load any hugging face pretrained models. 
# but here BertModel & BertTokenizer are specifically desinged for bert based models to have more customization. 


# inputs can be a setence, batch of sentence, 
# pair of setence, paris of sentences 

input_string = 'collection of tokens which will never be split during tokenization'
input_string_b = 'the separator token, which is used when building a sequence from multiple sequences'


# this tokenizer is inherited from tokenizer main class 
# Documentation : https://huggingface.co/docs/transformers/main_classes/tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# calling the tokenizer 
# can use encode, encode_plus methods 
sent = tokenizer(   
    input_string,input_string_b,
    return_tensors='pt', 
    # max_length=15, # will be ignored when padding is true 
    padding=True,
    add_special_tokens=True,
)

if __name__=='__main__':
    # print(bert_config)
    pprint(sent)


# BertConfig {
#   "attention_probs_dropout_prob": 0.1,
#   "classifier_dropout": null,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "pad_token_id": 0,
#   "position_embedding_type": "absolute",
#   "transformers_version": "4.32.1",
#   "type_vocab_size": 2,
#   "use_cache": true,
#   "vocab_size": 30522
#}
    