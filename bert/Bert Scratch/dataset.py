import random
import typing 
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class IMDBBertDataset(Dataset):

    # special tokens 
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    
    MASK_PERCENTAGE = 0.15
    MASKED_INDICES_COLUMN  = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE =70

    def __init__(self,path,ds_from=None,ds_to = None,should_include_text=False):
        self.ds = pd.read_csv(path)['review']

        # creating a subset if required 
        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]
        
        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None 
        self.optimal_sentence_lengh = None 
        self.should_include_text = should_include_text

        if should_include_text:
            self.column = [
                'masked_sentence',
                self.MASKED_INDICES_COLUMN,
                'sentence',
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN
            ]
        else:
            self.columns = [
                self.MASKED_INDICES_COLUMN,
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN
            ]
        
        

