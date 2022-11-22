import torch
import torch.nn as nn
import torch.optim as optim
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BytePairEmbeddings
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
import hydra

def load_flair_embedding():
    embedding = StackedEmbeddings(
        [
            # standard FastText word embeddings for English
            WordEmbeddings('en'),
            # Byte pair embeddings for English
            BytePairEmbeddings('en'),
        ]
    )
    return embedding

def load_model_weight(cfg: DictConfig, model):
    checkpoint = torch.load(cfg.models.path.weight_file_path, map_location='cuda:0') 
    model.load_state_dict(checkpoint['model'])
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    if cfg.train == True:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    epoch_last = checkpoint['epoch']
    return model, optimizer

def instantiate_model(cfg: DictConfig):
    model = hydra.utils.instantiate(list(cfg.models.items())[0][1])
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    embedding = load_flair_embedding()
    
    if cfg.models.pretrained == True:
        model, optimizer = load_model_weight(cfg, model)
    
    if cfg.decoupled == True:
        # load the 2nd model
        pass
    return model, embedding, optimizer