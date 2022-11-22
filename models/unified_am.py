import torch
import torch.nn as nn
import torch.optim as optim
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BytePairEmbeddings, TransformerDocumentEmbeddings
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from axial_positional_embedding import AxialPositionalEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class UnifiedAM_Paragraph(nn.Module): # for paragraph level, only difference with the essay level is the axial positional embedding dimensions

    def __init__(self, ntoken=None, ninp=None, label=None, dropout=None):
        super(UnifiedAM_Paragraph, self).__init__()
        
        self.pos_emb_ngram = AxialPositionalEmbedding( # not used
        dim = 400,
        axial_shape = (14, 14),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        self.pos_emb = AxialPositionalEmbedding( 
        dim = 400,
        axial_shape = (14, 14),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        self.pos_emb_custom = AxialPositionalEmbedding( # not used
        dim = 400,
        axial_shape = (14, 14),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        self.multi_head_attn = nn.MultiheadAttention(embed_dim=400, num_heads=4)
        self.lstm_fasttext = nn.LSTM(800, 512, bidirectional=True, num_layers=2, dropout=0.65)
        self.linear_3 = nn.Linear(in_features=1024, out_features=33)
       
        self.dropout = nn.Dropout(p=0.50)
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        initrange = 0.1
        self.linear_3.bias.data.zero_()
        self.linear_3.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, batch_size):
        hidden = self.hidden = (
            torch.zeros(2*2, batch_size, 512).cuda(),
            torch.zeros(2*2, batch_size, 512).cuda()
        )
        hidden_final = self.hidden_final = (
            torch.zeros(2*2, batch_size, 512).cuda(),
            torch.zeros(2*2, batch_size, 512).cuda()
        )
        return hidden, hidden_final

    def forward(self, src_fasttext, src_bert, src_glove, real_seq_len, hidden, hidden_final, concatenated_handcraft_features):
        # positional embedding, multi-head attention
        outputs = self.pos_emb(src_fasttext)
        outputs = outputs.permute(1,0,2)
        attn_output, attn_output_weights = self.multi_head_attn(outputs,outputs,outputs)
        outputs = attn_output.permute(1,0,2)
        outputs = torch.cat((outputs, src_fasttext), dim=2)
        
        # bi-LSTM
        output_fastext = pack_padded_sequence(outputs, real_seq_len, batch_first=True, enforce_sorted=False)
        output_fastext, hidden_fasttext = self.lstm_fasttext(output_fastext, hidden_final)
        output_fastext, lengths = pad_packed_sequence(output_fastext, batch_first=True, padding_value=0.0)

        # final linear
        output = output_fastext
        output = output.view(-1, 1024)
        output = self.linear_3(self.dropout(output))
        
        attn_output_weights = 0
        return output, hidden_final, attn_output_weights


class UnifiedAM_Essay(nn.Module):

    def __init__(self, ntoken, ninp, label, dropout):
        super(UnifiedAM_Essay, self).__init__()
    
        self.pos_emb_ngram = AxialPositionalEmbedding( # not used
        dim = 400,
        axial_shape = (14, 14),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        self.pos_emb = AxialPositionalEmbedding(
        dim = 400,
        axial_shape = (24, 24),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        self.pos_emb_custom = AxialPositionalEmbedding( # not used
        dim = 400,
        axial_shape = (14, 14),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
        axial_dims = (200, 200)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
        )

        #self.embedding = nn.Embedding(ntoken, 512, padding_idx=0)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=400, num_heads=4)
        self.lstm_fasttext = nn.LSTM(800, 512, bidirectional=True, num_layers=2, dropout=0.55)
        self.linear_3 = nn.Linear(in_features=1024, out_features=label)
       
        self.dropout = nn.Dropout(p=0.50)
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        initrange = 0.1
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_3.bias.data.zero_()
        self.linear_3.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, batch_size):
        hidden = self.hidden = (
            torch.zeros(2*2, batch_size, 512).cuda(),
            torch.zeros(2*2, batch_size, 512).cuda()
        )
        hidden_final = self.hidden_final = (
            torch.zeros(2*2, batch_size, 512).cuda(),
            torch.zeros(2*2, batch_size, 512).cuda()
        )
        return hidden, hidden_final

    def forward(self, src_fasttext, src_bert, src_glove, real_seq_len, hidden, hidden_final, concatenated_handcraft_features):
        # positional embedding, multi-head attention
        outputs = self.pos_emb(src_fasttext)
        outputs = outputs.permute(1,0,2)
        attn_output, attn_output_weights = self.multi_head_attn(outputs,outputs,outputs)
        outputs = attn_output.permute(1,0,2)        
        outputs = torch.cat((outputs, src_fasttext), dim=2)

        # bi-LSTM
        output_fastext = pack_padded_sequence(outputs, real_seq_len, batch_first=True, enforce_sorted=False)
        output_fastext, hidden_fasttext = self.lstm_fasttext(output_fastext, hidden_final)
        output_fastext, lengths = pad_packed_sequence(output_fastext, batch_first=True, padding_value=0.0)

        # final linear
        output = output_fastext
        output = output.view(-1, 1024)
        output = self.linear_3(self.dropout(output))
        
        attn_output_weights = 0
        return output, hidden_final, attn_output_weights
