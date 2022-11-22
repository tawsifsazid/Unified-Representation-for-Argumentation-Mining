import torch
import torch.nn as nn
import torch.optim as optim
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BytePairEmbeddings, TransformerDocumentEmbeddings
from torch.nn.utils.rnn import pad_sequence
from axial_positional_embedding import AxialPositionalEmbedding

def create_id_to_label_dictionary(): 
  id_to_label = {
      0: 'O',
      1: 'B',
      2: 'I',
      3: 'MajorClaim',
      4: 'Claim',
      5: 'Premise',
      6: 'Support',
      7: 'For',
      8: 'Attack',
      9: 'Against',
      10: -11,
      11: -10,
      12: -9,
      13: -8 ,
      14: -7 ,
      15: -6,
      16: -5,
      17: -4,
      18: -3 ,
      19: -2,
      20:  -1,
      21: 0,
      22:  1, 
      23: 2,
      24: 3,
      25: 4,
      26: 5,
      27: 6,
      28: 7,
      29: 8,
      30: 9,
      31: 10, 
      32: 11,
      33: 12,
      34: 'root'
  }
  return id_to_label

def score(tp, fp, fn):
  F1 = 2*tp/ (2*tp+fp+fn)
  return F1

def check_max_index(outputs, i, start, end):
  best_val = -100000000000
  best_ind = -1
  for j in range(start, end):
    if outputs[i][j] > best_val:
      best_val = outputs[i][j]
      best_ind = j
  return best_ind, best_val 

def change_indices(outputs, i, start, end, ind):
  for j in range(start, end):
    if j == ind:
      outputs[i][j] = 100000000000
    else:
      outputs[i][j] = -100000000000
  return outputs 

def change_to_neg(outputs, i, start, end):
  for j in range(start, end):
    outputs[i][j] = -100000000000
  return outputs

def gradually_update(outputs, y, i, fl):
  comp = {
    3 : 'mj',
    4 : 'claim',
    5 : 'premise'
  }
  ind, value = check_max_index(outputs, i, 1, 3) # check beginning/continuation
  if y[i][ind] != 1:
    outputs = change_indices(outputs, i, 1, 3, ind)
    fl = 1
  else:
    outputs = change_indices(outputs, i, 1, 3, ind) # correctly predicted the beginning/continuation index
    ind, value = check_max_index(outputs, i, 3, 6) # check components majorclaim, claim, premise
    if y[i][ind] != 1:
      outputs = change_indices(outputs, i, 3, 6, ind)
      fl = 1
    else:
      outputs = change_indices(outputs, i, 3, 6, ind) # correctly predicted the majorclaim, claim, or premise index
      comp_name = comp[ind] 
      if comp_name == 'claim' or comp_name == 'premise':
        ind, value = check_max_index(outputs, i, 6, 10) # check support, for, attack, against
        if y[i][ind] != 1:
          outputs = change_indices(outputs, i, 6, 10, ind)
          fl = 1
        else:
          outputs = change_indices(outputs, i, 6, 10, ind)
          if comp_name == 'premise':
            ind, value = check_max_index(outputs, i, 10, 33) # check distance -11 to + 11
            if y[i][ind] != 1:
              outputs = change_indices(outputs, i, 10, 33, ind)
              fl = 1
            else:
              outputs = change_indices(outputs, i, 10, 33, ind)  
          else:
            outputs = change_to_neg(outputs, i, 10, 33) 
            # else claim 
      else:
        outputs = change_to_neg(outputs, i, 6, 33)  
        # else mj
  return fl, outputs     

def threshold_selection(outputs, y, index):
  fully_correct=0
  mj = 0
  non_arg = 0
  cl = 0
  premise = 0
  for i in range(len(outputs)):
    fl = 0
    non_arg_index_value = outputs[i][0] 
    ind, value = check_max_index(outputs, i, 3, 6)
    if non_arg_index_value > value: # check if non argumentative has highest logit value
      ind = 0
    
    if ind == 0 and y[i][0] == 1 : 
      outputs = change_indices(outputs, i, 0, 33, ind)
      fl = 0
    elif ind == 0 and y[i][0] == 0:
      outputs = change_indices(outputs, i, 0, 33, ind)
      fl = 1
    elif ind != 0 and y[i][0] == 1: 
      fl = 1
      fl, outputs = gradually_update(outputs, y, i, fl) 
    else:
      fl = 0
      fl, outputs = gradually_update(outputs, y, i, fl) 
    
    if fl == 0:
      if y[i][3] == 1:
        mj += 1
      elif y[i][4] == 1:
        cl += 1
      elif y[i][5] == 1:
        premise += 1
      elif y[i][0] == 1:
        non_arg += 1
      fully_correct += 1
    else:
        pass
  return fully_correct, non_arg, mj, cl, premise, outputs


def create_tensor(x, dictionary, embedding):
  max_len = len(x)
  sentence = ' '.join(x)
  T = Sentence(sentence)
  embedding.embed(T)
  embed = []
  for i, token in enumerate(T):
    if i >= max_len:
      break
    embed.append(token.embedding)
  return_tensor = pad_sequence(embed, batch_first=True)
  return_tensor = return_tensor.view(1,-1,400)
  return return_tensor


def test_model(X, Y, model, embedding, word_to_ix, device):
  with torch.no_grad():
    model.eval()
    embedding.eval()
    correct_100_test = 0
    for i in range(len(X)):
      inputs = create_tensor(X[i], word_to_ix, embedding).to(device)
      seq_len = torch.tensor([inputs.size()[1]])
      hidden, hidden_final = model.init_hidden(1)
      outputs, hidden, attn = model(inputs, None, None, seq_len, hidden, hidden_final, None)
      
      fully_correct = threshold_selection(outputs, Y[i], i)
      correct_100_test += fully_correct[0]
    print('test token level accuarcy = {}'.format(100*(correct_100_test/29537)))
    return 100*(correct_100_test/29537)

def match_exists(pred_labels, true_labels, percentage): 
  component_pred = pred_labels[1]
  component_true = true_labels[1]

  if component_pred == component_true:
    start_pos_a = pred_labels[0]
    len_a = pred_labels[2] - pred_labels[0] 
    start_pos_b = true_labels[0]
    len_b = true_labels[2] - true_labels[0] 

    a_tok = set(range(start_pos_a, start_pos_a + len_a))
    b_tok = set(range(start_pos_b, start_pos_b + len_b))

    n = len(a_tok.intersection(b_tok)) * 1.0
    if (n / max(len(a_tok), len(b_tok)) )*100 >= percentage:
      return True
  return False

def match_exists_rel(pred_labels, true_labels, percentage):
  component_pred = pred_labels[1]
  component_pred_stance = pred_labels[4]
  component_pred_distance = pred_labels[5]

  component_true = true_labels[1]
  component_true_stance = true_labels[4]
  component_true_distance = true_labels[5]

  if component_pred == component_true and component_pred_stance == component_true_stance and  component_pred_distance == component_true_distance:
    start_pos_a = pred_labels[0]
    len_a = pred_labels[2] - pred_labels[0] 
    
    start_pos_b = true_labels[0]
    len_b = true_labels[2] - true_labels[0] 


    a_tok = set(range(start_pos_a, start_pos_a + len_a))
    b_tok = set(range(start_pos_b, start_pos_b + len_b))
    
    n = len(a_tok.intersection(b_tok)) * 1.0
    if (n / max(len(a_tok), len(b_tok)) )*100 >= percentage:
      return True
  return False


def calculate_F1(pred, true, percentage, length_dataset):
  tp = fp = fn = 0
  # FN calculation
  for i in range(length_dataset):  # total paragraph = 449, total essay = 79
    for j in range(len(true[i])):
      true_labels = true[i][j]
      flag = 0
      for k in range(len(pred[i])):
        pred_labels = pred[i][k]
        if match_exists(pred_labels, true_labels, percentage):
          tp += 1
          flag = 1
      if flag == 0:
        fn += 1
  # FP calculation
  for i in range(length_dataset): # total paragraph = 449, essay = 79
    for j in range(len(pred[i])):
      pred_labels = pred[i][j]
      flag = 0
      for k in range(len(true[i])):
        true_labels = true[i][k]
        if match_exists(pred_labels, true_labels, percentage):
          flag = 1
      if flag == 0:
        fp += 1
  return tp, fp, fn, percentage

def calculate_rel_F1(pred, true, percentage, length_dataset):
  tp = fp = fn = 0
  # FN calculation
  for i in range(length_dataset):  # total paragraph = 449, total essay = 79
    for j in range(len(true[i])):
      true_labels = true[i][j]
      flag = 0
      for k in range(len(pred[i])):
        pred_labels = pred[i][k]
        if match_exists_rel(pred_labels, true_labels, percentage):
          tp += 1
          flag = 1
      if flag == 0:
        fn += 1
  # FP calculation
  for i in range(length_dataset): # total paragraph = 449, essay = 79
    for j in range(len(pred[i])):
      pred_labels = pred[i][j]
      flag = 0
      for k in range(len(true[i])):
        true_labels = true[i][k]
        if match_exists_rel(pred_labels, true_labels, percentage):
          flag = 1
      if flag == 0:
        fp += 1
  return tp, fp, fn, percentage