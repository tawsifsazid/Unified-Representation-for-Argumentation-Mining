from evaluate.calculate_scores import *
import logging

def process_paragraph_true(outputs, y, ind, id_to_label):
  one_paragraph = []
  i = 0
  while i < len(outputs):
    length_true = 0
    if y[i][1] == 1:
      j = i 
      beginning_index = i 
      while(y[j][0] == 0): # untill it reaches non-argumentative (O)
        length_true += 1
        j += 1
      ending_index = j
      i = j 
      length = length_true

      comp_name = ""
      for k in range(3,6):
        if y[beginning_index][k] == 1:
          comp_name = id_to_label[k]
          break
      
      word = (beginning_index, comp_name, ending_index, length) 
      one_paragraph.append(word)
      
    i += 1

  return one_paragraph


def process_paragraph_pred(outputs, y, ind, id_to_label):
  one_paragraph = []
  cnt_effect = 0
  total_above =  0
  i = 0
  while(i < len(outputs)):
    if outputs[i][1] > 0:  # beginning of a predicted component 
      mj = 0
      cl = 0
      premise = 0
      beginning_index = i
      max_score = 0
      max_index = -1
      j = i
      while(j < len(outputs) and outputs[j][0] < 0): 
        if outputs[j][3] > 0 and y[j][3] == 1:
          mj += 1
        elif  outputs[j][4] > 0 and y[j][4] == 1:
          cl += 1
        elif outputs[j][5] > 0 and y[j][5] == 1:
          premise += 1
        j += 1
        if mj > max_score:
          max_score = mj
          max_index = 3
        if cl > max_score:
          max_score = cl
          max_index = 4
        if premise > max_score:
          max_score = premise
          max_index = 5

      if max_score == 0: 
        found_some_component = 0
        for k in range(3,6):
          if outputs[beginning_index][k] > 0:
            max_index = k
            found_some_component = 1
            break
        if found_some_component == 0:
          max_index = 0
      ending_index = j
      i = j
      word = (beginning_index, id_to_label[max_index], ending_index,  max_score)
      one_paragraph.append(word)
    i += 1
  return one_paragraph, cnt_effect, total_above

def process_pred_true_for_CF1(model, X_test, Y_test, embedding, word_to_ix, id_to_label, device):
  with torch.no_grad():
    model.eval()
    total_above_specified = 0
    total_effect_span = 0
    true_paragraph = []
    pred_paragrph = []
    TPS = FPS = FNS = 0
    correct_100_test = 0
    correct_mj_100 = 0
    fps = 0
    total_fully_correct_relation = 0
    for i in range(len(X_test)):
      inputs = create_tensor(X_test[i], word_to_ix, embedding).to(device)
      seq_len = torch.tensor([inputs.size()[1]])
      hidden, hidden_final = model.init_hidden(1)
      outputs, hidden, attn = model(inputs, None, None, seq_len, hidden, hidden_final, None)

      fully_correct = threshold_selection(outputs, Y_test[i], i)
      outputs = fully_correct[5]
      one_paragraph = process_paragraph_true(outputs, Y_test[i], i, id_to_label)
      pred_one_paragraph, effect, total_above = process_paragraph_pred(outputs, Y_test[i], i, id_to_label)
      total_above_specified += total_above
      total_effect_span += effect
      true_paragraph.append(one_paragraph)
      pred_paragrph.append(pred_one_paragraph)
      correct_100_test += fully_correct[0]
      correct_mj_100 += fully_correct[1]

    return pred_paragrph, true_paragraph

def calculate_cf1(X_test, Y_test, model, embedding, word_to_ix, device):
  id_to_label = create_id_to_label_dictionary()
  pred, true = process_pred_true_for_CF1(model, X_test, Y_test, embedding, word_to_ix, id_to_label, device)
  TPS, FPS, FNS, percentage = calculate_F1(pred, true, percentage = 100, length_dataset=len(X_test))
  print('C-F1({}) score =  {}, tp = {}, fp = {}, fn = {}'.format(percentage,score(TPS, FPS, FNS), TPS, FPS, FNS))
  logging.info(' C-F1({}) score =  {}, tp = {}, fp = {}, fn = {}'.format(percentage,score(TPS, FPS, FNS), TPS, FPS, FNS))
  s = 0
  for i in range(len(pred)):
      s += len(pred[i])
