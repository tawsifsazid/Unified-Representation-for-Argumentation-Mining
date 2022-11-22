from evaluate.calculate_scores import *
import logging

def check_dist(outputs, word):
  l = 23
  for l in range(10,33):
    if outputs[word][l] > 0:
      return l
  return l

def process_paragraph_true_rel(outputs, y, ind, id_to_label):
  one_paragraph = []
  i = 0
  while i < len(y):
    length_true = 0
    if y[i][1] == 1:
      j = i 
      beginning_index = i 
      while(y[j][0] == 0):
        length_true += 1
        j += 1
      ending_index = j
      i = j 
      length = length_true

      comp_name_ind = 34
      for k in range(3,6):
        if y[beginning_index][k] == 1:
          comp_name_ind = k
          break
      
      stance_ind = 34
      for k in range(6,10):
        if y[beginning_index][k] == 1:
          stance_ind = k
          break

      # distance number  
      distance = 34
      if y[beginning_index][5] == 1:
        for k in range(10,33):
          if y[beginning_index][k] == 1:
            distance = k
            break
      word = (beginning_index, id_to_label[comp_name_ind], ending_index, length, id_to_label[stance_ind], id_to_label[distance]) 
      one_paragraph.append(word)
    i += 1 

  return one_paragraph


def process_paragraph_pred_rel(outputs, y, ind, id_to_label):
  one_paragraph = []
  cnt_effect = 0
  total_above =  0
  i = 0
  while i < len(outputs):
    if outputs[i][1] > 0:  # beginning of a predicted component 
      mj = 0
      cl = 0
      premise = 0
      beginning_index = i
      max_score = 0
      max_index = -1
      max_stance_index = 34
      max_distance_ind = 34
      change_stance = 34
      change_distance = 34
      j = i
      while(j < len(outputs) and outputs[j][0] < 0): 
        if outputs[j][3] > 0 and y[j][3] == 1:
          mj += 1
          change_distance = 34
          change_stance = 34
        elif  outputs[j][4] > 0 and y[j][4] == 1 and ( (outputs[j][7] > 0 and y[j][7] == 1) or (outputs[j][9] > 0 and y[j][9] == 1) ):
          cl += 1
          if outputs[j][7] > 0:
            change_stance = 7
          elif outputs[j][9] > 0:
            change_stance = 9
          change_distance = 34
        elif outputs[j][5] > 0 and y[j][5] == 1 and ( (outputs[j][6] > 0 and y[j][6] == 1) or (outputs[j][8] > 0 and y[j][8] == 1) ):
          distance_ind = check_dist(outputs, j)
          if outputs[j][distance_ind] > 0 and y[j][distance_ind] == 1:
            premise += 1
            change_distance = distance_ind
            if outputs[j][6] > 0:
              change_stance = 6
            elif outputs[j][8] > 0:
              change_stance = 8
        j += 1
        if mj > max_score:
          max_score = mj
          max_index = 3
          max_stance_index = change_stance
          max_distance_ind = change_distance
        if cl > max_score:
          max_score = cl
          max_index = 4
          max_stance_index = change_stance
          max_distance_ind = change_distance
        if premise > max_score:
          max_score = premise
          max_index = 5
          max_stance_index = change_stance
          max_distance_ind = change_distance
          
        
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
      word = (beginning_index, id_to_label[max_index], ending_index,  max_score, id_to_label[max_stance_index], id_to_label[max_distance_ind]) # extra thing how much we have matched the span
      one_paragraph.append(word)
    i += 1   
  return one_paragraph, cnt_effect, total_above

def process_pred_true_for_RF1(model, X_test, Y_test, embedding, word_to_ix, id_to_label, device):
    with torch.no_grad():
        model.eval()
        total_above_specified = 0
        total_effect_span = 0
        true_paragraph_rel = []
        pred_paragrph_rel = []
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
            one_paragraph = process_paragraph_true_rel(outputs, Y_test[i], i, id_to_label)
            pred_one_paragraph, effect, total_above = process_paragraph_pred_rel(outputs, Y_test[i], i, id_to_label)
            total_above_specified += total_above
            total_effect_span += effect
            true_paragraph_rel.append(one_paragraph)
            pred_paragrph_rel.append(pred_one_paragraph)

            correct_100_test += fully_correct[0]
            correct_mj_100 += fully_correct[1]
        return pred_paragrph_rel, true_paragraph_rel 


def calculate_rf1(X_test, Y_test, model, embedding, word_to_ix, device):
    id_to_label = create_id_to_label_dictionary()
    pred, true = process_pred_true_for_RF1(model, X_test, Y_test, embedding, word_to_ix, id_to_label, device)
    TPS, FPS, FNS, percentage = calculate_rel_F1(pred, true, percentage = 100, length_dataset=len(X_test))
    print('R-F1({}) score =  {}, tp = {}, fp = {}, fn = {}'.format(percentage,score(TPS, FPS, FNS), TPS, FPS, FNS))
    logging.info(' R-F1({}) score =  {}, tp = {}, fp = {}, fn = {}'.format(percentage,score(TPS, FPS, FNS), TPS, FPS, FNS))
    s = 0
    for i in range(len(pred)):
        s += len(pred[i])
    