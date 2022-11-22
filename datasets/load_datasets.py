from omegaconf import DictConfig, OmegaConf
import hydra

def create_unified_representation(x):
    target = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
    target_only_distance = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    target_components_and_relation = [0,0,0,0,0,0,0,0,0,0]
    if x[0] == 'O':
        # non-argumentative
        target = [1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
        target_components_and_relation = [1,0,0,0,0,0,0,0,0,0]
    elif x[0] == 'I':
        # continuation
        if x[2] == 'P':
            target = [0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,0,1,0,0,1,0,0,0,0]
        elif x[2] == 'C':
            target = [0 ,0 ,1 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,0,1,0,1,0,0,0,0,0]
        else:
            # major claim
            target = [0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,0,1,1,0,0,0,0,0,0]
    elif x[0] == 'B':
        # begining
        if x[2] == 'P':
            target = [0 ,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,1,0,0,0,1,0,0,0,0]
        elif x[2] == 'C':
            target = [0 ,1 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,1,0,0,1,0,0,0,0,0]
        else:
            target = [0 ,1 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
            target_components_and_relation = [0,1,0,1,0,0,0,0,0,0]
   
    ################## STANCE SUPPORT/ATTACK ###############################################
    for i in range(len(x)):
        if x[i] == ':':
            if x[i +1] == 'S':
                # support
                target[6] = 1
                target_components_and_relation[6] = 1
            elif x[i + 1] == 'F':
                # for
                target[7] = 1
                target_components_and_relation[7] = 1
            elif x[ i +1] == 'A' and x[i + 2] == 't':
                # attack
                target[8] = 1
                target_components_and_relation[8] = 1
            elif x[ i +1] == 'A' and x[i + 2] == 'g':
                # against
                target[9] = 1
                target_components_and_relation[9] = 1

    ##### DISTANCE METRIC ################################################################
    distance_dictionary = {
        -11 : 10,
        -10 : 11,
        -9 : 12,
        -8 : 13,
        -7 : 14,
        -6 : 15,
        -5 : 16,
        -4 : 17,
        -3 : 18,
        -2 : 19,
        -1 : 20,
        0 : 21,
        1 : 22,
        2 : 23,
        3 : 24,
        4 : 25,
        5 : 26,
        6 : 27,
        7 : 28,
        8 : 29,
        9 : 30,
        10 : 31,
        11 : 32
    }

    import re
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", x)
    if nums:
        target[distance_dictionary[int(nums[0])] ] = 1
        target_only_distance[distance_dictionary[int(nums[0])]-10] = 1
    return target, target_components_and_relation, target_only_distance

def create_dataset(data, test_data_flag):
    word_to_ix = {}
    ix_to_word = {}
    word_to_ix['<pad>'] = len(word_to_ix)
    ix_to_word[len(ix_to_word)] = '<pad>'
    word_to_ix['OOV'] = len(word_to_ix)
    ix_to_word[len(ix_to_word)] = 'OOV'

    inputs = []
    targets = []
    targets_components_and_relation = []
    targets_only_distance = []
    X = []
    Y_target_full = []
    Y_target_components_and_relation = []
    Y_target_only_distance = []

    max_paragraph_length = -1
    token_count = 0
    cnt = 0
    for i in range(len(data)):
        if data[i] == None:
            break
        if data[i] != '\n':
            separate_input_label =  data[i].split('\t')
            inputs.append(separate_input_label[1])
            target, target_components_and_relation, target_only_distance =  create_unified_representation(separate_input_label[4])
            targets.append(target)
            targets_components_and_relation.append(target_components_and_relation)
            targets_only_distance.append(target_only_distance)

            token_count += 1
            cnt += 1
            # vocabulary creation (optional as we are using the flair pretrained embeddings)
            if separate_input_label[1] not in word_to_ix and  test_data_flag == 0:
                word_to_ix[separate_input_label[1]] = len(word_to_ix)
                ix_to_word[len(ix_to_word)] = separate_input_label[1]
            elif separate_input_label[1] not in word_to_ix and test_data_flag == 1:
                word_to_ix[separate_input_label[1]] = word_to_ix['OOV']
                ix_to_word[word_to_ix['OOV']] = separate_input_label[1]
        else:
            if inputs:
                X.append(inputs)
            if targets:  
                Y_target_full.append(targets)
                Y_target_components_and_relation.append(targets_components_and_relation)
                Y_target_only_distance.append(targets_only_distance)
            max_paragraph_length = max(cnt,max_paragraph_length)
            inputs = []
            targets = []
            targets_components_and_relation = []
            targets_only_distance = []
            cnt = 0
    return X, Y_target_full, Y_target_components_and_relation,  Y_target_only_distance, token_count, word_to_ix, ix_to_word

def load_dataset(cfg : DictConfig):
    # Train
    if cfg.datasets.name != 'augmented_paragraph_corpus':
        train_data = open(cfg.datasets.path.train_dataset_name, encoding="utf8")
        train_data = list(train_data)
        X_train, Y_train_full, Y_train_components_and_relation, Y_train_only_distance,  total_token_test, word_to_ix, ix_to_word = create_dataset(train_data, test_data_flag=0)
    
    # Val
    dev_data = open(cfg.datasets.path.val_dataset_name, encoding="utf8")
    dev_data = list(dev_data)
    X_dev, Y_dev_full, Y_dev_components_and_relation, Y_dev_only_distance,  total_token_test, word_to_ix, ix_to_word = create_dataset(dev_data, test_data_flag=0)

    # Test
    test_data = open(cfg.datasets.path.test_dataset_name, encoding="utf8")
    test_data = list(test_data)
    X_test, Y_test_full, Y_test_components_and_relation, Y_test_only_distance,  total_token_test, word_to_ix, ix_to_word = create_dataset(test_data, test_data_flag=1)

    return X_test, Y_test_full, Y_test_components_and_relation, Y_test_only_distance, total_token_test,  word_to_ix, ix_to_word