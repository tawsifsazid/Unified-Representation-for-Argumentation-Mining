from omegaconf import DictConfig, OmegaConf
import hydra
from datasets.load_datasets import *
from models.load_models import *
from evaluate.calculate_scores import *
from evaluate.C_F1 import *
from evaluate.R_F1 import *
import logging

@hydra.main(version_base='1.2', config_path="config", config_name="config")
def main(cfg : DictConfig):
    
    logging.basicConfig(filename='results.log', encoding='utf-8', level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(' device name = {}'.format(device))
    print('device name = {}'.format(device))

    # load dataset from config
    X_test, Y_test_full, Y_test_components_and_relation, Y_test_only_distance, total_token_test, word_to_ix, ix_to_word = load_dataset(cfg)
    logging.info(' train, test and validation datasets loading successful...')
    print('train, test and validation datasets loading successful...')
    
    # load model from config
    model, embedding, optimizer = instantiate_model(cfg)
    model = model.to(device)
    embedding = embedding.to(device)
    logging.info(' model loading successful...')
    print('model loading successful...')

    # evaluate model
    if cfg.evaluate == True:
        logging.info(' model evaluation starting with the unified representation...')
        print('model evaluation starting with the unified representation...')
        test_model(X_test, Y_test_full, model, embedding, word_to_ix, device) # test token level accuracy
        calculate_cf1(X_test, Y_test_full, model, embedding, word_to_ix, device)
        calculate_rf1(X_test, Y_test_full, model, embedding, word_to_ix, device)
        logging.info(' model evaluation end...')
        print('model evaluation end...')

if __name__ == "__main__":
    main()