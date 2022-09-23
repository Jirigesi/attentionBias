import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, RobertaModel
from VariableMisuseClassifier import VariableMisuseClassifier
from VariableMisuseDataset import VariableMisuseDataset
from tqdm import tqdm


def evaluate(model, tokenizer, data_loader, loss_fn, device):
    """
    Args:
        model : classification model.
        data_loader : 
        loss_fn : _description_
        device : _description_
    Return:
          eval_loss: average evaluation loss
          correct_labels: list of labels that correctly predicted
          predicted_labels: list of labels that predicted
    """
    # set model to evaluation mode
    model = model.eval()
    # initialize lists to store correct and predicted labels
    # correct_labels = []
    predicted_labels = []
    losses = []
    # no gradient calculation needed
    with torch.no_grad():
        
        for _, batch in enumerate(tqdm(data_loader, desc="Evaluation")):
            code_batch = batch[0]
            n_label_batch = batch[1]
            
            embeddings = tokenizer.batch_encode_plus(
                            code_batch,
                            max_length=32,
                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                            return_token_type_ids=False,
                            padding=True,
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',  # Return PyTorch tensors
                            )
            # get output from the model
            output = model(embeddings['input_ids'].to(device), embeddings['attention_mask'].to(device))
            
            predicted_labels.extend(output.to('cpu').numpy().tolist())
            # correct_labels.extend(n_label_batch.to('cpu').numpy().tolist())
            loss = loss_fn(output, n_label_batch.unsqueeze(1).to(device))
            
            losses.append(loss.to('cpu').numpy().tolist())
            
    return np.mean(losses), predicted_labels
            
            
            
        


if __name__ == "__main__":
    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define the model name
    PRE_TRAINED_MODEL_NAME = 'microsoft/codebert-base'

    tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    model = VariableMisuseClassifier(PRE_TRAINED_MODEL_NAME, 'roberta')
    model = model.to(device)
    
    json_file = '/home/fjiriges/attention_bias/attentionBias/data/variable_misuse/20200621_Python_variable_misuse_datasets_train.json'
    df = pd.read_json(json_file, lines=True)

    iterable_dataset = VariableMisuseDataset(df)

    dataloader = DataLoader(iterable_dataset, batch_size=512, shuffle=False)
    
    loss_fn = nn.BCELoss().to(device)
    
    eval_loss, predicted_labels = evaluate(model, tokenizer, dataloader, loss_fn, device)
    


    
    
    
    
    
