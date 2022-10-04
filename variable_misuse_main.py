from transformers import RobertaTokenizer
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from VariableMisuseClassifier import VariableMisuseClassifier
from VariableMisuseDataset import VariableMisuseDataset
from tqdm import tqdm
from transformers import AdamW
# , WarmupLinearSchedule
from torch.utils.tensorboard import SummaryWriter


def train(model, tokenizer, train_dataloader, dev_dataloader, loss_fn, optimizer, device, OUTPUT_DIR, PATIENCE):
    """
    Args:
        model : classification model.
        data_loader : 
        loss_fn : _description_
        optimizer : _description_
        scheduler : _description_
        device : _description_
    Return:
        train_loss: average training loss
    """
    writer = SummaryWriter()
    loss_history = []
    no_improvement = 0
    for epoch in range(NUM_TRAIN_EPOCHS, desc="Epoch"):
        model.train()
        train_loss = 0 
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            code_batch, n_label_batch = batch
            # get embeddings from the tokenizer
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
            # calculate loss
            predicted_labels.extend(output.to('cpu').numpy().tolist())
            loss = loss_fn(output, n_label_batch.unsqueeze(1).to(device))
            
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            # backpropagation
            loss.backward()
            train_loss += loss.item()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
        # get dev loss 
        dev_loss, _ = evaluate(model, tokenizer, dev_dataloader, loss_fn, device)
        # write both losses to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/dev", dev_loss, epoch)
        
        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            MODEL_FILE_NAME = "best_model.pt"
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
            
        if no_improvement >= PATIENCE: 
            print("No improvement on development set. Finish training.")
            break
        
        loss_history.append(dev_loss)
        
        writer.flush()
        writer.close()
        


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
            code_batch, n_label_batch = batch
            
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
    
    mode = "train"
    
    if mode == "train":
        # define the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # define the model name
        PRE_TRAINED_MODEL_NAME = 'microsoft/codebert-base'
        # define the tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model = VariableMisuseClassifier(PRE_TRAINED_MODEL_NAME, 'roberta')
        model = model.to(device)
        
        # buile the dataset 
        json_file = '/home/fjiriges/attention_bias/attentionBias/data/variable_misuse/20200621_Python_variable_misuse_datasets_train.json'
        df = pd.read_json(json_file, lines=True)

        iterable_dataset = VariableMisuseDataset(df)
        BATCH_SIZE = 512
        train_dataloader = DataLoader(iterable_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        GRADIENT_ACCUMULATION_STEPS = 1
        NUM_TRAIN_EPOCHS = 20
        LEARNING_RATE = 5e-5
        WARMUP_PROPORTION = 0.1
        MAX_GRAD_NORM = 5
        
        # num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
        # num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)
        
        
        OUTPUT_DIR = "/tmp/"
        # MODEL_FILE_NAME = "pytorch_model.bin"
        PATIENCE = 2
        
        loss_fn = nn.BCELoss().to(device)
        
        train(model, tokenizer, train_dataloader, loss_fn, optimizer, device, OUTPUT_DIR, PATIENCE)
        
        
    elif mode == "test":
        # define the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # define the model name
        PRE_TRAINED_MODEL_NAME = 'microsoft/codebert-base'
        # define the tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model = VariableMisuseClassifier(PRE_TRAINED_MODEL_NAME, 'roberta')
        model = model.to(device)
        
    
        json_file = '/home/fjiriges/attention_bias/attentionBias/data/variable_misuse/20200621_Python_variable_misuse_datasets_train.json'
        df = pd.read_json(json_file, lines=True)

        iterable_dataset = VariableMisuseDataset(df)

        dataloader = DataLoader(iterable_dataset, batch_size=512, shuffle=False)
        
        loss_fn = nn.BCELoss().to(device)
        
        eval_loss, predicted_labels = evaluate(model, tokenizer, dataloader, loss_fn, device)
    
