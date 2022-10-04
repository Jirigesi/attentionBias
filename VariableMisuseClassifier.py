from torch import nn, optim
import torch
from transformers import BertModel, RobertaModel


class VariableMisuseClassifier(nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, model_type):
        super(VariableMisuseClassifier, self).__init__()
        if model_type == 'bert':
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        elif model_type == 'roberta':
            self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # define the dropout layer
        self.drop = nn.Dropout(p=0.3)
        # define the dense layer
        self.Linear = nn.Linear(self.bert.config.hidden_size, 1)
        # define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_input_ids, batch_attention_mask):
        # pass the batched toekns to the bert mode
        output = self.bert(batch_input_ids, batch_attention_mask)
        # get cls token from the output of bert
        CLS_EMB = output['last_hidden_state'][:, 0, :]
        # pass the cls token to the dropout layer
        output = self.drop(CLS_EMB)
        # pass the output of the dropout layer to the dense layer
        output = self.Linear(output)
        # pass the output of the dense layer to the sigmoid activation function
        output = self.sigmoid(output)
        
        return output
        
        


    # def train(self, input_tensor, target_tensor):
    #     hidden = self.initHidden()
    #     self.zero_grad()
    #     loss = 0

    #     for i in range(input_tensor.size()[0]):
    #         output, hidden = self(input_tensor[i], hidden)
    #         loss += criterion(output, target_tensor[i].unsqueeze(0))

    #     loss.backward()
    #     optimizer.step()

    #     return output, loss.item() / input_tensor.size()[0]

    # def evaluate(self, input_tensor, target_tensor):
    #     hidden = self.initHidden()
    #     loss = 0

    #     for i in range(input_tensor.size()[0]):
    #         output, hidden = self(input_tensor[i], hidden)
    #         loss += criterion(output, target_tensor[i].unsqueeze(0))

    #     return output, loss.item() / input_tensor.size()[0]

    # def predict(self, input_tensor):
    #     hidden = self.initHidden()

    #     for i in range(input_tensor.size()[0]):
    #         output, hidden = self(input_tensor[i], hidden)

    #     return output

    # def save(self, path):
    #     torch.save(self.state_dict(), path)

    # def load(self, path):
    #     self.load_state_dict(torch.load(path))