import torch.nn as nn
from transformers import RobertaModel
import torch as torch

torch.set_printoptions(threshold=100_000)
torch.set_printoptions(profile="full")

'''
Class for a typical RoBERTa model + Multi layer perceptron, setting dropout and number of classes
'''
class RoBERTaThread(nn.Module):

    def __init__(self,
                 num_classes,
                 dropout,
                 ):

        super(RoBERTaThread, self).__init__()
        #BERT-based structure
        self.size = 768  # size of input embeddings
        self.BERT = RobertaModel.from_pretrained("roberta-base", output_attentions=True)

        #MLP structure
        self.drop = nn.Dropout(p=dropout)

        self.act_mlp = nn.ReLU()
        self.act_mlp2 = nn.Tanh()

        self.linear1 = nn.Linear(self.BERT.config.hidden_size, 200)
        self.linear2 = nn.Linear(200, 300)
        self.linear3 = nn.Linear(300, num_classes)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask):

        out_bert = self.BERT(input_ids=input_ids, attention_mask=attention_mask)
        #extract the last hidden state of the [CLS] token
        out_bert = out_bert.last_hidden_state[:, 0, :]

        output1 = self.drop(self.act_mlp(self.linear1(out_bert)))
        output2 = self.drop(self.act_mlp(self.linear2(output1)))
        output3 = self.act_mlp2(self.linear3(output2))

        return self.softmax(output3)