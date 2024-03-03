from transformers import AdamW, RobertaTokenizer
from dataset_experiments import create_data_loaderData_TC_U
from model_conversations import RoBERTaThread
import torch
import torch.nn as nn
from model_loop import train_epoch, eval_model
import json

#manually set the seed
basic_seed = 100
torch.manual_seed(basic_seed)

# load train and validation data
with open('train_sdk.json', 'r') as f:
    train_data = json.load(f)
with open('validation_sdk.json', 'r') as f:
    validation_data = json.load(f)
'''
data are reported as a dictionary with keys: conv_id, time, author_id, parent_author_id, dialogue, sequence_label,
target_label, each with the associated list of values
labels are mapped as 0: 'counter', 1: 'support'
'''

lr = 1e-5
weight_decay = 1e-4
dropout = 0.25

for count in range(1, 11):

    print("Experiment number: ", count)

    seed = count * basic_seed

    torch.manual_seed(int(seed))

    # define the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    MAX_LEN_BERT = 512
    BATCH_SIZE = 32
    # add tokens to the tokenizer for timestamp (t), origin user (o) and destination user (d)
    tokenizer.add_tokens(['<o>', '</o>'], special_tokens=True)

    # create dataloaders for training and validation sets
    print("Creating Training set...")
    train_loader, train_length = create_data_loaderData_TC_U(train_data, tokenizer, MAX_LEN_BERT, BATCH_SIZE)
    # no shuffle
    validation_loader, validation_length = create_data_loaderData_TC_U(validation_data, tokenizer, MAX_LEN_BERT,
                                                                          BATCH_SIZE, eval=True)

    #best loss inside the experimentq
    best_loss = 1000000.

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    # define the model
    model = RoBERTaThread(2, dropout)
    model.BERT.resize_token_embeddings(len(tokenizer))
    model.to(device)
    torch.cuda.empty_cache()
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # define the loss function (unweighted, the dataset is already balanced)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # define the number of epochs
    go = True
    epoch = 0
    no_increase = 0

    while go:
        epoch = epoch + 1

        print(f'Epoch {epoch}')
        print('-' * 10)
        #train the model
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, train_length)
        print(f'Train loss {train_loss} accuracy {train_acc}')
        torch.cuda.empty_cache()
        #evaluate the model
        val_acc, val_loss = eval_model(model, validation_loader, loss_fn, device, validation_length)
        print(f'Val loss {val_loss} accuracy {val_acc}')
        torch.cuda.empty_cache()

        # save model with best loss
        if val_loss < best_loss:
            torch.save(model, 'TC_U'+str(count) + '.pth')  # save best model
            best_loss = val_loss
            no_increase = 0
        else:
            no_increase += 1

        if no_increase == 2:
            go = False
            print('Early stopping')

        torch.cuda.empty_cache()

