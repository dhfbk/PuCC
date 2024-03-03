from transformers import RobertaTokenizer
from dataset_experiments import create_data_loaderData_TC
import torch
import torch.nn as nn
from model_loop import eval_model, get_predictions
from sklearn.metrics import classification_report
import json

#manually set the seed
basic_seed = 100
torch.manual_seed(basic_seed)

# load validation and test data
with open('validation_sdk.json', 'r') as f:
    validation_data = json.load(f)
with open('test_sdk.json', 'r') as f:
    test_data = json.load(f)
'''
data are reported as a dictionary with keys: conv_id, time, author_id, parent_author_id, dialogue, sequence_label,
target_label, each with the associated list of values
labels are mapped as 0: 'counter', 1: 'support'
'''

for count in range(1, 11):

    seed = count * basic_seed

    torch.manual_seed(int(seed))

    # define the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    MAX_LEN_BERT = 512
    BATCH_SIZE = 32
    # no shuffle
    validation_loader, validation_length = create_data_loaderData_TC(validation_data, tokenizer, MAX_LEN_BERT,
                                                                       BATCH_SIZE, eval=True)

    test_loader, test_length = create_data_loaderData_TC(test_data, tokenizer, MAX_LEN_BERT, BATCH_SIZE, eval=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loss_fn = nn.CrossEntropyLoss().to(device)

    model = torch.load('TC'+str(count) + '.pth')  # save best model
    model = model.to(device)

    #evaluate the model on the validation set
    print("Evaluating on validation set...")
    val_acc, val_loss = eval_model(model, validation_loader, loss_fn, device, validation_length)

    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc)

    # extract the predictions
    y_pred, y_true = get_predictions(model, validation_loader, device)
    report = classification_report(y_true, y_pred, digits=3)

    with open("report_TC_val" + str(seed) + ".txt", 'w') as f:
        f.write("VALIDATION TIME\n")
        f.write(report)

    # evaluate the model on the test set
    print("Evaluating on test set...")
    test_acc, test_loss = eval_model(model, test_loader, loss_fn, device, test_length)

    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_acc)

    # extract the predictions
    y_pred, y_true = get_predictions(model, test_loader, device)

    report = classification_report(y_true, y_pred, digits=3)

    with open("report_TC_test" + str(seed) + ".txt", 'w') as f:
        f.write("TEST TIME\n")
        f.write(report)

    # save prediction in a file txt
    with open('test_pred_TC' + str(seed) + '.txt', 'w') as f:
        for item in y_pred:
            f.write("%s\n" % item)

    # save true labels in a file txt
    with open('test_true.txt', 'w') as f:
        for item in y_true:
            f.write("%s\n" % item)
