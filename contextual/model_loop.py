import numpy as np
import torch as torch
import torch.nn as nn
# standard loops for training, evaluation and testing, or to simply get predictions

#training epoch procedure
def train_epoch(model, data_loader, loss_fn, optimizer, device, length):
    # set model to training mode
    model = model.train()
    # keep track of losses and correct predictions
    losses = []
    correct_predictions = 0.0
    # loop over the data
    for data in data_loader:
        # get the input ids and attention mask
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels.flatten().long())
        correct_predictions += torch.sum(preds == labels.flatten().long())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return correct_predictions/length, np.mean(losses)

#evaluation epoch procedure
def eval_model(model, data_loader, loss_fn, device, length):
    # set model to evaluation mode
    model = model.eval()
    # keep track of losses and correct predictions
    losses = []
    correct_predictions = 0.0

    with torch.no_grad():
        # loop over the data
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels.flatten().long())

            correct_predictions += torch.sum(preds == labels.flatten().long())
            losses.append(loss.item())

    return correct_predictions/length, np.mean(losses)

#get predictions of a model for each item
def get_predictions(model, data_loader, device):
    # set model to evaluation mode
    model = model.eval()
    # keep track of predictions and also the probabilities
    predictions = []
    prediction_probs = []
    real_values = []
    # no need to compute gradients
    with torch.no_grad():
        for data in data_loader:
            # get the input ids and attention mask
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            # get the outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels.flatten().long())

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    #return predictions and real values (1 to 1 correspondences)
    return predictions, real_values
