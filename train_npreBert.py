import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from Bert import Dataset, BertClassifier_npre

def training(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = DataLoader(train, batch_size=2, sampler=RandomSampler(train))
    val_dataloader = DataLoader(val, batch_size=2, sampler=SequentialSampler(val))

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.to(device)  # Move criterion to device

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        model.train()  # Set model to training mode

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.type(torch.LongTensor).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()  # Zero the gradients before backward pass
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item() 
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_acc_val = 0
        total_loss_val = 0
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):
                val_label = val_label.type(torch.LongTensor).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()  
                total_acc_val += acc
        print(
            f'''Epochs: {epoch + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Accuracy: {total_acc_train / len(train_data): .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

def evaluate(model, test_data):
    model.eval()
    test = Dataset(test_data)
    total_acc_test = 0
    #判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda:
        model = model.cuda()

    test_dataloader = DataLoader(test, batch_size=16, sampler=SequentialSampler(test))

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.type(torch.LongTensor).to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)


            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(
        f'''Test Accuracy: {total_acc_test / len(test_data): .3f}''')


if __name__ == '__main__':

    #读取数据
    data_train = pd.read_csv('train.csv')
    data_valid = pd.read_csv('valid.csv')
    data_test = pd.read_csv('test.csv')
   
    model = BertClassifier_npre()
    lr = 0.000001 
    epochs = 130
    training(model, data_train, data_valid, lr, epochs)
    evaluate(model, data_test)
