import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification
from tqdm import tqdm
from Bert import Dataset
from sklearn.preprocessing import MinMaxScaler

def training(model, train_data, val_data,learning_rate, epochs):
    #获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    #训练采用RandomSampler,验证采用SequentialSampler
    train_dataloader = DataLoader(train, batch_size=2,sampler=RandomSampler(train))
    val_dataloader = DataLoader(val,batch_size=2, sampler=SequentialSampler(val))
    #判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    total_step = len(train_dataloader)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0.1*total_step, num_training_steps=total_step)#参数待调

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        model = model.cuda()
        criterion = criterion.cuda()
    
    #循环训练
    for epoch in range(epochs):
        total_acc_train = 0 #记录训练集的准确率
        total_loss_train = 0 #记录训练集的损失
        #进度条函数
        for train_input, train_label in tqdm(train_dataloader):
            train_input = train_input.to(device)
            train_label = train_label.type(torch.LongTensor).to(device)
        #输出
            output = model(**train_input, labels= train_label)

            y_pred_prob = output[1]
            y_pred_label = y_pred_prob.argmax(dim=1)
        #计算损失
            batch_loss = criterion(y_pred_prob, train_label)
            total_loss_train += batch_loss.item()
        #计算精度
            acc = (y_pred_label == train_label).sum().item()
            total_acc_train += acc
        #模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
    #验证
        total_acc_val = 0#记录验证集的准确率
        total_loss_val = 0#记录验证集的损失
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_input = val_input.to(device)
                val_label = val_label.type(torch.LongTensor).to(device)


                output = model(**val_input, labels= val_label)

                y_pred_prob = output[1]
                y_pred_label = y_pred_prob.argmax(dim=1)

                batch_loss = criterion(y_pred_prob, val_label)
                total_loss_val += batch_loss.item()

                acc = (y_pred_label == val_label).sum().item()
                total_acc_val += acc
        print(
            f'''Epochs: {epoch + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Accuracy: {total_acc_train / len(train_data): .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Accuracy: {total_acc_val / len(val_data): .3f}''')


if __name__ == '__main__':

    data = pd.read_csv('final_data.csv')
    label = pd.read_csv('final_label.csv')
    #去除时间戳列
    data.drop('0',axis=1,inplace=True)
    label.drop('0',axis=1,inplace=True)
   
    # epsilon = 0.00000001
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])
    # data = data + epsilon
    data = data.round(2)

    data_sentence = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)

    input = pd.DataFrame()
    input['text'] = data_sentence
    input['label'] = label

    Classification_train = []
    Classification_valid = []
    for l in input['label'].unique():
        temp = input[input['label'] == l]
        train, valid= np.split(temp.sample(frac=1,random_state=42) ,[int(0.85*len(temp))],axis=0)
        Classification_train.append(train)
        Classification_valid.append(valid)
    data_train = pd.concat(Classification_train, axis=0, join='outer', ignore_index=True)
    data_valid = pd.concat(Classification_valid, axis=0, join='outer', ignore_index=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    lr = 0.00001 
    epochs = 10
    training(model, data_train, data_valid, lr, epochs)
    