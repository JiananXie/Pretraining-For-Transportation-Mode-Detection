import torch
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from Bert import Dataset, BertClassifier

def train(model, train_data, val_data,learning_rate, epochs):
    #获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    #训练采用RandomSampler,验证采用SequentialSampler
    train_dataloader = DataLoader(train, batch_size=1,sampler=RandomSampler(train))
    val_dataloader = DataLoader(val,batch_size=1, sampler=SequentialSampler(val))
    #判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer)#参数待调

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
            train_label = train_label.type(torch.LongTensor).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
        #输出
            output = model(input_id, mask)
        #计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
        #计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
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

if __name__ == '__main__':
    model = BertClassifier()
    lr = 0.00001
    epochs = 1
    data_train = pd.read_csv()
    data_val = pd.read_csv()
    train(model, data_train, data_val, lr, epochs)
    