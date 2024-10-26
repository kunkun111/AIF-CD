import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import arff
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import random
import time
import warnings
warnings.filterwarnings("ignore")




#--------------------------------
#  Base model training
#--------------------------------

# Define base model
class BaseModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        hidden_output = x.clone()
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden_output


# Train base model
def base_train(model, criterion, optimizer, x_train, y_train, num_epoche):
    
    for epoch in range(num_epochs):
        outputs, hidden_output = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return hidden_output
    

# Test base model
def base_test(model, criterion, optimizer, x_test, y_test):
    
    with torch.no_grad():
        outputs, hidden_output = model(x_test)
        loss = criterion(outputs, y_test).numpy()
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        f1 = f1_score(y_test.numpy(), predicted.numpy(), labels=labels, average = 'macro')
        mcc = matthews_corrcoef(y_test.numpy(), predicted.numpy())
        
    return accuracy, f1, mcc, loss, predicted.numpy()


# Fine-tune base model
def fine_tune(model, criterion, optimizer, x_test, y_test, num_epochs_finetune):
    
    for epoch in range(num_epochs_finetune):
        outputs, hidden_output = model(x_test)
        loss = criterion(outputs, y_test)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return hidden_output



# Fine-tune hidden layer
def fine_tune_hidden(model, criterion, optimizer, hidden_data, y_test, num_epochs_finetune):
    
    for epoch in range(num_epochs_finetune):
        outputs = model.fc2(hidden_data)
        loss = criterion(outputs, y_test)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()



#--------------------------------
#  Import data stream
#--------------------------------

def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name  + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multiple GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)



#--------------------------------
#  Parameter/experiment setting
#--------------------------------

# Parameter setting
hidden_size = 15
output_size = 2
learning_rate = 0.001
num_epochs = 1000
num_epochs_finetune = 100



# data path

path = '/data/kwang3/work8/realworld data/'


# Secnario 1
# s1_datasets = ['Credit_card_clients_S1']
# s2_datasets = ['Credit_card_clients_S2']
# s3_datasets = ['Credit_card_clients_S3']

# s1_output_size = 2
# s2_output_size = 2
# s3_output_size = 2

# labels = [0,1]


# Secnario 2
s1_datasets = ['mHealth_S1']
s2_datasets = ['mHealth_S2']
s3_datasets = ['mHealth_S3']

s1_output_size = 13
s2_output_size = 13
s3_output_size = 13

labels = [0,1,2,3,4,5,6,7,8,9,10,11,12]


# Secnario 3
# s1_datasets = ['covtypeNorm_S1']
# s2_datasets = ['covtypeNorm_S2']
# s3_datasets = ['covtypeNorm_S3']

# s1_output_size = 7
# s2_output_size = 7
# s3_output_size = 7

# labels = [1,2,3,4,5,6,7]


# Secnario 4
# s1_datasets = ['KDDCUP99_S1']
# s2_datasets = ['KDDCUP99_S2']
# s3_datasets = ['KDDCUP99_S3']

# s1_output_size = 23
# s2_output_size = 23
# s3_output_size = 23

# labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]



ini_train_size = 100
win_size = 100



#--------------------------------
#  Run the experiment
#--------------------------------

for i in range (len(s1_datasets)):
    
    
    s1_acc_total = []
    s2_acc_total = []
    s3_acc_total = []
    
    
    s1_f1_total = []
    s2_f1_total = []
    s3_f1_total = []
    
    
    s1_mcc_total = []
    s2_mcc_total = []
    s3_mcc_total = []
    
    time_total = []
    
    
    for seeds in range(5):  
        
        np.random.seed(seeds)
        

        s1_y_pred_cum = np.empty(0)
        s2_y_pred_cum = np.empty(0)
        s3_y_pred_cum = np.empty(0)
        
        
        print('----------------------------')
        print(s1_datasets[0], 'seeds:', seeds)
        print(s2_datasets[0], 'seeds', seeds)
        print(s3_datasets[0], 'seeds', seeds)
        
        

        #--------------------------------
        #  Data Preperation
        #--------------------------------

        # load s1 data
        s1_data = load_arff(path, s1_datasets[0])
        s1_data = s1_data.values
        
        
        # load s2 data
        s2_data = load_arff(path, s2_datasets[i])
        s2_data = s2_data.values
        
        
        # load s3 data
        s3_data = load_arff(path, s3_datasets[i])
        s3_data = s3_data.values
        
        
        # s1 data chunk
        s1_x_train = torch.FloatTensor(s1_data[0:ini_train_size, :-1])
        s1_y_train = torch.LongTensor(s1_data[0:ini_train_size, -1])
        s1_input_size = s1_x_train.shape[1]
        
        
        # s2 data chunk
        s2_x_train = torch.FloatTensor(s2_data[0:ini_train_size, :-1])
        s2_y_train = torch.LongTensor(s2_data[0:ini_train_size, -1])
        s2_input_size = s2_x_train.shape[1]
        
        
        # s3 data chunk
        s3_x_train = torch.FloatTensor(s3_data[0:ini_train_size, :-1])
        s3_y_train = torch.LongTensor(s3_data[0:ini_train_size, -1])
        s3_input_size = s3_x_train.shape[1]
        
        
        
        #--------------------------------
        #  Model initialize and training
        #--------------------------------
        
        start_time = time.time()
        
        # Initialize model for s1, loss, and optimizer
        model1 = BaseModel(s1_input_size, hidden_size, s1_output_size)
        optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
        criterion1 = nn.CrossEntropyLoss()
        
         
        # Initialize model for s2, loss, and optimizer
        model2 = BaseModel(s2_input_size, hidden_size, s2_output_size)
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
        criterion2 = nn.CrossEntropyLoss()
        
        
        # Initialize model for s3, loss, and optimizer
        model3 = BaseModel(s3_input_size, hidden_size, s3_output_size)
        optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
        criterion3 = nn.CrossEntropyLoss()
        

     
        # train base model for s1
        s1_train_hidden_output = base_train(model1, criterion1, optimizer1, s1_x_train, s1_y_train, num_epochs) 
        s1_train_label = torch.zeros(s1_train_hidden_output.shape[0], 1)
        s1_new_x_train = torch.cat((s1_train_hidden_output, s1_train_label), dim=1)
        # print('s1_new_train:', s1_new_train.shape)
        
        
        # train base model for s2
        s2_train_hidden_output = base_train(model2, criterion2, optimizer2, s2_x_train, s2_y_train, num_epochs) 
        s2_train_label = torch.zeros(s2_train_hidden_output.shape[0], 1)
        s2_new_x_train = torch.cat((s2_train_hidden_output, s2_train_label), dim=1)
        # print('s2_new_train:', s2_new_train.shape)
        
        
        # train base model for s3
        s3_train_hidden_output = base_train(model3, criterion3, optimizer3, s3_x_train, s3_y_train, num_epochs) 
        s3_train_label = torch.zeros(s3_train_hidden_output.shape[0], 1)
        s3_new_x_train = torch.cat((s3_train_hidden_output, s3_train_label), dim=1)
        # print('s3_new_train:', s3_new_train.shape)
        
        

        #--------------------------------
        #  Data split
        #--------------------------------
        
        # s1_k-fold
        s1_kf = KFold(int((s1_data.shape[0] - (ini_train_size)) / win_size))
        s1_stream = s1_data[ini_train_size:, :]
        
        s2_kf = KFold(int((s2_data.shape[0] - (ini_train_size)) / win_size))
        s2_stream = s2_data[ini_train_size:, :]
        
        s3_kf = KFold(int((s3_data.shape[0] - (ini_train_size)) / win_size))
        s3_stream = s3_data[ini_train_size:, :]



        #--------------------------------
        #  Data stream learning
        #--------------------------------
        
        for s1_train_index, s1_test_index in tqdm(s1_kf.split(s1_stream), total = s1_kf.get_n_splits(), desc = "#batch"):
        
            
            # s1 test data
            s1_x_test = torch.FloatTensor(s1_stream[s1_test_index, :-1])
            s1_y_test = torch.LongTensor(s1_stream[s1_test_index, -1])
            

            # test base model1
            s1_chunk_acc, s1_chunk_f1, s1_chunk_mcc, s1_loss, s1_y_pred = base_test(model1, criterion1, optimizer1, s1_x_test, s1_y_test)
            s1_y_pred_cum = np.hstack((s1_y_pred_cum, s1_y_pred))
            print(s1_y_pred)
            
            
            # s2 test data
            s2_x_test = torch.FloatTensor(s2_stream[s1_test_index, :-1])
            s2_y_test = torch.LongTensor(s2_stream[s1_test_index, -1])
     
        
            # test base model2
            s2_chunk_acc, s2_chunk_f1, s2_chunk_mcc, s2_loss, s2_y_pred = base_test(model2, criterion2, optimizer2, s2_x_test, s2_y_test)
            s2_y_pred_cum = np.hstack((s2_y_pred_cum, s2_y_pred))
            
            
            # s3 test data
            s3_x_test = torch.FloatTensor(s3_stream[s1_test_index, :-1])
            s3_y_test = torch.LongTensor(s3_stream[s1_test_index, -1])
     
        
            # test base model3
            s3_chunk_acc, s3_chunk_f1, s3_chunk_mcc, s3_loss, s3_y_pred = base_test(model3, criterion3, optimizer3, s3_x_test, s3_y_test)
            s3_y_pred_cum = np.hstack((s3_y_pred_cum, s3_y_pred))
            
          
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time cost: {elapsed_time:.6f} s")
        time_total.append(elapsed_time)
            
        
        # s1,s2,s3 results
        s1_y = s1_data[ini_train_size:,-1]
        s2_y = s2_data[ini_train_size:,-1]
        s3_y = s3_data[ini_train_size:,-1]
        
        
        s1_acc = accuracy_score(s1_y, s1_y_pred_cum)
        s2_acc = accuracy_score(s2_y, s2_y_pred_cum)
        s3_acc = accuracy_score(s3_y, s3_y_pred_cum)
        print('S1 acc:', s1_acc)
        print('S2 acc:', s2_acc)
        print('S3 acc:', s3_acc)
        
        
        s1_f1 = f1_score(s1_y, s1_y_pred_cum, average = 'macro', labels=labels)
        s2_f1 = f1_score(s2_y, s2_y_pred_cum, average = 'macro', labels=labels)
        s3_f1 = f1_score(s3_y, s3_y_pred_cum, average = 'macro', labels=labels)
        print('S1 f1:', s1_f1)
        print('S2 f1:', s2_f1)
        print('S3 f1:', s3_f1)
        
        
        s1_mcc = matthews_corrcoef(s1_y, s1_y_pred_cum)
        s2_mcc = matthews_corrcoef(s2_y, s2_y_pred_cum)
        s3_mcc = matthews_corrcoef(s3_y, s3_y_pred_cum)
        print('S1 mcc:', s1_mcc)
        print('S2 mcc:', s2_mcc)
        print('S3 mcc:', s3_mcc)
        
        
        s1_acc_total.append(s1_acc)
        s2_acc_total.append(s2_acc)
        s3_acc_total.append(s3_acc)
        
        
        s1_f1_total.append(s1_f1)
        s2_f1_total.append(s2_f1)
        s3_f1_total.append(s3_f1)
        
        
        s1_mcc_total.append(s1_mcc)
        s2_mcc_total.append(s2_mcc)
        s3_mcc_total.append(s3_mcc)
        
        
        # save results
        s1_result = np.zeros([s1_y.shape[0], 2])
        s1_result[:, 0] = s1_y_pred_cum
        s1_result[:, 1] = s1_y
        # np.savetxt(s1_datasets[0] + str(seeds) +'.out', s1_result, delimiter=',')   
        
        
        s2_result = np.zeros([s2_y.shape[0], 2])
        s2_result[:, 0] = s2_y_pred_cum
        s2_result[:, 1] = s2_y
        # np.savetxt(s2_datasets[0] + str(seeds) +'.out', s2_result, delimiter=',')  
        
        
        s3_result = np.zeros([s3_y.shape[0], 2])
        s3_result[:, 0] = s3_y_pred_cum
        s3_result[:, 1] = s3_y
        # np.savetxt(s3_datasets[0] + str(seeds) +'.out', s3_result, delimiter=',')  
        
        

    # acc
    print('-----------------------------------------')
    print('S1 AVE acc:', np.mean(s1_acc_total))
    print('S1 STD acc:', np.std(s1_acc_total))
    print('-----------------------------------------') 
    print('S2 AVE acc:', np.mean(s2_acc_total))
    print('S2 STD acc:', np.std(s2_acc_total))
    print('-----------------------------------------') 
    print('S3 AVE acc:', np.mean(s3_acc_total))
    print('S3 STD acc:', np.std(s3_acc_total))
    print('-----------------------------------------') 
    
    
    # f1
    print('-----------------------------------------')
    print('S1 AVE f1:', np.mean(s1_f1_total))
    print('S1 STD f1:', np.std(s1_f1_total))
    print('-----------------------------------------') 
    print('S2 AVE f1:', np.mean(s2_f1_total))
    print('S2 STD f1:', np.std(s2_f1_total))
    print('-----------------------------------------') 
    print('S3 AVE f1:', np.mean(s3_f1_total))
    print('S3 STD f1:', np.std(s3_f1_total))
    print('-----------------------------------------') 
    
    
    # mcc
    print('-----------------------------------------')
    print('S1 AVE mcc:', np.mean(s1_mcc_total))
    print('S1 STD mcc:', np.std(s1_mcc_total))
    print('-----------------------------------------') 
    print('S2 AVE mcc:', np.mean(s2_mcc_total))
    print('S2 STD mcc:', np.std(s2_mcc_total))
    print('-----------------------------------------') 
    print('S3 AVE mcc:', np.mean(s3_mcc_total))
    print('S3 STD mcc:', np.std(s3_mcc_total))
    print('-----------------------------------------') 
    
    
    # time
    print('-----------------------------------------')
    print('AVE Time:', np.mean(time_total))
    print('STD Time:', np.std(time_total))
    print('-----------------------------------------') 

    
 
    
        
        
        
        
        

    
    
    
    
    
    
    
    
    