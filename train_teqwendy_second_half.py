"""
Training file for the second half of TEQWENDY method
"""
from models import Teqwendy_second_half
from evaluation import auroc_auprc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
import gc
gc.collect()
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed): # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(6)  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_folder = 'training_data'
def load_data(start, end): # load covariance and A_1
    x_data = []
    y_data = []
    data_ranges = list(range(start, end))
    for i in data_ranges:
        ds_num = f'{data_ranges[i]:02d}'  # File name formatting
        xK_file = os.path.join(data_folder, f'dataset_{ds_num}_Kdata.npy')
        xA_file = os.path.join(data_folder, f'dataset_{ds_num}_teqwendyA.npy')
        y_file = os.path.join(data_folder, f'dataset_{ds_num}_A.npy')
        xK = np.load(xK_file)
        xA = np.load(xA_file)
        xA = np.expand_dims(xA, axis=1)
        x = np.concatenate((xK, xA), axis=1)
        y = np.load(y_file)
        x_data.append(x)
        y_data.append(y)    
        
    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)   
    return torch.tensor(x_data, dtype=torch.float32, device=device), torch.tensor(y_data, dtype=torch.float32, device=device)

x_train, y_train = load_data(0, 100)
batch_size = 16
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Teqwendy_second_half().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
model.train()
patience = 10
epochs_without_improvement = 0

best_score = 0.0
best_epoch = -1

val_A = np.load('training_data/dataset_val_A.npy')
val_cov = np.load('training_data/dataset_val_Kdata.npy')
val_QA = np.load('training_data/dataset_val_teqwendyA.npy')
val_cov = torch.tensor(val_cov, dtype=torch.float32, device=device)
val_QA = torch.tensor(val_QA, dtype=torch.float32, device=device)
val_QA = val_QA.unsqueeze(1)
val_total = torch.concat((val_cov, val_QA), dim=1) # validation data

for epoch in range(num_epochs):
    model.train() 
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    total_auroc = 0.0 
    total_auprc = 0.0 
    model.eval()
    for k in range(100):
        grn = model(val_total[10*k:10+10*k]).cpu().detach().numpy()
        for i in range(10):            
            t1, t2 = auroc_auprc(val_A[i+k*10], grn[i])
            total_auroc += t1 / 1000
            total_auprc += t2 / 1000 # evaluate model performance
            
    print('current measure, ', epoch, total_auroc, total_auprc)
    measure = total_auroc + total_auprc
    if measure > best_score:
        best_score = measure
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
        best_epoch = epoch
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
        
torch.save(model.state_dict(), "weights/teqwendy_2nd.pth")










