"""
Training file for the first half of TEQWENDY method
"""
from models import Teqwendy_first_half
from evaluation import auroc_auprc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from methods import Qwendy_Kx
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
set_seed(42)  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_folder = 'training_data'
def load_data(start, end): # load covariance and revised covariance
    x_data = []
    y_data = []
    data_ranges = list(range(start, end))
    for i in data_ranges:
        ds_num = f'{data_ranges[i]:02d}'  # File name formatting
        x_file = os.path.join(data_folder, f'dataset_{ds_num}_Kdata.npy')
        y_file = os.path.join(data_folder, f'dataset_{ds_num}_Kstar.npy')
        x = np.load(x_file)
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

model = Teqwendy_first_half().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
model.train()
patience = 10
epochs_without_improvement = 0


val_cov = np.load('training_data/dataset_val_Kdata.npy')
val_A = np.load('training_data/dataset_val_A.npy')
cov_torch = torch.tensor(val_cov, dtype=torch.float32, device=device)
val_x = np.load('training_data/dataset_val_xdata.npy')
val_x = val_x[:, :4] # validation data

best_score = 0.0
best_epoch = -1

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
        kstar = model(cov_torch[10*k:10+10*k]).cpu().detach().numpy()
        for i in range(10):            
            grn = Qwendy_Kx(kstar[i], val_x[i+k*10])
            t1, t2 = auroc_auprc(val_A[i+k*10], grn)
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
        
torch.save(model.state_dict(), "weights/teqwendy_1st.pth")

# training terminates here
# calculate A_1 from the above model
model = Teqwendy_first_half().to(device)
state_dict = torch.load('weights/teqwendy_1st.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

data_ranges = list(range(0, 100))
for i in data_ranges:
    ds_num = f'{data_ranges[i]:02d}'  # Format to 00, 01, ..., 99
    K_file = f'training_data/dataset_{ds_num}_Kdata.npy'
    K = np.load(K_file)
    K = torch.tensor(K, dtype=torch.float32, device=device)
    x_file = f'training_data/dataset_{ds_num}_xdata.npy'
    x = np.load(x_file)
    x = x[:, :4]
    qwendy_A = np.zeros((1000, 10, 10))
    for k in range(100):
        kstar = model(K[10*k:10+10*k]).cpu().detach().numpy()
        for j in range(10):            
            qwendy_A[j+k*10] = Qwendy_Kx(kstar[j], x[j+k*10])
    np.save(f'training_data/dataset_{ds_num}_teqwendyA.npy', qwendy_A)   

K_file = 'training_data/dataset_val_Kdata.npy'
K = np.load(K_file)
K = torch.tensor(K, dtype=torch.float32, device=device)
x_file = 'training_data/dataset_val_xdata.npy'
x = np.load(x_file)
qwendy_A = np.zeros((1000, 10, 10))
for k in range(100):
    kstar = model(K[10*k:10+10*k]).cpu().detach().numpy()
    for j in range(10):            
        qwendy_A[j+k*10] = Qwendy_Kx(kstar[j], x[j+k*10])
np.save('training_data/dataset_val_teqwendyA.npy', qwendy_A)  

