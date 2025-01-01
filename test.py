import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)

print(torch.__version__, torch.cuda.is_available())

# Loading the dataset as a pandas dataframe
df = pd.read_csv('./datasets/fmnist_small.csv')
print(df.head())

# train test split
X = df.iloc[:,1:].values
y = df.iloc[:,0].values

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# scaling the features
from sklearn.preprocessing import StandardScaler
# scaling the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# scaling the test data
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = X_train/255
# X_test_scaled = X_test/255


device = "cuda" if torch.cuda.is_available else "cpu"

class CutomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
train_data = CutomDataset(X_train_scaled, y_train)
test_data = CutomDataset(X_test_scaled, y_test)

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

class MyModel(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )
    
    def forward(self, X):
        return self.model(X)
    
model_0 = MyModel(X_train_scaled.shape[1]).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_0.parameters(),lr=0.01)

def accuracy(logits,actual):
    total = logits.shape[0]
    pred_lables = torch.argmax(logits, dim=1)
    correct = (pred_lables == actual).sum().item()
    return correct/total

epochs = 100


for i in range(epochs):
    model_0.train()
    
    epoch_train_accuracy = 0
    epoch_loss = 0
    
    for batch_features, batch_labels in train_data_loader:
        
        logits = model_0(batch_features)
        
        optimizer.zero_grad()
        
        loss = loss_fn(logits, batch_labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        epoch_train_accuracy += accuracy(logits,batch_labels)
        
    
    epoch_loss /= len(train_data_loader)
    epoch_train_accuracy /= len(train_data_loader)
    
    # Testing loop
    
    if i % 10 == 0:
        print(f"epoch : {i}    loss : {epoch_loss},  accuracy  :  {epoch_train_accuracy}")

# Evaluating the model on test data        
total = 0
correct = 0
