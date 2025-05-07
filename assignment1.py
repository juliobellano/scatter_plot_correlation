import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

torch.mps.empty_cache()

#MPS config
device = torch.device("mps" if torch.mps.is_available() else "cpu")
torch.manual_seed(42) #VERY VERY IMPORTANT SO CAN REPRODUCE THE SAME RESULT ON DIF PARAMS!!

#hyper params
n_epochs = 3
batchsize = 64
learning_rate = 0.001
gamma = 0.1

#ToTensor
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,),(0.5,))])

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_ids, labels, image_dir='input/images', transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        #get image pathhh
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.png")

        #load image
        image = Image.open(img_path).convert('1')

        #apply transforms
        if self.transform:
            image = self.transform(image)
        
        #get label
        label = self.labels[idx]

        return image, label
    
#ConvNet Class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        '''self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*17*17, 4624)
        self.fc2 = nn.Linear(4624, 1156)
        self.fc3 = nn.Linear(1156, 289)
        self.fc4 = nn.Linear(289, 1)
        self.tanh =nn.Tanh()
        '''
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16*36*36, 256)
        self.fc2 = nn.Linear(256, 1)
        #self.fc3 = nn.Linear(603, 1)
        #self.fc4 = nn.Linear(750, 1)
        self.tanh =nn.Tanh()
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #flatten
        x = x.view(-1, 16*36*36)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        x = self.tanh(x)
        x = x.squeeze(1)
        return x

#datafile
data_file = pd.read_csv('input/responses.csv')

image_id = data_file['id'].values
labels = data_file['corr'].values

print(len(image_id))

X_train, X_valtest, y_train, y_valtest = train_test_split(image_id,
                                                          labels,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, 
                                                y_valtest,
                                                test_size=0.5,
                                                random_state=42,
                                                shuffle=True)




print(f'len of train data: {len(X_train)}')
print(f'len of train label: {len(y_train)}')
print(f'len of val data: {len(X_val)}')
print(f'len of test data: {len(X_test)}')

print(f'the first id: {X_train[0]}, label: {y_train[0]}')

#create datasets
train_dataset = ImageDataset(X_train, y_train, image_dir = 'input/images', transform=transforms)
val_dataset = ImageDataset(X_val, y_val, image_dir='input/images', transform=transforms)
test_dataset = ImageDataset(X_test, y_test, image_dir='input/images', transform=transforms)

#create dataloaders
train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_load = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
test_load = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
                                        
#make data iter
dataiter = iter(train_load) 
images, labels = next(dataiter)

print(f"Batch images shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")
print(f"First image in batch shape: {images[0].shape}")
print(f"First few labels: {labels[:6]}")
'''
for i in range(6):
    plt.subplot(2,3,i+1)
    img = images[i]
    img = img.permute(1, 2, 0)

    plt.imshow(img, cmap='gray')
'''
#plt.show()

model = ConvNet().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_load)

for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_load):
        images = images.to(device)
        labels = labels.to(torch.float32).to(device)

        #forward 
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)

        #loss & optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{n_epochs}, steps: {i+1}/{n_total_steps}, loss: {loss.item()}')
        if (i+1) % 200 == 0:
            model.eval()
            all_prediction_labels = []
            all_val_labels = []
            val_loss=0
            with torch.no_grad():
                for val_images, val_labels in val_load:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(torch.float32).to(device)
                    
                    all_val_labels.extend(val_labels.cpu().numpy())

                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
                    all_prediction_labels.extend(val_outputs.cpu().numpy())

            corr = float(np.corrcoef(all_val_labels,all_prediction_labels)[0,1])
            val_loss = val_loss/len(val_load)
            print(f'val loss:{val_loss}')
            print(f'corr at epoch:{epoch+1}={corr}')

            model.train()
print('finished training')
print('starts testing')

model.eval()
test_loss = 0
all_predictions = []
all_labels = []
all_images = []

with torch.no_grad():
    for test_images, test_labels in test_load:

        test_images = test_images.to(device)
        test_labels = test_labels.to(torch.float32).to(device)

        #forward
        test_output = model(test_images)
        
        #calculate loss
        loss = criterion(test_output , test_labels)
        test_loss += loss.item()

        #store prediction and labels
        all_predictions.extend(test_output.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
        all_images.extend(test_images.cpu().numpy())

test_loss = test_loss/len(test_load)
print(f'Test Loss: {test_loss:.4f}')


# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

corr = float(np.corrcoef(all_labels,all_predictions)[0, 1])
print(f'corr:{corr}')

results_df = pd.DataFrame({
    'image_id': X_test,
    'actual': all_labels,
    'predicted': all_predictions,
    'difference': all_predictions - all_labels,
    'absolute_error': np.abs(all_predictions - all_labels)
})

results_df.to_csv('predictions_results.csv', index=False)


#visualize
plt.figure(figsize=(10, 6))
plt.scatter(all_labels, all_predictions, alpha=0.5)
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
plt.imshow
plt.close()

print('saving model')
os.makedirs('saved_models', exist_ok=True)

torch.save(model, 'saved_models/cnn_model.pth')
print('Model saved successfully!')












