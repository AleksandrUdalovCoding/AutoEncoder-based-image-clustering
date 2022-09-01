################## import libraries

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms.functional as F
from sklearn.decomposition import PCA
import pickle
import os
import pytorch_lightning as pl


#%%
################## defining device and current folder

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")
current_folder = os.path.dirname(os.path.abspath(__file__))
#print(current_folder)

'''
#%%
################## Data Loader

def load_data(data_folder, train_ratio=0.2):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose(
                       [transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize])
    
    data = datasets.ImageFolder(root=data_folder, transform=image_transform)
    
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size])
    return data_train, data_test


#%%
################## Default Model Loader (if we need to load one of the default models)

#model_1 = models.resnet18(pretrained=True)
#model_1 = models.resnet50(pretrained=True)
#model_1 = models.resnet152(pretrained=True)
#model_1 = models.vgg16_bn(pretrained=True)
#print(model_1)

################# Only features layers (all except lase fully-connected layer)

#model = torch.nn.Sequential(*(list(model_1.children())[:-1])) # for resnets
#model.train(False)
#print(model)


#%%
################## User Model Loader (if we need to load user model data, ckpt format)

class ResNet152(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.load(current_folder + '\\models (pretrained)\\resnet152.pt')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 9)
    def forward(self, x):
        return self.model(x)

model = ResNet152()
model.load_from_checkpoint(current_folder + '\\best_retrained_models\\sample-histology-epoch=08-val_acc=0.86.ckpt')

################# Only features layers (all except lase fully-connected layer)

model_1 = torch.nn.Sequential(*(list(model.model.children())[:-1])) # for resnets
model = model_1
model.train(False)
print(model)


#%%
################## Read features for test images (if needed)

torch.cuda.empty_cache()
data = current_folder + '\\datasets\\histology_cut\\histology_dataset\\'
dataset_train_all, dataset_test_all = load_data(data_folder = data)

print(len(dataset_test_all))

dataset_test_all_cut = [dataset_test_all[i] for i in range(400)]
dataloader_test = DataLoader(dataset_test_all_cut,batch_size=1,shuffle=True, num_workers=0, pin_memory = False)
model = model.to(device)
iterator = 1
outputs = []
labels_output = []
for inputs, labels in dataloader_test:                    
    print(iterator)
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model.forward(inputs)
    output_flattened_list = output.tolist()
    output_labels = labels.tolist()
    del output
    del labels
    outputs.append(output_flattened_list)
    labels_output.append(output_labels)
    del output_flattened_list
    del output_labels
    iterator += 1


#%%
################## Flattening features

outputs1 = np.array(outputs)
labels_output1 = np.array(labels_output)
outputs_flattened = []
for i in range(len(outputs1)):
    img_flattened = outputs1[i].flatten().tolist()
    outputs_flattened.append(img_flattened) 
outputs_flattened_1 = np.array(outputs_flattened)
labels_output = labels_output1.flatten()


#%%
################## Features save (if needed)

with open(current_folder + '/features and labels/features_resnet_152_histology_retrained_86.txt', 'wb') as filehandle:
    pickle.dump(outputs_flattened_1, filehandle)
with open(current_folder + '/features and labels/labels_resnet_152_histology_retrained_86.txt', 'wb') as filehandle:
    pickle.dump(labels_output, filehandle)

'''
#%%
################## Features loading

with open(current_folder + '/features and labels/features_resnet_152_histology.txt', 'rb') as filehandle:
    features_resnet_152_histology =  pickle.load(filehandle)
with open(current_folder + '/features and labels/labels_resnet_152_histology.txt', 'rb') as filehandle:
    labels_resnet_152_histology =  pickle.load(filehandle)
        
with open(current_folder + '/features and labels/features_resnet_152_histology_retrained_87.txt', 'rb') as filehandle:
    features_resnet_152_histology_retrained_87 =  pickle.load(filehandle)
with open(current_folder + '/features and labels/labels_resnet_152_histology_retrained_87.txt', 'rb') as filehandle:
    labels_resnet_152_histology_retrained_87 =  pickle.load(filehandle)

with open(current_folder + '/features and labels/features_resnet_152_histology_retrained_86.txt', 'rb') as filehandle:
    features_resnet_152_histology_retrained_86 =  pickle.load(filehandle)
with open(current_folder + '/features and labels/labels_resnet_152_histology_retrained_86.txt', 'rb') as filehandle:
    labels_resnet_152_histology_retrained_86 =  pickle.load(filehandle)

with open(current_folder + '/features and labels/features_resnet_152_histology_retrained_85.txt', 'rb') as filehandle:
    features_resnet_152_histology_retrained_85 =  pickle.load(filehandle)
with open(current_folder + '/features and labels/labels_resnet_152_histology_retrained_85.txt', 'rb') as filehandle:
    labels_resnet_152_histology_retrained_85 =  pickle.load(filehandle) 


#%%
################## PCA

target_names = np.array(['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'])
target_names = np.array(['American_staffordshire_terrier', 'Beagle', 'Bull_terrier', 'English_setter', 'German_pinscher', 'German_shepherd_dog', 'Golden_retriever', 'Greyhound', 'Labrador_retriever', 'Poodle'])
target_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
target_names = np.array(['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9'])
X1 = features_resnet_152_histology
y1 = labels_resnet_152_histology
pca1 = PCA(n_components=2)
X_r1 = pca1.fit(X1).transform(X1)

X2 = features_resnet_152_histology_retrained_85
y2 = labels_resnet_152_histology_retrained_85
pca2 = PCA(n_components=2)
X_r2 = pca2.fit(X2).transform(X2)

X3 = features_resnet_152_histology_retrained_86
y3 = labels_resnet_152_histology_retrained_86
pca3 = PCA(n_components=2)
X_r3 = pca3.fit(X3).transform(X3)

X4 = features_resnet_152_histology_retrained_87
y4 = labels_resnet_152_histology_retrained_87
pca4 = PCA(n_components=2)
X_r4 = pca4.fit(X4).transform(X4)


fig, axs = plt.subplots(1, 1, figsize=(15, 15))
fig.suptitle('data resolution matrix (N)')

colors = ["navy", "turquoise", "darkorange", 'royalblue', 'purple', 'pink', 'greenyellow', 'lightcoral', 'red', 'green']
lw = 2

for color, i, target_name in zip(colors, [0,1,2,3,4,5,6,7,8,9], target_names):
    axs.scatter(X_r1[y1 == i, 0], X_r1[y1 == i, 1], color=color, alpha=0.5, lw=lw, label=target_name)
    axs.set_title("ResNet-152")
    axs.legend()
    '''
    axs[0][1].scatter(X_r2[y2 == i, 0], X_r2[y2 == i, 1], color=color, alpha=0.5, lw=lw, label=target_name)
    axs[0][1].set_title("ResNet-152 retrained 85")
   
    axs[1][0].scatter(X_r3[y3 == i, 0], X_r3[y3 == i, 1], color=color, alpha=0.5, lw=lw, label=target_name)
    axs[1][0].set_title("ResNet-152 retrained 86")
    
    axs[1][1].scatter(X_r4[y4 == i, 0], X_r4[y4 == i, 1], color=color, alpha=0.5, lw=lw, label=target_name)
    axs[1][1].set_title("ResNet-152 retrained 87")
    '''
plt.show()