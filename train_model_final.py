from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

class ResNet152(pl.LightningModule):
  global model1
  def __init__(self, length = 0):
    super().__init__()
    self.length = length
    self.model = torch.load(current_folder + '\\models (pretrained)\\resnet152.pt')
    for param in self.model.parameters():
        param.requires_grad = True
    self.model.fc = nn.Linear(self.model.fc.in_features, 9)
    self.loss = nn.CrossEntropyLoss()
    self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_no):
    # implement single training step
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs).argmax(dim=-1)
    acc = (labels == preds).float().mean()
    # By default logs it per epoch (weighted average over batches)
    self.log("val_acc", acc)
    return [acc]

  def test_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs).argmax(dim=-1)
    acc = (labels == preds).float().mean()
    # By default logs it per epoch (weighted average over batches), and returns it afterwards
    self.log("test_acc", acc)
    return [acc]

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,steps_per_epoch=self.length, epochs=5)
    return [optimizer], [scheduler]

pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
 
#%%
################## Data Loader

def load_data(data_folder, train_ratio=[0.8, 0.1, 0.1]):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose(
                       [transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize])

    data = datasets.ImageFolder(root=data_folder, transform=image_transform)

    train_size = int(train_ratio[0] * len(data))
    test_size = int(train_ratio[1] * len(data))
    val_size = len(data) - train_size - test_size
    data_train, data_test, data_val = torch.utils.data.random_split(data, [train_size, test_size, val_size])
    return data_train, data_test, data_val

current_folder = os.path.dirname(os.path.abspath(__file__))

################## Datasets definition

data = current_folder + '\\datasets\\histology_cut\\histology_dataset\\'
dataset_train, dataset_test, dataset_val = load_data(data_folder = data)
print(len(dataset_train))
print(len(dataset_test))
print(len(dataset_val))

train_dl = DataLoader(dataset_train, batch_size=4, num_workers=0, drop_last=True, shuffle = True, pin_memory=True)
test_dl = DataLoader(dataset_test, batch_size=4, num_workers=0, pin_memory=True)
val_dl = DataLoader(dataset_val, batch_size=2, num_workers=0, pin_memory=True)
#%%
################## Model training

model = ResNet152(length = len(train_dl))

trainer = pl.Trainer(
    default_root_dir=os.path.join(current_folder + '\\models results\\weights', 'resnet152_histology'),
    gpus=1 if str(device) == "cuda:0" else 0,
    max_epochs=5, # set number of epochs
    auto_scale_batch_size = True,
    callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", dirpath=current_folder + '\\models results\\info\\', filename='sample-histology-{epoch:02d}-{val_acc:.2f}', save_top_k = -1)]
)

trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
trainer.fit(model, train_dl, val_dl)

#%%
################## Model validation

val_result = trainer.validate(model, val_dl, verbose=False)
test_result = trainer.test(model, test_dl, verbose=False)
result = {"test": test_result[0], "val": val_result[0]}
print(result)