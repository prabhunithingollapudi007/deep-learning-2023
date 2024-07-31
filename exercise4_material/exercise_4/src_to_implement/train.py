import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
import torchvision as tv
from sklearn.model_selection import train_test_split

#reproducibility
t.manual_seed(42)
t.cuda.empty_cache()
t.backends.cudnn.benchmark = False
t.backends.cudnn.deterministic = True
t.backends.cudnn.enabled = True

# loading the images and labels file
data_file = pd.read_csv('src_to_implement/data.csv', sep = ";")

# Perform a stratified train-test-split
mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
stratify_labels = [mapping[(x, y)] for x, y in data_file[['crack', 'inactive']].to_numpy()]
train, test = train_test_split(data_file, test_size=0.2, shuffle=True, random_state=42, stratify=stratify_labels)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data = t.utils.data.DataLoader(ChallengeDataset(train, mode='train'), batch_size=32, shuffle = True)
test_data = t.utils.data.DataLoader(ChallengeDataset(test, mode='test'), batch_size=1, shuffle = True)

# Create an instance of a pretrained ResNet model
# res_net = tv.models.resnet34(pretrained=True)
# num_ftrs = res_net.fc.in_features
# res_net.fc = t.nn.Sequential(t.nn.Linear(num_ftrs, 2), t.nn.Sigmoid())

res_net = model.ResNet()

# Optimizer: SGD with Momentum
optimizer = t.optim.SGD(res_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# Learning rate decay
scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20 , 30 , 40], gamma=0.1)
scheduler2 = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 100, 130], gamma=0.5)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(res_net, t.nn.BCELoss(), optimizer,scheduler=[scheduler, scheduler2], train_dl=train_data, val_test_dl=test_data, early_stopping_patience=7)
train_loss, val_loss = trainer.fit(epochs=100)
