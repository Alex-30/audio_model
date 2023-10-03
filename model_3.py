import sys
import timm

import os
from tqdm import tqdm
import itertools

import cv2
import csv
import librosa
import librosa.display
from PIL import Image
from torchaudio import transforms
from torchaudio.transforms import FrequencyMasking, TimeMasking

import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
# from torchtoolbox.transform import Cutout
import shutil
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True

# loading and processing dataset

data_ds = "./images"

classes = os.listdir(data_ds)
print(classes)

valid_num = 300		# 20%
for aclass in classes:
	print('Copying', aclass)

	src_ds = os.path.join(data_ds, aclass)

	train_cd = os.path.join('./tmp/train', aclass)
	valid_cd = os.path.join('./tmp/valid', aclass)

	if not os.path.exists(train_cd):
		os.makedirs(train_cd)
	if not os.path.exists(valid_cd):
		os.makedirs(valid_cd)

	files = os.listdir(src_ds)

	if len (files) < 600:
		valid_num = 0

	for f in files[valid_num:]:
		src = os.path.join(src_ds, f)
		dst = os.path.join(train_cd, f)
		shutil.copy(src, dst)

	valid_num = 300

	for f in files[:valid_num]:
		src = os.path.join(src_ds, f)
		dst = os.path.join(valid_cd, f)
		shutil.copy(src, dst)

# transform and show dataset

transforms = T.Compose([
	T.ToTensor(),
	#T.RandomAdjustSharpness(sharpness_factor = 2, p = 1),
	#T.RandomAutocontrast(p = 1),
])

train_ds = ImageFolder('./tmp/train', transform = transforms)
valid_ds = ImageFolder('./tmp/valid', transform = transforms)

random_seed = 42
torch.manual_seed(random_seed)
batch_size = 16							# try it

train_y = [label for img, label in train_ds.samples]
class_sample_count = np.array([len(np.where(train_y == t)[0]) for t in np.unique(train_y)])
print("class sample count: ", class_sample_count)

train_dl = DataLoader(train_ds, batch_size, num_workers = 2, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers = 2, pin_memory=True)

def show_batch(dl):
	for images, labels in dl:
		fig, ax = plt.subplots(figsize=(12, 6))
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
		plt.show()
		break

#show_batch(train_dl)				   	# show batch size image.

# setup GPU

def get_default_device():
	# pick gpu if available, else cpu

	if torch.cuda.is_available():

		return torch.device('cuda')
	else:
		return torch.device('cpu')

def to_device(data, device):
	# moving tensor to chosen device

	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
		
	return data.to(device, non_blocking=True)

class DeviceDataLoader():
	# wrap a dataloader to move data to a device

	def __init__(self, dl, device):
		self.dl = dl
		self.device = device
	
	def __iter__(self):
		# yield a batch of data after moving it to device

		for b in self.dl:
			yield to_device(b, self.device)

	def __len__(self):
		# number of batches
		return len(self.dl)

device = get_default_device()
print(device)

# train model's related funtion

def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim = 1)
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
	def training_step(self, batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)

		return loss

	'''
	def predict_step(self, batch, y_pred, y_true):
		images, labels = batch
		out = self(images)
		_, preds = torch.max(out, 1)
		loss = F.cross_entropy(out, labels)
		y_pred.extend(preds.view(-1).detach().cpu().numpy())
		y_true.extend(labels.view(-1).detach().cpu().numpy())

		return y_pred, y_true
	'''

	def predict_step(self, batch, y_pred, y_true):
		k = 5
		images, labels = batch
		out = self(images)
		_, topk_preds = torch.topk(out, k, dim=1)

		loss = F.cross_entropy(out, labels)
		topk_probs, topk_indices = torch.topk(torch.softmax(out, dim=1), k, dim=1)

		y_pred_labels = topk_preds.tolist()
		y_pred_probs = topk_probs.tolist()
		y_true = labels.view(-1).tolist()

		return y_pred_labels, y_pred_probs, y_true

	def validation_step(self, batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)
		acc = accuracy(out, labels)

		return {'val_loss': loss.detach(), 'val_acc': acc}

	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()
		batch_accs = [x['val_acc'] for x in outputs]
		epoch_acc = torch.stack(batch_accs).mean()

		return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

	def epoch_end(self, epoch, result):
		torch.save(self.network.state_dict(), './working/model-' + str(epoch) + '.pth')			# epoch plus one or not ? (epoch + 1)
		print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))

@torch.no_grad()
def predict(model, val_loader, y_pred, y_true):
	model.eval()

	for batch in val_loader:
		y_pred, y_true=model.predict_step(batch, y_pred, y_true)

	return y_pred, y_true

@torch.no_grad()
def evaluate(model, val_loader):
	model.eval()
	outputs = [model.validation_step(batch) for batch in val_loader]

	return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, y_pred, y_true, opt_func=torch.optim.AdamW):
	history = []
	optimizer = opt_func(model.parameters(), lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

	for epoch in tqdm(range(epochs)):
		model.train()
		train_losses = []

		for batch in train_loader:
			optimizer.zero_grad()
			loss = model.training_step(batch)
			train_losses.append(loss)
			loss.backward()
			optimizer.step()

		result = evaluate(model, val_loader)
		scheduler.step(result['val_loss'])
		result['train_loss'] = torch.stack(train_losses).mean().item()
		model.epoch_end(epoch, result)
		history.append(result)
		predict(model, valid_dl, y_pred, y_true)

		target_names = classes

		cnf_matrix = confusion_matrix(y_true, y_pred)
		per_cls_acc = cnf_matrix.diagonal() / cnf_matrix.sum(axis = 0)

		print(target_names)
		print(per_cls_acc)
		print("Plot confusion matrix")

		df_cm = pd.DataFrame(cnf_matrix, target_names, target_names)
		plt.figure(figsize = (40, 40))
		sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
		plt.xlabel("prediction")
		plt.ylabel("label (ground truth)")

		y_pred = []
		y_true = []

	return history

# ConvNeXt model

class EfficientNet(ImageClassificationBase):
	def __init__(self):
		super().__init__()
		self.network = timm.create_model('efficientnet_b0', pretrained = True)
		num_ftrs = self.network.classifier.in_features
		self.network.classifier = nn.Linear(num_ftrs, 40)

	@autocast()
	def forward(self, xb):
		return self.network(xb)

model = EfficientNet()

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
to_device(model, device)

model = to_device(EfficientNet(), device)
y_pred = []
y_true = []

num_epochs = 10
opt_func = torch.optim.AdamW
lr = 3e-4
history = fit(num_epochs, lr, model, train_dl, valid_dl, y_pred, y_true, opt_func)

# display acc of epochs
def plot_accuracies(history):
	accuracies = [x['val_acc'] for x in history]
	plt.plot(accuracies, '-x')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)

# display loss of epochs
def plot_losses(history):
	train_losses = [x.get('train_loss') for x in history]
	val_losses = [x['val_loss'] for x in history]
	plt.plot(train_losses, '-bx')
	plt.plot(val_losses, '-rx')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['Training', 'Validation'])
	plt.title('Loss vs. No. of epochs');

plot_losses(history)
