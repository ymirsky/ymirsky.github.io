import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


class LoanPredictor:
    def __init__(self):
        print("Setting up Victim Model")
        datapath = "loan_dataset.csv"
        D = pd.read_csv(datapath)
        X = D[D.columns[1:-1]]
        Y = D[D.columns[-1]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.Y_test = Y_test.to_numpy()

        self.__train_model()

        Y_pred = self.M.predict(self.X_test)
        self.ACC = np.round(accuracy_score(self.Y_test, Y_pred),3)
        self.AUC = np.round(roc_auc_score(self.Y_test, Y_pred),3)

        # Setup poison detector
        self.norm = np.array([70000, 64, 14000])
        self.A = NearestNeighbors(n_neighbors=2).fit(self.X_train / self.norm)

    def predict(self, x):
        return self.M.predict(x)

    # X must be in format of bxn where b is the batch size
    def __check_for_poison(self, X, verbose=True):
        distances, samples = self.A.kneighbors(X/self.norm)
        threshold = 0.8
        if (np.mean(distances,axis=0) > threshold).any():
            if verbose:
                print("Poison Detected! Samples rejected")
            return None
        else:
            if verbose:
                print("Data is Safe. Added to training set")
            return X

    def __train_model(self):
        M = RandomForestClassifier(n_estimators=10)
        self.M = M.fit(self.X_train, self.Y_train)

    #checks a model's performance against M
    # model must have a predict() method that accepts bx3 arrays (b:batch)
    def check_model_performance(self, M_t):
        Y_pred = M_t.predict(self.X_test)
        ACC = np.round(accuracy_score(self.Y_test, Y_pred), 3)
        AUC = np.round(roc_auc_score(self.Y_test, Y_pred), 3)
        print("Loan Predictor, Accuracy:", self.ACC, "AUC:", self.AUC)
        print("Stolen Model, Accuracy:", ACC, "AUC:", AUC)

    #updates internal model with given data (checks for poison first)
    #data must be in bxn format where b is batch size, and in np.array format
    def update_model(self, X_train_extra, Y_train_extra, verbose=True):
        X_train_extra = self.__check_for_poison(X_train_extra, verbose)
        if X_train_extra is None:
            return
        self.X_train = np.vstack((self.X_train, X_train_extra))
        self.Y_train = np.concatenate((self.Y_train, Y_train_extra))
        self.__train_model()
        if verbose:
            print("Model updated")


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

class TwitterAPI:
  def __init__(self):
    self.__target = "I really don't like what the government is doing :-("
    self.__sid = SentimentIntensityAnalyzer()

  def __get_jaccard_sim(self, str1, str2): 
      a = set(str1.lower().split()) 
      b = set(str2.lower().split())
      c = a.intersection(b)
      return float(len(c)) / (len(a) + len(b) - len(c))

  def sendTweet(self, tweet):
    #check if similar enough to target
    jac = self.__get_jaccard_sim(tweet, self.__target)
    senti = self.__sid.polarity_scores(tweet)

    if jac < 0.38 or ("government" not in tweet.lower()):
      print("Your Tweet was too diffrent from the target, try again")
      return

    if senti['compound'] < 0.9:
      print("You said something negative about the government, police have been dispatched!")
    else:
      print("Tweet sent successfuly. Long live the resistance!")

    return
    
    
    
## Standard libraries
import os
import json
import math
import time
import numpy as np 
import scipy.linalg

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import urllib.request
from urllib.error import HTTPError
import zipfile

class ModelAPI():
  def __init__(self):
    # Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
    DATASET_PATH = "../data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/tutorial10"

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", self.device)

    # Github URL where the dataset is stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
    # Files to download
    pretrained_files = [(DATASET_PATH, "TinyImageNet.zip"), (CHECKPOINT_PATH, "patches.zip")]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for dir_name, file_name in pretrained_files:
        file_path = os.path.join(dir_name, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)
            if file_name.endswith(".zip"):
                print("Unzipping file...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_path.rsplit("/",1)[0])
    
    # Load CNN architecture pretrained on ImageNet
    os.environ["TORCH_HOME"] = CHECKPOINT_PATH
    self.blackboxmodel = torchvision.models.mobilenet_v3_large(pretrained=True)
    self.blackboxmodel = self.blackboxmodel.to(self.device)

    # No gradients needed for the network
    self.blackboxmodel.eval()
    for p in self.blackboxmodel.parameters():
        p.requires_grad = False
 
    # Mean and Std from ImageNet
    self.NORM_MEAN = np.array([0.485, 0.456, 0.406])
    self.NORM_STD = np.array([0.229, 0.224, 0.225])
    # No resizing and center crop necessary as images are already preprocessed.
    plain_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=self.NORM_MEAN,
                            std=self.NORM_STD)
    ])

    # Load dataset and create data loader
    imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")
    assert os.path.isdir(imagenet_path), f"Could not find the ImageNet dataset at expected path \"{imagenet_path}\". " 
    dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=plain_transforms)
    self.data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8)

    # Load label names to interpret the label numbers 0 to 999
    with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
        self.label_names = json.load(f)
        
  def get_label_index(self, lab_str):
      assert lab_str in self.label_names, f"Label \"{lab_str}\" not found. Check the spelling of the class."
      return self.label_names.index(lab_str)

  def eval_model(self, dataset_loader, img_func=None):
    tp, tp_5, counter = 0., 0., 0.
    for imgs, labels in tqdm(dataset_loader, desc="Validating..."):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        if img_func is not None:
            imgs = img_func(imgs, labels) 
        with torch.no_grad():
            preds = self.blackboxmodel(imgs)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[...,None]).any(dim=-1).sum()
        counter += preds.shape[0]
    acc = tp.float().item()/counter
    top5 = tp_5.float().item()/counter
    print(f"Top-1 error: {(100.0 * (1 - acc)):4.2f}%")
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5

  def show_prediction(self, img, label, pred, K=5, adv_img=None, noise=None):
  
    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * self.NORM_STD[None,None]) + self.NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()
    
    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(10,2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(12,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})
    
    ax[0].imshow(img)
    ax[0].set_title(self.label_names[label])
    ax[0].axis('off')
    
    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_img = (adv_img * self.NORM_STD[None,None]) + self.NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        noise = noise.cpu().permute(1, 2, 0).numpy()
        noise = noise * 0.5 + 0.5 # Scale between 0 to 1 
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')
    
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([self.label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')
    
    plt.show()
    plt.close()

  def get_targets(self):
    exmp_batch, label_batch = next(iter(self.data_loader))
    samples = [5,7,8]
    x_samples = exmp_batch[samples]
    x_labels = label_batch[samples]
    return x_samples, x_labels
