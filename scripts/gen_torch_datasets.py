import random
import seaborn as sns
from torchvision import transforms

from nbformat import convert
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Make sure tqdm is installed
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

from hs_hackathon_drone_acoustics import CLASSES, RAW_DATA_DIR
from hs_hackathon_drone_acoustics.base import AudioDataset
from hs_hackathon_drone_acoustics.plot import plot_spectrogram

def convert_to_mtx(waveform):
    resample = T.Resample(waveform.sample_rate, 16_000)
    waveform_to_spectrogram = T.Spectrogram(win_length=4096, n_fft=4096, hop_length=2048)
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    spectrogram = db_transform(waveform_to_spectrogram(resample(waveform.data)))
    # spectrogram = db_transform(waveform_to_spectrogram(waveform.data))
    return spectrogram

def create_chunks(mtx, n):
    chunk_size = mtx.shape[0] // n
    chunks = [mtx[i*chunk_size:(i+1)*chunk_size, :] for i in range(n)]
    return chunks

def gen_datasets(ds_base_path, n_chunks):
    TRAIN_PATH = Path(ds_base_path) / "train"
    VAL_PATH = Path(ds_base_path) / "val"

    train_dataset = AudioDataset(root_dir=TRAIN_PATH)
    val_dataset = AudioDataset(root_dir=VAL_PATH)    
    
    spectogram_chunks_train = []
    labels_train = []

    print("Train samples:", len(train_dataset))
    for sample in train_dataset:
        waveform, label = sample
        mtx = convert_to_mtx(waveform)
        chunks = create_chunks(mtx, n_chunks)
        spectogram_chunks_train += chunks
        labels_train += [label for i in range(len(chunks))]
        
    print("Train labels", len(labels_train))
        
    spectogram_chunks_val = []
    labels_val = []

    print("Val samples:", len(val_dataset))
    for sample in val_dataset:
        waveform, label = sample
        mtx = convert_to_mtx(waveform)
        chunks = create_chunks(mtx, n_chunks)
        # print(len(chunks))
        spectogram_chunks_val += chunks
        labels_val += [label for i in range(len(chunks))]

    print("Val labels", len(labels_val))
    
    return spectogram_chunks_train, labels_train, spectogram_chunks_val, labels_val 

class FastTensorDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors  # List of torch.Tensor
        self.labels = torch.tensor(labels)  # List or tensor of labels

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

    def __len__(self):
        return len(self.tensors)


class AddUniformNoise:
    def __init__(self, min_val=-0.1, max_val=0.1, p=0.5):
        """
        min_val, max_val: range of uniform noise
        p: probability to apply the transform
        """
        self.min_val = min_val
        self.max_val = max_val
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            noise = torch.empty_like(x).uniform_(self.min_val, self.max_val)
            x = x + noise
        return x

def gen_dataloaders(path, n_chunks, batch_size):
    spect_chunks_train, labels_train, spect_chunks_val, labels_val = gen_datasets(path, n_chunks)
    train_ds_torch = FastTensorDataset(spect_chunks_train, labels_train)
    val_ds_torch = FastTensorDataset(spect_chunks_val, labels_val)
    train_loader = DataLoader(train_ds_torch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds_torch, batch_size=batch_size)
    input_shape = spect_chunks_train[0].shape
    num_classes = len(set(labels_train))
    return train_loader, val_loader, input_shape, num_classes


def plot_ds_examples(train_ds_torch, val_ds_torch):
    # Check dataset sizes
    print(f"Train dataset size: {len(train_ds_torch)}")
    print(f"Validation dataset size: {len(val_ds_torch)}")

    # Plot 10 samples from each dataset
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for i in range(10):
        # Train samples
        X_, y = train_ds_torch[i]
        X = X_.detach().numpy()
        axes[0, i].imshow(X, aspect="auto", origin="lower")
        axes[0, i].set_title(f"Train label: {y}")
        axes[0, i].axis("off")
        # Validation samples
        Xv, yv = val_ds_torch[i]
        axes[1, i].imshow(Xv, aspect="auto", origin="lower")
        axes[1, i].set_title(f"Val label: {yv}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

def to_cuda(model):
    #  List available CUDA devices
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
    
def plot_metrics(metrics):
    print("Max Acc:", np.amax(metrics["val_acc"]))
    plt.figure(figsize=(20, 10))

    # Plot train and val loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    if metrics["val_loss"]:
        plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Plot train accuracy
    plt.subplot(1, 3, 2)
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy")
    plt.legend()

    # Plot val accuracy
    plt.subplot(1, 3, 3)
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix(model, device, n_classes, ds_loader):
    # Suppose cm is your confusion matrix (2D numpy array)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Collect predictions and fill cm
    model.eval()
    with torch.no_grad():
        for X, y in ds_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(y.cpu().numpy(), predicted.cpu().numpy()):
                cm[t, p] += 1

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.show()