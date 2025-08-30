import random

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

from hs_hackathon_drone_acoustics import CLASSES, RAW_DATA_DIR
from hs_hackathon_drone_acoustics.base import AudioDataset
from hs_hackathon_drone_acoustics.plot import plot_spectrogram

def convert_to_mtx(waveform):
    resample = T.Resample(waveform.sample_rate, 16_000)
    waveform_to_spectrogram = T.Spectrogram(win_length=4096, n_fft=4096, hop_length=2048)
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    spectrogram = db_transform(waveform_to_spectrogram(resample(waveform.data)))
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
        X, y = train_ds_torch[i]
        axes[0, i].imshow(X.cpu().numpy(), aspect="auto", origin="lower")
        axes[0, i].set_title(f"Train label: {y.item()}")
        axes[0, i].axis("off")
        # Validation samples
        Xv, yv = val_ds_torch[i]
        axes[1, i].imshow(Xv.cpu().numpy(), aspect="auto", origin="lower")
        axes[1, i].set_title(f"Val label: {yv.item()}")
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
    
    
    