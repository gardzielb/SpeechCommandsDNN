from pathlib import Path
import pandas as pd

from speech_dnn.networks.AttRnn import AttRnn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from speech_dnn.data_loader import SoundFolder
import torch
from speech_dnn.data_augmentation import SpectrogramAugmentation
import torchaudio.transforms as aT
from tqdm import tqdm
import os


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        
        
        train_transform = torch.nn.Sequential(
            aT.MelSpectrogram(
                n_fft=1024, 
                win_length=1024, 
                hop_length=128, 
                f_min=40.0, 
                f_max=8000.0, 
                normalized=True, 
                n_mels=80
            ),
            SpectrogramAugmentation(
                freq_masks= 2,
                time_masks= 2,
                freq_width= 15,
                time_width= 25
            )
        ) 
        val_transform = torch.nn.Sequential(
            aT.MelSpectrogram(
                n_fft=1024, 
                win_length=1024, 
                hop_length=128, 
                f_min=40.0, 
                f_max=8000.0, 
                normalized=True, 
                n_mels=80
            )
        ) 
        self.train_dataset = SoundFolder(f'data/train/audio', transform=train_transform)
        self.test_dataset = SoundFolder(f'data/test2', transform=val_transform)
        self.train_fold = None
        self.val_fold = None

    def train_dataloader(self):
        return DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, num_workers = 14, pin_memory = True,
                          shuffle = True)

    def test_dataloader(self):
        num_gpus = torch.cuda.device_count()
        return DataLoader(dataset = self.test_dataset, batch_size=1024, shuffle=False, num_workers=14, pin_memory=True)


    def classes(self):
        return self.train_dataset.classes

if __name__ == '__main__':
    root = Path('./data')
    data_module = SpeechCommandsDataModule(64)
    model = AttRnn(len(data_module.classes()), lr = 0.001)
    
    trainer = pl.Trainer(accelerator = 'gpu', devices = 1, max_epochs = 40, precision=16, limit_val_batches=0.0, default_root_dir='./checkpoint')
    trainer.fit(model, datamodule = data_module)
    preds = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    test_dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for X, _ in tqdm(test_dataloader):
            X = X.to(device)
            preds.extend(model(X).argmax(dim=1).type(torch.int32).cpu().numpy())
    a = os.listdir('data/test2/audio')

    df = pd.DataFrame({'fname': sorted(a), 'label': preds})
    df['label'] = df['label'].apply(lambda x: data_module.classes()[x])
    df.to_csv('submission.csv', index=False)