from pathlib import Path
import pandas as pd

from speech_dnn.networks.AttRnn import AttRnn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import torch
from speech_dnn.data_augmentation import SpectrogramAugmentation
import torchaudio.transforms as aT
from torchvision.datasets import DatasetFolder
import torchaudio
from tqdm import tqdm
import os
import torchmetrics
import nemo
import nemo.collections.asr as nemo_asr
from typing import Optional
from omegaconf import OmegaConf
import numpy as np

SOUND_EXTENSIONS = (".wav")

class SoundFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        sample_rate = 16000        
    ):
        transform = torch.nn.Sequential(
            aT.MelSpectrogram(
                n_fft=512, 
                win_length=400, 
                hop_length=160, 
                f_min=40.0, 
                f_max=8000.0, 
                normalized=True, 
                n_mels=64
            )
        ) 
        
        def loader(path):
            waveform, sr = torchaudio.load(path)
            padded_audio = torch.nn.functional.pad(waveform, (0, 16000-waveform.shape[1]), value=0.0)
            if sample_rate != sr:
                raise ValueError(f"sample rate should be {sample_rate}, but got {sr}")
            return padded_audio

        super().__init__(
            root,
            loader,
            SOUND_EXTENSIONS,
            transform=None,
            target_transform=None,
            is_valid_file=None,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int):

      features, labels = super().__getitem__(index)
      
      features = torch.squeeze(features)

      fl = torch.tensor(features.shape[0]).long()
      tl = torch.tensor(1).long()

      return features, fl, labels, tl

class MatchboxNet(nemo_asr.models.EncDecClassificationModel):	
    def __init__(self, cfg):
        super().__init__(cfg=cfg)        
        self.__last_cm__: Optional[torch.FloatTensor] = None

    def get_test_confusion_matrix(self) -> np.ndarray:
        return self.__last_cm__.detach().cpu().numpy()

    def on_validation_epoch_end(self):
        self.__last_cm__ = self.cm.compute()
        self.cm.reset()

    def validation_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        val_loss = self.loss(logits=logits, labels=labels)        
        self.log("val_loss", val_loss)        
        self.cm.update(logits, labels)

class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_dataset = SoundFolder(f'data/train/audio')
        self.test_dataset = SoundFolder(f'data/test')
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
    data_module = SpeechCommandsDataModule(256)
    config_path = f"matchboxnet_3x1x64_v1.yaml"
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    trainer = pl.Trainer(accelerator = 'gpu', devices = 1, max_epochs = 20, precision=16, limit_val_batches=0.0, default_root_dir='./checkpoint')
    config.model.train_ds.batch_size = 256
    config.model.optim.lr = 0.01
    train_dataset = os.path.join("google_dataset_v1/google_speech_recognition_v1", 'train_manifest.json')
    val_dataset = os.path.join("google_dataset_v1/google_speech_recognition_v1", 'validation_manifest.json')
    test_dataset = os.path.join("google_dataset_v1/google_speech_recognition_v1", 'validation_manifest.json')
    config.model.train_ds.manifest_filepath = train_dataset
    config.model.validation_ds.manifest_filepath = val_dataset
    config.model.test_ds.manifest_filepath = test_dataset
    model = MatchboxNet(cfg=config.model, trainer=trainer)
    trainer.fit(model, datamodule = data_module)
    preds = []   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    test_dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for X, xl, _, _ in tqdm(test_dataloader):
            X = X.to(device)
            preds.extend(model.forward(input_signal=X, input_signal_length=xl).argmax(dim=1).type(torch.int32).cpu().numpy())
    a = os.listdir('data/test/audio')

    df = pd.DataFrame({'fname': sorted(a), 'label': preds})
    df['label'] = df['label'].apply(lambda x: data_module.classes()[x])
    df.to_csv('submission_matchbox.csv', index=False)