import os
from pathlib import Path

root = Path('./data')
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import RepeatedKFold
from itertools import islice
from pytorch_lightning import LightningDataModule
from torchvision.datasets import DatasetFolder
import torch
import torchaudio.transforms as aT
import torchaudio

SOUND_EXTENSIONS = (".wav")

class SoundFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        sample_rate = 16000        
    ):
        transform = torch.nn.Sequential(
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
            transform=transform,
            target_transform=None,
            is_valid_file=None,
        )
        self.imgs = self.samples


def iter_nth(iterable, nth):
	return next(islice(iterable, nth, nth + 1))

class KFoldImageDataModule(LightningDataModule):
	def __init__(
			self,
			i: int,  # fold id
			batch_size: int = 32,
			k: int = 5,  # fold count,
			n_repeats = 5,
			split_seed: int = 12345
	):
		super().__init__()
		self.batch_size = batch_size
		self.k = k
		self.split_seed = split_seed
		self.i = i
		self.n_repeats = n_repeats
		data_path = os.getenv('DATA_PATH', './data')
		self.train_dataset = SoundFolder(f'{data_path}/train/audio')
		self.validation_dataset = SoundFolder(f'{data_path}/train/audio')
		self.train_fold = None
		self.val_fold = None

	def setup(self, stage = None):
		if not self.train_fold and not self.val_fold:
			# choose fold to train on
			kf = RepeatedKFold(n_splits = self.k, n_repeats = self.n_repeats, random_state = self.split_seed)
			train_indexes, val_indexes = iter_nth(kf.split(self.train_dataset), self.i)
			self.train_fold = Subset(self.train_dataset, train_indexes)
			self.val_fold = Subset(self.validation_dataset, val_indexes)
			

	def train_dataloader(self):
		return DataLoader(dataset = self.train_fold, batch_size = self.batch_size, num_workers = 10, pin_memory = True,
						  shuffle = True)

	def val_dataloader(self):
		return DataLoader(dataset = self.val_fold, batch_size = self.batch_size, num_workers = 10, pin_memory = True)

	def classes(self):
		return self.train_dataset.classes