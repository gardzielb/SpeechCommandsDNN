import os
import warnings

import torch

from speech_dnn.experiments import grid_search
from speech_dnn.networks.AttRnn import AttRnn

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category = UserWarning)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	
	grid_search(
		n_epochs = 1, experiment_title = 'test-3',
		seed = 2137, fold_count = 5, repeat_count = 3,
		network_cls = AttRnn,
		param_grid = {
			'batch_size': [128],
			'lr': [0.003, 0.01],
			'l2': [0.0]
		}
	)
