import os
import warnings

import torch

from speech_dnn.experiments import grid_search
from speech_dnn.networks.FakeNetwork import FakeNetwork

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category = UserWarning)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	
	grid_search(
		n_epochs = 150, out_path = f'results/bs{32}-lr{0.003}-l2',
		seed = 2137, fold_count = 5, repeat_count = 5,
		network_cls = FakeNetwork,
		param_grid = {
			'batch_size': [32, 64],
			'learn_rate': [0.003, 0.01],
			'l2': [0.0, 0.0005]
		}
	)
