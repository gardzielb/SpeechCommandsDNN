import argparse
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('src_path')
	arg_parser.add_argument('-o', '--out-path', type = str)
	arg_parser.add_argument('-f', '--format', type = str, default = 'wav')
	arg_parser.add_argument('-n', '--n-samples', type = int, default = 16_000)
	args = arg_parser.parse_args()

	src_path = Path(args.src_path)
	if not src_path.exists():
		print('Source directory does not exist')
		exit(1)

	out_path = Path(args.out_path)
	out_path.mkdir(parents = True, exist_ok = True)

	n_samples = args.n_samples
	audio_fmt = args.format

	for path in src_path.rglob(f'*.{audio_fmt}'):
		audio, sample_rate = sf.read(path)
		rec_title = path.stem

		samples_range = range(0, len(audio), n_samples)
		for i, from_idx in enumerate(tqdm(samples_range, desc = rec_title)):
			to_idx = min(from_idx + n_samples, len(audio))
			sf.write(out_path.joinpath(f'{rec_title}_{i + 1}.{audio_fmt}'), audio[from_idx:to_idx], sample_rate)
