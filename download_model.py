import subprocess
import os
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # python -m pip install [package|[-r requirements.txt]]

try:
   import torch as pytorch
except ImportError:
    print('Error [torch] module not installed. Please run download_model_prereqs.py to install [torch]')    
    sys.exit(1)

try:
    import requests
except ImportError:
    print('Error [requests] module not installed. Please run download_model_prereqs.py to install [requests]')
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print('Error [tqdm] module not installed. Please run download_model_prereqs.py to install [tqdm]')
    sys.exit(1)

if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 124M')
    sys.exit(1)

model = sys.argv[1]

subdir = os.path.join('models', model)
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:

    r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/" + subdir + "/" + filename, stream=True)

    with open(os.path.join(subdir, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
