import subprocess
import os
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # python -m pip install [[package]|[-r requirements.txt]]

print('Installing missing packages...')

modules_to_import = [
    ('fire', 'fire'),
    ('regex', 're'),
    ('aiohttp', 'aiohttp'),
    ('aiofiles', 'aiofiles'),
    ('asyncio', 'asyncio'),
    ('torch', 'pytorch'),
    ('requests', 'requests'),
    ('tqdm', 'tqdm', 'from tqdm import tqdm')
]

for module in modules_to_import:
    try:
        if len(module) == 3:
            exec(module[2])
        else:
            exec(f"import {module[0]} as {module[1]}")
    except ImportError:
        install(module[0])
        #install("torchvision") # pip install torchvision 
        #install("torchaudio") # pip install torchaudio
        if module[0] == "torch":
            print('***INFO***: the [torch] module also requires some additional DLLs which can be downloaded with Powershell: curl -o vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe')

print("\nDownloads Complete!")
