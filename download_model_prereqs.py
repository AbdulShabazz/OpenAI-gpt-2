import subprocess
import os
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # python -m pip install [package|-r requirements.txt]

print('Installing missing packages...')

try:
    import fire
except ImportError:
    install("fire")

try:
    import regex as re
except ImportError:
    install("regex")

try:
    import requests
except ImportError:
    install("requests") # pip install requests

try:
    from tqdm import tqdm
except ImportError:
    install("tqdm") # pip install tqdm

try:
    import tensorflow as tf
except ImportError:
    install("tensorflow")

try:
   import torch as pytorch
except ImportError:
   install("torch") # pip install torch
   #install("torchvision") # pip install torchvision 
   #install("torchaudio") # pip install torchaudio
   print('***INFO***: the [torch] module also requires some additional DLLs which can be downloaded with Powershell: curl -o vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe')

print("\nDownloads Complete!")
