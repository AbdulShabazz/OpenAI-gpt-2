import subprocess
import os
import sys
from modules_to_import import modules_to_import

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # python -m pip install [[package]|[-r requirements.txt]]

print('Installing missing packages...')

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
            print("\033[33mWARNING: The [torch] module also requires additional DLLs to be installed, which can collectively be downloaded in Powershell (Windows) with the command: curl -o vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe\033[0m]")

print("\nDownloads Complete!")
