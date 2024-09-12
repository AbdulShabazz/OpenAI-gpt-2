import os
import sys

try:
    import aiohttp
except ImportError:
    print('Error [aiohttp] module not installed. Please run download_model_prereqs.py to install [aiohttp]')    
    sys.exit(1)

try:
    import aiofiles
except ImportError:
    print('Error [aiofiles] module not installed. Please run download_model_prereqs.py to install [aiofiles]')    
    sys.exit(1)

try:
    import asyncio
except ImportError:
    print('Error [asyncio] module not installed. Please run download_model_prereqs.py to install [asyncio]')    
    sys.exit(1)

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
subdir = subdir.replace('\\','/') # needed for Windowsimport asyncio

async def download_chunk(session, url, start, end, file, pbar):
    headers = {'Range': f'bytes={start}-{end}'}
    async with session.get(url, headers=headers) as response:
        chunk = await response.read()
        await file.seek(start)
        await file.write(chunk)
        pbar.update(len(chunk))

async def download_file(session, url, filename, subdir, chunk_size=1024*1024):
    async with session.head(url) as response:
        file_size = int(response.headers.get('Content-Length', 0))

    path = os.path.join(subdir, filename)
    async with aiofiles.open(path, 'wb') as file:
        tasks = []
        with tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"Fetching {filename}") as pbar:
            for start in range(0, file_size, chunk_size):
                end = min(start + chunk_size - 1, file_size - 1)
                task = asyncio.create_task(download_chunk(session, url, start, end, file, pbar))
                tasks.append(task)
            await asyncio.gather(*tasks)

async def main():
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/"
    filenames = ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']

    os.makedirs(subdir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for filename in filenames:
            url = base_url + subdir + "/" + filename
            task = asyncio.create_task(download_file(session, url, filename, subdir))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
