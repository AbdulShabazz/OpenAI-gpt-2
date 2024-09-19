import os
import sys
from modules_to_import import modules_to_import

for module in modules_to_import:
    try:
        if len(module) == 3:
            exec(module[2])
        else:
            exec(f"import {module[0]} as {module[1]}")
    except ImportError:
        print(f"Error [{module[0]}] module not installed. Please run download_model_prereqs.py to install [{module[0]}]")
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
