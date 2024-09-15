#!/usr/bin/env python3

"""interactive GPT prompt environment"""

import os
import fire
#import numpy as np
#import tensorflow as tf
import gpt

def interactive_model(
    model_name='124M',
    seed=None,
    nsamples=3,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=1,
    top_p=1,
    models_dir='../models' # Adjust as needed during DEBUG mode
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    # if batch_size is None:
    #     batch_size = 1
    # assert nsamples % batch_size == 0

    # gpt_codec = gpt.get_codec(model_name, models_dir)
    # hparams = gpt.get_default_hparams(model_name, models_dir, 'hparams.json')

    # if length is None:
    #     length = hparams.n_ctx // 2
    # elif length > hparams.n_ctx:
    #     raise ValueError(f"Can't capture samples wider than the window size: {hparams.n_ctx}")

    # Set up the seed for reproducibility
    seed = 42  # Or whatever seed value you were using before
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    # Create the model
    # gpt2_instance = gpt.create_model(hparams)  # Assuming you have a function to create the model

    # # Set up the checkpoint manager
    # checkpoint_dir = os.path.join(models_dir, model_name)
    # checkpoint = tf.train.Checkpoint(model=gpt2_instance)
    # ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # # Restore the latest checkpoint
    # if ckpt_manager.latest_checkpoint:
    #     checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    #     print(f"Model restored from {ckpt_manager.latest_checkpoint}")
    # else:
    #     print("Initializing model...")

    print("Initializing model...")

    # Interactive prompt loop
    while True:
        raw_text = input("Model prompt >>> ") # "Hello, how are you ?"
        while not raw_text:
            print('Please supply a text Prompt to the model!')
            raw_text = input("Model prompt >>> ")

        generated = 0
        while nsamples == 0 or generated < nsamples:
            text_output = gpt.submit_text_query(
                context = raw_text,
                length = length,
                batch_size = None,
                models_dir = models_dir,
                model_name = model_name,
                seed=seed )
            
            print("=" * 40 + f" COMPLETION {i} " + "=" * 40)
            for _, text in enumerate(text_output):
                print(text)

            generated += 1
        
        print("=" * 80)


if __name__ == '__main__':
    fire.Fire(interactive_model)
    
