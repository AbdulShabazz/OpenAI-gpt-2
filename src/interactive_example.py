#!/usr/bin/env python3

"""interactive GPT prompt environment"""

#from importlib.resources import open_text
import json
import os
import fire
import numpy as np
import tensorflow as tf
import codec
import gpt

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='../models'
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
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = codec.get_encoder(model_name, models_dir)
    
    hparams = gpt.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json'), encoding="UTF-8") as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError(f"Can't capture samples longer than window size: {hparams.n_ctx}")

    # Set up the seed for reproducibility
    seed = 42  # Or whatever seed value you were using before
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create the model
    gpt2_instance = gpt.create_model(hparams)  # Assuming you have a function to create the model

    # Set up the checkpoint manager
    checkpoint_dir = os.path.join(models_dir, model_name)
    checkpoint = tf.train.Checkpoint(model=gpt2_instance)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # Restore the latest checkpoint
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Model restored from {ckpt_manager.latest_checkpoint}")
    else:
        print("Initializing model...")

    # Interactive prompt loop
    while True:
        raw_text = input("Model prompt >>> ")
        while not raw_text:
            print('Please supply a text Prompt to the model!')
            raw_text = input("Model prompt >>> ")
        
        context_tokens = enc.encode(raw_text)
        tokens_length = len(context_tokens)
        context_tokens_tensor = tf.convert_to_tensor([context_tokens] * batch_size, dtype=tf.int32)    
        generated = 0

        for _ in range(nsamples // batch_size):
            output = gpt.submit_query(context = context_tokens_tensor, length = tokens_length, batch_size = batch_size)
            output = output[:, tokens_length:].numpy()
            
            for i in range(batch_size):
                generated += 1
                text = enc.decode(output[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        
        print("=" * 80)


if __name__ == '__main__':
    fire.Fire(interact_model)

