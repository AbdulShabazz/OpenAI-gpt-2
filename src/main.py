"""test gpt2 model"""

import gpt2

model_name = '124M'
model_dir = '../models'
prompt = "Hello, how are you today?"

text_completion = gpt2.sample_sequence(model_name=model_name, model_dir=model_dir, context=prompt, length=50)

print("Generated text:", text_completion)