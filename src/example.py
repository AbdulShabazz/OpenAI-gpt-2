"""test gpt2 model"""

import gpt

model_name = '124M'
models_dir = '../models'

prompt = "Hello, how are you today?"

text_completion = gpt.submit_text_query(model_name=model_name, models_dir=models_dir, context=prompt, length=50)

print("Generated text:", text_completion)
