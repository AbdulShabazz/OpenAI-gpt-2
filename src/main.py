"""test case encoding/decoding utility"""

import sample
import model
import encoder

# Define the prompt
prompt = "Hello, how are you today?"

gpt2 = encoder.get_encoder('124M','../models')

tokens = gpt2.encode(prompt) # [15496, 11, 703, 389, 345, 1909, 30]

# Generate a response (optional)
hparams = model.default_hparams()
output = sample.sample_sequence(hparams,tokens, length=50)

# Decode the generated output
generated_text = gpt2.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
