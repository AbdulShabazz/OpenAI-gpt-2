# gpt-2

(This repository updates the entire gpt-2 codebase to tensorflow v2.17.0.)

Code and models from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

You can read about GPT-2 and its staged release in our [original blog post](https://openai.com/research/better-language-models/), [6 month follow-up post](https://openai.com/blog/gpt-2-6-month-follow-up/), and [final post](https://www.openai.com/blog/gpt-2-1-5b-release/).

We have also [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.

<sup>*</sup> *Note that our original parameter counts were wrong due to an error (in our previous blog posts and paper).  Thus you may have seen small referred to as 117M and medium referred to as 345M.*

# GPT-2 Python Model Directory

This repository contains the refactored implementation of a GPT-2 model in Python. The project has been organized into multiple files, each serving a distinct purpose to streamline both model functionality and API interaction.

## Project Structure

### 1. `gpt.py`
   - **Purpose**: 
     This file houses the core GPT-2 model logic. It handles tasks like loading pre-trained models, training new models, fine-tuning on datasets, saving/loading models from disk, and managing model parameters.
   - **Key Functions**:
     - `load_model()`: Loads the GPT-2 model.
     - `train_model()`: Trains or fine-tunes the GPT-2 model on a given dataset.
     - `save_model()`: Saves the trained model for later use.

### 2. `gpt_core_v2.py`
   - **Purpose**:
     This file contains all the APIs that define how the GPT-2 model is queried, trained, or interacted with by external applications. Any updates to the APIs in this file will reflect updates to how the GPT-2 model can be accessed.
   - **Key Functions**:
     - `query_model(prompt)`: Takes in a user-provided text prompt and returns a text completion from the GPT-2 model.
     - `train_via_api()`: An API endpoint that allows training the GPT-2 model through external calls.
     - `save_model_via_api()`: An API endpoint to save the current state of the model through external calls.

### 3. `codec.py`
   - **Purpose**:
     This file contains the encoder and decoder modules, responsible for parsing prompts into tokenized formats and converting model outputs back into human-readable text.
   - **Key Functions**:
     - `encode(text)`: Converts text into tokenized input to be fed into the model.
     - `decode(tokens)`: Converts tokenized output from the model back into readable text.

### 4. `example.py`
   - **Purpose**:
     This file is an example script to demonstrate how to interact with the GPT-2 model using the API from `gpt_core_v2.py`. It shows sample queries and potential use cases.
   - **Key Functions**:
     - Example use of `query_model()` to generate text completions.
     - Sample inputs and outputs from the GPT-2 model.

## Usage

- To interact with the GPT-2 model through APIs, use the endpoints defined in `gpt_core_v2.py`.
- To load or train the model, refer to the functions in `gpt.py`.
- For encoding/decoding tasks, utilize the helper functions in `codec.py`.
- Refer to `example.py` for a basic demonstration of how to query the model.

## Future Updates

- The APIs in `gpt_core_v2.py` will be updated as needed to reflect the latest interaction methods for the GPT-2 model.
- Future versions of the API file will follow a similar naming convention (e.g., `gpt_core_v3.py` for subsequent updates).

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

For basic information, see our [model card](./model_card.md).

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[Modified MIT](./LICENSE)
