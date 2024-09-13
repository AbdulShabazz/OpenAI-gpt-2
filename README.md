# GPT-2: Language Models as Unsupervised Multitask Learners

## Overview

This repository contains an updated implementation of the GPT-2 model, **refactored for TensorFlow v2.17.0**. It is based on the research presented in ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

For comprehensive information about GPT-2 and its staged release, please refer to:
- [Original blog post](https://openai.com/research/better-language-models/)
- [6-month follow-up post](https://openai.com/blog/gpt-2-6-month-follow-up/)
- [Final release post](https://www.openai.com/blog/gpt-2-1-5b-release/)

Additionally, a [dataset](https://github.com/openai/gpt-2-output-dataset) has been released for researchers to study the model's behaviors.

## Project Structure

The repository is organized into multiple files, each serving a specific purpose:

1. **gpt.py**
   - Core GPT-2 model logic
   - Key functions:
     ```python
     load_model()  # Loads the GPT-2 model
     train_model()  # Trains or fine-tunes the GPT-2 model on a given dataset
     save_model()  # Saves the trained model for later use
     query_model(prompt)  # Takes a user-provided text prompt and returns a completion
     ```

2. **gpt_core_v2.py**
   - API definitions for GPT architecture
   - Key functions:
     ```python
     train_via_api()  # API endpoint for training the GPT-2 model through external calls
     save_model_via_api()  # API endpoint to save the current model state through external calls
     ```

3. **codec.py**
   - Encoder and decoder modules
   - Key functions:
     ```python
     encode(text)  # Converts text into tokenized input for the model
     decode(tokens)  # Converts tokenized output from the model into readable text
     ```

4. **example.py**
   - Demonstration script for model interaction
   - Contains:
     ```python
     # Example usage of query_model() to generate text completions
     # Sample inputs and outputs from the GPT-2 model
     ```

## Usage Guidelines

This repository serves as a starting point for researchers and engineers to experiment with GPT-2. Please refer to our [model card](./model_card.md) for basic information.

- Model updates: Utilize endpoints in `gpt_core_v2.py`
- Model loading/training: Use functions in `gpt.py`
- Encoding/decoding: Employ helper functions in `codec.py`
- Basic querying: See `example.py` for demonstration

## Important Considerations

1. **Robustness**: GPT-2 models' worst-case behaviors are not fully understood. Evaluate carefully for your use case, especially in safety-critical applications.
2. **Bias**: The training dataset contains texts with biases and inaccuracies. GPT-2 models may reflect these issues.
3. **Synthetic Text Labeling**: To prevent misidentification, clearly label model-generated samples as synthetic before wide dissemination.

## Research Collaboration

We welcome collaboration on interesting research or applications of GPT-2. We are particularly interested in:
- Studying potential misuse cases and developing defenses
- Investigating and mitigating problematic content (e.g., bias) in model outputs

Please contact us at [languagequestions@openai.com](mailto:languagequestions@openai.com) for inquiries or collaborations.

## Development and Contributions

- For development guidelines, see [DEVELOPERS.md](./DEVELOPERS.md)
- For a list of contributors, see [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Citation

Please use the following BibTeX entry:

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future Work

- Potential release of code for model evaluation on various benchmarks
- Considering the release of larger models

## License

This project is licensed under the [Modified MIT License](./LICENSE).
