import transformers # GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import datasets # load_dataset


model_name = "gpt2" # You can choose from 'gpt2', 'gpt2-medium', 'gpt2-large', etc.
model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

optional_custom_training = """

# Adjust the tokenizer for code (optional step)
# Adding a special [PAD] token if not already included
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize model embeddings to accommodate any new special tokens
model.resize_token_embeddings(len(tokenizer))

# Load and prepare your dataset (replace 'java_code_dataset.txt' with your dataset)
dataset = load_dataset('text', data_files={'train': 'java_code_dataset.txt'})

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
    inputs["labels"] = inputs["input_ids"].copy()
    
    # Debugging: Print out token IDs to ensure they're within bounds
    print("Token IDs:", inputs['input_ids'][0])  # Print the first example for inspection
    
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-java-instrumentation",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced batch size to prevent memory issues
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-java-instrumentation")
tokenizer.save_pretrained("./gpt2-java-instrumentation")

# Generate OpenTelemetry instrumentation suggestions for a given Java code
model = GPT2LMHeadModel.from_pretrained("./gpt2-java-instrumentation")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-java-instrumentation")

# Example non-instrumented Java code
prompt = ""
instrument with opentelemetry:

public class ExampleService {
    public void process() {
        System.out.println("Processing data...");
    }
}
""
"""

# Example prompt string
prompt = """
Once upon a time long ago, 
"""

prompt_length_in_tokens = len(prompt) * 2

max_length = prompt_length_in_tokens if prompt_length_in_tokens < 1024 else 1024

# Generate instrumented code suggestion
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)  # max_length=1024

# Decode and print the generated code
print(tokenizer.decode(outputs[0], skip_special_tokens=True))