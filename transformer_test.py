from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

from transformers import GPT2Tokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=16)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]