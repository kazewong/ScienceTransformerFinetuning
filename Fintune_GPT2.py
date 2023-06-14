from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

# files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = ["data/GWtext/Backpop.raw"]
tokenizer.train(files, trainer)

from transformers import PreTrainedTokenizerFast, GPT2Tokenizer, GPT2Model

# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


from datasets import load_dataset, DatasetDict
raw_dataset = load_dataset("text", data_files={"train": ["./data/GWtext/Backpop.raw"]})['train'].train_test_split(test_size=0.1, train_size=0.9, shuffle=True, seed=42)
valid_test_dataset = raw_dataset['test'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True, seed=42)

ds_train = DatasetDict({
    'train': raw_dataset['train'],
    'test': valid_test_dataset['test'],
    'valid': valid_test_dataset['train']})
    
context_length = 64
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = ds_train.map(
    tokenize, batched=True,batch_size=100, remove_columns=ds_train["train"].column_names
)


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from torch.nn import CrossEntropyLoss
import torch

keytoken_ids = []
for keyword in [
    "gravitational",
    "wave",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss

from torch.utils.data.dataloader import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=4, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=4)

weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='fp16')

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 50
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

from tqdm.notebook import tqdm

gradient_accumulation_steps = 8
eval_steps = 5_000


model.train()
completed_steps = 0
output_dir = "./data/"
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "steps": completed_steps,
                    "loss/eval": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                
from transformers import pipeline
model = model.eval()
untuned_generator = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id), tokenizer=tokenizer, max_new_tokens= context_length)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device=0, max_new_tokens= context_length)

prompt = "The population of black holes provides"
generate_result = generator([prompt for i in range(3)])
untuned_generate_result = untuned_generator([prompt for i in range(3)])

for i in range(3):
    print("Tuned: ", generate_result[i][0]["generated_text"])
    print("")
    print("Untuned: ", untuned_generate_result[i][0]["generated_text"])
    print("")