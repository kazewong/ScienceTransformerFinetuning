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
fast_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


from datasets import load_dataset, DatasetDict
ds_train = load_dataset("text", data_files={"train": ["./data/GWtext/Backpop.raw"]})

context_length = 128
def tokenize(element):
    outputs = fast_tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length > 0:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = ds_train.map(
    tokenize, batched=True,batch_size=100, remove_columns=ds_train["train"].column_names
)

model = GPT2Model.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = fast_tokenizer(text, return_tensors='pt')
output = model(**encoded_input)