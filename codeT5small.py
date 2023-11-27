import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

checkpoint = "Salesforce/codet5-small"
device = "cuda"  if torch.cuda.is_available() else "cpu"# for GPU usage or "cpu" for CPU usage
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float16,
trust_remote_code=True,
force_download=True,
resume_download=True)

text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "user: {user.name}"


# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

tokenizer = RobertaTokenizer.from_pretrained.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True,
                                              force_download=True,
                                              resume_download=True
                                              ).to(device)

tokenizer.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small")
model.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small")

tokenizer = RobertaTokenizer.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small", local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small", local_files_only=True)

from transformers import AutoConfig

device = "cuda"  if torch.cuda.is_available() else "cpu"# for GPU usage or "cpu" for CPU usage

config = AutoConfig.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b/Salesforce/codet5p-16b/config.json")

function = "give me in java a book class"
encoding = tokenizer(function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

