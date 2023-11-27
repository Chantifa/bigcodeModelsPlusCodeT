import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="Salesforce/codet5p-16b", filename="config.json", cache_dir="C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b/")

checkpoint = "Salesforce/codet5p-16b"
device = "cuda"  if torch.cuda.is_available() else "cpu"# for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True,
                                              force_download=True,
                                              resume_download=True
                                              ).to(device)

tokenizer.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b")
model.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b")

tokenizer = AutoTokenizer.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b", local_files_only=True)

from transformers import AutoConfig

device = "cuda"  if torch.cuda.is_available() else "cpu"# for GPU usage or "cpu" for CPU usage

config = AutoConfig.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5p-16b/Salesforce/codet5p-16b/config.json")

function = "give me in java a book class"
encoding = tokenizer(function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

