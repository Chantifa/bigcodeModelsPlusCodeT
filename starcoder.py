import requests
import torch

# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# download the code
checkpoint = ("bigcode/starcoderbase-1b")

# chose gpu or cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# save the model to the local cache
tokenizer.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--bigcode--starcoderbase-1b")
model.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--bigcode--starcoderbase-1b")

tokenizer = AutoTokenizer.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--bigcode--starcoderbase-1b", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("C:/Users/X/.cache/huggingface/hub/models--bigcode--starcoderbase-1b", local_files_only=True).to(device)

input = "show in python cosinus function"
inputs = tokenizer.encode(input, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))