import requests
import torch

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoderbase-1b"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Can you please let us know more details about your ",
})

# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = ("bigcode/starcoderbase-1b")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))