# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="bigcode/santacoder", trust_remote_code=True)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
device

tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("bigcode/santacoder", trust_remote_code=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/santacoder"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
input_text = "<fim-prefix>def print_hello_world():\n    <fim-suffix>\n    print('Hello world!')<fim-middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))