import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"# for GPU usage or "cpu" for CPU usage

checkpoint = "Salesforce/codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float16,
trust_remote_code=True,
force_download=True,
resume_download=True)


tokenizer.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small")
model.save_pretrained("C:/Users/X/.cache/huggingface/hub/models--Salesforce--codet5-small")

tokenizer = RobertaTokenizer.from_pretrained("C:/Users/X/.cache\huggingface/hub/models--Salesforce--codet5-small/snapshots/a642dc934e5475185369d09ac07091dfe72a31fc", local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained("C:/Users/X/.cache\huggingface/hub/models--Salesforce--codet5-small/snapshots/a642dc934e5475185369d09ac07091dfe72a31fc", local_files_only=True)
text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "user: {user.name}"


