# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="bigcode/santacoder", trust_remote_code=True)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

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
# Author Santa Coder --> Hugging Face
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import pipeline
import os
import torch

description = """# <p style="text-align: center; color: white;"> 🎅 <span style='color: #ff75b3;'>SantaCoder:</span> Code Generation </p>
<span style='color: white;'>This is a demo to generate code with <a href="https://huggingface.co/bigcode/santacoder" style="color: #ff75b3;">SantaCoder</a>,
a 1.1B parameter model for code generation in Python, Java & JavaScript. The model can also do infilling, just specify where you would like the model to complete code
with the <span style='color: #ff75b3;'>&lt;FILL-HERE&gt;</span> token.</span>"""

token = os.environ["HUB_TOKEN"]
device="cuda:0"


FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

GENERATION_TITLE= "<p style='font-size: 16px; color: white;'>Generated code:</p>"

tokenizer_fim = AutoTokenizer.from_pretrained("bigcode/santacoder", use_auth_token=token, padding_side="left")

tokenizer_fim.add_special_tokens({
    "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
    "pad_token": EOD,
})

tokenizer = AutoTokenizer.from_pretrained("bigcode/christmas-models", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("bigcode/christmas-models", trust_remote_code=True, use_auth_token=token).to(device)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def post_processing(prompt, completion):
    completion = "<span style='color: #ff75b3;'>" + completion + "</span>"
    prompt = "<span style='color: #727cd6;'>" + prompt + "</span>"
    code_html = f"<br><hr><br><pre style='font-size: 12px'><code>{prompt}{completion}</code></pre><br><hr>"
    return GENERATION_TITLE + code_html

def post_processing_fim(prefix, middle, suffix):
    prefix = "<span style='color: #727cd6;'>" + prefix + "</span>"
    middle = "<span style='color: #ff75b3;'>" + middle + "</span>"
    suffix = "<span style='color: #727cd6;'>" + suffix + "</span>"
    code_html = f"<br><hr><br><pre style='font-size: 12px'><code>{prefix}{middle}{suffix}</code></pre><br><hr>"
    return GENERATION_TITLE + code_html

def fim_generation(prompt, max_new_tokens, temperature):
    prefix = prompt.split("<FILL-HERE>")[0]
    suffix = prompt.split("<FILL-HERE>")[1]
    [middle] = infill((prefix, suffix), max_new_tokens, temperature)
    return post_processing_fim(prefix, middle, suffix)

def extract_fim_part(s: str):
    # Find the index of
    start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    stop = s.find(EOD, start) or len(s)
    return s[start:stop]

def infill(prefix_suffix_tuples, max_new_tokens, temperature):
    if type(prefix_suffix_tuples) == tuple:
        prefix_suffix_tuples = [prefix_suffix_tuples]

    prompts = [f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}" for prefix, suffix in prefix_suffix_tuples]
    # `return_token_type_ids=False` is essential, or we get nonsense output.
    inputs = tokenizer_fim(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    # WARNING: cannot use skip_special_tokens, because it blows away the FIM special tokens.
    return [
        extract_fim_part(tokenizer_fim.decode(tensor, skip_special_tokens=False)) for tensor in outputs
    ]


def code_generation(prompt, max_new_tokens, temperature=0.2, seed=42):
    #set_seed(seed)

    if "<FILL-HERE>" in prompt:
        return fim_generation(prompt, max_new_tokens, temperature=0.2)
    else:
        completion = pipe(prompt, do_sample=True, top_p=0.95, temperature=temperature, max_new_tokens=max_new_tokens)[0]['generated_text']
        completion = completion[len(prompt):]
        return post_processing(prompt, completion)


demo = gr.Blocks(
    css=".gradio-container {background-color: #20233fff; color:white}"
)
with demo:
    with gr.Row():
        _, colum_2, _ = gr.Column(scale=1), gr.Column(scale=6), gr.Column(scale=1)
        with colum_2:
            gr.Markdown(value=description)
            code = gr.Code(lines=5, language="python", label="Input code", value="def all_odd_elements(sequence):\n    \"\"\"Returns every odd element of the sequence.\"\"\"")

            with gr.Accordion("Advanced settings", open=False):
                max_new_tokens= gr.Slider(
                    minimum=8,
                    maximum=1024,
                    step=1,
                    value=48,
                    label="Number of tokens to generate",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.5,
                    step=0.1,
                    value=0.2,
                    label="Temperature",
                )
                seed = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    step=1,
                    label="Random seed to use for the generation"
                )
            run = gr.Button()
            output = gr.HTML(label="Generated code")

    event = run.click(code_generation, [code, max_new_tokens, temperature, seed], output, api_name="predict")
    gr.HTML(label="Contact", value="<img src='https://huggingface.co/datasets/bigcode/admin/resolve/main/bigcode_contact.png' alt='contact' style='display: block; margin: auto; max-width: 800px;'>")

demo.launch()