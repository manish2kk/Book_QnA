import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#model_name = "gpt2"
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

prompt = "Capital of India is"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
    )

print(
    "GPT-2 Output:\n",
    tokenizer.decode(output[0], skip_special_tokens=True)
)


'''from llama_cpp import Llama

model = Llama(
    model_path="50m-q8_0.gguf",
    n_ctx=1024,
    verbose=False,
)

result = model(
    "Artificial intelligence is",
    max_tokens=64,
    temperature=0.8,
    top_p=0.95,
    repeat_penalty=1.1,
)

print("LLaMA Output: \n", result["choices"][0]["text"])
'''

'''from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
'''

'''import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "exnivo/LoafLM-10M"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float32,
).eval()

messages = [
    {"role": "user", "content": "How does the cat sound?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.4,
        top_k=50,
        do_sample=True,
    )

new_tokens = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
'''