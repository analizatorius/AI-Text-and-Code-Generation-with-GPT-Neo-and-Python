from transformers import GPTNeoForCausalLM, GPT2Tokenizer


path = "D:/_Coding/Python/AI/Text Generators/AI Text and Code Generation with GPT Neo and Python/Transformers/gpt neo 1.3B"

model = GPTNeoForCausalLM.from_pretrained(path)
tokenizer = GPT2Tokenizer.from_pretrained(path)

prompt = "nasty shit is going on here"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
