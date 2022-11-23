from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

prompt = "hacker man"

res = generator(prompt, do_sample=True, min_length=50, temperature=0.9)

print(res[0]["generated_text"])