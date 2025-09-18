import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .captioning import generate_caption

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_story(image, max_story_length=250):
    caption = generate_caption(image)
    story_prompt = f"Write a short story about: {caption}\n\nOnce upon a time,"
    input_ids = gpt_tokenizer.encode(story_prompt, return_tensors="pt")

    story_output = gpt_model.generate(
        input_ids,
        max_length=max_story_length,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    story = gpt_tokenizer.decode(story_output[0], skip_special_tokens=True)
    return caption, story
