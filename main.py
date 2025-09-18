import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
import clip

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load models
# BLIP Captioning
blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# BLIP VQA
blip_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# GPT-2 for story
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# CLIP for evaluation
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# ===== Functions =====
def generate_caption(image):
    inputs = blip_caption_processor(image, return_tensors="pt").to(device)
    outputs = blip_caption_model.generate(**inputs)
    caption = blip_caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def generate_multiple_captions(image, num_beams=5, num_return_sequences=5):
    inputs = blip_caption_processor(image, return_tensors="pt").to(device)
    beam_outputs = blip_caption_model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences)
    captions = [blip_caption_processor.decode(out, skip_special_tokens=True) for out in beam_outputs]

    # CLIP evaluation
    clip_image_input = clip_preprocess(image).unsqueeze(0).to(device)
    clip_text_inputs = clip.tokenize(captions).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(clip_image_input).expand(len(captions), -1)
        text_features = clip_model.encode_text(clip_text_inputs)
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)

    best_idx = similarities.argmax().item()
    best_caption = captions[best_idx]
    best_score = similarities[best_idx].item()
    return best_caption, best_score, captions, similarities


def vqa(image, question):
    inputs = blip_vqa_processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = blip_vqa_model.generate(**inputs)
    answer = blip_vqa_processor.decode(outputs[0], skip_special_tokens=True)
    return answer


def generate_story(caption_seed, max_length=250):
    story_prompt = f"Write a short story about: {caption_seed}\n\nOnce upon a time,"
    inputs = gpt_tokenizer.encode(story_prompt, return_tensors="pt").to(device)
    
    # Suppress attention mask warning by explicitly passing attention_mask
    attention_mask = torch.ones_like(inputs)
    
    story_output = gpt_model.generate(inputs, attention_mask=attention_mask, max_length=max_length, do_sample=True, temperature=0.8)
    story = gpt_tokenizer.decode(story_output[0], skip_special_tokens=True)
    return story


# ===== Main Program =====
def main():
    # Use a fixed sample image
    image_path = "inputs/sample.jpg"  # make sure you have a sample image here
    image = Image.open(image_path).convert('RGB')

    print("\nChoose task:")
    print("1. Single Caption")
    print("2. Multiple Captions (with CLIP)")
    print("3. VQA")
    print("4. Story")

    choice = input("Enter choice: ")

    if choice == "1":
        caption = generate_caption(image)
        print("\nCaption:", caption)
        with open("outputs/caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)
        print("Saved to outputs/caption.txt")

    elif choice == "2":
        best_caption, best_score, captions, similarities = generate_multiple_captions(image)
        print("\nBest Caption:", best_caption)
        print("CLIP Similarity:", best_score)
        with open("outputs/multiple_captions.txt", "w", encoding="utf-8") as f:
            for cap, score in zip(captions, similarities):
                f.write(f"{cap} | CLIP Score: {score.item():.2f}\n")
        print("Saved to outputs/multiple_captions.txt")

    elif choice == "3":
        question = input("Enter your question: ")
        answer = vqa(image, question)
        print("\nAnswer:", answer)
        with open("outputs/vqa.txt", "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\nAnswer: {answer}\n")
        print("Saved to outputs/vqa.txt")

    elif choice == "4":
        caption_seed = generate_caption(image)
        story = generate_story(caption_seed)
        print("\nStory:\n", story)
        with open("outputs/story.txt", "w", encoding="utf-8") as f:
            f.write(f"Caption Seed: {caption_seed}\nStory: {story}\n")
        print("Saved to outputs/story.txt")

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
