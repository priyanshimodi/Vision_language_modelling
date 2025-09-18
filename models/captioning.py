import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image):
    inputs = caption_processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return caption_processor.decode(outputs[0], skip_special_tokens=True)

def generate_multiple_captions(image, num_beams=5, num_return=5):
    inputs = caption_processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return)
    return [caption_processor.decode(out, skip_special_tokens=True) for out in outputs]
