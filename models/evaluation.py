import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def evaluate_with_clip(image, texts):
    """Score one or multiple texts against an image"""
    if isinstance(texts, str):
        texts = [texts]

    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
        image_features = image_features.expand(len(texts), -1)
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)

    return similarities.cpu().numpy()
