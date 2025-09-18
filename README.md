# Vision Language Modelling

This project is a multi-modal AI application that uses **BLIP, CLIP, and GPT-2** to process images and generate meaningful outputs.  
It allows users to:  
1. Generate **captions** for images  
2. Ask **questions about images** (Visual Question Answering, VQA)  
3. Generate **short stories** based on image content  
4. Evaluate outputs with **CLIP** to measure alignment between image and text  

---

## Features

- **Single Caption**: BLIP generates a caption describing the image.  
- **Multiple Captions + CLIP Reranking**: Generates multiple captions and selects the best one using CLIP similarity.  
- **Visual Question Answering (VQA)**: Ask natural language questions about the image and get AI-generated answers.  
- **Story Generation**: Generates a short story from a BLIP-generated caption seed using GPT-2.  
- **CLIP Evaluation**: All outputs are scored for relevance to the image.  

---

## Project Structure

```bash
vision_language_app/
│── main.py                 # Main program to run tasks
│── models/                 # Modules for image captioning, VQA, storytelling and evaluation
│ ├── captioning.py
│ ├── vqa.py
│ ├── storytelling.py
│ └── evaluation.py
│── inputs/                 # Sample images
│── outputs/                # Generated captions/stories
└── requirements.txt        # Python dependencies

```
---

## Usage

- Run the main program: python main.py
- You will see a menu. Choose a task:
    1. Caption → Generates a single caption and shows CLIP similarity score.
    2. Multiple Captions (with CLIP reranking) → Generates several captions and selects the best using CLIP.
    3. VQA → Enter a question about the image to get an answer.
    4. Story → Generates a short story based on the image.
- Enter choice 
- Output displayed and saved in the outputs folder
