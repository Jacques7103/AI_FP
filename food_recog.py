from PIL import Image
import requests
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline
import torch
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading
from ttkthemes import ThemedTk

def get_recipe(url):
    # url = 'https://huggingface.co/Jacques7103/Food-Recognition/resolve/main/273350.jpg'
    repo_name = "Jacques7103/Food-Recognition"

    feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    image = Image.open(requests.get(url, stream=True).raw)
    encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    pipe = pipeline("image-classification", "Jacques7103/Food-Recognition")
    pipe(image)

    df = pd.read_csv("ingredients.csv", encoding = "windows-1252")

    for i in range(10):
        if df['Name'][i] == model.config.id2label[predicted_class_idx]:
            return model.config.id2label[predicted_class_idx], df['Ingredients'][i]
        
def run_function():
    link = link_entry.get()

    def fetch_recipe():
        name, recipe = get_recipe(link)
        recipe_text = f"Recipe for {name} is: {''.join(recipe)}"
        result_label.config(text=recipe_text)
        fetch_button.config(state=tk.NORMAL)

    fetch_button.config(state=tk.DISABLED)
    result_label.config(text="Fetching...", foreground="gray")  # Set text color to gray

    threading.Thread(target=fetch_recipe).start()

root = ThemedTk(theme="arc")  # Use a ttkthemes theme, e.g., "arc"
root.title("Recipe Fetcher")

link_label = ttk.Label(root, text="Enter Recipe Link:")
link_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

link_entry = ttk.Entry(root, width=40)
link_entry.grid(row=0, column=1, pady=10, padx=10, sticky="w")

fetch_button = ttk.Button(root, text="Fetch Recipe", command=run_function)
fetch_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = ttk.Label(root, text="", wraplength=500, justify="left")
result_label.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

root.mainloop()