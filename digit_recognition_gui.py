import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn as nn
import pickle
from classif_model import NN
from cluster_model import AutoEncoder


def binarize_image(image, threshold=65):
    return (image > threshold).astype(np.uint8)

def crop_image(image, margin=1):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    if not rows.any() or not cols.any():
        return image
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    rmin = max(rmin - margin, 0)
    rmax = min(rmax + margin, image.shape[0] - 1)
    cmin = max(cmin - margin, 0)
    cmax = min(cmax + margin, image.shape[1] - 1)
    
    return image[rmin:rmax+1, cmin:cmax+1]

def resize_image(image, new_shape=(12, 6)):
    new_height, new_width = new_shape
    old_height, old_width = image.shape
    
    row_indices = (np.linspace(0, old_height, new_height, endpoint=False)).astype(int)
    col_indices = (np.linspace(0, old_width, new_width, endpoint=False)).astype(int)
    
    return image[np.ix_(row_indices, col_indices)]

class DigitRecognitionGUI:
    def __init__(self, classification_path, clustering_path):
        self.root = tk.Tk()
        self.root.title("Digit Recognition & Clustering")
        
        self.canvas_size = 280
        self.grid = np.zeros((28, 28))
        self.drawing = False
        
        # Load both models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classification model
        self.classification_model = NN().to(self.device)
        checkpoint = torch.load(classification_path, map_location=self.device)
        self.classification_model.load_state_dict(checkpoint['model_state_dict'])
        self.classification_model.eval()
        
        # Load clustering models
        self.autoencoder = AutoEncoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load(f"{clustering_path}/autoencoder.pth", 
                                                   map_location=self.device))
        self.autoencoder.eval()
        
        with open(f"{clustering_path}/kmeans.pkl", 'rb') as f:
            self.kmeans = pickle.load(f)
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model selection frame
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Model selection radio buttons
        self.model_var = tk.StringVar(value="classification")
        ttk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Classification", 
                       variable=self.model_var, 
                       value="classification").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Clustering", 
                       variable=self.model_var, 
                       value="clustering").pack(side=tk.LEFT, padx=5)
        
        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, 
                              width=self.canvas_size,
                              height=self.canvas_size,
                              bg='white')
        self.canvas.grid(row=1, column=0, columnspan=2)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Buttons
        ttk.Button(main_frame, text="Clear", 
                  command=self.clear_canvas).grid(row=2, column=0, pady=10)
        ttk.Button(main_frame, text="Predict", 
                  command=self.predict).grid(row=2, column=1, pady=10)
        
        # Prediction label
        self.prediction_label = ttk.Label(main_frame, 
                                        text="Draw a digit and click Predict")
        self.prediction_label.grid(row=3, column=0, columnspan=2)
        
        # Preview label
        self.preview_label = ttk.Label(main_frame, 
                                     text="Processed image will be shown here")
        self.preview_label.grid(row=4, column=0, columnspan=2)
    
    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            r = 3
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
            
            grid_x = int(x * 28 / self.canvas_size)
            grid_y = int(y * 28 / self.canvas_size)
            
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < 28 and 0 <= nx < 28:
                        self.grid[ny, nx] = 255
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = np.zeros((28, 28))
        self.prediction_label.config(text="Draw a digit and click Predict")
        self.preview_label.config(text="Processed image will be shown here")
    
    def process_drawn_image(self):
        binary = binarize_image(self.grid)
        cropped = crop_image(binary)
        resized = resize_image(cropped, new_shape=(12, 6))
        return resized
    
    def predict(self):
        processed_image = self.process_drawn_image()
        self.preview_label.config(text=f"Processed image shape: {processed_image.shape}")
        
        input_tensor = torch.FloatTensor(processed_image.flatten()).unsqueeze(0).to(self.device)
        
        if self.model_var.get() == "classification":
            # Classification prediction
            with torch.no_grad():
                outputs = self.classification_model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()
            
            self.prediction_label.config(
                text=f"Predicted Digit: {predicted.item()} (Confidence: {confidence:.2%})"
            )
        else:
            # Clustering prediction
            with torch.no_grad():
                encoded, _ = self.autoencoder(input_tensor)
                cluster = self.kmeans.predict(encoded.cpu().numpy())[0]
            
            self.prediction_label.config(
                text=f"Assigned to Cluster: {cluster}"
            )
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    CLASSIFICATION_MODEL_PATH = "./models/classification/classif_model.pth"
    CLUSTERING_MODEL_PATH = "./models/clustering"
    
    app = DigitRecognitionGUI(CLASSIFICATION_MODEL_PATH, CLUSTERING_MODEL_PATH)
    app.run()