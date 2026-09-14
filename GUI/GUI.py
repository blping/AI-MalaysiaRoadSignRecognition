import os
import random
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import joblib  
from skimage.feature import hog

YOLO_MODEL_PATH = '../YOLO/runs/detect/malaysia_road_sign_noise_preprocessed_50/weights/best.pt'
CNN_JSON_PATH = '../CNN/config.json'
CNN_WEIGHTS_PATH = '../CNN/model.weights.h5'
SVM_MODEL_PATH = '../SVM/SVM_Phase2_0Clean.joblib' 

CLASS_NAMES = [
    'Bumps', 'No U-turns', 'No entry', 'No parking', 'Obstruction',
    'Pass on the left', 'Roadway diverges', 'Speed limit', 'Stop', 'U turn'
]

try:
    #yolo model
    yolo_model = YOLO(YOLO_MODEL_PATH)

    #cnn model
    with open(CNN_JSON_PATH, 'r') as f:
        cnn_model = model_from_json(f.read())
    cnn_model.load_weights(CNN_WEIGHTS_PATH)

    #svm model
    svm_entry = joblib.load(SVM_MODEL_PATH)
    svm_model = svm_entry['model'] 
    svm_pca   = svm_entry['pca']    
    
    print("All models (YOLO, CNN, SVM) loaded successfully!")
except Exception as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Model Error", f"Failed to load models:\n{str(e)}")
    exit()

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Malaysia Road Sign Recognition System")
        self.root.geometry("1600x900") 

        bg_color = "#FFF9E3" 
        self.root.configure(bg=bg_color)
        
        self.file_path = None
        self.tk_image = None

        tk.Label(root, text="Malaysia Road Sign Recognition System", 
                 font=("Arial", 24, "bold"), bg=bg_color).pack(pady=(10, 20))

        content_wrapper = tk.Frame(root, bg=bg_color)
        content_wrapper.pack(fill=tk.BOTH, expand=True, padx=20)

        left_sidebar = tk.LabelFrame(content_wrapper, text=" Recognizable Signs ", 
                                     font=("Arial", 14, "bold"), bg=bg_color, fg="#5D4037")
        left_sidebar.pack(side=tk.LEFT, padx=(0, 20), anchor='n', pady=10)

        self.class_icons = {} 
        for name in CLASS_NAMES:
            item_frame = tk.Frame(left_sidebar, bg=bg_color)
            item_frame.pack(fill=tk.X, pady=4, padx=10)
            
            icon_path = os.path.join("icons", f"{name}.png")
            
            if os.path.exists(icon_path):
                img = Image.open(icon_path).resize((40, 40), Image.Resampling.LANCZOS)
                icon_img = ImageTk.PhotoImage(img)
                self.class_icons[name] = icon_img 
                tk.Label(item_frame, image=icon_img, bg=bg_color).pack(side=tk.LEFT)
            else:
                tk.Label(item_frame, text="●", fg="#A1887F", bg=bg_color, width=3).pack(side=tk.LEFT)

            tk.Label(item_frame, text=name, font=("Arial", 11, "bold"), 
                     bg=bg_color, anchor="w").pack(side=tk.LEFT, padx=10)

        main_area = tk.Frame(content_wrapper, bg=bg_color)
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)#
        self.image_frame = tk.Frame(main_area, highlightbackground="#f8f8f8", highlightthickness=2)
        self.image_frame.pack(pady=10)
        
        self.image_canvas = tk.Canvas(self.image_frame, width=400, height=400, bg="#f8f8f8", highlightthickness=0)
        self.image_canvas.pack()

        btn_frame = tk.Frame(main_area, bg=bg_color)
        btn_frame.pack(pady=15)
        self.btn_load = tk.Button(btn_frame, text="Upload Image", font=("Arial", 12, "bold"), 
                                  command=self.load_image, width=15, height=2)
        self.btn_load.pack(side=tk.LEFT, padx=15)
        
        self.btn_recognize = tk.Button(btn_frame, text="Start Comparison", font=("Arial", 12, "bold"), 
                                       command=self.recognize_image, bg="#2196F3", fg="white", 
                                       width=15, height=2, state=tk.DISABLED)
        self.btn_recognize.pack(side=tk.LEFT, padx=15)

        display_frame = tk.Frame(main_area, bg=bg_color)
        display_frame.pack(pady=10, fill=tk.X)

        # --- YOLO ---
        yolo_frame = tk.LabelFrame(display_frame, text=" YOLOv8 (Detection) ", font=("Arial", 14, "bold"), fg="#1B5E20", bg=bg_color)
        yolo_frame.pack(side=tk.LEFT, expand=True, padx=5, fill=tk.BOTH)
        self.yolo_text = tk.Text(yolo_frame, height=5, width=28, font=("Consolas", 14, "bold"), state=tk.DISABLED, bg="#E8F5E9")
        self.yolo_text.pack(pady=10, padx=5)

        # --- CNN ---
        cnn_frame = tk.LabelFrame(display_frame, text=" CNN (Classification) ", font=("Arial", 14, "bold"), fg="#B71C1C", bg=bg_color)
        cnn_frame.pack(side=tk.LEFT, expand=True, padx=5, fill=tk.BOTH)
        self.cnn_text = tk.Text(cnn_frame, height=5, width=28, font=("Consolas", 14, "bold"), state=tk.DISABLED, bg="#FFEBEE")
        self.cnn_text.pack(pady=10, padx=5)

        # --- SVM ---
        svm_frame = tk.LabelFrame(display_frame, text=" SVM (Machine Learning) ", font=("Arial", 14, "bold"), fg="#0D47A1", bg=bg_color)
        svm_frame.pack(side=tk.LEFT, expand=True, padx=5, fill=tk.BOTH)
        self.svm_text = tk.Text(svm_frame, height=5, width=28, font=("Consolas", 14, "bold"), state=tk.DISABLED, bg="#E3F2FD")
        self.svm_text.pack(pady=10, padx=5)

        right_sidebar = tk.LabelFrame(content_wrapper, text="XXXXX", 
                                      font=("Arial", 14, "bold"), bg=bg_color, fg=bg_color)
        right_sidebar.pack(side=tk.LEFT, padx=90, anchor='n', pady=10)
    
    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")])
        if not self.file_path: return
        
        img = Image.open(self.file_path).convert("RGB")
        img.thumbnail((400, 400))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(200, 200, anchor=tk.CENTER, image=self.tk_image)
        
        self.btn_recognize.config(state=tk.NORMAL)
        self.update_ui_text("", "", "")

    def get_yolo_pred(self):
        results = yolo_model.predict(source=self.file_path, conf=0.25, verbose=False)
        boxes = results[0].boxes
        if not boxes: return "No objects detected."
        lines = []
        for box in boxes:
            name = CLASS_NAMES[int(box.cls[0])]
            conf = float(box.conf[0])
            lines.append(f"[{conf:.2%}] {name}")
        return "\n".join(lines)

    def get_cnn_pred(self):
        try:
            img = cv2.imread(self.file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) 
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            preds = cnn_model.predict(img, verbose=0)
            idx = np.argmax(preds[0])
            return f"[{preds[0][idx]:.2%}] {CLASS_NAMES[idx]}"
        except Exception as e: return f"Error: {str(e)}"

    def get_svm_pred(self):
        try:
            img = cv2.imread(self.file_path)
            img = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = cv2.createCLAHE(clipLimit=2.0).apply(l)
            enhanced_rgb = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
            blurred = cv2.GaussianBlur(enhanced_rgb, (3, 3), 0)
            
            normalized = blurred.astype(np.float32) / 255.0
            gray = np.mean(normalized, axis=2)
            
            final_img = cv2.resize(gray, (64, 64))
    
            features_hog = hog(final_img, pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), feature_vector=True)
            
            features_pca = svm_pca.transform(features_hog.reshape(1, -1))
    
            pred_idx = svm_model.predict(features_pca)[0]
            
            try:
                probs = svm_model.predict_proba(features_pca)
                conf = np.max(probs)
                return f"[{conf:.2%}] {CLASS_NAMES[pred_idx]}"
            except:
                return CLASS_NAMES[pred_idx]
    
        except Exception as e:
            return f"Error: {str(e)}"

    def recognize_image(self):
        yolo_res = self.get_yolo_pred()
        cnn_res = self.get_cnn_pred()
        svm_res = self.get_svm_pred()
        self.update_ui_text(yolo_res, cnn_res, svm_res)

    def update_ui_text(self, yolo_msg, cnn_msg, svm_msg):
        widgets = [(self.yolo_text, yolo_msg), (self.cnn_text, cnn_msg), (self.svm_text, svm_msg)]
        for widget, msg in widgets:
            widget.config(state=tk.NORMAL)
            widget.delete('1.0', tk.END)
            widget.insert(tk.END, msg)
            widget.tag_configure("center", justify='center')
            widget.tag_add("center", "1.0", "end")
            widget.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()