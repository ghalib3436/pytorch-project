import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageTk
from torchvision import models


weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

camera = None
frame_lock = threading.Lock()
latest_frame = None
analysis_overlay = None
events = queue.Queue()
stop_camera = threading.Event()
analysis_running = False


def camera_loop():
    global latest_frame

    if not camera.isOpened():
        events.put(("error", "Camera unavailable. Check that it is connected and not in use."))
        camera.release()
        return

    try:
        while not stop_camera.is_set():
            ok, captured_frame = camera.read()
            if not ok:
                events.put(("error", "Could not read a frame from the camera."))
                break
            with frame_lock:
                latest_frame = captured_frame
    finally:
        camera.release()


def class_activation_map(input_tensor, class_index, image_size):
    with torch.enable_grad():
        input_tensor = input_tensor.detach().requires_grad_(True)
        features = model.features(input_tensor)
        pooled = model.avgpool(features)
        logits = model.classifier(torch.flatten(pooled, 1))
        gradients, = torch.autograd.grad(logits[0, class_index], features)

    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    activation = torch.relu((channel_weights * features).sum(dim=1, keepdim=True))
    activation = functional.interpolate(
        activation,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    activation = activation.detach().cpu().numpy()
    minimum = float(activation.min())
    maximum = float(activation.max())
    if maximum > minimum:
        activation = (activation - minimum) / (maximum - minimum)
    else:
        activation = np.zeros_like(activation)
    return activation


def make_overlay(frame, activation):
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(activation, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def analyze_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = model(input_tensor).softmax(dim=1)[0]
        scores, indices = probabilities.topk(3)

    candidates = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        activation = class_activation_map(input_tensor, index, image.size)
        candidates.append(
            (weights.meta["categories"][index], score, index, activation)
        )

    winner_name, _, _, winner_activation = candidates[0]
    report = [
        "1. Détection : trois hypothèses d'objet sont proposées par MobileNetV3.",
        "",
        "2. Caractéristiques visuelles : Grad-CAM repère, pour chaque hypothèse, "
        "les zones de l'image qui contribuent à sa classification. "
        "La superposition affichée correspond à l'hypothèse retenue.",
        "",
        "3. Recherche et comparaison : les zones visuelles sont calculées "
        "pour chacune des trois options. Le modèle compare ensuite les "
        "scores de classe; le pourcentage est une probabilité relative "
        "(pas une certitude garantie).",
        "",
        "4. Classement des options :",
    ]
    report.extend(
        f"   {rank}. {name} — {score:.1%} (zones visuelles analysées)"
        for rank, (name, score, _, _) in enumerate(candidates, start=1)
    )
    report.extend(
        [
            "",
            "5. Chemin de recherche : image capturée → prétraitement → "
            "classement des 3 classes → comparaison de leurs scores → "
            f"visualisation des zones associées à « {winner_name} ».",
            "",
            f"6. Décision finale : {winner_name} ({candidates[0][1]:.1%}).",
            "",
            "Les zones colorées indiquent l'influence visuelle, pas le contour "
            "exact de l'objet.",
        ]
    )
    return make_overlay(frame, winner_activation), "\n".join(report)


def begin_analysis():
    global analysis_running, analysis_overlay

    if analysis_running:
        return
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        report_text.configure(state="normal")
        report_text.delete("1.0", tk.END)
        report_text.insert(tk.END, "En attente d'une image de la caméra.")
        report_text.configure(state="disabled")
        return

    analysis_overlay = None
    analysis_running = True
    analyze_button.configure(state="disabled", text="Analyse en cours…")
    threading.Thread(target=analysis_worker, args=(frame,), daemon=True).start()


def analysis_worker(frame):
    try:
        events.put(("analysis", analyze_frame(frame)))
    except Exception as error:
        events.put(("error", f"Analyse impossible : {error}"))


def show_report(text):
    report_text.configure(state="normal")
    report_text.delete("1.0", tk.END)
    report_text.insert(tk.END, text)
    report_text.configure(state="disabled")


def process_events():
    global analysis_running, analysis_overlay

    while True:
        try:
            event_type, payload = events.get_nowait()
        except queue.Empty:
            break
        if event_type == "analysis":
            analysis_overlay, report = payload
            show_report(report)
            analysis_running = False
            analyze_button.configure(state="normal", text="Analyser l'image")
        elif event_type == "error":
            show_report(payload)
            if analysis_running:
                analysis_running = False
                analyze_button.configure(state="normal", text="Analyser l'image")

    root.after(100, process_events)


def draw_frame():
    if analysis_overlay is not None:
        image_source = analysis_overlay
    else:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        image_source = None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if image_source is not None:
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        image = Image.fromarray(image_source)
        image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER,
        )
        canvas.image = photo

    root.after(30, draw_frame)


def close_app():
    stop_camera.set()
    root.destroy()


def main():
    global analyze_button, camera, canvas, report_text, root

    camera = cv2.VideoCapture(0)
    root = tk.Tk()
    root.title("AI Vision — analyse explicable")
    root.geometry("1000x680")
    root.minsize(680, 480)
    root.columnconfigure(0, weight=3)
    root.columnconfigure(1, weight=2)
    root.rowconfigure(0, weight=1)

    canvas = tk.Canvas(root, background="#171717", highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

    sidebar = ttk.Frame(root, padding=(10, 10, 10, 10))
    sidebar.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    sidebar.columnconfigure(0, weight=1)
    sidebar.rowconfigure(1, weight=1)

    ttk.Label(
        sidebar,
        text="Analyse de l'image",
        font=("Segoe UI", 15, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 10))

    report_text = tk.Text(
        sidebar,
        wrap="word",
        font=("Segoe UI", 10),
        relief="flat",
        padx=8,
        pady=8,
    )
    report_text.grid(row=1, column=0, sticky="nsew")
    report_text.insert(
        tk.END,
        "L'image s'adapte à la fenêtre. Cliquez sur « Analyser l'image » "
        "pour comparer trois hypothèses et afficher les zones pertinentes.",
    )
    report_text.configure(state="disabled")

    analyze_button = ttk.Button(
        sidebar,
        text="Analyser l'image",
        command=begin_analysis,
    )
    analyze_button.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    threading.Thread(target=camera_loop, daemon=True).start()
    root.protocol("WM_DELETE_WINDOW", close_app)
    draw_frame()
    process_events()
    root.mainloop()


if __name__ == "__main__":
    main()
