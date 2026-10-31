# pytorch-project

## Adaptive, explainable camera classification

Run `python main_adaptive.py` to open the camera classifier. The preview
resizes to fit the window while preserving the image's aspect ratio.

Select **Analyser l'image** to rank the three most likely ImageNet classes.
The report explains the classification steps and shows a Grad-CAM overlay for
the highest-ranked class. The displayed percentages are model probabilities,
not guaranteed confidence levels; the overlay indicates influential image
regions and is not an object outline.

The application uses the same PyTorch, torchvision, OpenCV, and Pillow
dependencies as `main.py`. The pretrained MobileNetV3 weights are downloaded
by torchvision the first time they are needed.
