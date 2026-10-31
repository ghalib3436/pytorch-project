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

## Build a Windows application

From PowerShell in the project directory, create an isolated CPU-only build
environment and package the application:

```powershell
python -m venv .venv-packaging
.\.venv-packaging\Scripts\python.exe -m pip install torch==2.11.0+cpu torchvision==0.26.0+cpu numpy pillow opencv-python pyinstaller==6.17.0 --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv-packaging\Scripts\python.exe -m PyInstaller --noconfirm --distpath .\dist --workpath .\build .\main_adaptive.spec
```

The standalone application is written to `dist/AI Vision/AI Vision.exe`.
The build includes the cached pretrained weights and required PyTorch,
torchvision, and OpenCV files. On another computer, copy the complete
`dist/AI Vision` folder; the camera and a compatible Windows runtime are
still required. If the weights are not cached yet, the build downloads them
once. The build outputs and packaging environment are excluded from Git.
