from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Small_Weights


project_dir = Path(SPECPATH)
weights_file = (
    Path(torch.hub.get_dir())
    / "checkpoints"
    / "mobilenet_v3_small-047dcff4.pth"
)
if not weights_file.is_file():
    MobileNet_V3_Small_Weights.DEFAULT.get_state_dict(
        progress=True,
        check_hash=True,
    )
if not weights_file.is_file():
    raise FileNotFoundError(f"MobileNetV3 weights were not cached at {weights_file}")

datas = [(str(weights_file), "model")]
datas.append((str(project_dir / "README.md"), "."))

a = Analysis(
    [str(project_dir / "main_adaptive.py")],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Vision",
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AI Vision",
)
