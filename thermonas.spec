# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import tomllib

project_root = Path(SPECPATH)
package_root = project_root / "src" / "thermonas"
with (project_root / "pyproject.toml").open("rb") as project_file:
    version = tomllib.load(project_file)["project"]["version"]
artifact_name = f"ThermoNAS-v{version}-Windows-x64"

a = Analysis(
    [str(project_root / "scripts" / "thermonas_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[
        (str(package_root / "models" / "timestep_predictor.npz"), "thermonas/models"),
    ],
    hiddenimports=["matplotlib.backends.backend_qtagg"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "IPython",
        "joblib",
        "pandas",
        "scipy",
        "sklearn",
        "tensorflow",
        "tkinter",
        "torch",
    ],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=artifact_name,
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
