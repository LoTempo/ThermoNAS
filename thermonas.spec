# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH)
package_root = project_root / "src" / "thermonas"

a = Analysis(
    [str(project_root / "scripts" / "thermonas_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[
        (str(package_root / "models" / "timestep_predictor.h5"), "thermonas/models"),
        (str(package_root / "models" / "timestep_scaler.pkl"), "thermonas/models"),
    ],
    hiddenimports=["PySide6", "matplotlib.backends.backend_qtagg"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ThermoNAS",
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
