# ThermoNAS

ThermoNAS is a desktop application for simulating transient, two-dimensional
conjugate heat transfer in a flat porous channel whose internal geometry is
based on triply periodic minimal surfaces (TPMS), specifically a Schwarz P
topology. It computes the temperature fields of the fluid and solid phases,
estimates the Darcy pressure drop, and visualizes the thermal evolution over
time.

The application combines an explicit finite-difference solver with an optional
neural-network classifier that suggests a stable time step for the selected
material and flow properties.

## Physical problem

An incompressible liquid flows through a porous TPMS channel of length \(L\) and
height \(H\). The model uses a local thermal non-equilibrium formulation: the
fluid and porous solid have separate temperature fields, \(T_f(x,y,t)\) and
\(T_s(x,y,t)\), coupled by interphase heat transfer.

The formulation assumes:

- low flow rates, so thermal dispersion is neglected;
- no phase change;
- steady, laminar Darcy flow in the \(x\) direction;
- no transverse velocity component;
- constant fluid density and viscosity.

The solid energy balance is

$$
(1-\phi)(\rho c_p)_s\frac{\partial T_s}{\partial t}
= k_s\left(\frac{\partial^2T_s}{\partial x^2}
+\frac{\partial^2T_s}{\partial y^2}\right)
+A_0h_{fs}(T_f-T_s),
$$

and the fluid energy balance is

$$
\phi(\rho c_p)_f\left(
\frac{\partial T_f}{\partial t}
+u_x\frac{\partial T_f}{\partial x}\right)
= k_f\left(\frac{\partial^2T_f}{\partial x^2}
+\frac{\partial^2T_f}{\partial y^2}\right)
+A_0h_{fs}(T_s-T_f).
$$

The pressure gradient follows Darcy's law,

$$
u_x=-\frac{K}{\mu}\frac{dP}{dx},
$$

with constant axial velocity. The fluid temperature is prescribed at the
inlet, its outlet uses a zero axial-gradient condition, and both phases have
fixed temperatures on the upper and lower walls. At the two axial ends, the
solid exchanges heat with external environments through convective boundary
conditions.

The complete source formulation, notation, schematic, initial conditions, and
boundary conditions are retained in
[`docs/problem_statement.pdf`](docs/problem_statement.pdf).

## Features

- coupled fluid/solid transient temperature solution;
- temperature contour plots with time-step navigation;
- Darcy pressure-drop calculation;
- neural-network-based time-step suggestion;
- input validation and explicit-scheme instability detection;
- support for non-square computational grids;
- warning before simulations that require large amounts of memory.

## Installation

ThermoNAS supports Python 3.10-3.12. A fresh virtual environment is
recommended.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the application

After installation, use either command:

```bash
thermonas
```

```bash
python -m thermonas
```

Enter the geometry, material properties, boundary conditions, grid resolution,
and time settings. **Suggest time step** evaluates the bundled classifier;
**Compute** runs the finite-difference model. Use the slider to inspect the
temperature field at each stored time level.

The time-step predictor is an engineering aid rather than a mathematical
stability guarantee. If the solver reports an unstable solution, decrease
\(\Delta t\). Results should be checked with grid- and time-step-independence
studies before being used for engineering decisions.

## Numerical method and memory use

ThermoNAS uses forward Euler integration, central differences for diffusion,
and a first-order upwind discretization for fluid advection. The full fluid and
solid temperature histories are retained for interactive visualization.
Approximate storage is therefore

$$
2N_tN_xN_y\times 8\ \text{bytes}.
$$

For example, \(N_t=32{,}000\), \(N_x=100\), and \(N_y=100\) require about
4.77 GiB for the two temperature arrays alone. The application asks for
confirmation when the estimated temperature storage exceeds 512 MiB.

## Project structure

```text
ThermoNAS/
├── docs/
│   └── problem_statement.pdf
├── scripts/
│   └── thermonas_entry.py
├── src/thermonas/
│   ├── models/
│   │   └── timestep_predictor.npz
│   ├── resources/
│   │   └── main_window.ui
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── predictor.py
│   ├── solver.py
│   └── ui_main_window.py
├── tests/
│   ├── test_predictor.py
│   └── test_solver.py
├── pyproject.toml
└── thermonas.spec
```

`ui_main_window.py` is generated from the Qt Designer form. After editing
`main_window.ui`, regenerate it with:

```bash
pyside6-uic src/thermonas/resources/main_window.ui -o src/thermonas/ui_main_window.py
```

## Tests

Install the project and run:

```bash
python -m unittest discover -s tests -v
```

The tests cover equilibrium preservation, validation, grid spacing, and the
pressure field on a non-square mesh.

## Building a Windows executable

```powershell
python -m pip install -e ".[build]"
pyinstaller thermonas.spec
```

The versioned executable is written to
`dist/ThermoNAS-v<version>-Windows-x64.exe`.

## Model inputs and limitations

The bundled classifier receives six inputs in this order:

1. fluid effective thermal conductivity;
2. solid effective thermal conductivity;
3. fluid volumetric heat capacity;
4. solid volumetric heat capacity;
5. fluid velocity;
6. candidate time step.

Its validity is limited to the parameter space represented by its training
data. The repository does not currently include that training dataset or a
model card, so predictions outside the original operating range should be
treated cautiously.
