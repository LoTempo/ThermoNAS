"""Qt desktop application for the ThermoNAS porous-channel model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FormatStrFormatter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QMainWindow, QMessageBox, QVBoxLayout

from .solver import SimulationParameters, solve
from .ui_main_window import Ui_MainWindow

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIRECTORY / "timestep_predictor.h5"
SCALER_PATH = MODEL_DIRECTORY / "timestep_scaler.pkl"
LARGE_SIMULATION_BYTES = 512 * 1024**2


class TimeStepPredictor:
    """Lazy loader for the optional neural-network stability estimator."""

    def __init__(self) -> None:
        self._model = None
        self._scaler = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import joblib
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise RuntimeError(
                "The time-step predictor requires the 'joblib' and 'tensorflow' packages."
            ) from exc

        if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
            raise RuntimeError("The packaged time-step model or scaler is missing.")

        self._model = load_model(MODEL_PATH, compile=False)
        self._scaler = joblib.load(SCALER_PATH)

    def suggest(self, parameters: SimulationParameters) -> float:
        self._load()
        assert self._model is not None and self._scaler is not None

        minimum, maximum, increment = 1e-5, 8e-3, 1e-5
        last_stable = minimum
        for time_step in np.arange(minimum, maximum + increment / 2, increment):
            features = np.array(
                [
                    [
                        parameters.fluid_thermal_conductivity,
                        parameters.solid_effective_thermal_conductivity,
                        parameters.fluid_volumetric_heat_capacity,
                        parameters.solid_volumetric_heat_capacity,
                        parameters.velocity,
                        time_step,
                    ]
                ]
            )
            prediction = float(
                self._model.predict(self._scaler.transform(features), verbose=0)[0][0]
            )
            if prediction < 0.5:
                break
            last_stable = float(time_step)

        return float(f"{last_stable:.2g}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.canvas = MatplotlibCanvas()
        layout = QVBoxLayout(self.ui.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.result = None
        self.predictor = TimeStepPredictor()
        self.ui.horizontalSlider.setEnabled(False)
        self.ui.horizontalSlider.valueChanged.connect(self.update_graph_from_slider)
        self.ui.pushButton_2.clicked.connect(self.run_calculation)
        self.ui.pushButton_3.clicked.connect(self.suggest_time_step)

    def _read_parameters(self) -> SimulationParameters:
        def value(field: QLineEdit) -> float:
            return float(field.text())

        def integer(field: QLineEdit) -> int:
            return int(field.text())

        specific_area = value(self.ui.lineEdit_A_0)
        interphase_coefficient = value(self.ui.lineEdit_h_sf)

        parameters = SimulationParameters(
            length=value(self.ui.lineEdit_L),
            height=value(self.ui.lineEdit_W),
            ambient_temperature_inlet=value(self.ui.lineEdit_T_amb1),
            ambient_temperature_outlet=value(self.ui.lineEdit_T_amb2),
            initial_fluid_temperature=value(self.ui.lineEdit_F_T_init),
            initial_solid_temperature=value(self.ui.lineEdit_S_T_init),
            fluid_thermal_conductivity=value(self.ui.lineEdit_F_keff),
            solid_effective_thermal_conductivity=value(self.ui.lineEdit_S_keff),
            solid_thermal_conductivity=value(self.ui.lineEdit_S_k),
            volumetric_heat_transfer_coefficient=specific_area * interphase_coefficient,
            inlet_heat_transfer_coefficient=value(self.ui.lineEdit_h_1),
            outlet_heat_transfer_coefficient=value(self.ui.lineEdit_h_2),
            fluid_volumetric_heat_capacity=value(self.ui.lineEdit_F_den)
            * value(self.ui.lineEdit_F_capac),
            solid_volumetric_heat_capacity=value(self.ui.lineEdit_S_den)
            * value(self.ui.lineEdit_S_capac),
            porosity=value(self.ui.lineEdit_poros),
            velocity=value(self.ui.lineEdit_F_vel),
            viscosity=value(self.ui.lineEdit_F_visc),
            permeability=value(self.ui.lineEdit_Perm),
            time_steps=integer(self.ui.lineEdit_t),
            x_points=integer(self.ui.lineEdit_X),
            y_points=integer(self.ui.lineEdit_Y),
            time_step=value(self.ui.lineEdit_t_step),
            lower_wall_temperature=value(self.ui.lineEdit_T_low),
            upper_wall_temperature=value(self.ui.lineEdit_T_up),
        )
        parameters.validate()
        return parameters

    def suggest_time_step(self) -> None:
        try:
            parameters = self._read_parameters()
            suggested = self.predictor.suggest(parameters)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Time-step predictor", str(exc))
            return

        self.ui.lineEdit_t_step.setText(f"{suggested:g}")

    def run_calculation(self) -> None:
        try:
            parameters = self._read_parameters()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        if parameters.estimated_temperature_storage_bytes > LARGE_SIMULATION_BYTES:
            gibibytes = parameters.estimated_temperature_storage_bytes / 1024**3
            answer = QMessageBox.question(
                self,
                "Large simulation",
                f"The temperature history requires about {gibibytes:.2f} GiB of RAM. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.result = solve(parameters)
        except (MemoryError, ValueError) as exc:
            QMessageBox.critical(self, "Calculation failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not self.result.converged:
            QMessageBox.critical(
                self,
                "Unstable solution",
                "The explicit solution became unstable. Decrease the time step and try again.",
            )
            self.result = None
            return

        self.canvas.plot_contours(
            self.result.fluid_temperature,
            self.result.solid_temperature,
            parameters.length,
            parameters.height,
            parameters.time_step,
        )
        self.ui.horizontalSlider.setRange(0, self.result.completed_steps - 1)
        self.ui.horizontalSlider.setTickInterval(1)
        self.ui.horizontalSlider.setSingleStep(1)
        self.ui.horizontalSlider.setValue(self.result.completed_steps - 1)
        self.ui.horizontalSlider.setEnabled(True)
        self.ui.label_61.setText(f"{self.result.pressure[0, 0]:.3f}")
        self._update_outlet_temperature(self.result.completed_steps - 1)

    def _update_outlet_temperature(self, step: int) -> None:
        assert self.result is not None
        average = np.mean(self.result.fluid_temperature[step, -1, :])
        self.ui.label_62.setText(f"{average:.3f}")

    def update_graph_from_slider(self, value: int) -> None:
        if self.result is None:
            return
        current_time = value * float(self.ui.lineEdit_t_step.text())
        self.canvas.update_step(
            self.result.fluid_temperature[value],
            self.result.solid_temperature[value],
            current_time,
        )
        self._update_outlet_temperature(value)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        super().__init__(self.figure)
        self.x_grid = None
        self.y_grid = None
        self.minimum_temperature = None
        self.maximum_temperature = None
        self.time_text = None

    def plot_contours(
        self,
        fluid: np.ndarray,
        solid: np.ndarray,
        length: float,
        height: float,
        time_step: float,
    ) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(1, 2)
        self.minimum_temperature = float(min(fluid.min(), solid.min()))
        self.maximum_temperature = float(max(fluid.max(), solid.max()))
        if np.isclose(self.minimum_temperature, self.maximum_temperature):
            self.minimum_temperature -= 0.5
            self.maximum_temperature += 0.5
        x = np.linspace(0, length, fluid.shape[1])
        y = np.linspace(0, height, fluid.shape[2])
        self.x_grid, self.y_grid = np.meshgrid(x, y, indexing="ij")

        contours = self._draw_temperatures(fluid[-1], solid[-1])
        for axis, contour in zip(self.axes, contours, strict=True):
            self.figure.colorbar(contour, ax=axis, orientation="vertical", aspect=10)

        final_time = time_step * (fluid.shape[0] - 1)
        self.time_text = self.figure.text(
            0.5, 0.95, f"Time: {final_time:.3f} s", ha="center", fontsize=12
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.93))
        self.draw_idle()

    def update_step(
        self, fluid_temperature: np.ndarray, solid_temperature: np.ndarray, time: float
    ) -> None:
        for axis in self.axes:
            axis.clear()
        self._draw_temperatures(fluid_temperature, solid_temperature)
        assert self.time_text is not None
        self.time_text.set_text(f"Time: {time:.3f} s")
        self.draw_idle()

    def _draw_temperatures(
        self, fluid_temperature: np.ndarray, solid_temperature: np.ndarray
    ) -> tuple[object, object]:
        assert self.x_grid is not None and self.y_grid is not None
        contours = []
        for axis, values, title in zip(
            self.axes,
            (fluid_temperature, solid_temperature),
            ("Fluid temperature", "Solid temperature"),
            strict=True,
        ):
            contour = axis.contourf(
                self.x_grid,
                self.y_grid,
                values,
                levels=100,
                cmap="jet",
                vmin=self.minimum_temperature,
                vmax=self.maximum_temperature,
            )
            axis.set_title(title)
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]", rotation=0, labelpad=16)
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            contours.append(contour)
        return contours[0], contours[1]


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
