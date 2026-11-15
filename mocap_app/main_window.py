from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import CameraWorker
from .miobject import MiobjectError, export_miobject, load_template


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mine-imator Motion Capture")
        self.setMinimumSize(1080, 700)
        self.resize(1240, 780)

        self.camera_worker: CameraWorker | None = None
        self.camera_ready = False
        self.template_path: Path | None = None
        self.last_image: QImage | None = None

        self.state = "idle"
        self.countdown_value = 3
        self.record_start = 0.0
        self.record_limit = 10.0
        self.recorded_duration = 0.0
        self.samples: list[tuple[float, dict]] = []

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.record_timer = QTimer(self)
        self.record_timer.setInterval(50)
        self.record_timer.timeout.connect(self._update_recording)

        self._build_ui()
        self._apply_style()
        self._update_actions()
        self._set_status("Conecta una cámara y carga una plantilla humana de Mine-imator.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(24, 20, 24, 20)
        page.setSpacing(16)

        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("MINE-IMATOR MOTION CAPTURE")
        title.setObjectName("title")
        subtitle = QLabel("Captura movimiento corporal y conviértelo en keyframes.")
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()
        version = QLabel("PREVIEW 0.1.0")
        version.setObjectName("version")
        header.addWidget(version, alignment=Qt.AlignmentFlag.AlignTop)
        page.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        page.addLayout(content, 1)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        content.addLayout(left_column, 3)

        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(10)

        preview_header = QHBoxLayout()
        preview_title = QLabel("VISTA DE CÁMARA")
        preview_title.setObjectName("sectionTitle")
        self.camera_state_label = QLabel("SIN CONEXIÓN")
        self.camera_state_label.setObjectName("statePill")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.camera_state_label)
        preview_layout.addLayout(preview_header)

        preview_host = QWidget()
        preview_grid = QGridLayout(preview_host)
        preview_grid.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel("Conecta una cámara para ver el seguimiento corporal.")
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(640, 390)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.overlay = QLabel("")
        self.overlay.setObjectName("recordOverlay")
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.overlay.hide()
        preview_grid.addWidget(self.preview, 0, 0)
        preview_grid.addWidget(self.overlay, 0, 0)
        preview_layout.addWidget(preview_host, 1)

        self.status_label = QLabel("Esperando cámara.")
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)
        preview_layout.addWidget(self.status_label)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.time_label = QLabel("Sin grabación")
        self.time_label.setObjectName("timeText")
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.time_label)
        preview_layout.addLayout(progress_row)
        left_column.addWidget(preview_panel, 1)

        info = QLabel(
            "Consejo: mantén la cabeza, las manos y los pies dentro del encuadre. "
            "La primera pose detectada se utiliza como referencia inicial."
        )
        info.setObjectName("hint")
        info.setWordWrap(True)
        left_column.addWidget(info)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        content.addLayout(right_column, 2)

        capture_panel = self._panel("CONFIGURACIÓN DE CAPTURA")
        capture_layout = capture_panel.layout()

        capture_layout.addWidget(self._field_label("Dispositivo de cámara"))
        self.camera_combo = QComboBox()
        for index in range(6):
            self.camera_combo.addItem(f"Cámara {index}", index)
        capture_layout.addWidget(self.camera_combo)

        capture_layout.addWidget(self._field_label("Duración de grabación"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(" segundos")
        capture_layout.addWidget(self.duration_spin)

        capture_layout.addWidget(self._field_label("FPS de exportación"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(12, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" FPS")
        capture_layout.addWidget(self.fps_spin)

        self.smooth_check = QCheckBox("Suavizar movimiento")
        self.smooth_check.setChecked(True)
        capture_layout.addWidget(self.smooth_check)

        self.optimize_check = QCheckBox("Optimizar cantidad de keyframes")
        self.optimize_check.setChecked(True)
        capture_layout.addWidget(self.optimize_check)

        capture_layout.addSpacing(8)
        self.camera_button = QPushButton("Conectar cámara")
        self.camera_button.setObjectName("secondaryButton")
        self.camera_button.clicked.connect(self._toggle_camera)
        capture_layout.addWidget(self.camera_button)

        self.record_button = QPushButton("GRABAR")
        self.record_button.setObjectName("primaryButton")
        self.record_button.clicked.connect(self._start_countdown)
        capture_layout.addWidget(self.record_button)

        self.stop_button = QPushButton("Cancelar / detener")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop_capture)
        self.stop_button.setEnabled(False)
        capture_layout.addWidget(self.stop_button)
        right_column.addWidget(capture_panel)

        template_panel = self._panel("PLANTILLA MINE-IMATOR")
        template_layout = template_panel.layout()
        self.template_label = QLabel("No se ha seleccionado ningún archivo.")
        self.template_label.setObjectName("fileLabel")
        self.template_label.setWordWrap(True)
        template_layout.addWidget(self.template_label)

        self.load_template_button = QPushButton("Cargar plantilla .miobject")
        self.load_template_button.setObjectName("secondaryButton")
        self.load_template_button.clicked.connect(self._load_template)
        template_layout.addWidget(self.load_template_button)

        self.export_button = QPushButton("EXPORTAR .MIOBJECT")
        self.export_button.setObjectName("exportButton")
        self.export_button.clicked.connect(self._export)
        template_layout.addWidget(self.export_button)
        right_column.addWidget(template_panel)

        notice = QLabel(
            "La profundidad de una webcam es estimada por el modelo. "
            "Para movimientos más precisos, usa buena iluminación y evita tapar las articulaciones."
        )
        notice.setObjectName("hint")
        notice.setWordWrap(True)
        right_column.addWidget(notice)
        right_column.addStretch()

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _panel(title: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background: #11151b;
                color: #e8edf3;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QFrame#panel {
                background: #1a2029;
                border: 1px solid #2b3542;
                border-radius: 12px;
            }
            QLabel#title {
                color: #f3f6fa;
                font-size: 19pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#subtitle, QLabel#hint {
                color: #9aa8b8;
            }
            QLabel#version {
                color: #75d7b5;
                font-size: 9pt;
                font-weight: 700;
                padding: 7px 10px;
                background: #1d302d;
                border-radius: 7px;
            }
            QLabel#sectionTitle {
                color: #dce5ed;
                font-weight: 700;
                font-size: 9pt;
                letter-spacing: 0.7px;
            }
            QLabel#preview {
                background: #080b10;
                color: #798797;
                border: 1px solid #303b49;
                border-radius: 8px;
                font-size: 11pt;
            }
            QLabel#recordOverlay {
                background: rgba(5, 9, 14, 150);
                color: #ffffff;
                font-size: 58pt;
                font-weight: 800;
                border-radius: 8px;
            }
            QLabel#statusText, QLabel#fileLabel {
                color: #b6c2ce;
            }
            QLabel#timeText {
                color: #75d7b5;
                font-weight: 700;
                min-width: 100px;
            }
            QLabel#statePill {
                color: #9aa8b8;
                background: #27313d;
                border-radius: 7px;
                padding: 5px 8px;
                font-size: 8pt;
                font-weight: 700;
            }
            QLabel#fieldLabel {
                color: #9eacba;
                font-size: 9pt;
                margin-top: 3px;
            }
            QComboBox, QSpinBox {
                background: #111821;
                color: #e8edf3;
                border: 1px solid #344253;
                border-radius: 7px;
                padding: 8px;
                min-height: 24px;
            }
            QComboBox:focus, QSpinBox:focus {
                border: 1px solid #70d5b3;
            }
            QCheckBox {
                color: #c4ced8;
                spacing: 8px;
                padding: 4px 0;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QPushButton {
                min-height: 38px;
                border: 0;
                border-radius: 7px;
                padding: 0 12px;
                font-weight: 700;
            }
            QPushButton:disabled {
                color: #677383;
                background: #29313c;
            }
            QPushButton#primaryButton {
                color: #08130f;
                background: #75d7b5;
                font-size: 12pt;
            }
            QPushButton#primaryButton:hover { background: #8be5c5; }
            QPushButton#secondaryButton {
                color: #e1e8ef;
                background: #303b49;
            }
            QPushButton#secondaryButton:hover { background: #3b4a5b; }
            QPushButton#stopButton {
                color: #ffd9d9;
                background: #512b33;
            }
            QPushButton#stopButton:hover { background: #67333e; }
            QPushButton#exportButton {
                color: #dceaff;
                background: #31547c;
            }
            QPushButton#exportButton:hover { background: #3a6696; }
            QProgressBar {
                border: 0;
                border-radius: 4px;
                background: #303b49;
                min-height: 8px;
                max-height: 8px;
            }
            QProgressBar::chunk {
                background: #75d7b5;
                border-radius: 4px;
            }
        """)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _toggle_camera(self) -> None:
        if self.camera_worker is not None:
            self._disconnect_camera()
            return

        camera_index = int(self.camera_combo.currentData())
        worker = CameraWorker(camera_index, self)
        self.camera_worker = worker
        worker.frame_ready.connect(self._show_frame)
        worker.pose_ready.connect(self._receive_pose)
        worker.status.connect(self._set_status)
        worker.ready.connect(self._camera_ready)
        worker.failed.connect(self._camera_error)
        worker.finished.connect(lambda worker=worker: self._camera_finished(worker))

        self.camera_button.setText("Conectando...")
        self.camera_button.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.camera_state_label.setText("CONECTANDO")
        worker.start()

    def _camera_ready(self) -> None:
        self.camera_ready = True
        self.camera_button.setText("Desconectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_state_label.setText("CONECTADA")
        self._update_actions()

    def _camera_error(self, message: str) -> None:
        self._set_status(f"Error de cámara: {message}")
        QMessageBox.critical(self, "No se pudo iniciar la cámara", message)

    def _camera_finished(self, worker: CameraWorker) -> None:
        if self.camera_worker is not worker:
            return
        self.camera_worker = None
        self.camera_ready = False
        self.camera_button.setText("Conectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.camera_state_label.setText("SIN CONEXIÓN")
        self.record_button.setEnabled(False)
        if self.state != "idle":
            self._stop_capture()
        worker.deleteLater()

    def _disconnect_camera(self) -> None:
        if self.state != "idle":
            self._stop_capture()
        worker = self.camera_worker
        self.camera_worker = None
        self.camera_ready = False
        if worker is not None:
            worker.stop()
            worker.wait(4000)
            worker.deleteLater()
        self.camera_button.setText("Conectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.camera_state_label.setText("SIN CONEXIÓN")
        self._update_actions()
        self._set_status("Cámara desconectada.")

    def _show_frame(self, image: QImage) -> None:
        self.last_image = image
        self._render_preview()

    def _render_preview(self) -> None:
        if self.last_image is None:
            return
        pixmap = QPixmap.fromImage(self.last_image)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview()

    def _receive_pose(self, pose: dict, captured_at: float) -> None:
        if self.state != "recording":
            return
        elapsed = captured_at - self.record_start
        if 0.0 <= elapsed <= self.record_limit:
            self.samples.append((elapsed, pose))
            if len(self.samples) % 15 == 0:
                self.time_label.setText(f"{len(self.samples)} muestras")

    def _start_countdown(self) -> None:
        if not self.camera_ready or self.state != "idle":
            return
        self.samples.clear()
        self.record_limit = float(self.duration_spin.value())
        self.countdown_value = 3
        self.state = "countdown"
        self.duration_spin.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.overlay.setText("3")
        self.overlay.show()
        self.time_label.setText("Preparando...")
        self.progress.setValue(0)
        self._set_status("La grabación comenzará en 3 segundos.")
        self.countdown_timer.start()

    def _countdown_tick(self) -> None:
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.overlay.setText(str(self.countdown_value))
            self._set_status(f"La grabación comenzará en {self.countdown_value}...")
            return
        self.countdown_timer.stop()
        self._begin_recording()

    def _begin_recording(self) -> None:
        self.state = "recording"
        self.record_start = time.monotonic()
        self.recorded_duration = self.record_limit
        self.overlay.setText("REC")
        self.overlay.show()
        self.time_label.setText(f"{self.record_limit:.1f} s restantes")
        self._set_status("Grabando movimiento...")
        self.record_timer.start()
        self.stop_button.setText("Detener grabación")

    def _update_recording(self) -> None:
        if self.state != "recording":
            self.record_timer.stop()
            return
        elapsed = time.monotonic() - self.record_start
        remaining = max(0.0, self.record_limit - elapsed)
        self.progress.setValue(int(min(1.0, elapsed / self.record_limit) * 1000))
        self.time_label.setText(f"{remaining:04.1f} s restantes")
        if remaining <= 0.0:
            self._finish_recording(automatic=True)

    def _stop_capture(self) -> None:
        if self.state == "countdown":
            self.countdown_timer.stop()
            self.state = "idle"
            self.overlay.hide()
            self.time_label.setText("Grabación cancelada")
            self._set_status("Cuenta regresiva cancelada.")
            self._restore_controls()
        elif self.state == "recording":
            self._finish_recording(automatic=False)

    def _finish_recording(self, automatic: bool) -> None:
        if self.state != "recording":
            return
        self.record_timer.stop()
        elapsed = time.monotonic() - self.record_start
        self.recorded_duration = self.record_limit if automatic else max(0.05, min(elapsed, self.record_limit))
        self.state = "idle"
        self.overlay.hide()
        self.progress.setValue(1000 if automatic else int(self.recorded_duration / self.record_limit * 1000))
        self.time_label.setText(f"{len(self.samples)} muestras")
        self._restore_controls()

        if not self.samples:
            self._set_status("No se detectó una pose durante la grabación. Ajusta el encuadre e inténtalo de nuevo.")
        else:
            self._set_status(
                f"Grabación terminada: {self.recorded_duration:.1f} segundos y "
                f"{len(self.samples)} muestras de pose."
            )
        self._update_actions()

    def _restore_controls(self) -> None:
        self.duration_spin.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.camera_combo.setEnabled(self.camera_worker is None)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Cancelar / detener")
        self.record_button.setEnabled(self.camera_ready)

    def _load_template(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar plantilla humana de Mine-imator",
            "",
            "Mine-imator Object (*.miobject);;Todos los archivos (*.*)",
        )
        if not filename:
            return
        try:
            load_template(filename)
        except MiobjectError as exc:
            QMessageBox.warning(self, "Plantilla no compatible", str(exc))
            return

        self.template_path = Path(filename)
        self.template_label.setText(self.template_path.name)
        self._set_status("Plantilla humana cargada y validada.")
        self._update_actions()

    def _update_actions(self) -> None:
        self.record_button.setEnabled(self.camera_ready and self.state == "idle")
        self.export_button.setEnabled(
            self.template_path is not None and bool(self.samples) and self.state == "idle"
        )

    def _export(self) -> None:
        if self.template_path is None or not self.samples:
            return
        suggested = "MineimatorMotionCapture.miobject"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar animación",
            suggested,
            "Mine-imator Object (*.miobject)",
        )
        if not filename:
            return
        output_path = Path(filename)
        if output_path.suffix.lower() != ".miobject":
            output_path = output_path.with_suffix(".miobject")

        try:
            result = export_miobject(
                self.template_path,
                output_path,
                self.samples,
                fps=self.fps_spin.value(),
                duration_seconds=self.recorded_duration,
                optimize_keyframes=self.optimize_check.isChecked(),
                smooth_motion=self.smooth_check.isChecked(),
            )
        except (MiobjectError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "No se pudo exportar", str(exc))
            return

        counts = ", ".join(
            f"{part}: {count}" for part, count in result["keyframe_counts"].items()
        )
        message = (
            f"Archivo exportado correctamente.\n\n"
            f"Duración: {result['duration']:.2f} s\n"
            f"FPS: {result['fps']}\n"
            f"Fotograma final: {result['final_frame']}\n"
            f"Keyframes generados: {result['total_keyframes']}\n\n"
            f"{counts}\n\n"
            f"{output_path}"
        )
        QMessageBox.information(self, "Exportación terminada", message)
        self._set_status(f"Exportación terminada: {output_path.name}")

    def closeEvent(self, event) -> None:
        self.countdown_timer.stop()
        self.record_timer.stop()
        worker = self.camera_worker
        if worker is not None:
            worker.stop()
            worker.wait(4000)
            self.camera_worker = None
        event.accept()
