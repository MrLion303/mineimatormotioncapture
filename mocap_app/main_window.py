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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .camera import (
    BEST_MODEL_VARIANT,
    DEFAULT_PREVIEW_FPS,
    CameraDevice,
    CameraWorker,
    enumerate_available_cameras,
)
from .miobject import MiobjectError, export_miobject, load_template
from .video import VideoWorker, pose_timestamp_is_valid

EXPORT_FPS = 30


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mine-imator Motion Capture")
        self.setMinimumSize(900, 650)
        self.resize(1180, 800)

        self.camera_worker: CameraWorker | None = None
        self.video_worker: VideoWorker | None = None
        self.video_path: Path | None = None
        self._processing_source = "camera"
        self._capture_error: str | None = None
        self.camera_devices: list[CameraDevice] = []
        self.camera_ready = False
        self._camera_switch_pending = False
        self.template_path: Path | None = None
        self.last_image: QImage | None = None

        self.state = "idle"
        self.countdown_value = 3
        self.record_start = 0.0
        self.record_limit = 10.0
        self.recorded_duration = 0.0
        self.record_end = 0.0
        self.samples: list[tuple[float, dict]] = []

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.record_timer = QTimer(self)
        self.record_timer.setInterval(50)
        self.record_timer.timeout.connect(self._update_recording)

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(round(1000.0 / DEFAULT_PREVIEW_FPS))
        self.preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self._last_preview_sequence = 0

        self._build_ui()
        self._apply_style()
        self._update_actions()
        self._set_status(
            "Selecciona un vídeo y una plantilla para procesarlo, o conecta una cámara para grabar."
        )

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        central = QWidget()
        central.setObjectName("central")
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        page = QVBoxLayout(central)
        page.setContentsMargins(18, 16, 18, 16)
        page.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)

        brand = QLabel("NS")
        brand.setObjectName("brandMark")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedSize(48, 48)
        header.addWidget(brand)

        heading = QVBoxLayout()
        heading.setSpacing(2)
        eyebrow = QLabel("NEGATIVE STUDIOS  /  TOOLS")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("MOTION CAPTURE")
        title.setObjectName("title")
        subtitle = QLabel("Convierte tus movimientos en animaciones para Mine-imator.")
        subtitle.setObjectName("subtitle")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()

        version = QLabel(f"BETA  {__version__}")
        version.setObjectName("version")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(version, alignment=Qt.AlignmentFlag.AlignTop)
        page.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(18)
        page.addLayout(content, 1)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        content.addLayout(left_column, 7)

        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(16, 15, 16, 14)
        preview_layout.setSpacing(12)

        preview_header = QHBoxLayout()
        preview_heading = QVBoxLayout()
        preview_heading.setSpacing(2)
        preview_title = QLabel("VISTA EN DIRECTO")
        preview_title.setObjectName("sectionTitle")
        preview_subtitle = QLabel("Seguimiento corporal en tiempo real")
        preview_subtitle.setObjectName("sectionHint")
        preview_heading.addWidget(preview_title)
        preview_heading.addWidget(preview_subtitle)
        preview_header.addLayout(preview_heading)
        preview_header.addStretch()

        self.preview_fps_label = QLabel("CAM  -- FPS")
        self.preview_fps_label.setObjectName("fpsBadge")
        self.preview_fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_header.addWidget(self.preview_fps_label)

        self.pose_fps_label = QLabel("TRACK  -- FPS")
        self.pose_fps_label.setObjectName("poseBadge")
        self.pose_fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_header.addWidget(self.pose_fps_label)

        self.camera_state_label = QLabel("SIN CONEXIÓN")
        self.camera_state_label.setObjectName("statePill")
        self.camera_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_header.addWidget(self.camera_state_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        preview_layout.addLayout(preview_header)

        preview_host = QWidget()
        preview_grid = QGridLayout(preview_host)
        preview_grid.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel("Selecciona un vídeo o conecta una cámara para comenzar.")
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(480, 320)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        status_row.addWidget(self.status_dot, alignment=Qt.AlignmentFlag.AlignTop)

        self.status_label = QLabel("Esperando conexión con la cámara.")
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        preview_layout.addLayout(status_row)

        progress_row = QHBoxLayout()
        self.progress_caption = QLabel("PROGRESO DE CAPTURA")
        self.progress_caption.setObjectName("microLabel")
        self.time_label = QLabel("Sin grabación")
        self.time_label.setObjectName("timeText")
        progress_row.addWidget(self.progress_caption)
        progress_row.addStretch()
        progress_row.addWidget(self.time_label)
        preview_layout.addLayout(progress_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setObjectName("captureProgress")
        preview_layout.addWidget(self.progress)
        left_column.addWidget(preview_panel, 1)

        tip = QFrame()
        tip.setObjectName("tipPanel")
        tip_layout = QHBoxLayout(tip)
        tip_layout.setContentsMargins(14, 10, 14, 10)
        tip_layout.setSpacing(10)
        tip_mark = QLabel("TIP")
        tip_mark.setObjectName("tipMark")
        tip_layout.addWidget(tip_mark, alignment=Qt.AlignmentFlag.AlignTop)
        tip_text = QLabel(
            "Mantén el cuerpo completo dentro del encuadre y evita ocultar hombros, codos, "
            "muñecas, caderas, rodillas y tobillos. Al comenzar REC, conserva una postura "
            "neutra y quieta durante un instante para calibrar las articulaciones. Después "
            "realiza el movimiento. El personaje permanecerá en la posición de la plantilla."
        )
        tip_text.setObjectName("hint")
        tip_text.setWordWrap(True)
        tip_layout.addWidget(tip_text, 1)
        left_column.addWidget(tip)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        content.addLayout(right_column, 4)

        video_panel = self._panel("IMPORTAR VÍDEO")
        video_layout = video_panel.layout()
        self.video_label = QLabel("Ningún vídeo seleccionado")
        self.video_label.setObjectName("fileLabel")
        self.video_label.setWordWrap(True)
        video_layout.addWidget(self.video_label)

        self.select_video_button = QPushButton("Seleccionar vídeo")
        self.select_video_button.setObjectName("secondaryButton")
        self.select_video_button.clicked.connect(self._select_video)
        video_layout.addWidget(self.select_video_button)

        self.process_video_button = QPushButton("PROCESAR VÍDEO")
        self.process_video_button.setObjectName("primaryButton")
        self.process_video_button.setMinimumHeight(44)
        self.process_video_button.clicked.connect(self._process_video)
        video_layout.addWidget(self.process_video_button)
        right_column.addWidget(video_panel)

        capture_panel = self._panel("CAPTURA POR CÁMARA")
        capture_layout = capture_panel.layout()

        capture_layout.addWidget(self._field_label("DISPOSITIVO"))
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(0)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_selection_changed)
        camera_row.addWidget(self.camera_combo, 1)

        self.refresh_camera_button = QPushButton("Actualizar")
        self.refresh_camera_button.setObjectName("secondaryButton")
        self.refresh_camera_button.setMinimumWidth(78)
        self.refresh_camera_button.setMaximumWidth(112)
        self.refresh_camera_button.clicked.connect(self._refresh_camera_list)
        camera_row.addWidget(self.refresh_camera_button)
        capture_layout.addLayout(camera_row)
        self._refresh_camera_list()

        model_note = QLabel("SEGUIMIENTO DE ALTA PRECISIÓN · MODELO HEAVY")
        model_note.setObjectName("fixedSetting")
        model_note.setWordWrap(True)
        capture_layout.addWidget(model_note)

        numeric_grid = QGridLayout()
        duration_label = self._field_label("DURACIÓN")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(" s")
        numeric_grid.addWidget(duration_label, 0, 0)
        numeric_grid.addWidget(self.duration_spin, 1, 0)
        numeric_grid.setColumnStretch(0, 1)
        capture_layout.addLayout(numeric_grid)

        fixed_fps_note = QLabel("Parámetros automáticos: captura optimizada y exportación a 30 FPS.")
        fixed_fps_note.setObjectName("hint")
        fixed_fps_note.setWordWrap(True)
        capture_layout.addWidget(fixed_fps_note)


        options_panel = QFrame()
        options_panel.setObjectName("innerPanel")
        options_layout = QVBoxLayout(options_panel)
        options_layout.setContentsMargins(12, 10, 12, 10)
        options_layout.setSpacing(8)

        options_title = QLabel("PROCESAMIENTO")
        options_title.setObjectName("microLabel")
        options_layout.addWidget(options_title)

        self.smooth_check = QCheckBox("Estabilizar sin perder gestos suaves")
        self.smooth_check.setChecked(True)
        options_layout.addWidget(self.smooth_check)

        self.optimize_check = QCheckBox("Optimizar cantidad de keyframes")
        self.optimize_check.setChecked(True)
        options_layout.addWidget(self.optimize_check)

        capture_layout.addWidget(options_panel)

        capture_layout.addSpacing(2)

        self.camera_button = QPushButton("Conectar cámara")
        self.camera_button.setObjectName("secondaryButton")
        self.camera_button.clicked.connect(self._toggle_camera)
        capture_layout.addWidget(self.camera_button)

        self.record_button = QPushButton("GRABAR")
        self.record_button.setObjectName("primaryButton")
        self.record_button.setMinimumHeight(48)
        self.record_button.clicked.connect(self._start_countdown)
        capture_layout.addWidget(self.record_button)

        self.stop_button = QPushButton("Cancelar / detener")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop_capture)
        self.stop_button.setEnabled(False)
        capture_layout.addWidget(self.stop_button)
        right_column.addWidget(capture_panel)

        template_panel = self._panel("EXPORTACIÓN")
        template_layout = template_panel.layout()

        target_header = QHBoxLayout()
        target_title = QLabel("PLANTILLA DE MINE-IMATOR")
        target_title.setObjectName("microLabel")
        target_header.addWidget(target_title)
        target_header.addStretch()
        target_layout_badge = QLabel(".MIOBJECT")
        target_layout_badge.setObjectName("fileBadge")
        target_header.addWidget(target_layout_badge)
        template_layout.addLayout(target_header)

        self.template_label = QLabel("Ninguna plantilla seleccionada")
        self.template_label.setObjectName("fileLabel")
        self.template_label.setWordWrap(True)
        template_layout.addWidget(self.template_label)

        self.load_template_button = QPushButton("Seleccionar plantilla")
        self.load_template_button.setObjectName("secondaryButton")
        self.load_template_button.clicked.connect(self._load_template)
        template_layout.addWidget(self.load_template_button)

        self.export_button = QPushButton("EXPORTAR ANIMACIÓN")
        self.export_button.setObjectName("exportButton")
        self.export_button.clicked.connect(self._export)
        template_layout.addWidget(self.export_button)
        right_column.addWidget(template_panel)

        notice = QLabel(
            "El seguimiento usa siempre el modelo Heavy. La vista previa y la "
            "exportación mantienen sus parámetros recomendados automáticamente."
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
        layout.setContentsMargins(16, 15, 16, 15)
        layout.setSpacing(11)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background: #11120f;
                color: #e9e9e2;
                font-family: "Segoe UI";
                font-size: 9pt;
            }
            QWidget#central {
                background: #11120f;
            }
            QFrame#panel {
                background: #1b1c18;
                border: 1px solid #30312a;
                border-radius: 13px;
            }
            QFrame#innerPanel {
                background: #151611;
                border: 1px solid #2d2f26;
                border-radius: 9px;
            }
            QFrame#tipPanel {
                background: #191b15;
                border: 1px solid #303328;
                border-radius: 10px;
            }
            QLabel#brandMark {
                color: #c5dda0;
                background: #24291d;
                border: 1px solid #3b4630;
                border-radius: 12px;
                font-size: 15pt;
                font-weight: 900;
                letter-spacing: 1px;
            }
            QLabel#eyebrow {
                color: #8f967f;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1.5px;
            }
            QLabel#title {
                color: #f3f3ed;
                font-size: 20pt;
                font-weight: 800;
                letter-spacing: 1.2px;
            }
            QLabel#subtitle {
                color: #a4a69a;
                font-size: 9pt;
            }
            QLabel#version {
                color: #c2d69f;
                background: #24291d;
                border: 1px solid #38432c;
                border-radius: 7px;
                padding: 7px 10px;
                font-size: 8pt;
                font-weight: 800;
                letter-spacing: 0.6px;
            }
            QLabel#sectionTitle {
                color: #e7e8df;
                font-weight: 800;
                font-size: 9pt;
                letter-spacing: 1.1px;
            }
            QLabel#sectionHint {
                color: #85877b;
                font-size: 8pt;
            }
            QScrollArea#pageScroll {
                background: #11120f;
                border: 0;
            }
            QLabel#preview {
                background: #090a08;
                color: #77796e;
                border: 1px solid #303128;
                border-radius: 9px;
                font-size: 10pt;
            }
            QLabel#recordOverlay {
                background: rgba(8, 9, 6, 188);
                color: #d0e6aa;
                font-size: 58pt;
                font-weight: 900;
                border: 1px solid rgba(180, 207, 137, 90);
                border-radius: 9px;
            }
            QLabel#statusDot {
                color: #a9c87d;
                font-size: 9pt;
                padding-top: 2px;
            }
            QLabel#statusText {
                color: #c3c4b9;
                font-size: 9pt;
            }
            QLabel#hint {
                color: #96998b;
                font-size: 8pt;
            }
            QLabel#timeText {
                color: #c6dba0;
                font-weight: 800;
                min-width: 100px;
            }
            QLabel#microLabel {
                color: #898d7c;
                font-size: 7.5pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#statePill {
                color: #b6b9a9;
                background: #292b23;
                border: 1px solid #3a3c30;
                border-radius: 7px;
                padding: 6px 9px;
                font-size: 7.5pt;
                font-weight: 800;
                letter-spacing: 0.5px;
            }
            QLabel#fpsBadge {
                color: #c2d69f;
                background: #25291e;
                border: 1px solid #3b422e;
                border-radius: 7px;
                padding: 6px 8px;
                font-size: 7.5pt;
                font-weight: 800;
            }
            QLabel#poseBadge {
                color: #c3c5b8;
                background: #25261f;
                border: 1px solid #3a3b30;
                border-radius: 7px;
                padding: 6px 8px;
                font-size: 7.5pt;
                font-weight: 800;
            }
            QLabel#tipMark {
                color: #c0d699;
                background: #29301f;
                border-radius: 5px;
                padding: 4px 6px;
                font-size: 7pt;
                font-weight: 900;
                letter-spacing: 0.5px;
            }
            QLabel#fileLabel {
                color: #d7d8ce;
                background: #151611;
                border: 1px solid #2d2f26;
                border-radius: 7px;
                padding: 10px;
                font-size: 9pt;
            }
            QLabel#fileBadge {
                color: #aebf8c;
                background: #25291e;
                border: 1px solid #3b422e;
                border-radius: 5px;
                padding: 4px 6px;
                font-size: 7pt;
                font-weight: 800;
            }
            QLabel#fieldLabel {
                color: #96998a;
                font-size: 7.5pt;
                font-weight: 800;
                letter-spacing: 0.7px;
                margin-top: 2px;
            }
            QComboBox, QSpinBox {
                min-width: 0;
                background: #13140f;
                color: #ecece4;
                border: 1px solid #36382e;
                border-radius: 7px;
                padding: 8px 10px;
                min-height: 23px;
                selection-background-color: #53633b;
            }
            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #626b4c;
                background: #171911;
            }
            QComboBox:focus, QSpinBox:focus {
                border: 1px solid #9dbb6d;
            }
            QComboBox::drop-down {
                border: 0;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                background: #1b1c18;
                color: #ecece4;
                border: 1px solid #424536;
                selection-background-color: #3d4930;
                selection-color: #f4f5ed;
                outline: 0;
                padding: 4px;
            }
            QCheckBox {
                color: #c3c5b8;
                spacing: 8px;
                padding: 2px 0;
                font-size: 8.5pt;
            }
            QCheckBox:hover {
                color: #e7e9dc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #565849;
                border-radius: 4px;
                background: #11120e;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #9dbb6d;
            }
            QCheckBox::indicator:checked {
                background: #a9c87d;
                border: 1px solid #a9c87d;
            }
            QPushButton {
                min-height: 35px;
                min-width: 0;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 5px 10px;
                font-weight: 800;
                letter-spacing: 0.3px;
            }
            QPushButton:disabled {
                color: #696b60;
                background: #24251f;
                border-color: #2d2e27;
            }
            QPushButton#primaryButton {
                color: #171b10;
                background: #b2ce85;
                border-color: #b2ce85;
                font-size: 11pt;
            }
            QPushButton#primaryButton:hover {
                background: #c1dc95;
                border-color: #c1dc95;
            }
            QPushButton#primaryButton:pressed {
                background: #95b36a;
            }
            QPushButton#secondaryButton {
                color: #e2e3d9;
                background: #2c2e26;
                border-color: #3b3d32;
            }
            QPushButton#secondaryButton:hover {
                background: #373a2e;
                border-color: #5b6247;
            }
            QPushButton#stopButton {
                color: #e4b6a8;
                background: #35251f;
                border-color: #4a3027;
            }
            QPushButton#stopButton:hover {
                background: #493027;
                border-color: #76503c;
            }
            QPushButton#exportButton {
                color: #c9dda3;
                background: #292f20;
                border-color: #465238;
            }
            QPushButton#exportButton:hover {
                color: #e2f1c4;
                background: #343d27;
                border-color: #64784a;
            }
            QProgressBar#captureProgress {
                border: 0;
                border-radius: 4px;
                background: #303127;
                min-height: 7px;
                max-height: 7px;
            }
            QProgressBar#captureProgress::chunk {
                background: #a9c87d;
                border-radius: 4px;
            }
            QToolTip {
                color: #eeeeE7;
                background: #25261f;
                border: 1px solid #535641;
                padding: 5px;
            }
        """)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _refresh_camera_list(self) -> None:
        if self.camera_worker is not None:
            self._set_status("Desconecta la cámara antes de actualizar la lista.")
            return

        previous = self.camera_combo.currentData()
        previous_key = (
            (previous.index, previous.backend)
            if isinstance(previous, CameraDevice)
            else None
        )
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_devices = enumerate_available_cameras()

        selected_index = -1
        for device in self.camera_devices:
            display_name = f"{device.name}  ·  {device.backend_name}"
            self.camera_combo.addItem(display_name, device)
            if previous_key == (device.index, device.backend):
                selected_index = self.camera_combo.count() - 1

        if not self.camera_devices:
            self.camera_combo.addItem("No se detectaron cámaras", None)
        elif selected_index >= 0:
            self.camera_combo.setCurrentIndex(selected_index)

        self.camera_combo.blockSignals(False)
        self.camera_combo.setEnabled(bool(self.camera_devices))
        self.refresh_camera_button.setEnabled(True)

        if self.camera_devices:
            self._set_status(
                f"Se detectaron {len(self.camera_devices)} cámara(s). "
                "Selecciona el dispositivo por su nombre."
            )
        else:
            self._set_status(
                "No se detectaron cámaras. Conecta una webcam y pulsa Actualizar."
            )

    def _toggle_camera(self) -> None:
        if self.camera_worker is not None:
            self._disconnect_camera()
            return
        self._start_camera_worker()

    def _start_camera_worker(self) -> None:
        if self.camera_worker is not None:
            return

        device = self.camera_combo.currentData()
        if not isinstance(device, CameraDevice):
            self._set_status("No hay una cámara disponible para conectar.")
            return

        model_variant = BEST_MODEL_VARIANT
        worker = CameraWorker(
            device.index,
            model_variant,
            self,
            camera_backend=device.backend,
            camera_name=device.name,
            preview_fps=DEFAULT_PREVIEW_FPS,
        )
        self.camera_worker = worker
        worker.pose_ready.connect(self._receive_pose)
        worker.capture_drained.connect(self._capture_drained)
        worker.preview_fps.connect(self._update_preview_fps)
        worker.pose_fps.connect(self._update_pose_fps)
        worker.status.connect(self._set_status)
        worker.ready.connect(self._camera_ready)
        worker.failed.connect(self._camera_error)
        worker.finished.connect(lambda worker=worker: self._camera_finished(worker))

        self.camera_ready = False
        self._last_preview_sequence = 0
        self.last_image = None
        self.preview_fps_label.setText("CAM  -- FPS")
        self.pose_fps_label.setText("TRACK  -- FPS")
        self.preview_timer.start()
        self.camera_button.setText("Conectando...")
        self.camera_button.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.refresh_camera_button.setEnabled(False)
        self.camera_state_label.setText("CONECTANDO")
        self._set_status(f"Conectando a {device.name}...")
        worker.start()

    def _on_camera_selection_changed(self, _index: int) -> None:
        if self.state != "idle" or self.camera_worker is None or not self.camera_ready:
            return

        device = self.camera_combo.currentData()
        if not isinstance(device, CameraDevice):
            return
        if (
            device.index == self.camera_worker.camera_index
            and device.backend == self.camera_worker.camera_backend
        ):
            return

        self._camera_switch_pending = True
        self.camera_ready = False
        self.preview_timer.stop()
        self.camera_button.setText("Cambiando cámara...")
        self.camera_button.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.camera_state_label.setText("CAMBIANDO")
        self.preview.clear()
        self.preview.setText(f"Conectando a {device.name}...")
        self._set_status(f"Cambiando a {device.name}...")
        self.camera_worker.stop()

    def _camera_ready(self) -> None:
        self.camera_ready = True
        self.camera_button.setText("Desconectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_combo.setEnabled(self.state == "idle")
        self.refresh_camera_button.setEnabled(False)
        self.camera_state_label.setText("CONECTADA")
        self._update_actions()

    def _camera_error(self, message: str) -> None:
        self._set_status(f"Error de cámara: {message}")
        QMessageBox.critical(self, "No se pudo iniciar la cámara", message)
        if self.state == "finalizing":
            self._complete_recording()

    def _camera_finished(self, worker: CameraWorker) -> None:
        if self.camera_worker is not worker:
            return

        self.camera_worker = None
        self.camera_ready = False
        worker.deleteLater()

        if self._camera_switch_pending:
            self._camera_switch_pending = False
            self._last_preview_sequence = 0
            self.last_image = None
            device = self.camera_combo.currentData()
            camera_name = (
                device.name if isinstance(device, CameraDevice)
                else "la cámara seleccionada"
            )
            self._set_status(f"Conectando a {camera_name}...")
            self._start_camera_worker()
            return

        self.preview_timer.stop()
        self.last_image = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Conecta una cámara para comenzar el seguimiento.")
        self.preview_fps_label.setText("CAM  -- FPS")
        self.pose_fps_label.setText("TRACK  -- FPS")
        self.camera_button.setText("Conectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_combo.setEnabled(bool(self.camera_devices))
        self.refresh_camera_button.setEnabled(True)
        self.camera_state_label.setText("SIN CONEXIÓN")
        self.record_button.setEnabled(False)
        if self.state != "idle":
            self._stop_capture()

    def _disconnect_camera(self) -> None:
        if self.state != "idle":
            self._stop_capture()
        worker = self.camera_worker
        self.camera_worker = None
        self.camera_ready = False
        self._camera_switch_pending = False
        self.preview_timer.stop()
        if worker is not None:
            worker.stop()
            worker.wait()
            worker.deleteLater()
        self.last_image = None
        self._last_preview_sequence = 0
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Conecta una cámara para comenzar el seguimiento.")
        self.preview_fps_label.setText("CAM  -- FPS")
        self.pose_fps_label.setText("TRACK  -- FPS")
        self.camera_button.setText("Conectar cámara")
        self.camera_button.setEnabled(True)
        self.camera_combo.setEnabled(bool(self.camera_devices))
        self.refresh_camera_button.setEnabled(True)
        self.camera_state_label.setText("SIN CONEXIÓN")
        self._update_actions()
        self._set_status("Cámara desconectada.")

    def _refresh_preview(self) -> None:
        worker = self.camera_worker
        if worker is None:
            return

        latest = worker.latest_preview_frame()
        if latest is None:
            return

        sequence, image = latest
        if sequence == self._last_preview_sequence:
            return

        self._last_preview_sequence = sequence
        self.last_image = image
        self._render_preview(smooth=False)

    def _update_preview_fps(self, fps: float) -> None:
        self.preview_fps_label.setText(f"CAM  {fps:04.1f} FPS")

    def _update_pose_fps(self, fps: float) -> None:
        self.pose_fps_label.setText(f"TRACK  {fps:04.1f} FPS")

    def _render_preview(self, smooth: bool = False) -> None:
        if self.last_image is None:
            return
        pixmap = QPixmap.fromImage(self.last_image)
        transformation = (
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation
        )
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                transformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview(smooth=True)

    def _receive_pose(self, pose: dict, captured_at: float) -> None:
        if self.state not in ("recording", "finalizing"):
            return
        elapsed = captured_at - self.record_start
        if pose_timestamp_is_valid(
            captured_at,
            self.record_start,
            self.recorded_duration,
            self._processing_source,
        ):
            if self.samples and elapsed <= self.samples[-1][0]:
                return
            self.samples.append((elapsed, pose))
            if len(self.samples) % 15 == 0 and self._processing_source == "camera":
                self.time_label.setText(f"{len(self.samples)} muestras")

    def _start_countdown(self) -> None:
        if not self.camera_ready or self.state != "idle":
            return
        self.samples.clear()
        self._capture_error = None
        self._processing_source = "camera"
        self.record_limit = float(self.duration_spin.value())
        self.countdown_value = 3
        self.state = "countdown"
        self.duration_spin.setEnabled(False)
        self.camera_button.setEnabled(False)
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
        self.record_end = self.record_start + self.record_limit
        if self.camera_worker is not None:
            self.camera_worker.begin_recording_capture()
        self.overlay.setText("REC")
        self.overlay.show()
        self.time_label.setText(f"{self.record_limit:.1f} s restantes")
        self._set_status(
            "Calibrando la postura inicial; mantente neutro un instante y después realiza el movimiento."
        )
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
        elif self.state == "finalizing":
            self._complete_recording()

    def _finish_recording(self, automatic: bool) -> None:
        if self.state != "recording":
            return
        self.record_timer.stop()
        elapsed = time.monotonic() - self.record_start
        self.recorded_duration = (
            self.record_limit
            if automatic
            else max(0.05, min(elapsed, self.record_limit))
        )
        self.record_end = self.record_start + self.recorded_duration
        self.state = "finalizing"
        self.overlay.setText("...")
        self.overlay.show()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Procesando...")
        self._set_status("Procesando los últimos fotogramas de la captura...")

        worker = self.camera_worker
        if worker is None:
            self._complete_recording()
            return
        worker.finish_recording_capture()

    def _capture_drained(self) -> None:
        if self.state == "finalizing":
            self._complete_recording()

    def _complete_recording(self) -> None:
        if self.state != "finalizing":
            return
        self.state = "idle"
        self.overlay.hide()
        self.progress.setRange(0, 1000)
        capture_error = self._capture_error
        self._capture_error = None
        self.camera_state_label.setText(
            "CONECTADA" if self.camera_ready else "SIN CONEXIÓN"
        )

        if capture_error:
            self.progress_caption.setText("PROCESAMIENTO INTERRUMPIDO")
            self.progress.setValue(0)
            self.time_label.setText("Procesamiento fallido")
            self.samples.clear()
        elif not self.samples:
            self.progress_caption.setText("SIN POSES DETECTADAS")
            self.progress.setValue(0)
            self.time_label.setText("Sin poses válidas")
        else:
            self.progress_caption.setText("PROCESAMIENTO COMPLETADO")
            if self._processing_source == "video":
                self.progress.setValue(1000)
            else:
                ratio = (
                    self.recorded_duration / self.record_limit
                    if self.record_limit > 0.0 else 0.0
                )
                self.progress.setValue(int(max(0.0, min(1.0, ratio)) * 1000))
            self.time_label.setText(
                f"{len(self.samples)} muestras · Listo para exportar"
            )

        self._restore_controls()

        if capture_error:
            self._set_status(f"No se pudo procesar la captura: {capture_error}")
            QMessageBox.critical(self, "Falló el procesamiento", capture_error)
        elif not self.samples:
            self._set_status(
                "No se detectó ninguna pose válida. Comprueba que el cuerpo "
                "aparezca completo, con buena iluminación y sin oclusiones."
            )
        elif self._processing_source == "video":
            self._set_status(
                f"Vídeo procesado: {self.recorded_duration:.2f} segundos y "
                f"{len(self.samples)} muestras. Ya puedes exportar la animación."
            )
        else:
            self._set_status(
                f"Grabación terminada: {self.recorded_duration:.1f} segundos y "
                f"{len(self.samples)} muestras de pose."
            )
        self._update_actions()

    def _restore_controls(self) -> None:
        self.duration_spin.setEnabled(True)
        self.camera_button.setEnabled(self.camera_ready)
        self.camera_combo.setEnabled(
            bool(self.camera_devices)
            and self.state == "idle"
            and (self.camera_worker is None or self.camera_ready)
        )
        self.refresh_camera_button.setEnabled(self.camera_worker is None)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Cancelar / detener")
        self.record_button.setEnabled(self.camera_ready)
        self.select_video_button.setEnabled(True)
        self.load_template_button.setEnabled(True)
        self._update_actions()

    def _update_processing_progress(
        self,
        completed_frames: int,
        total_frames: int,
        _fps: float,
        remaining_seconds: float,
    ) -> None:
        if self.state != "finalizing" or total_frames <= 0:
            return
        completed_frames = max(0, min(completed_frames, total_frames))
        ratio = completed_frames / total_frames
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(round(ratio * 1000)))
        minutes, seconds = divmod(max(0, int(remaining_seconds)), 60)
        percent = int(round(ratio * 100))
        self.time_label.setText(
            f"{completed_frames}/{total_frames} · {percent}% · "
            f"{minutes:02d}:{seconds:02d} restantes"
        )

    def _select_video(self) -> None:
        if self.state != "idle":
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar vídeo para Motion Capture",
            "",
            "Vídeos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;Todos los archivos (*.*)",
        )
        if not filename:
            return
        self.video_path = Path(filename)
        self.video_label.setText(self.video_path.name)
        self._set_status(
            "Vídeo seleccionado. Carga una plantilla humana y pulsa PROCESAR VÍDEO."
        )
        self._update_actions()

    def _process_video(self) -> None:
        if self.state != "idle":
            return
        if self.video_path is None:
            self._set_status("Selecciona primero un vídeo.")
            return
        if self.template_path is None:
            self._set_status("Selecciona primero una plantilla humana de Mine-imator.")
            return
        if self.camera_worker is not None:
            self._set_status(
                "Desconecta la cámara antes de procesar un vídeo para no duplicar el motor."
            )
            return

        self.samples.clear()
        self._capture_error = None
        self.recorded_duration = 0.0
        self.record_limit = 0.0
        self.record_start = 0.0
        self._processing_source = "video"
        self.state = "finalizing"
        self.progress_caption.setText("PROCESO DE VÍDEO")
        self.progress.setRange(0, 0)
        self.time_label.setText("Preparando vídeo...")
        self.camera_state_label.setText("PROCESANDO VÍDEO")
        self.preview_timer.stop()
        self.preview.setPixmap(QPixmap())
        self.preview.setText(f"Procesando: {self.video_path.name}")
        self.overlay.hide()
        self.duration_spin.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.refresh_camera_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.load_template_button.setEnabled(False)
        self.select_video_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._set_status(
            "Abriendo el vídeo y preparando MediaPipe PoseLandmarker Heavy. "
            "No hace falta conectar una cámara."
        )

        worker = VideoWorker(self.video_path, self)
        self.video_worker = worker
        worker.pose_ready.connect(self._receive_pose)
        worker.processing_progress.connect(self._update_processing_progress)
        worker.pose_fps.connect(self._update_pose_fps)
        worker.status.connect(self._set_status)
        worker.failed.connect(self._video_error)
        worker.completed.connect(self._video_completed)
        worker.finished.connect(lambda worker=worker: self._video_finished(worker))
        self._update_actions()
        worker.start()

    def _video_error(self, message: str) -> None:
        self._capture_error = message
        self.samples.clear()
        self._set_status(f"Error al procesar el vídeo: {message}")

    def _video_completed(self, _pose_count: int, duration: float) -> None:
        self.recorded_duration = max(0.0, float(duration))
        self.record_limit = self.recorded_duration

    def _video_finished(self, worker: VideoWorker) -> None:
        if self.video_worker is not worker:
            return
        self.video_worker = None
        worker.deleteLater()
        if self.state == "finalizing":
            self._complete_recording()

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
        idle = self.state == "idle"
        self.record_button.setEnabled(self.camera_ready and idle)
        self.export_button.setEnabled(
            self.template_path is not None and bool(self.samples) and idle
        )
        self.select_video_button.setEnabled(idle)
        self.load_template_button.setEnabled(idle)
        self.process_video_button.setEnabled(
            idle
            and self.video_path is not None
            and self.template_path is not None
            and self.camera_worker is None
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
                fps=EXPORT_FPS,
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
        self.preview_timer.stop()
        video_worker = self.video_worker
        if video_worker is not None:
            video_worker.stop()
            video_worker.wait()
            self.video_worker = None
        worker = self.camera_worker
        if worker is not None:
            worker.stop()
            worker.wait()
            self.camera_worker = None
        event.accept()
