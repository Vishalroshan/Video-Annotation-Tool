from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QDockWidget, QListWidget, QListWidgetItem, QPushButton,
    QLineEdit, QSlider, QFileDialog, QMessageBox, QTextEdit,
    QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QAction,
    QInputDialog,
)
import sys
import os
import numpy as np
import torch
import cv2
from PIL import Image

# cv2 sets QT_QPA_PLATFORM_PLUGIN_PATH to its bundled Qt plugins on import,
# which are incompatible with PyQt5. Override immediately after importing cv2.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
)

from sam2.build_sam import build_sam2_video_predictor

# ── Device setup ─────────────────────────────────────────────────────────────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "cpu":
    torch.autocast("cpu", dtype=torch.bfloat16).__enter__()

# ── Annotation classes ────────────────────────────────────────────────────────
OBJ_NAMES = [
    "Head Surgeon", "Assistant Surgeon", "Nurse", "Tool 1", "Tool 2",
]

# BGR for OpenCV drawing; RGB derived on the fly as (B,G,R)->(R,G,B)
OBJ_COLORS_BGR = [
    (0, 200, 0),    # Head Surgeon   — green
    (200, 100, 0),  # Asst Surgeon   — blue
    (0, 0, 220),    # Nurse          — red
    (0, 180, 255),  # Tool 1         — orange
    (180, 0, 220),  # Tool 2         — purple
]

MAX_OBJECTS = len(OBJ_NAMES)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SAM2_CHECKPOINT_REL = os.path.join("checkpoints", "sam2.1_hiera_large.pt")
SAM2_CONFIG_CANDIDATES = [
    os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml"),
    os.path.join("sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
]


def _candidate_sam2_roots():
    candidates = []
    env_root = os.environ.get("SAM2_ROOT")
    if env_root:
        candidates.append(os.path.abspath(env_root))
    candidates.extend([
        BASE_DIR,
        os.path.abspath(os.path.join(BASE_DIR, "..")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "sam2_repo")),
        os.getcwd(),
    ])
    seen, unique = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def resolve_sam2_paths():
    for root in _candidate_sam2_roots():
        ckpt = os.path.join(root, SAM2_CHECKPOINT_REL)
        if not os.path.isfile(ckpt):
            continue
        for cfg_rel in SAM2_CONFIG_CANDIDATES:
            cfg = os.path.join(root, cfg_rel)
            if os.path.isfile(cfg):
                hydra_cfg = cfg_rel.replace("\\", "/")
                if hydra_cfg.startswith("sam2/"):
                    hydra_cfg = hydra_cfg[len("sam2/"):]
                return hydra_cfg, ckpt
    checked = "\n  - ".join(_candidate_sam2_roots())
    raise FileNotFoundError(
        "Could not locate SAM2 config/checkpoint. Set SAM2_ROOT.\n"
        f"Checked roots:\n  - {checked}"
    )


def get_total_frame_count(path: str) -> int:
    if os.path.isdir(path):
        return len([
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def load_frames_from_folder(folder: str, start: int, count: int):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    )
    return [np.array(Image.open(f)) for f in files[start:start + count]]


def load_video_frames(path: str, start: int, count: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_frames(path: str, start: int, count: int):
    if os.path.isdir(path):
        return load_frames_from_folder(path, start, count)
    return load_video_frames(path, start, count)


# ── VideoWidget ───────────────────────────────────────────────────────────────

class VideoWidget(QWidget):
    frame_changed = pyqtSignal(int)
    point_added = pyqtSignal(int, int, int)   # x, y, label
    point_removed = pyqtSignal(int, int)      # x, y (nearest)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = []
        self.current_frame = 0
        self._current_pixmap = None

        # Shared references — updated by MainWindow after each batch load
        self.objects_data = {}
        self.active_obj = 0

        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background: black;")
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.display_label)

    def load_frames(self, frames: list):
        self.frames = frames
        self.current_frame = 0
        self.refresh()
        self.frame_changed.emit(0)

    def seek_to_frame(self, frame_idx: int):
        if not self.frames:
            return
        frame_idx = max(0, min(frame_idx, len(self.frames) - 1))
        self.current_frame = frame_idx
        self.refresh()
        self.frame_changed.emit(self.current_frame)

    def refresh(self):
        if not self.frames or self.current_frame >= len(self.frames):
            return
        frame = self._apply_overlays(self.frames[self.current_frame].copy())
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self._render_pixmap()

    def _apply_overlays(self, display: np.ndarray) -> np.ndarray:
        fidx = self.current_frame

        for obj_id, data in self.objects_data.items():
            if fidx in data["masks"]:
                mask = data["masks"][fidx]
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                b, g, r = OBJ_COLORS_BGR[obj_id]
                color_rgb = (r, g, b)
                overlay = np.zeros_like(display)
                overlay[mask > 0] = color_rgb
                display = cv2.addWeighted(display, 1.0, overlay, 0.6, 0)
                contours, _ = cv2.findContours(
                    (mask * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(display, contours, -1, color_rgb, 2)

        for obj_id, data in self.objects_data.items():
            b, g, r = OBJ_COLORS_BGR[obj_id]
            color_rgb = (r, g, b)
            for point, label in zip(
                data["points"].get(fidx, []),
                data["labels"].get(fidx, []),
            ):
                x, y = int(point[0]), int(point[1])
                if label == 1:
                    cv2.circle(display, (x, y), 10, (255, 255, 255), -1)
                    cv2.circle(display, (x, y), 8, color_rgb, -1)
                    cv2.circle(display, (x, y), 10, (0, 0, 0), 2)
                else:
                    cv2.circle(display, (x, y), 10, color_rgb, -1)
                    cv2.circle(display, (x, y), 10, (255, 255, 255), 2)

        return display

    def _render_pixmap(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.display_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pixmap()

    def _frame_coords(self, wx: int, wy: int):
        """Map widget-space coordinates to frame-pixel coordinates."""
        if not self.frames or self._current_pixmap is None:
            return None, None
        fh, fw = self.frames[self.current_frame].shape[:2]
        lw = self.display_label.width()
        lh = self.display_label.height()
        scale = min(lw / fw, lh / fh)
        off_x = (lw - fw * scale) / 2
        off_y = (lh - fh * scale) / 2
        # display_label fills the widget, so widget coords == label coords
        fx = (wx - off_x) / scale
        fy = (wy - off_y) / scale
        if 0 <= fx < fw and 0 <= fy < fh:
            return int(fx), int(fy)
        return None, None

    def mousePressEvent(self, event):
        if not self.frames:
            return
        fx, fy = self._frame_coords(event.x(), event.y())
        if fx is None:
            return
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self.point_removed.emit(fx, fy)
            else:
                self.point_added.emit(fx, fy, 1)
        elif event.button() == Qt.MiddleButton:
            self.point_added.emit(fx, fy, 0)


# ── Stream redirector (captures print() into the log box) ────────────────────

class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def __init__(self, original_stream=None):
        super().__init__()
        self._original = original_stream

    def write(self, text: str):
        if text:
            self.text_written.emit(text)
        if self._original:
            self._original.write(text)

    def flush(self):
        if self._original:
            self._original.flush()


# ── ObjectSelectorPanel ───────────────────────────────────────────────────────

class ObjectSelectorPanel(QWidget):
    object_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_obj = 0

        self.batch_label = QLabel("Batch 1 / 1")
        self.batch_label.setStyleSheet("color: black; font-size: 11px;")

        self.list_widget = QListWidget()
        self.list_widget.setSpacing(1)
        for i, name in enumerate(OBJ_NAMES):
            b, g, r = OBJ_COLORS_BGR[i]
            item = QListWidgetItem(name)
            item.setBackground(QColor(r, g, b, 160))
            item.setForeground(QColor(0, 0, 0))
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)

        controls = QLabel(
            "<b>Controls</b><br>"
            "Left click — positive point<br>"
            "Middle click — negative point<br>"
            "Shift+click — remove nearest<br>"
            "D — delete last point<br>"
            "A / Z — switch object<br>"
            "; / ' — navigate (step)<br>"
            "[ / ] — navigate (5× step)<br>"
            ", / . — decrease / increase step<br>"
            "G — go to global frame<br>"
            "Space — play / pause<br>"
            "R — SAM2 current frame<br>"
            "M — merge &amp; propagate<br>"
            "T — save masks<br>"
            "N / P — next / prev batch"
        )
        controls.setStyleSheet("color: black; font-size: 10px;")
        controls.setWordWrap(True)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(180)
        self.log_box.setStyleSheet(
            "background: #f8f8f8; color: black; font-family: monospace; font-size: 10px;"
        )
        self.log_box.setPlaceholderText("Terminal output…")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("<b>Object Selection</b>"))
        layout.addWidget(self.batch_label)
        layout.addWidget(self.list_widget)
        layout.addWidget(controls)
        layout.addWidget(QLabel("<b>Log</b>"))
        layout.addWidget(self.log_box)
        layout.addStretch()

    def _on_row_changed(self, row: int):
        if 0 <= row < MAX_OBJECTS:
            self.active_obj = row
            self.object_selected.emit(row)

    def set_active_object(self, obj_id: int):
        self.active_obj = obj_id
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(obj_id)
        self.list_widget.blockSignals(False)

    def update_batch_info(self, current: int, total: int):
        self.batch_label.setText(f"Batch {current} / {total}")

    def update_point_counts(self, objects_data: dict, num_frames: int):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            total_pts = sum(
                len(objects_data[i]["points"].get(f, []))
                for f in range(num_frames)
            )
            item.setText(f"{OBJ_NAMES[i]}  [{total_pts}]" if total_pts else OBJ_NAMES[i])

    def append_log(self, text: str):
        from PyQt5.QtGui import QTextCursor
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(text)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )


# ── ControlsBar ───────────────────────────────────────────────────────────────

class ControlsBar(QWidget):
    goto_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.setFixedWidth(70)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setFixedWidth(70)

        self.next_btn = QPushButton("Next >")
        self.next_btn.setFixedWidth(70)

        self.step_edit = QLineEdit("1")
        self.step_edit.setFixedWidth(45)
        self.step_edit.setAlignment(Qt.AlignCenter)
        self.step_edit.setToolTip("Frames to skip per Prev / Next click")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTracking(True)

        self.goto_edit = QLineEdit()
        self.goto_edit.setFixedWidth(75)
        self.goto_edit.setAlignment(Qt.AlignCenter)
        self.goto_edit.setPlaceholderText("Go to…")
        self.goto_edit.setToolTip("Type a frame number and press Enter to jump")
        self.goto_edit.returnPressed.connect(self._on_goto)

        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFixedWidth(160)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(QLabel("Step:"))
        layout.addWidget(self.step_edit)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.goto_edit)
        layout.addWidget(self.frame_label)
        self.setFixedHeight(55)

    def _on_goto(self):
        try:
            self.goto_requested.emit(int(self.goto_edit.text()))
        except ValueError:
            pass
        self.goto_edit.clear()

    def get_step(self) -> int:
        try:
            return max(1, int(self.step_edit.text()))
        except ValueError:
            return 1

    def setup_range(self, total_frames: int):
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, total_frames - 1))
        self.slider.setValue(0)
        self.frame_label.setText(f"Frame: 0 / {total_frames - 1}")

    def update_display(self, frame_idx: int, total_frames: int):
        self.frame_label.setText(f"Frame: {frame_idx} / {max(0, total_frames - 1)}")
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

    def set_playing_state(self, is_playing: bool):
        self.play_pause_btn.setText("Pause" if is_playing else "Play")


# ── SAM2 action toolbar ───────────────────────────────────────────────────────

class SAMToolbar(QWidget):
    run_single_requested = pyqtSignal()
    propagate_requested = pyqtSignal()
    save_masks_requested = pyqtSignal()
    next_batch_requested = pyqtSignal()
    prev_batch_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.run_btn = QPushButton("R: Run Frame")
        self.propagate_btn = QPushButton("M: Propagate")
        self.save_btn = QPushButton("T: Save Masks")
        self.prev_batch_btn = QPushButton("P: Prev Batch")
        self.next_batch_btn = QPushButton("N: Next Batch")
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #88aaff; font-size: 11px;")

        for btn in (self.run_btn, self.propagate_btn, self.save_btn,
                    self.prev_batch_btn, self.next_batch_btn):
            btn.setFixedHeight(32)

        self.run_btn.clicked.connect(self.run_single_requested)
        self.propagate_btn.clicked.connect(self.propagate_requested)
        self.save_btn.clicked.connect(self.save_masks_requested)
        self.prev_batch_btn.clicked.connect(self.prev_batch_requested)
        self.next_batch_btn.clicked.connect(self.next_batch_requested)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.propagate_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.prev_batch_btn)
        layout.addWidget(self.next_batch_btn)
        layout.addStretch()
        layout.addWidget(self.status_label)
        self.setFixedHeight(45)

    def set_status(self, msg: str):
        self.status_label.setText(msg)
        QApplication.processEvents()


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")
        self.resize(1280, 760)

        # Annotation state
        self.objects_data = {
            i: {"points": {}, "labels": {}, "masks": {}}
            for i in range(MAX_OBJECTS)
        }
        self.video_segments = {}
        self.active_obj = 0
        self.frames = []
        self.current_batch = 0
        self.total_batches = 1
        self.batch_start_idx = 0
        self.total_frame_count = 0
        self.video_path = None
        self.previous_batch_masks = {}
        self.sam2_predictor = None
        self.sam2_inference_state = None
        self.nav_step = 1
        self.is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._setup_log_redirect()

    def _setup_ui(self):
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.video_widget = VideoWidget()
        self.controls_bar = ControlsBar()
        self.sam_toolbar = SAMToolbar()

        v.addWidget(self.video_widget, stretch=1)
        v.addWidget(self.controls_bar)
        v.addWidget(self.sam_toolbar)
        self.setCentralWidget(central)

        self.obj_panel = ObjectSelectorPanel()
        scroll = QScrollArea()
        scroll.setWidget(self.obj_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(240)

        dock = QDockWidget("Objects", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _setup_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("File")

        open_video = QAction("Open Video…", self)
        open_video.setShortcut("Ctrl+O")
        open_video.triggered.connect(self.open_video)

        open_folder = QAction("Open Folder of Frames…", self)
        open_folder.setShortcut("Ctrl+Shift+O")
        open_folder.triggered.connect(self.open_folder)

        save_masks = QAction("Save Masks…", self)
        save_masks.setShortcut("Ctrl+S")
        save_masks.triggered.connect(self.save_masks)

        file_menu.addAction(open_video)
        file_menu.addAction(open_folder)
        file_menu.addSeparator()
        file_menu.addAction(save_masks)

    def _connect_signals(self):
        self.video_widget.frame_changed.connect(self._on_frame_changed)
        self.video_widget.point_added.connect(self._on_point_added)
        self.video_widget.point_removed.connect(self._on_point_removed)

        self.controls_bar.slider.valueChanged.connect(self.video_widget.seek_to_frame)
        self.controls_bar.play_pause_btn.clicked.connect(self._toggle_play)
        self.controls_bar.prev_btn.clicked.connect(self._on_prev_step)
        self.controls_bar.next_btn.clicked.connect(self._on_next_step)
        self.controls_bar.goto_requested.connect(self.video_widget.seek_to_frame)

        self.obj_panel.object_selected.connect(self._on_object_selected)

        self.sam_toolbar.run_single_requested.connect(self.run_prediction_single_frame)
        self.sam_toolbar.propagate_requested.connect(self.merge_and_propagate)
        self.sam_toolbar.save_masks_requested.connect(self.save_masks)
        self.sam_toolbar.next_batch_requested.connect(self.next_batch)
        self.sam_toolbar.prev_batch_requested.connect(self.prev_batch)

    # ── Log redirect ─────────────────────────────────────────────────────────

    def _setup_log_redirect(self):
        self._stdout_redir = StreamRedirector(sys.stdout)
        self._stderr_redir = StreamRedirector(sys.stderr)
        self._stdout_redir.text_written.connect(self.obj_panel.append_log)
        self._stderr_redir.text_written.connect(self.obj_panel.append_log)
        sys.stdout = self._stdout_redir
        sys.stderr = self._stderr_redir

    # ── Playback ──────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.is_playing:
            self._play_timer.stop()
            self.is_playing = False
            self.controls_bar.set_playing_state(False)
        else:
            self.is_playing = True
            self._play_timer.start(40)
            self.controls_bar.set_playing_state(True)

    def _on_play_tick(self):
        cur = self.video_widget.current_frame
        if cur >= len(self.frames) - 1:
            self._play_timer.stop()
            self.is_playing = False
            self.controls_bar.set_playing_state(False)
            return
        self.video_widget.seek_to_frame(cur + 1)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_frame_changed(self, frame_idx: int):
        self.controls_bar.update_display(frame_idx, len(self.frames))
        self.obj_panel.update_point_counts(self.objects_data, len(self.frames))

    def _on_prev_step(self):
        step = self.controls_bar.get_step()
        self.video_widget.seek_to_frame(self.video_widget.current_frame - step)

    def _on_next_step(self):
        step = self.controls_bar.get_step()
        self.video_widget.seek_to_frame(self.video_widget.current_frame + step)

    # ── Object selection ──────────────────────────────────────────────────────

    def _on_object_selected(self, obj_id: int):
        self.active_obj = obj_id
        self.video_widget.active_obj = obj_id

    # ── Point handling ────────────────────────────────────────────────────────

    def _on_point_added(self, x: int, y: int, label: int):
        fidx = self.video_widget.current_frame
        pts = self.objects_data[self.active_obj]["points"]
        lbs = self.objects_data[self.active_obj]["labels"]
        if fidx not in pts:
            pts[fidx] = []
            lbs[fidx] = []
        pts[fidx].append([x, y])
        lbs[fidx].append(label)
        self.video_widget.refresh()
        self.obj_panel.update_point_counts(self.objects_data, len(self.frames))

    def _on_point_removed(self, x: int, y: int):
        fidx = self.video_widget.current_frame
        pts = self.objects_data[self.active_obj]["points"].get(fidx, [])
        if not pts:
            return
        arr = np.array(pts)
        idx = int(np.argmin(np.linalg.norm(arr - np.array([x, y]), axis=1)))
        self.objects_data[self.active_obj]["points"][fidx].pop(idx)
        self.objects_data[self.active_obj]["labels"][fidx].pop(idx)
        self.video_widget.refresh()
        self.obj_panel.update_point_counts(self.objects_data, len(self.frames))

    def _delete_last_point(self):
        fidx = self.video_widget.current_frame
        pts = self.objects_data[self.active_obj]["points"].get(fidx, [])
        if pts:
            self.objects_data[self.active_obj]["points"][fidx].pop()
            self.objects_data[self.active_obj]["labels"][fidx].pop()
            self.video_widget.refresh()
            self.obj_panel.update_point_counts(self.objects_data, len(self.frames))

    # ── File loading ──────────────────────────────────────────────────────────

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
        )
        if path:
            self._load_source(path)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder of Frames", "")
        if folder:
            self._load_source(folder)

    def _load_source(self, path: str):
        self.video_path = path
        self.total_frame_count = get_total_frame_count(path)
        self.total_batches = max(1, (self.total_frame_count + BATCH_SIZE - 1) // BATCH_SIZE)
        self.current_batch = 0
        self.previous_batch_masks = {}
        self._load_batch(0)
        self.setWindowTitle(f"Video Annotation Tool — {os.path.basename(path)}")

    def _load_batch(self, batch_idx: int):
        self.batch_start_idx = batch_idx * BATCH_SIZE
        count = min(BATCH_SIZE, self.total_frame_count - self.batch_start_idx)
        self.sam_toolbar.set_status(f"Loading batch {batch_idx + 1}/{self.total_batches}…")

        self.frames = load_frames(self.video_path, self.batch_start_idx, count)
        self.objects_data = {
            i: {"points": {}, "labels": {}, "masks": {}}
            for i in range(MAX_OBJECTS)
        }
        self.video_segments = {}
        self.sam2_inference_state = None

        if self.previous_batch_masks:
            self.video_segments[0] = {}
            for obj_id, mask in self.previous_batch_masks.items():
                self.video_segments[0][obj_id] = mask.copy()
                self.objects_data[obj_id - 1]["masks"][0] = mask.copy().astype(np.uint8)
            print(f"Restored {len(self.previous_batch_masks)} masks from previous batch to frame 0")

        # Hand shared references to the video widget
        self.video_widget.objects_data = self.objects_data
        self.video_widget.load_frames(self.frames)

        self.controls_bar.setup_range(len(self.frames))
        self.obj_panel.update_batch_info(batch_idx + 1, self.total_batches)
        self.obj_panel.update_point_counts(self.objects_data, len(self.frames))
        self.sam_toolbar.set_status(
            f"Batch {batch_idx + 1}/{self.total_batches}  "
            f"(frames {self.batch_start_idx}–{self.batch_start_idx + len(self.frames) - 1})"
        )
        print(f"Loaded batch {batch_idx + 1}/{self.total_batches} ({len(self.frames)} frames)")

    # ── Batch management ──────────────────────────────────────────────────────

    def next_batch(self):
        if self.current_batch >= self.total_batches - 1:
            QMessageBox.information(self, "Batch", "Already at last batch.")
            return
        if self.video_segments:
            self._auto_save_masks()
            last_fidx = len(self.frames) - 1
            if last_fidx in self.video_segments:
                self.previous_batch_masks = {
                    oid: mask.copy()
                    for oid, mask in self.video_segments[last_fidx].items()
                }
        self.current_batch += 1
        self._load_batch(self.current_batch)

    def prev_batch(self):
        if self.current_batch <= 0:
            QMessageBox.information(self, "Batch", "Already at first batch.")
            return
        self.previous_batch_masks = {}
        self.current_batch -= 1
        self._load_batch(self.current_batch)

    def _goto_global_frame(self, global_1based: int):
        global_zero = global_1based - 1
        target_batch = global_zero // BATCH_SIZE
        target_local = global_zero % BATCH_SIZE
        if target_batch != self.current_batch:
            if target_batch > self.current_batch and self.video_segments:
                self._auto_save_masks()
            self.current_batch = target_batch
            self._load_batch(self.current_batch)
        self.video_widget.seek_to_frame(min(target_local, len(self.frames) - 1))

    # ── SAM2: single frame ────────────────────────────────────────────────────

    def run_prediction_single_frame(self):
        fidx = self.video_widget.current_frame
        has_pts = any(
            self.objects_data[oid]["points"].get(fidx)
            for oid in range(MAX_OBJECTS)
        )
        if not has_pts:
            QMessageBox.warning(
                self, "No Points",
                f"No points on frame {fidx}. Add points first."
            )
            return

        self.sam_toolbar.set_status("Running SAM2 on current frame…")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_cfg, sam2_ckpt = resolve_sam2_paths()
            predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, sam2_ckpt, device=device))
            predictor.set_image(self.frames[fidx])

            for obj_id, data in self.objects_data.items():
                if not data["points"].get(fidx):
                    continue
                points = np.array(data["points"][fidx], dtype=np.float32)
                labels = np.array(data["labels"][fidx], dtype=np.int32)
                masks, _, _ = predictor.predict(
                    point_coords=points, point_labels=labels, multimask_output=False
                )
                mask = masks[0]
                self.video_segments.setdefault(fidx, {})[obj_id + 1] = mask
                data["masks"][fidx] = mask.astype(np.uint8)
                print(f"Mask: {OBJ_NAMES[obj_id]} @ frame {self.batch_start_idx + fidx}")

            self.video_widget.refresh()
            self.sam_toolbar.set_status("Single-frame prediction done.")
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.sam_toolbar.set_status(f"Error: {exc}")
            QMessageBox.critical(self, "SAM2 Error", str(exc))

    # ── SAM2: propagate ───────────────────────────────────────────────────────

    def merge_and_propagate(self):
        if not self.video_segments:
            QMessageBox.warning(
                self, "No Masks",
                "No masks to propagate. Run SAM2 on a frame first (R)."
            )
            return

        start_frame = self.video_widget.current_frame
        self.sam_toolbar.set_status("Initialising SAM2 for propagation…")
        try:
            sam2_cfg, sam2_ckpt = resolve_sam2_paths()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.sam2_predictor = build_sam2_video_predictor(
                sam2_cfg, sam2_ckpt, device=device
            )

            frames_dir = "./frames_from_video"
            if os.path.exists(frames_dir):
                import shutil; shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)

            self.sam_toolbar.set_status(f"Saving {len(self.frames)} frames to disk…")
            for i, frame in enumerate(self.frames):
                Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:05d}.jpg"))

            self.sam2_inference_state = self.sam2_predictor.init_state(video_path=frames_dir)

            relevant = [f for f in sorted(self.video_segments) if f <= start_frame]
            if not relevant:
                QMessageBox.warning(
                    self, "No Masks",
                    f"No masks at or before frame {start_frame}."
                )
                return

            init_frame = relevant[-1]
            objects_added = set()
            for obj_id, mask in self.video_segments[init_frame].items():
                try:
                    self.sam2_predictor.add_new_mask(
                        inference_state=self.sam2_inference_state,
                        frame_idx=init_frame,
                        obj_id=obj_id,
                        mask=np.squeeze(mask),
                    )
                    objects_added.add(obj_id)
                    print(f"Added mask for {OBJ_NAMES[obj_id-1]} at frame {self.batch_start_idx + init_frame}")
                except Exception as e:
                    print(f"Error adding mask for {OBJ_NAMES[obj_id-1]}: {e}")

            if not objects_added:
                QMessageBox.warning(self, "Error", "No masks could be added for propagation.")
                return

            preserved = {f: s.copy() for f, s in self.video_segments.items() if f < start_frame}
            for f in [k for k in self.video_segments if k >= start_frame]:
                del self.video_segments[f]

            self.sam_toolbar.set_status(
                f"Propagating from frame {self.batch_start_idx + start_frame}…"
            )
            for out_fidx, out_obj_ids, out_logits in self.sam2_predictor.propagate_in_video(
                inference_state=self.sam2_inference_state,
                start_frame_idx=start_frame,
            ):
                segs = self.video_segments.setdefault(out_fidx, {})
                for i, oid in enumerate(out_obj_ids):
                    mask = (out_logits[i] > 0.0).cpu().numpy()
                    segs[oid] = mask
                    self.objects_data[oid - 1]["masks"][out_fidx] = mask.astype(np.uint8)

            for f, segs in preserved.items():
                self.video_segments[f] = segs
                for oid, mask in segs.items():
                    self.objects_data[oid - 1]["masks"][f] = mask.astype(np.uint8)

            self.video_widget.refresh()
            self.sam_toolbar.set_status(
                f"Propagation done — {len(self.video_segments)} frames have masks."
            )
            print(f"Propagation done from frame {self.batch_start_idx + start_frame}.")
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.sam_toolbar.set_status(f"Error: {exc}")
            QMessageBox.critical(self, "SAM2 Error", str(exc))

    # ── Mask saving ───────────────────────────────────────────────────────────

    def _write_masks(self, out_dir: str) -> int:
        """Save one PNG per frame that has at least one mask. Returns frame count."""
        os.makedirs(out_dir, exist_ok=True)

        # Collect every frame index that has any mask across all objects.
        all_frames = set()
        for data in self.objects_data.values():
            all_frames.update(data["masks"].keys())

        for fidx in sorted(all_frames):
            global_idx = self.batch_start_idx + fidx
            h, w = self.frames[fidx].shape[:2]
            combined_bgr = np.zeros((h, w, 3), dtype=np.uint8)

            for obj_id, data in self.objects_data.items():
                if fidx not in data["masks"]:
                    continue
                mask_2d = np.squeeze(data["masks"][fidx]).astype(bool)
                if not mask_2d.any():
                    continue
                b, g, r = OBJ_COLORS_BGR[obj_id]  # obj_id is already 0-based
                combined_bgr[mask_2d, 0] = b
                combined_bgr[mask_2d, 1] = g
                combined_bgr[mask_2d, 2] = r

            cv2.imwrite(
                os.path.join(out_dir, f"frame_{global_idx:05d}_combined.png"),
                combined_bgr,
            )

        return len(all_frames)

    def _has_any_masks(self) -> bool:
        return any(data["masks"] for data in self.objects_data.values())

    def _auto_save_masks(self):
        out_dir = os.path.join(BASE_DIR, "output_video_masks")
        n = self._write_masks(out_dir)
        print(f"Auto-saved {n} mask frames to {out_dir}")

    def save_masks(self):
        if not self._has_any_masks():
            QMessageBox.information(self, "Save Masks", "No masks to save.")
            return
        folder, ok = QInputDialog.getText(
            self, "Save Masks", "Output folder name:", text="output_video_masks"
        )
        if not ok or not folder.strip():
            return
        out_dir = os.path.join(BASE_DIR, folder.strip())
        n = self._write_masks(out_dir)
        QMessageBox.information(
            self, "Save Masks", f"Saved {n} mask frames to:\n{out_dir}"
        )
        self.sam_toolbar.set_status(f"Saved {n} mask frames.")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        cur = self.video_widget.current_frame

        if key == Qt.Key_A:
            new_obj = (self.active_obj - 1) % MAX_OBJECTS
            self.active_obj = new_obj
            self.video_widget.active_obj = new_obj
            self.obj_panel.set_active_object(new_obj)

        elif key == Qt.Key_Z:
            new_obj = (self.active_obj + 1) % MAX_OBJECTS
            self.active_obj = new_obj
            self.video_widget.active_obj = new_obj
            self.obj_panel.set_active_object(new_obj)

        elif key == Qt.Key_R:
            self.run_prediction_single_frame()

        elif key == Qt.Key_M:
            self.merge_and_propagate()

        elif key == Qt.Key_T:
            self.save_masks()

        elif key == Qt.Key_N:
            self.next_batch()

        elif key == Qt.Key_P:
            self.prev_batch()

        elif key == Qt.Key_D:
            self._delete_last_point()

        elif key == Qt.Key_Space:
            self._toggle_play()

        elif key == Qt.Key_Semicolon:
            self.video_widget.seek_to_frame(max(0, cur - self.nav_step))

        elif key == Qt.Key_Apostrophe:
            self.video_widget.seek_to_frame(min(len(self.frames) - 1, cur + self.nav_step))

        elif key == Qt.Key_BracketLeft:
            self.video_widget.seek_to_frame(max(0, cur - self.nav_step * 5))

        elif key == Qt.Key_BracketRight:
            self.video_widget.seek_to_frame(min(len(self.frames) - 1, cur + self.nav_step * 5))

        elif key == Qt.Key_Comma:
            self.nav_step = max(1, self.nav_step - 1)
            self.controls_bar.step_edit.setText(str(self.nav_step))
            self.sam_toolbar.set_status(f"Step size: {self.nav_step}")

        elif key == Qt.Key_Period:
            self.nav_step = min(100, self.nav_step + 1)
            self.controls_bar.step_edit.setText(str(self.nav_step))
            self.sam_toolbar.set_status(f"Step size: {self.nav_step}")

        elif key == Qt.Key_G:
            if not self.frames:
                return
            val, ok = QInputDialog.getInt(
                self, "Go to Frame",
                f"Global frame (1–{self.total_frame_count}):",
                self.batch_start_idx + cur + 1, 1, self.total_frame_count,
            )
            if ok:
                self._goto_global_frame(val)

        else:
            super().keyPressEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    app.setApplicationName("Video Annotation Tool")
    window = MainWindow()
    window.show()
    # Optional: pass a video / folder path as the first argument to open it on launch.
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window._load_source(sys.argv[1])
    sys.exit(app.exec_())
