import sys
import cv2
from moviepy import VideoFileClip
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFontMetrics


class VideoCutter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Cutter")
        self.setGeometry(100, 100, 800, 600)

        # 视频对象
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # 视频状态
        self.playing = False
        self.total_frames = 0
        self.fps = 30
        self.start_frame = 0
        self.end_frame = 0
        self.current_frame_idx = 0
        self.video_path = None

        # UI组件

        self.label = QLabel("Load a video to start")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_label = QLabel(" 0 / 0 ")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.frame_label.font()
        # font.setFamily("Monospace")  # 用等宽字体更精确
        # font.setPointSize(10)  # 字号可按需调整
        # self.frame_label.setFont(font)

        # 计算10个字符的宽度
        fm = QFontMetrics(font)
        char_width = fm.horizontalAdvance("0")  # 单个字符宽度
        self.frame_label.setFixedWidth(char_width * 10)

        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_pause)

        self.cut_btn = QPushButton("Cut Video")
        self.cut_btn.clicked.connect(self.cut_video)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setFixedHeight(int(1.2 * fm.height()))

        # self.start_input = QLineEdit()
        # self.start_input.setPlaceholderText("Start frame")
        # self.end_input = QLineEdit()
        # self.end_input.setPlaceholderText("End frame")

        # 原来的start_input和end_input换成按钮
        self.start_btn = QPushButton("Set Start")
        self.start_btn.clicked.connect(self.set_start_frame)

        self.end_btn = QPushButton("Set End")
        self.end_btn.clicked.connect(self.set_end_frame)

        # # 显示剪辑起止帧的label
        # self.start_label = QLabel("Start: 0")
        # self.end_label = QLabel(f"End: {self.total_frames - 1}")

        # # 按钮布局
        # clip_layout = QHBoxLayout()
        # clip_layout.addWidget(self.start_btn)
        # clip_layout.addWidget(self.start_label)
        # clip_layout.addStretch()
        # clip_layout.addWidget(self.end_btn)
        # clip_layout.addWidget(self.end_label)

        # 布局
        h_layout = QHBoxLayout()

        h_layout.addWidget(self.load_btn)
        h_layout.addWidget(self.play_btn)
        h_layout.addWidget(self.cut_btn)

        # Start / End QLineEdit（保留原有输入框）
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("Start frame")
        self.start_input.editingFinished.connect(self.edit_start_frame)

        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("End frame")
        self.end_input.editingFinished.connect(self.edit_end_frame)

        time_layout = QHBoxLayout()
        time_layout.addWidget(self.start_btn)
        time_layout.addWidget(self.start_input)
        time_layout.addWidget(self.end_btn)
        time_layout.addWidget(self.end_input)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.frame_label)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label, stretch=8)
        # v_layout.addWidget(self.label)
        v_layout.addLayout(slider_layout, stretch=1)

        # v_layout.addWidget(self.frame_label)
        # v_layout.addWidget(self.slider)
        v_layout.addLayout(time_layout, stretch=1)
        v_layout.addLayout(h_layout, stretch=1)

        self.setLayout(v_layout)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.end_frame = self.total_frames - 1
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setEnabled(True)
            self.current_frame_idx = 0
            self.show_frame(self.current_frame_idx)
            self.update_frame_label()

    def show_frame(self, idx):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.flip(frame, 0)  # 修正上下倒置
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.label.setPixmap(pix.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.update_frame_label()

    def update_frame_label(self):
        self.frame_label.setText(f"{self.current_frame_idx} / {self.total_frames - 1}")

    def next_frame(self):
        if self.current_frame_idx >= self.total_frames:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("Play")
            return
        self.show_frame(self.current_frame_idx)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self.current_frame_idx += 1

    def play_pause(self):
        if self.cap is None:
            return
        if self.playing:
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(int(1000 / self.fps))
            self.play_btn.setText("Pause")
        self.playing = not self.playing

    def slider_changed(self, value):
        self.current_frame_idx = value
        self.show_frame(self.current_frame_idx)

    def set_start_frame(self):
        self.clip_start_frame = self.slider.value()
        self.start_input.setText(str(self.clip_start_frame))

    def set_end_frame(self):
        self.clip_end_frame = self.slider.value()
        self.end_input.setText(f"{self.clip_end_frame}")

    def edit_start_frame(self):
        # 用户在输入框手动编辑
        try:
            val = int(self.start_input.text())
            if 0 <= val < self.total_frames:
                self.clip_start_frame = val
            else:
                self.start_input.setText(str(self.clip_start_frame))
        except ValueError:
            self.start_input.setText(str(self.clip_start_frame))

    def edit_end_frame(self):
        try:
            val = int(self.end_input.text())
            if 0 <= val < self.total_frames:
                self.clip_end_frame = val
            else:
                self.end_input.setText(str(self.clip_end_frame))
        except ValueError:
            self.end_input.setText(str(self.clip_end_frame))

    def cut_video(self):
        if not self.video_path:
            return
        try:
            start_frame = int(self.start_input.text())
            end_frame = int(self.end_input.text())
        except ValueError:
            print("Invalid start/end frame")
            return
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, self.total_frames - 1)
        if start_frame >= end_frame:
            print("Start frame must be smaller than end frame")
            return

        # 使用 VideoFileClip 按帧裁剪
        clip = VideoFileClip(self.video_path)
        start_sec = start_frame / self.fps
        end_sec = end_frame / self.fps
        subclip = clip.subclipped(start_sec, end_sec)

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4)")
        if save_path:
            subclip.write_videofile(save_path, codec="libx264")
            print(f"Saved cut video to {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoCutter()
    window.show()
    sys.exit(app.exec())
