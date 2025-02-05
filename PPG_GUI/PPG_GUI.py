import sys
import os
import serial.tools.list_ports
import serial
import threading
import numpy as np
import torch
from collections import deque
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox
from PyQt6.QtCore import pyqtSignal, QObject, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 載入 PPG 模型
from PPG_model import DilatedDenoisingAutoencoder
# 載入 BPM 計算
import BPM_process as BPM

# 串口設置
DEFAULT_BAUD_RATE = 115200
SAMPLE_RATE = 64  # 64Hz 取樣頻率
MAX_DATA_POINTS = 300  # 顯示最近 300 筆數據（約 5 秒）

# 加載 PyTorch 模型
if getattr(sys, 'frozen', False):  # 是否在 .exe 中執行
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "best_model_hybrid_ssim.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DilatedDenoisingAutoencoder().to(device)
model = torch.load(model_path, map_location=device)

model.eval()  # 設定為推論模式


class SerialReader(QObject):
    data_received = pyqtSignal(str)
    numeric_data_received = pyqtSignal(float)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.running = False

    def start_reading(self):
        """開啟 UART 串口並開始讀取數據"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True
            threading.Thread(target=self.read_data, daemon=True).start()
        except serial.SerialException as e:
            self.data_received.emit(f"串口錯誤: {str(e)}")

    def read_data(self):
        """背景執行緒讀取 UART 數據"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    data = self.serial_conn.readline().decode("utf-8").strip()
                    self.data_received.emit(data)

                    try:
                        numeric_value = float(data)
                        self.numeric_data_received.emit(numeric_value)
                    except ValueError:
                        pass
            except Exception as e:
                self.data_received.emit(f"讀取錯誤: {str(e)}")

    def stop_reading(self):
        """停止讀取 UART 並關閉串口"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        self.data_received.emit("串口已關閉")


class SerialMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.serial_reader = None  # 延遲初始化 SerialReader
        self.selected_port = None

        # 存儲圖表數據
        self.data_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.time_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.processed_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.current_time = 0.0

        self.raw_data_window = deque(maxlen=512)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(1500)  # 1.5 秒更新一次

    def initUI(self):
        """建立 GUI 介面"""
        
        self.setWindowTitle("UART Raw Data Monitor with AI Inference")
        self.setGeometry(200, 200, 900, 600)

        self.layout = QVBoxLayout()

        # 串口選擇區域
        port_layout = QHBoxLayout()
        self.port_label = QLabel("選擇串口:")
        self.port_selector = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(self.port_label)
        port_layout.addWidget(self.port_selector)

        self.layout.addLayout(port_layout)

        # Raw Data 顯示框
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.layout.addWidget(self.text_display)

        # Matplotlib 圖表 (兩個子圖)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.ax1.set_ylim(0, 4095)
        self.ax2.set_ylim(0, 1)

        self.layout.addWidget(self.canvas)

        # BPM 顯示區域
        bpm_layout = QHBoxLayout()
        self.bpm_label_kal = QLabel("Heart Rate kalman: -- BPM")
        self.bpm_label_kal.setStyleSheet("font-size: 18px; color: red; font-weight: bold;")

        self.bpm_label_raw = QLabel("Heart Rate raw: -- BPM")
        self.bpm_label_raw.setStyleSheet("font-size: 18px; color: green; font-weight: bold;")

        bpm_layout.addWidget(self.bpm_label_kal)
        bpm_layout.addWidget(self.bpm_label_raw)
        self.layout.addLayout(bpm_layout)

        # 按鈕區域
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("開始接收")
        self.start_button.clicked.connect(self.start_reading)
        self.button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止接收")
        self.stop_button.clicked.connect(self.stop_reading)
        self.button_layout.addWidget(self.stop_button)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def refresh_ports(self):
        """獲取可用的 COM Port 並更新下拉選單"""
        self.port_selector.clear()
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        self.port_selector.addItems(available_ports)
        if available_ports:
            self.selected_port = available_ports[0]

    def start_reading(self):
        """開始串口讀取"""
        selected_port = self.port_selector.currentText()
        self.text_display.append(f"正在連接串口 {selected_port} ...")

        self.serial_reader = SerialReader(selected_port, DEFAULT_BAUD_RATE)
        self.serial_reader.data_received.connect(self.update_display)
        self.serial_reader.numeric_data_received.connect(self.update_plot)
        self.serial_reader.start_reading()

    def stop_reading(self):
        """停止串口讀取"""
        if self.serial_reader:
            self.serial_reader.stop_reading()

    def update_display(self, data):
        """更新 GUI 文字框內容"""
        self.text_display.append(data)

    def update_plot(self, value):
        """更新數據緩衝區，準備繪製圖表"""
        self.current_time += 1.0 / SAMPLE_RATE
        self.time_buffer.append(self.current_time)

        normalized_value = value / 4095.0
        self.data_buffer.append(value)

        self.raw_data_window.append(normalized_value)
        if len(self.raw_data_window) == 512:
            self.run_model_inference()

    def run_model_inference(self):
        """將 512 點的數據送入模型進行推論"""
        input_tensor = torch.tensor(list(self.raw_data_window), dtype=torch.float32).view(1, 1, 1, 512).to(device)
        with torch.no_grad():
            processed_value = model(input_tensor).cpu().numpy().flatten()
        self.processed_buffer.extend(processed_value)

    def refresh_plot(self):
        """刷新圖表"""
        if len(self.data_buffer) > 0:
            self.ax1.clear()
            self.ax1.plot(self.time_buffer, self.data_buffer, "r-")

            # 計算 Raw Data 的最小/最大值，並給 10% padding
            min_val = min(self.data_buffer)
            max_val = max(self.data_buffer)
            padding = (max_val - min_val) * 0.1  # 增加 10% 的範圍，防止數據貼邊
            self.ax1.set_ylim(min_val - padding, max_val + padding)

        if len(self.processed_buffer) > 0:  # 確保 processed_buffer 有數據
            valid_time = list(self.time_buffer)[-len(self.processed_buffer):]  # 時間對應處理後的數據
            self.ax2.clear()
            self.ax2.plot(valid_time, list(self.processed_buffer), "b-")

            # 計算模型輸出的最小/最大值，並給 10% padding
            min_val_proc = min(self.processed_buffer)
            max_val_proc = max(self.processed_buffer)
            padding_proc = (max_val_proc - min_val_proc) * 0.1  # 增加 10% 的範圍
            self.ax2.set_ylim(min_val_proc - padding_proc, max_val_proc + padding_proc)
            # self.ax2.set_ylim(0, 1)

            # 列印推論結果（最新 5 筆）
            # print("推論結果 (最新 5 筆):", list(self.processed_buffer)[-5:])

            heart_rate_kal, heart_rate_raw = BPM.process_heart_rate(self.processed_buffer)
            # print(f"Heart Rate (BPM): {heart_rate}")
            self.bpm_label_kal.setText(f"Heart Rate kalman: {heart_rate_kal} BPM")
            self.bpm_label_raw.setText(f"Heart Rate raw: {heart_rate_raw} BPM")

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialMonitor()
    window.show()
    sys.exit(app.exec())
