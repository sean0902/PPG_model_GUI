import sys
import os
import serial.tools.list_ports
import serial
import threading
import numpy as np
import torch
import pandas as pd
import queue
from collections import deque
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QComboBox, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, QObject, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 載入 PPG 模型
from PPG_model import DilatedDenoisingAutoencoder
# 載入 BPM 計算
import BPM_process as BPM
# 載入 filter
import filter as lpf

# 串口設置
DEFAULT_BAUD_RATE = 115200
SAMPLE_RATE = 64
MAX_DATA_POINTS = 512

# 加載模型
if getattr(sys, 'frozen', False):  # 是否在 .exe 中執行
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "best_model_hybrid_ssim25_all_and_segment.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()


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
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True
            threading.Thread(target=self.read_data, daemon=True).start()
        except serial.SerialException as e:
            self.data_received.emit(f"串口錯誤: {str(e)}")

    def read_data(self):
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
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None


class SerialMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.serial_reader = None
        self.selected_port = None
        
        self.data_sample = deque(maxlen=MAX_DATA_POINTS)
        self.data_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.time_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.processed_buffer = deque(maxlen=MAX_DATA_POINTS)
        self.raw_data_window = deque(maxlen=MAX_DATA_POINTS)
        self.current_time = 0.0

        self.raw_data = []  # 用於儲存錄製的原始數據
        self.processed_data = [] # 用於儲存推論後的數據
        self.data_queue = queue.Queue()

        self.is_recording = False
        self.is_load_data = False
        self.file_path = None
        self.start_index = 0

        # 定期刷新串口
        threading.Thread(target=self.update_ports_thread, daemon=True).start()

        self.processing_thread = None
        self.processing_event = threading.Event()  # 用來控制執行緒狀態

        # threading.Thread(target=self.refresh_plot, daemon=True).start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(1000)  # 1s 刷新一次圖表

    def initUI(self):
        self.setWindowTitle("PPG GUI Monitor")
        self.setGeometry(200, 200, 1200, 600)  # 擴展寬度以適應右邊的 TextDisplay

        # 設置整體背景顏色和字體樣式
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F5F5;  /* 深白色背景 */
                color: black;              /* 預設文字顏色為黑色 */
                font-size: 18px;           /* 預設文字大小為 18px */
                font-weight: bold;         /* 預設文字粗體 */
                font-family: Microsoft YaHei;
            }
            QTextEdit {
                background-color: #FFFFFF; /* 純白背景的輸出訊息區域 */
                border: 1px solid #CCCCCC;
                font-weight: bold
                font-family: Microsoft YaHei;
            }
            QLabel {
                font-weight: bold
                font-family: Microsoft YaHei;        /* 字體設置為 Inter-Regular */
            }
        """)

        # 主水平佈局
        main_layout = QHBoxLayout()

        # 左側垂直佈局
        left_layout = QVBoxLayout()

        # 串口選擇
        port_layout = QHBoxLayout()
        self.port_label = QLabel("選擇串口:")
        self.port_selector = QComboBox()
        self.port_selector.setStyleSheet("""
            QComboBox {
                background-color: white;  /* 設定白色背景 */
                color: black;  /* 設定文字顏色為黑色 */
                font-size: 18px;  /* 字體大小 */
                font-weight: bold;  /* 字體加粗 */
                border: 2px solid white;  /* 邊框顏色 */
                border-radius: 10px;  /* 圓角 */
                padding: 5px;  /* 內距，增加可讀性 */
            }
            QComboBox:hover {
                background-color: #f0f0f0;  /* 滑鼠懸停時，背景變淺灰色 */
            }
        """)

        self.refresh_ports()
        port_layout.addWidget(self.port_label)
        port_layout.addWidget(self.port_selector)
        left_layout.addLayout(port_layout)

        # 功能按鈕
        button_layout = QHBoxLayout()

        self.start_record_button = QPushButton("開始錄製")
        self.start_record_button.clicked.connect(self.start_recording_and_reading)  # 錄製功能
        self.start_record_button.setStyleSheet("""
            QPushButton {
            background-color: #eafaf1;
            color: #229954;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid #229954;
            border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #00CC00;
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)
        button_layout.addWidget(self.start_record_button)

        self.stop_record_button = QPushButton("停止錄製")
        self.stop_record_button.clicked.connect(self.stop_recording_and_reading)  # 結合停止錄製與存儲功能
        self.stop_record_button.setStyleSheet("""
            QPushButton {
            background-color: #fce4e4;
            color: #c0392b;
            font-size: 18px;
            font-weight: bold;
            border: 2px solid #c0392b;
            border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #FF6666;
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)
        button_layout.addWidget(self.stop_record_button)

        self.save_data_button = QPushButton("儲存資料")
        self.save_data_button.clicked.connect(self.save_data)
        self.save_data_button.setStyleSheet("""
            QPushButton {
                background-color: #d6eaf8;  /* 淡藍色背景 */
                color: #3498db;  /* 深藍色文字 */
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #3498db;  /* 深藍色邊框 */
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #aed6f1;  /* 滑鼠懸停時的背景顏色 */
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)

        button_layout.addWidget(self.save_data_button)

        self.load_data_button = QPushButton("讀取資料")
        self.load_data_button.clicked.connect(self.load_data)
        self.load_data_button.setStyleSheet("""
            QPushButton {
                background-color: #fdebd0;  /* 淡橙色背景 */
                color: #e67e22;  /* 深橙色文字 */
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #e67e22;  /* 深橙色邊框 */
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #f5b041;  /* 滑鼠懸停時的背景顏色（較深橙色） */
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)
        button_layout.addWidget(self.load_data_button)

        self.delete_data_button = QPushButton("刪除資料")
        self.delete_data_button.clicked.connect(self.delete_data)
        self.delete_data_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: balck;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid balck;  /* 深藍色邊框 */
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: balck;  /* 滑鼠懸停時的背景顏色 */
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)
        button_layout.addWidget(self.delete_data_button)

        left_layout.addLayout(button_layout)
        
        # 向左/向右移動按鈕
        move_button_layout = QHBoxLayout()
        self.move_left_button = QPushButton("向左移動")
        self.move_left_button.clicked.connect(self.move_left)
        self.move_left_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: balck;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid balck;  /* 深藍色邊框 */
                border-radius: 10px;
                padding: 5px 10px; /* 按鈕內間距 */
            }
            QPushButton:hover {
                background-color: balck;  /* 滑鼠懸停時的背景顏色 */
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)

        move_button_layout.addWidget(self.move_left_button)

        self.move_right_button = QPushButton("向右移動")
        self.move_right_button.clicked.connect(self.move_right)
        self.move_right_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: balck;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid balck;
                border-radius: 10px;
                padding: 5px 10px; /* 按鈕內間距 */
            }
            QPushButton:hover {
                background-color: balck;  /* 滑鼠懸停時的背景顏色 */
                color: white;  /* 滑鼠懸停時文字變成白色 */
            }
        """)
        move_button_layout.addWidget(self.move_right_button)

        left_layout.addLayout(move_button_layout)  # 將移動按鈕加入左下部分

        # Matplotlib 圖表
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.ax1.set_ylim(0, 4095)
        self.ax2.set_ylim(0, 1)
        left_layout.addWidget(self.canvas)

        # BPM 顯示區域
        bpm_layout = QHBoxLayout()
        self.bpm_label_kal = QLabel("Heart Rate kalman: -- BPM")
        self.bpm_label_kal.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bpm_label_kal.setFixedHeight(40)  # 設定固定高度（可根據需求調整）
        self.bpm_label_kal.setStyleSheet("""
            QLabel {
                background-color: #eafaf1; /* 淡綠色背景 */
                color: #27ae60; /* 鮮綠色文字 */
                font-size: 20px; /* 字體大小 */
                font-weight: bold; /* 粗體 */
                border: 2px solid #229954; /* 深綠色邊框 */
                border-radius: 10px; /* 圓角邊框 */
                padding: 5px 10px; /* 內間距 */
            }
        """)

        self.bpm_label_raw = QLabel("Heart Rate raw: -- BPM")
        self.bpm_label_raw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bpm_label_raw.setFixedHeight(40)  # 設定固定高度
        self.bpm_label_raw.setStyleSheet("""
            QLabel {
                background-color: #fce4e4; /* 淡紅色背景 */
                color: #e74c3c; /* 鮮紅色文字 */
                font-size: 20px; /* 字體大小 */
                font-weight: bold; /* 粗體 */
                border: 2px solid #c0392b; /* 深紅色邊框 */
                border-radius: 10px; /* 圓角邊框 */
                padding: 5px 10px; /* 內間距 */
            }
        """)

        bpm_layout.addWidget(self.bpm_label_kal)
        bpm_layout.addWidget(self.bpm_label_raw)
        left_layout.addLayout(bpm_layout)  # 將 BPM 顯示區域加入左側底部

        # 右側佈局：Text Display
        right_layout = QVBoxLayout()

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        right_layout.addWidget(QLabel("輸出訊息:"))
        right_layout.addWidget(self.text_display)

        # 將左右佈局加入主佈局
        main_layout.addLayout(left_layout, stretch=3)  # 左側佔 3/4
        main_layout.addLayout(right_layout, stretch=1)  # 右側佔 1/4

        self.setLayout(main_layout)

    def refresh_ports(self):
        self.port_selector.clear()
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        self.port_selector.addItems(available_ports)
    
    def start_recording_and_reading(self):
        """開始錄製並讀取資料"""
        self.is_recording = True
        self.raw_data = []  # 清空錄製數據
        self.start_reading()  # 開始讀取資料

    def stop_recording_and_reading(self):
        """停止錄製並讀取資料"""
        self.is_recording = False
        self.stop_reading()  # 停止讀取資料
        self.save_data()

    def save_data(self):
        if len(list(self.raw_data)) >= MAX_DATA_POINTS:
            self.start_index = len(self.raw_data) - MAX_DATA_POINTS
            print(f"raw_data:{len(self.raw_data)}")
            print(f"processed_data:{len(self.processed_data)}")
        else:
            self.start_index = 0

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        
        default_filename  = f"ppg_data_{timestamp}.csv"

        reply = QMessageBox.question(
            self, "儲存數據", "是否要儲存錄製的數據？", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            file_path, _ = QFileDialog.getSaveFileName(self, "保存 PPG 數據", default_filename, "CSV Files (*.csv)")
            if file_path:
                # 確保 raw_data 和 processed_data 長度一致
                min_length = min(len(self.raw_data), len(self.processed_data))
                raw_trimmed = self.raw_data[:min_length]
                processed_trimmed = self.processed_data[:min_length]

                # 合併數據並存檔
                df = pd.DataFrame({"Raw Data": raw_trimmed, "Processed Data": processed_trimmed})
                df.to_csv(file_path, index=False)

                self.text_display.append(f"數據已保存到: {file_path}")

    def delete_data(self):
        """刪除數據並重置所有緩衝區，確保圖表清空"""
        reply = QMessageBox.question(
            self, "刪除資料", "是否要刪除所有數據？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes: 
            # 清空 Matplotlib 圖表
            self.ax1.clear()
            self.ax1.set_ylabel("Raw Data")
            self.ax1.set_xlim(0, MAX_DATA_POINTS)
            self.ax1.set_ylim(0, 4095)

            self.ax2.clear()
            self.ax2.set_ylabel("Processed Data")
            self.ax2.set_xlim(0, MAX_DATA_POINTS)
            self.ax2.set_ylim(0, 1)

            # 清空數據但保留大小
            self.raw_data = []
            self.processed_data = []
            self.data_buffer.clear()
            self.processed_buffer.clear()
            self.raw_data_window.clear()
            self.time_buffer.clear()

            # 重置時間
            self.current_time = 0.0

            # 刷新畫布
            self.canvas.draw()

            # 清空文本顯示區
            self.text_display.clear()
            self.text_display.append("所有數據已被刪除！")

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

    def update_ports_thread(self):
        """定期檢查並更新可用的串口"""
        previous_ports = set()
        while True:
            current_ports = {port.device for port in serial.tools.list_ports.comports()}
            if current_ports != previous_ports:  # 如果串口列表有變化
                previous_ports = current_ports
                self.update_ports_list(list(current_ports))  # 將 set 轉為 list

    def update_ports_list(self, available_ports):
        """更新串口選單"""
        self.port_selector.clear()
        self.port_selector.addItems(available_ports)
        if available_ports:  # 檢查串口列表是否不為空
            self.selected_port = available_ports[0]

    def load_data(self):
        """讀取 CSV 檔案並拆分成 raw_data 和 processed_data"""
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 PPG 數據檔案", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                # 讀取 CSV 檔案
                df = pd.read_csv(file_path)

                # 確保 CSV 內有 "Raw Data" 和 "Processed Data" 這兩欄
                if "Raw Data" in df.columns and "Processed Data" in df.columns:
                    self.raw_data = df["Raw Data"].tolist()
                    self.processed_data = df["Processed Data"].tolist()
                    self.is_load_data = True
                    self.start_index = 0  # 重置索引
                    self.plot_data()

                    self.text_display.append(f"成功載入數據:\n  - 檔案: {file_path}")
                else:
                    QMessageBox.warning(self, "讀取失敗", "檔案格式錯誤，缺少必要數據欄位！")

            except Exception as e:
                QMessageBox.warning(self, "讀取錯誤", f"讀取 CSV 失敗: {str(e)}")


    def move_left(self):
        if self.start_index > 0:
            self.start_index = max(0, self.start_index - SAMPLE_RATE)
            self.plot_data()

    def move_right(self):
        max_length = len(self.raw_data)
        if self.start_index + MAX_DATA_POINTS < max_length:
            self.start_index = min(max_length - MAX_DATA_POINTS, self.start_index + SAMPLE_RATE)
            self.plot_data()

    def plot_data(self):
        self.ax1.clear()
        self.ax2.clear()

        end_index = self.start_index + MAX_DATA_POINTS
        time_x = np.linspace(self.start_index / SAMPLE_RATE, end_index / SAMPLE_RATE, MAX_DATA_POINTS)

        if len(self.raw_data) >= end_index:
            data_to_plot = self.raw_data[self.start_index:end_index]
            processed_to_plot = self.processed_data[self.start_index:end_index]
        else:
            data_to_plot = self.raw_data[self.start_index:]
            processed_to_plot = self.processed_data[self.start_index:]

        self.ax1.plot(time_x, data_to_plot, "r-")
        self.ax1.set_ylabel("Raw Data")

        self.ax2.plot(time_x, processed_to_plot, "b-")
        self.ax2.set_ylabel("Processed Data")

        self.canvas.draw()

    def update_plot(self, value):
        """更新數據緩衝區並將推論數據加入隊列"""
        if self.is_recording:
            if len(self.data_sample) < SAMPLE_RATE:
                self.data_sample.append(value)
            else:
                self.data_queue.put(self.data_sample)
                self.data_processed()
                self.data_sample.clear()

    def data_processed(self):
        """ 啟動獨立執行緒處理數據 """
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_event.set()  # 啟動處理事件
            self.processing_thread = threading.Thread(target=self._process_data, daemon=True)
            self.processing_thread.start()

    def _process_data(self):
        """ 真正執行數據處理的函數，當 `Queue` 清空後關閉執行緒 """
        while self.processing_event.is_set():
            batch = []    
            while  not self.data_queue.empty() and len(batch) < SAMPLE_RATE:
                batch.extend(self.data_queue.get())
                self.data_queue.task_done()

                # 轉換成正規化數據
                batch = list(batch)
                normalized_values = [float(v) / 4095.00 for v in batch]

                # 存入數據緩衝區
                self.raw_data_window.extend(normalized_values)
                self.data_buffer.extend(batch)
                self.raw_data.extend(batch)

                # 更新時間數據
                new_times = [self.current_time + (i / SAMPLE_RATE) for i in range(len(batch))]
                self.time_buffer.extend(new_times)
                self.current_time = self.time_buffer[-1]  # 更新到最新時間

                if len(self.raw_data_window) == MAX_DATA_POINTS:
                    self.run_model_inference()
            else:
                # 當 Queue 空了，關閉處理執行緒
                self.processing_event.clear()

    def run_model_inference(self):
        """將 512 點的數據送入模型進行推論"""
        input_tensor = torch.tensor(list(self.raw_data_window), dtype=torch.float32).view(1, 1, 1, 512).to(device)
        with torch.no_grad():
            processed_value = model(input_tensor).cpu().numpy().flatten()
        value_array = np.array(processed_value)

        value_filter = lpf.low_pass_filter(value_array)

        buffer = deque(value_filter, maxlen=MAX_DATA_POINTS)

        self.processed_buffer.extend(buffer)

        if len(self.processed_data) == 0:
            # 第一次推論，儲存 512 筆資料
            self.processed_data.extend(list(value_filter))
            self.text_display.append("第一次推論：儲存所有 512 筆資料")
        else:
            # 非第一次推論，儲存最後 64 筆資料
            self.processed_data.extend(value_filter[-64:])

    def refresh_plot(self):
        """刷新圖表"""
        if self.is_recording:
            if len(self.data_buffer) > 0:
                valid_time = list(self.time_buffer)[-len(self.data_buffer):]
                self.ax1.clear()
                self.ax1.plot(valid_time, self.data_buffer, "r-")
               
                # 計算 Raw Data 的最小/最大值，並給 10% padding
                min_val = min(self.data_buffer)
                max_val = max(self.data_buffer)
                padding = (max_val - min_val) * 0.1  # 增加 10% 的範圍，防止數據貼邊
                self.ax1.set_ylim(min_val - padding, max_val + padding)
                self.ax1.set_ylabel("Raw Data")
                
                self.ax1.relim()
                self.ax1.autoscale_view()

            if len(self.processed_buffer) > 0:  # 確保 processed_buffer 有數據
                valid_time = list(self.time_buffer)[-len(self.processed_buffer):]  # 時間對應處理後的數據
                self.ax2.clear()
                self.ax2.plot(valid_time, list(self.processed_buffer), "b-")
                
                # 計算模型輸出的最小/最大值，並給 10% padding
                min_val_proc = min(self.processed_buffer)
                max_val_proc = max(self.processed_buffer)
                padding_proc = (max_val_proc - min_val_proc) * 0.1  # 增加 10% 的範圍
                self.ax2.set_ylim(min_val_proc - padding_proc, max_val_proc + padding_proc)
                self.ax2.set_ylabel("Processed Data") 
                     
                #計算心率
                heart_rate_kal, heart_rate_raw = BPM.process_heart_rate(self.processed_buffer)
                self.bpm_label_kal.setText(f"Heart Rate kalman: {heart_rate_kal} BPM")
                self.bpm_label_raw.setText(f"Heart Rate raw: {round(heart_rate_raw, 1)} BPM")

            self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialMonitor()
    window.show()
    sys.exit(app.exec())
