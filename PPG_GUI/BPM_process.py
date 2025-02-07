import numpy as np
from scipy import signal
import math
from scipy.signal import find_peaks
from scipy.signal import stft
from scipy.signal import lfilter
from scipy.signal import butter, filtfilt
def bandpass_filter(signal, lowcut=0.5, highcut=3.0, fs=64, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
 
class SciPyKalmanFilter:
    def __init__(self, process_variance=1, measurement_variance=10):
        """
        使用 Kalman 濾波平滑心率變化
        :param process_variance: 狀態轉移過程的變異數 (越小，濾波越平滑)
        :param measurement_variance: 測量噪聲的變異數 (越大，允許 HR 變動越大)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.kalman_gain = None
        self.estimate = 75

    def update(self, measured_hr):
        """
        進行 Kalman 濾波更新
        :param measured_hr: 當前計算出的 HR (BPM)
        :return: 平滑後的 HR
        """
        if self.estimate is None:
            self.estimate = measured_hr
            self.kalman_gain = 1
        else:
            prediction = self.estimate
            error = measured_hr - prediction
            self.kalman_gain = self.process_variance / (self.process_variance + self.measurement_variance)
            self.estimate += self.kalman_gain * error
        return self.estimate


def compute_heart_rate_stft(filtered_signal, fs=64, window_size=6, overlap=2, method="weighted_mean"):
    """
    使用 STFT (短時傅立葉變換) 計算心率 (Heart Rate)。

    :param filtered_signal: 經過預處理的 PPG 信號 (1D numpy array)
    :param fs: 取樣頻率 (Hz)，默認為 64 Hz
    :param window_size: STFT 視窗大小 (秒)
    :param overlap: STFT 視窗重疊時間 (秒)
    :param method: 選擇心率計算方法 ("max_peak" / "weighted_mean")
    :return: 計算出的心率 (BPM)
    """

    # 1. 設定 STFT 參數
    nperseg = window_size * fs  # 視窗大小 (樣本數)
    noverlap = overlap * fs  # 重疊樣本數

    # 2. 計算 STFT
    f, t, Zxx = stft(filtered_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=512)

    # 3. 取頻譜幅值
    magnitude = np.abs(Zxx)

    # 4. 限制頻率範圍 (0.5 Hz ~ 3 Hz) 對應於 30 BPM ~ 180 BPM
    valid_range = (f >= 0.5) & (f <= 3)
    valid_freqs = f[valid_range]
    valid_magnitude = magnitude[valid_range, :]

    # 5. 找出 STFT 最大功率的頻率分量
    if method == "max_peak":
        dominant_freqs = valid_freqs[np.argmax(valid_magnitude, axis=0)]  # 選擇最大峰值的頻率
    elif method == "weighted_mean":
        weighted_sum = np.sum(valid_freqs[:, None] * valid_magnitude, axis=0)
        total_power = np.sum(valid_magnitude, axis=0)
        dominant_freqs = weighted_sum / total_power  # 加權平均頻率
    else:
        raise ValueError("Invalid method! Use 'max_peak' or 'weighted_mean'.")

    # 6. 轉換為 BPM
    heart_rate_bpm = np.mean(dominant_freqs) * 60  # 取均值

    ## 7. 可視化 STFT 頻譜
    #plt.figure(figsize=(10, 6))
    #plt.pcolormesh(t, f * 60, np.abs(Zxx), shading='gouraud')  # 轉換 Hz -> BPM
    #plt.colorbar(label='Magnitude')
    #plt.scatter(t, dominant_freqs * 60, color='red', label="Detected HR", marker='o')
    #plt.xlabel("Time (seconds)")
    #plt.ylabel("Heart Rate (BPM)")
    #plt.title("STFT Heart Rate Estimation")
    #plt.legend()
    #plt.show()

    # print(f"Estimated Heart Rate (Before Kalman): {heart_rate_bpm:.2f} BPM")
    return heart_rate_bpm


kalman_filter = SciPyKalmanFilter()

def process_heart_rate(filtered_signal, fs=64, method="max_peak"):
    """
    經過 STFT 計算心率後，使用 Kalman 濾波器平滑心率。
    :param filtered_signal: 預處理後的 PPG 訊號
    :param fs: 取樣頻率
    :param method: "max_peak" 或 "weighted_mean" 計算心率
    """
    filtered_signals = bandpass_filter(filtered_signal, lowcut=0.5, highcut=3.0, fs=64)
    hr = compute_heart_rate_stft(filtered_signals, fs, method=method)
    
    if hr is not None:
        hr_kal = round(kalman_filter.update(hr),0)  # Kalman 濾波平滑
    # print(f"Final Smoothed HR (After Kalman): {hr:.2f} BPM")
    return hr_kal, hr
