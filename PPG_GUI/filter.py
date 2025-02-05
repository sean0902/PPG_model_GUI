from scipy.signal import butter, filtfilt

def low_pass_filter(data, cutoff=10, fs=50, order=4):
    """
    應用低通濾波器處理數據。
    :param data: 原始數據 (1D numpy array)
    :param cutoff: 截止頻率 (Hz)
    :param fs: 取樣頻率 (Hz)
    :param order: 濾波器階數
    :return: 濾波後的數據
    """
    nyquist = 0.5 * fs  # 奈奎斯特頻率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data