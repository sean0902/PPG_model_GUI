# 綠光PPG_model_GUI

**功能**
1. raw data
2. 模型推論後的結果
3. 計算心率BMP

**接收資料方式**
1. 使用UART讀取COM資料

**規格**
1. sample rate = 64Hz
2. ADC 解析度 = 12bit
3. model size = 512
4. 1.5秒更新一次，每次保留5秒資料
5. 上圖是原始資料，下圖是辨識後資料
6. 心率計算有kalman與原始心率，分別以紅色與綠色表示
