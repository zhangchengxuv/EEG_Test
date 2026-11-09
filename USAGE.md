# EEG_Test 使用教程

本文说明：**MNE 与时间信息**、**Python 依赖安装**、**FIFO_plot 与 cue_paradigm 联机采集/打标**，以及 **FIF 读取与 TRIG 解码**。

---

## 1. MNE 是否需要「时间戳」？要不要去掉？

**结论：MNE 的 `Raw` 对象里，数据矩阵本身不需要、也不应该再带一列「逐采样时间戳」。**

- **时间轴如何定义**：`Raw` 中为形状 `(n_channels, n_times)` 的连续采样，配合 **`sfreq`（采样率）**，第 `k` 个采样点对应时间  
  `t_k = k / sfreq`（相对本段录制的起点）。这是标准用法。
- **录制开始时刻**：FIF 可包含 **`meas_date`** 元数据，用于与外部事件对齐；这是**文件头信息**，不是每个采样点一列时间。
- **本项目的 9 通道**：前 8 路为 EEG（幅值），第 9 路为软件打标 **TRIG**。请勿把「日期时间字符串」当作 EEG 通道写入；若从旧版 CSV 转换，只保留 8 路电压 + 1 路打标对齐即可。
- **与 Cue 对齐**：优先用 **第 9 通道 TRIG 码**（见 `trigger_protocol.py`），必要时结合 **`Plot/logs/cue_markers_*.csv`** 的 wall-clock 时间。

因此：**不必为了 MNE「去除时间戳」**——规范做法是矩阵仅存通道幅值，用 `sfreq` 推时间；需避免的是**误把时间文本当脑电数据列**。

---

## 2. Python 环境与依赖安装

### 2.1 建议环境

- Python 3.9+（64 位推荐）
- 仅在 **PC 端运行 Python 脚本**需要下列包；固件（PlatformIO）与 pip 无关。

### 2.2 一键安装

在项目根目录 `EEG_Test` 下：

```bash
cd c:\Users\ROG\Documents\PlatformIO\Projects\EEG_Test
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2.3 `requirements.txt` 包含的包

| 包名 | 用途 |
|------|------|
| mne | 读写 FIF、分析 |
| numpy | 数值数组 |
| PyQt5 | GUI（FIFO_plot、cue_paradigm） |
| pyqtgraph | FIFO_plot 实时曲线 |
| pyserial | 串口 |
| scipy | 滤波等 |
| PyQt-Fluent-Widgets | FIFO_plot 的 Fluent 控件 |

`mne` 会拉取自身依赖（如 matplotlib、pooch 等），一般无需再单独指定。

---

## 3. 硬件与采样率

- 固件 `src/main.cpp` 中 ADS1299 配置为 **500 SPS**。
- `Plot/FIFO_plot.py` 中 **`SAMPLE_RATE = 500`** 须与固件一致。
- 串口波特率等与固件一致即可（见工程内代码注释）。

---

## 4. 推荐启动顺序（在线 UDP 打标）

UDP 打标端口由 **FIFO_plot 监听**（默认见 `trigger_protocol.py`：`127.0.0.1:45454` 为发送目标；采集端绑定 `0.0.0.0:端口`）。

1. **先**启动采集端（保证端口已监听）：
   ```bash
   cd c:\Users\ROG\Documents\PlatformIO\Projects\EEG_Test
   .venv\Scripts\activate
   python Plot\FIFO_plot.py
   ```
2. 选择正确 **COM 口**，确认有数据；界面 **「打标: n」** 会随 UDP 更新（未运行 cue 时多为 `0`）。
3. **再**启动范式端：
   ```bash
   python UI_cue\cue_paradigm.py
   ```
4. 在 cue 中确认 **主机 / 端口** 与 `trigger_protocol.py` 一致，并勾选 **「向采集端发送打标码」**。
5. 点击 **开始范式**，与 FIFO_plot 同时录制即可。

**若先开 cue 后开 FIFO_plot**：在 FIFO_plot 启动前发出的 UDP 包无法被接收，打标可能丢失；**务必先开 FIFO_plot**。

---

## 5. 范式与左右随机

- **每次试次独立随机左/右**：保持 cue 中 **「左右试次数量相等」不勾选**（默认）。
- 若需要 **左右试次数相同**：勾选该项（需 **偶数** 试次，程序可能自动调整试次数）。

---

## 6. 输出文件说明

| 内容 | 典型路径 |
|------|----------|
| 连续脑电 + 第 9 路打标（FIF） | `Plot/logs/EEG_*-raw.fif` |
| Cue 事件 CSV（可选，用于对齐核对） | `Plot/logs/cue_markers_*.csv` |
| 打标码定义（与代码共用） | 项目根目录 `trigger_protocol.py` |

打标码含义以 **`trigger_protocol.py`** 为准（如：`0` 关、`1` 左想象、`2` 右想象、`10` 休息、`11` 准备等）。

---

## 7. 用 MNE 读取 FIF 并解码 TRIG

第 9 通道在文件中按 **伏特** 存储：`value_V = code * TRIG_VOLT_SCALE`（`TRIG_VOLT_SCALE` 见 `trigger_protocol.py`）。

```python
import numpy as np
import mne
from trigger_protocol import TRIG_VOLT_SCALE, TRIG_MI_LEFT, TRIG_MI_RIGHT

raw = mne.io.read_raw_fif("Plot/logs/你的文件名-raw.fif", preload=True)
trig = np.rint(raw.get_data(picks="TRIG")[0] / TRIG_VOLT_SCALE).astype(int)
eeg = raw.copy().pick("eeg")  # 仅前 8 路电极，用于特征

# 示例：想象期「左」「右」的采样掩码
mask_left = trig == TRIG_MI_LEFT
mask_right = trig == TRIG_MI_RIGHT
```

分类流程中通常：**前 8 通道**做特征，**TRIG** 作标签或切段依据。

---

## 8. 常见问题

| 现象 | 建议 |
|------|------|
| 打标始终为 0 | 先启动 FIFO_plot；检查防火墙；cue 中 IP/端口与协议一致且已勾选发送。 |
| FIF 无法读取 | 确认文件为当前脚本保存的 `*-raw.fif`；升级 `mne`。 |
| 通道数不对 | 应为 **9**（8×EEG + TRIG）；分析时 EEG 用前 8 路，TRIG 仅标签。 |
| 路径 `import trigger_protocol` 失败 | 在 **`EEG_Test` 根目录**下运行 `python Plot\FIFO_plot.py`，勿随意改工作目录。 |

---

## 9. 相关路径速查

- 采集与录制：`Plot/FIFO_plot.py`
- 范式与 UDP 发送：`UI_cue/cue_paradigm.py`
- 打标协议（端口、码表、缩放系数）：`trigger_protocol.py`
- 依赖列表：`requirements.txt`
