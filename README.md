# 📊 Chantism Pro V2: 缠论智能量化交易系统

**Chantism Pro** 是一款基于 Python 的高性能量化分析工具，旨在将《缠中说禅》理论中的几何形态（分型、笔、线段、中枢）转化为严格的数学定义，并提供实时的可视化图表与策略回测。

本项目采用 **Streamlit** 作为交互前端，**Pyecharts** 进行交互式绘图，核心逻辑包含向量化回测引擎，相比传统逐行回测效率提升 100+ 倍。

---

## 🚀 核心功能 (Key Features)

1.  **严格的 K 线包含处理**
    * 遵循“趋势高高/低低”原则，递归处理 K 线包含关系，有效过滤市场噪音。
2.  **笔 (Bi) 的自动生成**
    * 基于顶底分型识别，严格执行“中间至少 1 根独立 K 线（总跨度≥5）”的标准定义。
    * 包含分型动态延伸逻辑，处理连续同向分型的情况。
3.  **动态中枢 (Pivot) 识别**
    * 自动识别由三笔重叠构成的中枢区间 $[ZD, ZG]$。
    * **中枢延伸**：支持中枢震荡的自动延伸逻辑，直到出现有效破坏。
4.  **智能信号捕捉**
    * **三类买点 (B3)**：自动检测突破中枢后的回踩确认信号。
    * **MACD 辅助**：集成 MACD 指标辅助判断背驰。
5.  **向量化回测 (Vectorized Backtest)**
    * 内置回测引擎，计算信号触发后的最大潜在收益与持有期回报，提供胜率统计。

---

## 🛠️ 环境依赖 (Prerequisites)

请确保您的系统安装了 **Python 3.8** 或更高版本。

项目主要依赖库：
* `streamlit` (Web 框架)
* `yfinance` (数据源)
* `pandas` & `numpy` (数据处理)
* `pandas_ta` (技术指标)
* `pyecharts` & `streamlit-echarts` (图表可视化)

---

## 📥 安装与运行 (Installation & Usage)

### 1. 准备项目文件
将核心代码保存为 `app.py`，并确保 `requirements.txt` 包含以下内容：
```text
streamlit
yfinance
pandas
numpy
pandas_ta
pyecharts
streamlit-echarts

