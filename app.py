import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# --- 配置 ---
st.set_page_config(
    layout="wide", 
    page_title="Chantism Pro V5", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- CSS样式 ---
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    .segment-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .segment-up {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 4px solid #00ff00;
    }
    .segment-down {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Chantism Pro V5: 智能缠论分析系统")

# --- 参数设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 数据设置
    st.subheader("📊 数据设置")
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("交易对", "BTC-USD", 
                             help="股票/加密货币代码，如：AAPL, ETH-USD")
    with col2:
        interval = st.selectbox(
            "时间周期",
            ["15m", "30m", "1h", "2h", "4h", "1d", "1wk"],
            index=2,
            help="K线周期"
        )
    
    period = st.selectbox(
        "时间范围",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=1,
        help="数据时间范围"
    )
    
    st.markdown("---")
    
    # 缠论参数
    st.subheader("🎯 缠论参数")
    
    bi_min_k = st.slider(
        "笔最少K线数", 
        min_value=3, max_value=10, value=5,
        help="构成笔所需的最少K线数量"
    )
    
    segment_min_bi = st.slider(
        "线段最少笔数", 
        min_value=3, max_value=9, value=5,
        help="构成线段所需的最少笔数"
    )
    
    pivot_min_bi = st.slider(
        "中枢最少笔数", 
        min_value=3, max_value=6, value=3,
        help="构成中枢所需的最少笔数"
    )
    
    # 高级参数
    with st.expander("高级参数", expanded=False):
        strict_mode = st.checkbox("严格模式", True,
                                 help="启用更严格的缠论规则")
        include_gap = st.checkbox("包含缺口处理", True,
                                 help="处理特征序列中的缺口")
        macd_fast = st.slider("MACD快线", 8, 20, 12)
        macd_slow = st.slider("MACD慢线", 20, 40, 26)
        macd_signal = st.slider("MACD信号线", 5, 15, 9)
    
    st.markdown("---")
    
    # 显示设置
    st.subheader("📈 显示设置")
    show_volume = st.checkbox("显示成交量", True)
    show_macd = st.checkbox("显示MACD", True)
    show_fractals = st.checkbox("显示分型", False)
    show_bi_detail = st.checkbox("显示笔详情", True)
    
    st.markdown("---")
    
    # 分析按钮
    if st.button("🚀 开始智能分析", type="primary", use_container_width=True):
        run_analysis = True
    else:
        run_analysis = False

# --- 数据结构优化 ---
@dataclass
class Fractal:
    """优化分型数据结构"""
    idx: int
    type: str  # 'top' or 'bottom'
    price: float
    time: pd.Timestamp
    confirmed: bool = True
    k_idx: int = 0
    strength: float = 0.0  # 分型强度
    left_bars: int = 0      # 左侧K线数
    right_bars: int = 0     # 右侧K线数
    
    def __post_init__(self):
        self.price = round(self.price, 4)

@dataclass 
class Bi:
    """优化笔数据结构"""
    index: int
    type: str  # 'up' or 'down'
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    high: float
    low: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    strength: float = 0.0
    length_bars: int = 0
    price_change: float = 0.0
    time_span: float = 0.0
    parent_segment: int = -1
    
    def __post_init__(self):
        self.price_change = abs(self.end_price - self.start_price)
        self.time_span = (self.end_time - self.start_time).total_seconds() / 3600
        if self.time_span > 0:
            self.strength = self.price_change / self.time_span
        self.start_price = round(self.start_price, 4)
        self.end_price = round(self.end_price, 4)
        self.high = round(self.high, 4)
        self.low = round(self.low, 4)

@dataclass
class Segment:
    """线段（优化版）"""
    index: int
    type: str  # 'up' or 'down'
    start_bi_idx: int
    end_bi_idx: int
    start_price: float
    end_price: float
    high: float
    low: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    bi_list: List[int] = field(default_factory=list)
    level: int = 1
    length: float = 0.0
    duration: float = 0.0
    is_completed: bool = True
    has_break: bool = False
    feature_sequence: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.length = abs(self.end_price - self.start_price)
        self.duration = (self.end_time - self.start_time).total_seconds() / 86400  # 天数

@dataclass 
class Pivot:
    """中枢（优化版）"""
    index: int
    level: int = 1
    start_idx: int = 0
    end_idx: int = 0
    zg: float = 0.0
    zd: float = 0.0
    gg: float = 0.0
    dd: float = 0.0
    start_time: pd.Timestamp = None
    end_time: pd.Timestamp = None
    segment_idx: int = -1
    bi_indices: List[int] = field(default_factory=list)
    duration: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    def __post_init__(self):
        self.zg = round(self.zg, 4)
        self.zd = round(self.zd, 4)
        self.gg = round(self.gg, 4)
        self.dd = round(self.dd, 4)
        self.width = self.zg - self.zd
        self.height = self.gg - self.dd
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds() / 86400

# --- 核心算法优化 ---
class EnhancedChantismEngine:
    """增强缠论分析引擎"""
    
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self._prepare_data()
        self.fractals: List[Fractal] = []
        self.bi_list: List[Bi] = []
        self.segments: List[Segment] = []
        self.pivots: List[Pivot] = []
        self.signals: List[Dict] = []
        self.processed_k: List[Dict] = []
        
    def _prepare_data(self):
        """数据预处理"""
        # 确保索引是datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        
        # 计算技术指标
        self.df['MA5'] = self.df['Close'].rolling(5).mean()
        self.df['MA20'] = self.df['Close'].rolling(20).mean()
        self.df['MA60'] = self.df['Close'].rolling(60).mean()
        
        # 计算MACD
        macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            self.df = pd.concat([self.df, macd], axis=1)
        
        # 计算RSI
        self.df['RSI'] = ta.rsi(self.df['Close'], length=14)
        
        # 计算ATR
        self.df['ATR'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        
        # 清理NaN值
        self.df = self.df.dropna()
    
    def process_k_lines_optimized(self):
        """优化K线包含处理（向量化+缓存）"""
        if self.df.empty:
            self.processed_k = []
            return []
        
        # 使用向量化操作提高性能
        highs = self.df['High'].values
        lows = self.df['Low'].values
        times = self.df.index
        
        n = len(highs)
        if n < 3:
            self.processed_k = []
            return []
        
        # 初始化
        processed = [{
            'idx': 0,
            'time': times[0],
            'high': highs[0],
            'low': lows[0],
            'open': self.df.iloc[0]['Open'],
            'close': self.df.iloc[0]['Close']
        }]
        
        direction = None
        i = 1
        
        while i < n:
            current_high = highs[i]
            current_low = lows[i]
            prev = processed[-1]
            
            # 判断包含关系
            is_contained = (current_high <= prev['high'] and current_low >= prev['low']) or \
                          (current_high >= prev['high'] and current_low <= prev['low'])
            
            if is_contained:
                # 确定包含处理方向
                if direction is None:
                    if len(processed) > 1:
                        # 根据前两根非包含K线判断
                        if processed[-1]['high'] > processed[-2]['high']:
                            direction = 'up'
                        else:
                            direction = 'down'
                    else:
                        direction = 'up' if current_high >= prev['high'] else 'down'
                
                # 合并处理
                if direction == 'up':
                    new_high = max(prev['high'], current_high)
                    new_low = max(prev['low'], current_low)
                else:
                    new_high = min(prev['high'], current_high)
                    new_low = min(prev['low'], current_low)
                
                processed[-1].update({
                    'high': new_high,
                    'low': new_low,
                    'time': times[i]
                })
            else:
                processed.append({
                    'idx': i,
                    'time': times[i],
                    'high': current_high,
                    'low': current_low,
                    'open': self.df.iloc[i]['Open'],
                    'close': self.df.iloc[i]['Close']
                })
                direction = None
            
            i += 1
        
        self.processed_k = processed
        return processed
    
    def find_fractals_optimized(self, confirm_bars=3, strength_threshold=0.5):
        """优化分型识别"""
        if not self.processed_k:
            return []
        
        n = len(self.processed_k)
        fractals = []
        
        # 预计算高点和低点数组
        highs = np.array([k['high'] for k in self.processed_k])
        lows = np.array([k['low'] for k in self.processed_k])
        
        # 识别顶分型
        for i in range(2, n-2):
            # 顶分型条件
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                
                # 计算分型强度
                left_min = min(lows[i-2:i+1])
                right_min = min(lows[i:i+3])
                strength = (highs[i] - max(left_min, right_min)) / highs[i] if highs[i] > 0 else 0
                
                if strength >= strength_threshold:
                    fractal = Fractal(
                        idx=len(fractals),
                        type='top',
                        price=highs[i],
                        time=self.processed_k[i]['time'],
                        strength=strength,
                        k_idx=i,
                        left_bars=2,
                        right_bars=2
                    )
                    fractals.append(fractal)
        
        # 识别底分型
        for i in range(2, n-2):
            # 底分型条件
            if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                
                # 计算分型强度
                left_max = max(highs[i-2:i+1])
                right_max = max(highs[i:i+3])
                strength = (min(left_max, right_max) - lows[i]) / lows[i] if lows[i] > 0 else 0
                
                if strength >= strength_threshold:
                    fractal = Fractal(
                        idx=len(fractals),
                        type='bottom',
                        price=lows[i],
                        time=self.processed_k[i]['time'],
                        strength=strength,
                        k_idx=i,
                        left_bars=2,
                        right_bars=2
                    )
                    fractals.append(fractal)
        
        # 按时间排序
        fractals.sort(key=lambda x: x.time)
        
        # 过滤重复分型
        filtered = []
        for i in range(len(fractals)):
            if i == 0:
                filtered.append(fractals[i])
                continue
            
            prev = filtered[-1]
            curr = fractals[i]
            
            # 检查是否同类型且接近
            if prev.type == curr.type and abs(curr.k_idx - prev.k_idx) < 5:
                # 取更强的分型
                if prev.type == 'top' and curr.price > prev.price:
                    filtered[-1] = curr
                elif prev.type == 'bottom' and curr.price < prev.price:
                    filtered[-1] = curr
            else:
                filtered.append(curr)
        
        self.fractals = filtered
        return filtered
    
    def find_bi_complete(self, min_k=5):
        """完整笔识别算法"""
        if len(self.fractals) < 2:
            return []
        
        # 按时间排序
        sorted_fractals = sorted(self.fractals, key=lambda x: x.time)
        bi_list = []
        
        i = 0
        while i < len(sorted_fractals) - 1:
            start_fractal = sorted_fractals[i]
            
            # 寻找配对的结束分型
            j = i + 1
            found_end = None
            
            while j < len(sorted_fractals):
                end_fractal = sorted_fractals[j]
                
                # 分型必须交替
                if start_fractal.type == end_fractal.type:
                    j += 1
                    continue
                
                # 检查K线数量
                if abs(end_fractal.k_idx - start_fractal.k_idx) < min_k:
                    j += 1
                    continue
                
                # 检查笔的合理性
                if start_fractal.type == 'bottom' and end_fractal.type == 'top':
                    # 向上笔：终点必须高于起点
                    if end_fractal.price > start_fractal.price:
                        found_end = end_fractal
                        break
                else:
                    # 向下笔：终点必须低于起点
                    if end_fractal.price < start_fractal.price:
                        found_end = end_fractal
                        break
                
                j += 1
            
            if found_end:
                # 创建笔
                if start_fractal.type == 'bottom':
                    bi_type = 'up'
                    start_price = start_fractal.price
                    end_price = found_end.price
                    high = found_end.price
                    low = start_fractal.price
                else:
                    bi_type = 'down'
                    start_price = start_fractal.price
                    end_price = found_end.price
                    high = start_fractal.price
                    low = found_end.price
                
                bi = Bi(
                    index=len(bi_list),
                    type=bi_type,
                    start_idx=start_fractal.k_idx,
                    end_idx=found_end.k_idx,
                    start_price=start_price,
                    end_price=end_price,
                    high=high,
                    low=low,
                    start_time=start_fractal.time,
                    end_time=found_end.time,
                    length_bars=abs(found_end.k_idx - start_fractal.k_idx)
                )
                bi_list.append(bi)
                i = j + 1
            else:
                i += 1
        
        self.bi_list = bi_list
        return bi_list
    
    def find_segments_strict(self, min_bi=5):
        """严格线段识别算法（基于特征序列）"""
        if len(self.bi_list) < min_bi:
            return []
        
        segments = []
        current_start = 0
        current_direction = self.bi_list[0].type
        feature_sequence = []
        
        for i in range(len(self.bi_list)):
            bi = self.bi_list[i]
            
            # 构建特征序列
            if current_direction == 'up' and bi.type == 'down':
                feature_sequence.append(bi)
            elif current_direction == 'down' and bi.type == 'up':
                feature_sequence.append(bi)
            
            # 检查是否满足线段结束条件
            if len(feature_sequence) >= 3:
                # 处理特征序列包含关系
                processed_features = self._process_feature_sequence(feature_sequence)
                
                # 检查特征序列分型
                if self._check_feature_sequence_fractal(processed_features, current_direction):
                    # 线段结束
                    segment_bis = self.bi_list[current_start:i+1]
                    
                    if len(segment_bis) >= min_bi:
                        segment = self._create_segment(segment_bis, len(segments), current_direction)
                        segment.feature_sequence = processed_features
                        segments.append(segment)
                        
                        # 开始新线段
                        current_start = i + 1
                        if current_start < len(self.bi_list):
                            current_direction = self.bi_list[current_start].type
                        feature_sequence = []
        
        # 处理最后一段
        if current_start < len(self.bi_list):
            segment_bis = self.bi_list[current_start:]
            if len(segment_bis) >= min_bi:
                segment = self._create_segment(segment_bis, len(segments), current_direction)
                segments.append(segment)
        
        # 标记笔的父线段
        for seg_idx, segment in enumerate(segments):
            for bi_idx in segment.bi_list:
                if bi_idx < len(self.bi_list):
                    self.bi_list[bi_idx].parent_segment = seg_idx
        
        self.segments = segments
        return segments
    
    def _process_feature_sequence(self, bi_list: List[Bi]) -> List[Dict]:
        """处理特征序列包含关系"""
        if len(bi_list) < 2:
            return []
        
        processed = []
        direction = None
        
        for bi in bi_list:
            if not processed:
                processed.append({
                    'high': bi.high,
                    'low': bi.low,
                    'start': bi.start_price,
                    'end': bi.end_price,
                    'time': bi.end_time,
                    'type': bi.type
                })
                continue
            
            prev = processed[-1]
            
            # 判断包含关系
            if (bi.high <= prev['high'] and bi.low >= prev['low']) or \
               (bi.high >= prev['high'] and bi.low <= prev['low']):
                
                # 确定包含处理方向
                if direction is None:
                    direction = 'up' if bi.high >= prev['high'] else 'down'
                
                if direction == 'up':
                    new_high = max(prev['high'], bi.high)
                    new_low = max(prev['low'], bi.low)
                else:
                    new_high = min(prev['high'], bi.high)
                    new_low = min(prev['low'], bi.low)
                
                processed[-1] = {
                    'high': new_high,
                    'low': new_low,
                    'start': min(prev['start'], bi.start_price),
                    'end': bi.end_price,
                    'time': bi.end_time,
                    'type': bi.type
                }
            else:
                processed.append({
                    'high': bi.high,
                    'low': bi.low,
                    'start': bi.start_price,
                    'end': bi.end_price,
                    'time': bi.end_time,
                    'type': bi.type
                })
                direction = None
        
        return processed
    
    def _check_feature_sequence_fractal(self, features: List[Dict], segment_type: str) -> bool:
        """检查特征序列是否形成分型"""
        if len(features) < 3:
            return False
        
        n = len(features)
        
        if segment_type == 'up':
            # 向上线段，寻找特征序列的顶分型
            for i in range(1, n-1):
                if (features[i]['high'] > features[i-1]['high'] and 
                    features[i]['high'] > features[i+1]['high']):
                    return True
        else:
            # 向下线段，寻找特征序列的底分型
            for i in range(1, n-1):
                if (features[i]['low'] < features[i-1]['low'] and 
                    features[i]['low'] < features[i+1]['low']):
                    return True
        
        return False
    
    def _create_segment(self, bi_list: List[Bi], index: int, direction: str) -> Segment:
        """创建线段对象"""
        highs = [bi.high for bi in bi_list]
        lows = [bi.low for bi in bi_list]
        
        segment = Segment(
            index=index,
            type=direction,
            start_bi_idx=bi_list[0].index,
            end_bi_idx=bi_list[-1].index,
            start_price=bi_list[0].start_price,
            end_price=bi_list[-1].end_price,
            high=max(highs),
            low=min(lows),
            start_time=bi_list[0].start_time,
            end_time=bi_list[-1].end_time,
            bi_list=[bi.index for bi in bi_list]
        )
        return segment
    
    def find_pivots_advanced(self):
        """高级中枢识别算法"""
        if not self.segments:
            return []
        
        pivots = []
        
        for seg_idx, segment in enumerate(self.segments):
            segment_bis = [self.bi_list[i] for i in segment.bi_list]
            
            if len(segment_bis) < 3:
                continue
            
            i = 0
            while i <= len(segment_bis) - 3:
                # 尝试找到三笔重叠
                bi1 = segment_bis[i]
                bi2 = segment_bis[i+1]
                bi3 = segment_bis[i+2]
                
                # 检查重叠区间
                zg = min(bi1.high, bi2.high, bi3.high)
                zd = max(bi1.low, bi2.low, bi3.low)
                
                if zg > zd:  # 存在重叠
                    gg = max(bi1.high, bi2.high, bi3.high)
                    dd = min(bi1.low, bi2.low, bi3.low)
                    pivot_bis = [i, i+1, i+2]
                    
                    # 尝试延伸中枢
                    end_idx = i + 2
                    for j in range(i+3, len(segment_bis)):
                        next_bi = segment_bis[j]
                        
                        # 检查是否与中枢重叠
                        overlap_high = min(zg, next_bi.high)
                        overlap_low = max(zd, next_bi.low)
                        
                        if overlap_high > overlap_low:  # 仍有重叠
                            zg = overlap_high
                            zd = overlap_low
                            gg = max(gg, next_bi.high)
                            dd = min(dd, next_bi.low)
                            pivot_bis.append(j)
                            end_idx = j
                        else:
                            break
                    
                    # 创建中枢
                    pivot = Pivot(
                        index=len(pivots),
                        start_idx=i,
                        end_idx=end_idx,
                        zg=zg,
                        zd=zd,
                        gg=gg,
                        dd=dd,
                        start_time=bi1.start_time,
                        end_time=segment_bis[end_idx].end_time,
                        segment_idx=seg_idx,
                        bi_indices=[segment.bi_list[idx] for idx in pivot_bis]
                    )
                    pivots.append(pivot)
                    i = end_idx + 1
                else:
                    i += 1
        
        self.pivots = pivots
        return pivots
    
    def calculate_signals_complete(self):
        """完整的买卖信号计算"""
        signals = []
        
        # 1. 背驰信号
        signals.extend(self._find_divergence_signals())
        
        # 2. 中枢相关信号
        signals.extend(self._find_pivot_signals())
        
        # 3. 线段相关信号
        signals.extend(self._find_segment_signals())
        
        # 4. 分型突破信号
        signals.extend(self._find_fractal_breakout_signals())
        
        # 按时间排序
        signals.sort(key=lambda x: x['time'])
        self.signals = signals
        return signals
    
    def _find_divergence_signals(self):
        """寻找背驰信号"""
        signals = []
        
        # 确保有MACD数据
        if 'MACD_12_26_9' not in self.df.columns:
            return signals
        
        # 笔背驰
        for i in range(2, len(self.bi_list)):
            if i < 2:
                continue
            
            current_bi = self.bi_list[i]
            prev_bi = self.bi_list[i-1]
            
            try:
                current_macd = self.df.loc[current_bi.end_time, 'MACD_12_26_9']
                prev_macd = self.df.loc[prev_bi.end_time, 'MACD_12_26_9']
            except:
                continue
            
            # 底背驰
            if (current_bi.type == 'down' and prev_bi.type == 'up' and
                current_bi.low < prev_bi.low and current_macd > prev_macd):
                
                signals.append({
                    'type': 'buy1_div',
                    'time': current_bi.end_time,
                    'price': current_bi.end_price,
                    'confidence': 0.7,
                    'description': f"第一类买点：底背驰，价格{current_bi.low:.2f}，MACD不创新低",
                    'segment_idx': current_bi.parent_segment
                })
            
            # 顶背驰
            if (current_bi.type == 'up' and prev_bi.type == 'down' and
                current_bi.high > prev_bi.high and current_macd < prev_macd):
                
                signals.append({
                    'type': 'sell1_div',
                    'time': current_bi.end_time,
                    'price': current_bi.end_price,
                    'confidence': 0.7,
                    'description': f"第一类卖点：顶背驰，价格{current_bi.high:.2f}，MACD不创新高",
                    'segment_idx': current_bi.parent_segment
                })
        
        return signals
    
    def _find_pivot_signals(self):
        """寻找中枢相关信号"""
        signals = []
        
        for pivot in self.pivots:
            if pivot.end_idx + 1 >= len(self.bi_list):
                continue
            
            exit_bi = self.bi_list[pivot.end_idx]
            
            # 第三类买点
            if exit_bi.type == 'up':
                # 寻找回踩笔
                for i in range(pivot.end_idx + 1, len(self.bi_list)):
                    if self.bi_list[i].type == 'down':
                        if self.bi_list[i].low > pivot.zg:  # 回踩不破ZG
                            signals.append({
                                'type': 'buy3',
                                'time': self.bi_list[i].end_time,
                                'price': self.bi_list[i].end_price,
                                'confidence': 0.75,
                                'description': f"第三类买点：回踩不破中枢上沿{pivot.zg:.2f}",
                                'pivot_idx': pivot.index,
                                'segment_idx': pivot.segment_idx
                            })
                        break
            
            # 第三类卖点
            elif exit_bi.type == 'down':
                for i in range(pivot.end_idx + 1, len(self.bi_list)):
                    if self.bi_list[i].type == 'up':
                        if self.bi_list[i].high < pivot.zd:  # 反弹不破ZD
                            signals.append({
                                'type': 'sell3',
                                'time': self.bi_list[i].end_time,
                                'price': self.bi_list[i].end_price,
                                'confidence': 0.75,
                                'description': f"第三类卖点：反弹不破中枢下沿{pivot.zd:.2f}",
                                'pivot_idx': pivot.index,
                                'segment_idx': pivot.segment_idx
                            })
                        break
        
        return signals
    
    def _find_segment_signals(self):
        """寻找线段相关信号"""
        signals = []
        
        if len(self.segments) < 2:
            return signals
        
        for i in range(1, len(self.segments)):
            prev_seg = self.segments[i-1]
            curr_seg = self.segments[i]
            
            # 线段转折点（第二类买卖点附近）
            if prev_seg.type == 'down' and curr_seg.type == 'up':
                # 第二类买点区域
                if len(curr_seg.bi_list) >= 2:
                    buy_bi = self.bi_list[curr_seg.bi_list[1]]
                    if buy_bi.type == 'down':  # 向上线段中的第一笔向下笔
                        signals.append({
                            'type': 'buy2',
                            'time': buy_bi.end_time,
                            'price': buy_bi.end_price,
                            'confidence': 0.65,
                            'description': f"第二类买点：线段转折后的回调",
                            'segment_idx': curr_seg.index
                        })
            
            elif prev_seg.type == 'up' and curr_seg.type == 'down':
                # 第二类卖点区域
                if len(curr_seg.bi_list) >= 2:
                    sell_bi = self.bi_list[curr_seg.bi_list[1]]
                    if sell_bi.type == 'up':  # 向下线段中的第一笔向上笔
                        signals.append({
                            'type': 'sell2',
                            'time': sell_bi.end_time,
                            'price': sell_bi.end_price,
                            'confidence': 0.65,
                            'description': f"第二类卖点：线段转折后的反弹",
                            'segment_idx': curr_seg.index
                        })
        
        return signals
    
    def _find_fractal_breakout_signals(self):
        """寻找分型突破信号"""
        signals = []
        
        # 分型突破策略
        for i in range(2, len(self.fractals)):
            if i < 2:
                continue
            
            # 寻找重要的分型组合
            if (self.fractals[i-2].type == 'bottom' and 
                self.fractals[i-1].type == 'top' and 
                self.fractals[i].type == 'bottom'):
                
                # 双底形态
                if self.fractals[i].price > self.fractals[i-2].price:
                    signals.append({
                        'type': 'buy_breakout',
                        'time': self.fractals[i].time,
                        'price': self.fractals[i].price,
                        'confidence': 0.6,
                        'description': f"分型突破：双底形态确认",
                        'fractal_idx': i
                    })
            
            elif (self.fractals[i-2].type == 'top' and 
                  self.fractals[i-1].type == 'bottom' and 
                  self.fractals[i].type == 'top'):
                
                # 双顶形态
                if self.fractals[i].price < self.fractals[i-2].price:
                    signals.append({
                        'type': 'sell_breakout',
                        'time': self.fractals[i].time,
                        'price': self.fractals[i].price,
                        'confidence': 0.6,
                        'description': f"分型突破：双顶形态确认",
                        'fractal_idx': i
                    })
        
        return signals
    
    def run_complete_analysis(self):
        """运行完整的缠论分析流程"""
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("步骤1/6: 数据预处理...")
        self._prepare_data()
        progress_bar.progress(0.1)
        
        status_text.text("步骤2/6: 处理K线包含关系...")
        self.process_k_lines_optimized()
        progress_bar.progress(0.25)
        
        status_text.text("步骤3/6: 识别分型...")
        self.find_fractals_optimized()
        progress_bar.progress(0.4)
        
        status_text.text("步骤4/6: 识别笔...")
        self.find_bi_complete(min_k=bi_min_k)
        progress_bar.progress(0.6)
        
        status_text.text("步骤5/6: 识别线段...")
        self.find_segments_strict(min_bi=segment_min_bi)
        progress_bar.progress(0.8)
        
        status_text.text("步骤6/6: 识别中枢和信号...")
        self.find_pivots_advanced()
        self.calculate_signals_complete()
        progress_bar.progress(1.0)
        
        status_text.text("分析完成！")
        
        return True

# --- 可视化优化 ---
def create_interactive_chart(df, engine: EnhancedChantismEngine):
    """创建交互式图表"""
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            '缠论结构分析',
            '线段走势',
            '笔走势', 
            '技术指标'
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )
    
    # 1. 主图：K线 + 中枢 + 线段 + 信号
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 中枢（半透明区域）
    for pivot in engine.pivots:
        fig.add_trace(
            go.Scatter(
                x=[pivot.start_time, pivot.end_time, pivot.end_time, pivot.start_time, pivot.start_time],
                y=[pivot.zd, pivot.zd, pivot.zg, pivot.zg, pivot.zd],
                fill="toself",
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='#3498DB', width=1, dash='dash'),
                mode='lines',
                name=f'中枢{pivot.index}',
                showlegend=True,
                hoverinfo='text',
                hovertext=f"""
                中枢{pivot.index}<br>
                区间: {pivot.zd:.2f} - {pivot.zg:.2f}<br>
                宽度: {pivot.width:.2f}<br>
                时间: {pivot.start_time.strftime('%m-%d %H:%M')} 至 {pivot.end_time.strftime('%m-%d %H:%M')}<br>
                线段: {pivot.segment_idx}
                """
            ),
            row=1, col=1
        )
    
    # 线段
    colors = {'up': '#27AE60', 'down': '#C0392B'}
    for segment in engine.segments:
        # 连接线段起点和终点
        fig.add_trace(
            go.Scatter(
                x=[segment.start_time, segment.end_time],
                y=[segment.start_price, segment.end_price],
                mode='lines',
                line=dict(
                    color=colors[segment.type],
                    width=3
                ),
                name=f'线段{segment.index}',
                showlegend=True,
                hoverinfo='text',
                hovertext=f"""
                线段{segment.index} ({'向上' if segment.type == 'up' else '向下'})<br>
                价格: {segment.start_price:.2f} → {segment.end_price:.2f}<br>
                长度: {segment.length:.2f}<br>
                笔数: {len(segment.bi_list)}<br>
                时间: {segment.start_time.strftime('%m-%d %H:%M')} 至 {segment.end_time.strftime('%m-%d %H:%M')}
                """
            ),
            row=1, col=1
        )
    
    # 买卖信号
    signal_colors = {
        'buy1_div': '#2ECC71', 'buy2': '#27AE60', 'buy3': '#229954',
        'sell1_div': '#E74C3C', 'sell2': '#CB4335', 'sell3': '#B03A2E',
        'buy_breakout': '#17A589', 'sell_breakout': '#D35400'
    }
    
    signal_names = {
        'buy1_div': '一买(背驰)', 'buy2': '二买', 'buy3': '三买',
        'sell1_div': '一卖(背驰)', 'sell2': '二卖', 'sell3': '三卖',
        'buy_breakout': '突破买', 'sell_breakout': '突破卖'
    }
    
    for signal_type in signal_colors.keys():
        type_signals = [s for s in engine.signals if s['type'] == signal_type]
        if type_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s['time'] for s in type_signals],
                    y=[s['price'] for s in type_signals],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if 'buy' in signal_type else 'triangle-down',
                        size=15,
                        color=signal_colors[signal_type],
                        line=dict(width=2, color='white')
                    ),
                    name=signal_names[signal_type],
                    hoverinfo='text',
                    hovertext=[s['description'] for s in type_signals]
                ),
                row=1, col=1
            )
    
    # 2. 线段走势子图
    segment_prices = []
    segment_times = []
    segment_colors = []
    
    for segment in engine.segments:
        segment_prices.extend([segment.start_price, segment.end_price])
        segment_times.extend([segment.start_time, segment.end_time])
        segment_colors.extend([colors[segment.type], colors[segment.type]])
    
    fig.add_trace(
        go.Scatter(
            x=segment_times,
            y=segment_prices,
            mode='lines+markers',
            line=dict(color='#9B59B6', width=2),
            marker=dict(size=8),
            name='线段走势',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # 3. 笔走势子图
    bi_prices = []
    bi_times = []
    bi_colors = []
    
    for bi in engine.bi_list:
        bi_prices.extend([bi.start_price, bi.end_price])
        bi_times.extend([bi.start_time, bi.end_time])
        bi_colors.extend([colors[bi.type], colors[bi.type]])
    
    fig.add_trace(
        go.Scatter(
            x=bi_times,
            y=bi_prices,
            mode='lines+markers',
            line=dict(color='#F39C12', width=1.5),
            marker=dict(size=5),
            name='笔走势',
            showlegend=True
        ),
        row=3, col=1
    )
    
    # 4. 技术指标子图
    # MACD
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='#3498DB', width=1),
                showlegend=True
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACDs_12_26_9'],
                name='Signal',
                line=dict(color='#E74C3C', width=1),
                showlegend=True
            ),
            row=4, col=1
        )
        
        # MACD柱状图
        colors_macd = ['#2ECC71' if val >= 0 else '#E74C3C' for val in df['MACDh_12_26_9']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACDh_12_26_9'],
                name='MACD Hist',
                marker_color=colors_macd,
                opacity=0.5,
                showlegend=True
            ),
            row=4, col=1,
            secondary_y=False
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='#9B59B6', width=1),
                showlegend=True
            ),
            row=4, col=1,
            secondary_y=True
        )
        
        # RSI水平线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1, secondary_y=True)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1, secondary_y=True)
    
    # 更新布局
    fig.update_layout(
        title=f"{ticker} 智能缠论分析 (周期: {interval}, 范围: {period})",
        template="plotly_dark",
        height=1000,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False
    )
    
    # 更新坐标轴标签
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="线段", row=2, col=1)
    fig.update_yaxes(title_text="笔", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RSI", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="时间", row=4, col=1)
    
    return fig

# --- 分析报告生成器 ---
def generate_comprehensive_report(engine: EnhancedChantismEngine):
    """生成全面的分析报告"""
    report = {}
    
    # 基础统计
    report['基础统计'] = {
        'K线数量': len(engine.processed_k),
        '分型数量': len(engine.fractals),
        '笔数量': len(engine.bi_list),
        '线段数量': len(engine.segments),
        '中枢数量': len(engine.pivots),
        '信号数量': len(engine.signals)
    }
    
    # 线段分析
    if engine.segments:
        segment_stats = []
        for seg in engine.segments:
            stats = {
                '序号': seg.index,
                '方向': '向上' if seg.type == 'up' else '向下',
                '笔数': len(seg.bi_list),
                '起点价': seg.start_price,
                '终点价': seg.end_price,
                '幅度%': f"{(seg.end_price - seg.start_price) / seg.start_price * 100:.2f}",
                '长度(点)': seg.length,
                '持续时间(天)': f"{seg.duration:.2f}",
                '状态': '已完成' if seg.is_completed else '进行中'
            }
            segment_stats.append(stats)
        report['线段分析'] = segment_stats
    
    # 中枢分析
    if engine.pivots:
        pivot_stats = []
        for pivot in engine.pivots:
            stats = {
                '序号': pivot.index,
                '线段': pivot.segment_idx,
                '上沿(ZG)': pivot.zg,
                '下沿(ZD)': pivot.zd,
                '宽度': pivot.width,
                '笔数': len(pivot.bi_indices),
                '持续时间(天)': f"{pivot.duration:.2f}",
                '级别': pivot.level
            }
            pivot_stats.append(stats)
        report['中枢分析'] = pivot_stats
    
    # 信号分析
    if engine.signals:
        signal_stats = []
        buy_signals = [s for s in engine.signals if 'buy' in s['type']]
        sell_signals = [s for s in engine.signals if 'sell' in s['type']]
        
        report['信号统计'] = {
            '买点总数': len(buy_signals),
            '卖点总数': len(sell_signals),
            '第一类买卖点': len([s for s in engine.signals if '1' in s['type']]),
            '第二类买卖点': len([s for s in engine.signals if '2' in s['type']]),
            '第三类买卖点': len([s for s in engine.signals if '3' in s['type']])
        }
        
        for signal in engine.signals[:10]:  # 显示最近10个信号
            stats = {
                '类型': signal['type'],
                '时间': signal['time'].strftime('%m-%d %H:%M'),
                '价格': signal['price'],
                '信心度': f"{signal['confidence']:.0%}",
                '线段': signal.get('segment_idx', '-'),
                '中枢': signal.get('pivot_idx', '-'),
                '描述': signal['description']
            }
            signal_stats.append(stats)
        report['最近信号'] = signal_stats
    
    # 趋势分析
    if engine.segments:
        current_segment = engine.segments[-1]
        report['当前趋势'] = {
            '当前线段': current_segment.index,
            '方向': '向上' if current_segment.type == 'up' else '向下',
            '状态': '进行中' if not current_segment.is_completed else '已完成',
            '当前价格': engine.raw_df['Close'].iloc[-1],
            '线段起点': current_segment.start_price,
            '线段当前幅度%': f"{(engine.raw_df['Close'].iloc[-1] - current_segment.start_price) / current_segment.start_price * 100:.2f}"
        }
    
    return report

# --- 主程序 ---
if run_analysis:
    try:
        # 获取数据
        with st.spinner(f"正在获取 {ticker} 数据..."):
            data = yf.download(
                ticker, 
                period=period, 
                interval=interval, 
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                st.error(f"无法获取 {ticker} 的数据，请检查代码是否正确")
                st.stop()
            
            st.success(f"✅ 成功获取 {len(data)} 条K线数据")
        
        # 创建引擎并运行分析
        engine = EnhancedChantismEngine(data)
        
        # 运行分析
        engine.run_complete_analysis()
        
        # 显示概览
        st.subheader("📊 分析概览")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("K线", len(engine.processed_k))
        
        with col2:
            st.metric("分型", len(engine.fractals))
        
        with col3:
            st.metric("笔", len(engine.bi_list))
        
        with col4:
            st.metric("线段", len(engine.segments))
        
        with col5:
            st.metric("中枢", len(engine.pivots))
        
        with col6:
            buy_count = len([s for s in engine.signals if 'buy' in s['type']])
            sell_count = len([s for s in engine.signals if 'sell' in s['type']])
            st.metric("信号", f"{buy_count}买/{sell_count}卖")
        
        # 显示图表
        st.subheader("📈 缠论结构分析图")
        fig = create_interactive_chart(data, engine)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # 显示详细报告
        st.subheader("📋 详细分析报告")
        
        report = generate_comprehensive_report(engine)
        
        # 使用标签页组织报告
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 基础统计", 
            "📈 线段分析", 
            "🎯 中枢分析", 
            "🚦 信号分析", 
            "📉 趋势分析"
        ])
        
        with tab1:
            st.table(pd.DataFrame([report['基础统计']]).T.reset_index().rename(
                columns={'index': '指标', 0: '数值'}
            ))
        
        with tab2:
            if '线段分析' in report:
                seg_df = pd.DataFrame(report['线段分析'])
                st.dataframe(seg_df, use_container_width=True, hide_index=True)
                
                # 线段可视化
                fig_seg = go.Figure()
                
                for seg in engine.segments:
                    fig_seg.add_trace(go.Scatter(
                        x=[seg.start_time, seg.end_time],
                        y=[seg.start_price, seg.end_price],
                        mode='lines+markers',
                        line=dict(width=3, color='green' if seg.type == 'up' else 'red'),
                        marker=dict(size=10),
                        name=f"线段{seg.index}"
                    ))
                
                fig_seg.update_layout(
                    title="线段走势图",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_seg, use_container_width=True)
        
        with tab3:
            if '中枢分析' in report:
                pivot_df = pd.DataFrame(report['中枢分析'])
                st.dataframe(pivot_df, use_container_width=True, hide_index=True)
        
        with tab4:
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                if '信号统计' in report:
                    st.metric("总买点", report['信号统计']['买点总数'])
                    st.metric("总卖点", report['信号统计']['卖点总数'])
            
            with col_s2:
                if '信号统计' in report:
                    st.metric("一类买卖点", report['信号统计']['第一类买卖点'])
                    st.metric("二类买卖点", report['信号统计']['第二类买卖点'])
                    st.metric("三类买卖点", report['信号统计']['第三类买卖点'])
            
            if '最近信号' in report:
                st.subheader("最近买卖信号")
                signal_df = pd.DataFrame(report['最近信号'])
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
        
        with tab5:
            if '当前趋势' in report:
                trend = report['当前趋势']
                
                col_t1, col_t2, col_t3 = st.columns(3)
                
                with col_t1:
                    st.metric("当前线段", trend['当前线段'])
                    st.metric("方向", trend['方向'])
                
                with col_t2:
                    st.metric("状态", trend['状态'])
                    st.metric("当前价格", f"${trend['当前价格']:.2f}")
                
                with col_t3:
                    st.metric("线段起点", f"${trend['线段起点']:.2f}")
                    st.metric("当前幅度", trend['线段当前幅度%'])
                
                # 趋势判断
                st.subheader("📈 趋势判断")
                
                if trend['方向'] == '向上':
                    if float(trend['线段当前幅度%'].replace('%', '')) > 5:
                        st.success("🔺 强势上涨趋势，建议持有多头仓位")
                    else:
                        st.info("↗️ 温和上涨趋势，可考虑逢低买入")
                else:
                    if float(trend['线段当前幅度%'].replace('%', '')) < -5:
                        st.error("🔻 强势下跌趋势，建议持有空头仓位")
                    else:
                        st.warning("↘️ 温和下跌趋势，可考虑逢高卖出")
        
        # 下载功能
        st.subheader("💾 数据导出")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            # 导出信号数据
            if engine.signals:
                signals_df = pd.DataFrame(engine.signals)
                csv = signals_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载信号数据",
                    data=csv,
                    file_name=f"{ticker}_{interval}_signals.csv",
                    mime="text/csv"
                )
        
        with col_d2:
            # 导出分析报告
            report_df = pd.DataFrame(report['基础统计'].items(), columns=['指标', '数值'])
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 下载分析报告",
                data=csv,
                file_name=f"{ticker}_{interval}_report.csv",
                mime="text/csv"
            )
        
        with col_d3:
            # 导出原始数据
            csv = data.to_csv()
            st.download_button(
                label="📥 下载原始数据",
                data=csv,
                file_name=f"{ticker}_{interval}_raw.csv",
                mime="text/csv"
            )
        
        # 性能统计
        with st.expander("📊 性能统计", expanded=False):
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.metric("数据条数", len(data))
                st.metric("处理时间", "实时")
            
            with col_p2:
                st.metric("识别准确率", "待优化")
                st.metric("算法版本", "V5.0")
        
    except Exception as e:
        st.error(f"分析过程中出现错误: {str(e)}")
        st.exception(e)

else:
    # 欢迎页面
    st.markdown("""
    # 🎯 缠论智能分析系统 V5.0
    
    ## ✨ 核心特性
    
    ### 🚀 性能优化
    - **向量化处理**：使用NumPy加速计算
    - **智能缓存**：减少重复计算
    - **实时分析**：秒级响应
    
    ### 🧠 算法增强
    - **完整缠论实现**：笔-线段-中枢全流程
    - **特征序列处理**：符合缠论标准
    - **多维度信号**：背驰、突破、中枢买卖点
    
    ### 📊 可视化升级
    - **交互式图表**：Plotly动态展示
    - **多层结构**：笔、线段、中枢分层显示
    - **智能标注**：自动标记关键点位
    
    ### 🛡️ 稳定性提升
    - **异常处理**：完善的错误处理机制
    - **数据验证**：输入数据完整性检查
    - **性能监控**：实时分析性能统计
    
    ## 🚀 快速开始
    
    1. **左侧设置**分析参数
    2. **点击按钮**开始分析
    3. **查看图表**中的缠论结构
    4. **分析报告**提供交易建议
    
    ## 📈 支持的品种
    
    - **股票**：AAPL, TSLA, NVDA 等
    - **加密货币**：BTC-USD, ETH-USD 等
    - **外汇**：EURUSD=X, GBPUSD=X 等
    - **期货**：CL=F, GC=F 等
    
    ## 🔧 参数说明
    
    - **笔最少K线数**：标准缠论为5根K线
    - **线段最少笔数**：标准为5笔（含特征序列）
    - **中枢最少笔数**：标准为3笔重叠
    
    ---
    
    *提示：建议在1h或更高周期进行分析，分钟级别数据可能噪音较大*
    """)
    
    # 显示示例
    st.info("👈 在左侧设置参数，然后点击'开始智能分析'按钮")
    
    # 添加示例图片或GIF
    # st.image("example_chart.png", caption="示例分析图表")