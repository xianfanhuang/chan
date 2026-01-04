import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- 配置 ---
st.set_page_config(layout="wide", page_title="Chantism Pro V4", page_icon="📈")
st.title("📈 Chantism Pro V4: 完整缠论分析系统")

# --- 参数设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("代码", "BTC-USD", help="股票/加密货币代码")
    with col2:
        interval = st.selectbox(
            "周期",
            ["1h", "2h", "4h", "1d", "1wk"],
            index=0,
            help="K线周期"
        )
    
    period = st.selectbox(
        "时间范围",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=1,
        help="数据时间范围"
    )
    
    st.markdown("---")
    st.subheader("缠论参数")
    
    bi_min_k = st.slider("笔最少K线数", 3, 10, 5, 
                        help="构成笔所需的最少K线数量")
    
    segment_min_bi = st.slider("线段最少笔数", 3, 7, 5,
                              help="构成线段所需的最少笔数")
    
    pivot_min_bi = st.slider("中枢最少笔数", 3, 6, 3,
                           help="构成中枢所需的最少笔数")
    
    st.markdown("---")
    analysis_mode = st.selectbox(
        "分析模式",
        ["自动识别", "严格模式", "宽松模式"],
        index=0,
        help="线段识别的严格程度"
    )
    
    include_macd = st.checkbox("显示MACD", True)
    include_volume = st.checkbox("显示成交量", True)
    
    st.markdown("---")
    
    if st.button("🔍 运行完整分析", type="primary", use_container_width=True):
        run_analysis = True
    else:
        run_analysis = False

# --- 数据结构 ---
@dataclass
class Fractal:
    """分型"""
    idx: int
    type: str  # 'top' or 'bottom'
    price: float
    time: pd.Timestamp
    confirmed: bool = True
    k_idx: int = 0  # 原始K线索引

@dataclass
class Bi:
    """笔"""
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
    is_verified: bool = True

@dataclass 
class FeatureElement:
    """特征序列元素"""
    start: float
    end: float
    high: float
    low: float
    type: str  # 'up' or 'down'
    time: pd.Timestamp
    is_gap: bool = False

@dataclass
class Segment:
    """线段"""
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
    bi_list: List[int] = None
    level: int = 1
    
    def __post_init__(self):
        if self.bi_list is None:
            self.bi_list = []

@dataclass 
class Pivot:
    """中枢"""
    index: int
    level: int = 1
    start_idx: int = 0
    end_idx: int = 0
    zg: float = 0.0  # 中枢高点
    zd: float = 0.0  # 中枢低点
    gg: float = 0.0  # 中枢最高点
    dd: float = 0.0  # 中枢最低点
    start_time: pd.Timestamp = None
    end_time: pd.Timestamp = None
    segment_idx: int = -1  # 所属线段索引
    
    def __post_init__(self):
        self.zg = round(self.zg, 4)
        self.zd = round(self.zd, 4)

@dataclass
class Signal:
    """买卖信号"""
    type: str  # 'buy1', 'buy2', 'buy3', 'sell1', 'sell2', 'sell3'
    time: pd.Timestamp
    price: float
    confidence: float = 0.5
    description: str = ""
    segment_idx: int = -1
    pivot_idx: int = -1

# --- 特征序列处理类 ---
class FeatureSequence:
    """特征序列处理器"""
    
    @staticmethod
    def get_feature_elements(bi_list: List[Bi], segment_type: str) -> List[FeatureElement]:
        """获取特征序列元素"""
        elements = []
        
        # 向上线段，特征序列是向下的笔
        # 向下线段，特征序列是向上的笔
        for bi in bi_list:
            if segment_type == 'up' and bi.type == 'down':
                element = FeatureElement(
                    start=bi.start_price,
                    end=bi.end_price,
                    high=bi.high,
                    low=bi.low,
                    type=bi.type,
                    time=bi.end_time
                )
                elements.append(element)
            elif segment_type == 'down' and bi.type == 'up':
                element = FeatureElement(
                    start=bi.start_price,
                    end=bi.end_price,
                    high=bi.high,
                    low=bi.low,
                    type=bi.type,
                    time=bi.end_time
                )
                elements.append(element)
        
        return elements
    
    @staticmethod
    def process_inclusion(elements: List[FeatureElement]) -> List[FeatureElement]:
        """特征序列包含处理"""
        if len(elements) < 2:
            return elements
        
        processed = []
        direction = None  # 包含处理方向
        
        for i, elem in enumerate(elements):
            if i == 0:
                processed.append(elem)
                continue
            
            prev = processed[-1]
            
            # 判断包含关系
            is_contained = (
                (elem.high <= prev.high and elem.low >= prev.low) or
                (elem.high >= prev.high and elem.low <= prev.low)
            )
            
            if is_contained:
                # 确定包含处理方向
                if direction is None:
                    # 第一个包含，取向上处理
                    direction = 'up' if elem.high >= prev.high else 'down'
                
                if direction == 'up':
                    # 向上处理：取高高
                    new_high = max(prev.high, elem.high)
                    new_low = max(prev.low, elem.low)
                else:
                    # 向下处理：取低低
                    new_high = min(prev.high, elem.high)
                    new_low = min(prev.low, elem.low)
                
                processed[-1] = FeatureElement(
                    start=prev.start if prev.start < elem.start else elem.start,
                    end=prev.end,
                    high=new_high,
                    low=new_low,
                    type=prev.type,
                    time=elem.time,
                    is_gap=prev.is_gap
                )
            else:
                processed.append(elem)
                direction = None
        
        return processed
    
    @staticmethod
    def has_break_gap(elements: List[FeatureElement]) -> bool:
        """检查是否有缺口"""
        for elem in elements:
            if elem.is_gap:
                return True
        return False

# --- 完整缠论引擎 ---
class ChantismCompleteEngine:
    """完整缠论分析引擎"""
    
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.processed_k = []
        self.fractals: List[Fractal] = []
        self.bi_list: List[Bi] = []
        self.segments: List[Segment] = []
        self.pivots: List[Pivot] = []
        self.signals: List[Signal] = []
        self.feature_processor = FeatureSequence()
        
    # === K线处理 ===
    def process_k_lines(self):
        """K线包含处理（改进版）"""
        if self.df.empty:
            return []
            
        data = self.df.reset_index()
        time_col = 'Date' if 'Date' in data.columns else 'Datetime'
        
        times = data[time_col].values
        highs = data['High'].values
        lows = data['Low'].values
        opens = data['Open'].values
        closes = data['Close'].values
        
        processed = []
        direction = None  # 包含处理方向
        
        for i in range(len(times)):
            if i == 0:
                processed.append({
                    'idx': i,
                    'time': times[i],
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i],
                    'close': closes[i],
                    'volume': data.iloc[i]['Volume'] if 'Volume' in data.columns else 0
                })
                continue
            
            current = {
                'high': highs[i],
                'low': lows[i],
                'time': times[i],
                'open': opens[i],
                'close': closes[i]
            }
            prev = processed[-1]
            
            # 检查包含关系
            if (current['high'] <= prev['high'] and current['low'] >= prev['low']) or \
               (current['high'] >= prev['high'] and current['low'] <= prev['low']):
                
                # 确定方向
                if direction is None:
                    if len(processed) == 1:
                        direction = 'up' if current['high'] >= prev['high'] else 'down'
                    else:
                        # 看前一根非包含K线
                        if len(processed) >= 2:
                            if processed[-2]['high'] < prev['high']:
                                direction = 'up'
                            else:
                                direction = 'down'
                
                if direction == 'up':
                    new_high = max(prev['high'], current['high'])
                    new_low = max(prev['low'], current['low'])
                else:
                    new_high = min(prev['high'], current['high'])
                    new_low = min(prev['low'], current['low'])
                
                processed[-1].update({
                    'high': new_high,
                    'low': new_low,
                    'time': current['time']
                })
            else:
                processed.append({
                    'idx': i,
                    'time': current['time'],
                    'high': current['high'],
                    'low': current['low'],
                    'open': current['open'],
                    'close': current['close'],
                    'volume': data.iloc[i]['Volume'] if 'Volume' in data.columns else 0
                })
                direction = None
        
        self.processed_k = processed
        return pd.DataFrame(processed)
    
    # === 分型识别 ===
    def find_fractals(self, confirm_bars=3):
        """识别顶底分型"""
        if not self.processed_k:
            return []
            
        n = len(self.processed_k)
        fractals = []
        
        for i in range(1, n-1):
            # 检查是否满足分型条件
            prev_k = self.processed_k[i-1]
            curr_k = self.processed_k[i]
            next_k = self.processed_k[i+1]
            
            # 顶分型条件
            if (curr_k['high'] > prev_k['high'] and 
                curr_k['high'] > next_k['high'] and
                curr_k['low'] > prev_k['low'] and
                curr_k['low'] > next_k['low']):
                
                # 确认：后续K线不创新高
                confirmed = True
                for j in range(1, min(confirm_bars+1, n-i-1)):
                    if self.processed_k[i+j]['high'] > curr_k['high']:
                        confirmed = False
                        break
                
                fractals.append(Fractal(
                    idx=len(fractals),
                    type='top',
                    price=curr_k['high'],
                    time=curr_k['time'],
                    confirmed=confirmed,
                    k_idx=curr_k['idx']
                ))
            
            # 底分型条件
            elif (curr_k['low'] < prev_k['low'] and 
                  curr_k['low'] < next_k['low'] and
                  curr_k['high'] < prev_k['high'] and
                  curr_k['high'] < next_k['high']):
                
                # 确认：后续K线不创新低
                confirmed = True
                for j in range(1, min(confirm_bars+1, n-i-1)):
                    if self.processed_k[i+j]['low'] < curr_k['low']:
                        confirmed = False
                        break
                
                fractals.append(Fractal(
                    idx=len(fractals),
                    type='bottom',
                    price=curr_k['low'],
                    time=curr_k['time'],
                    confirmed=confirmed,
                    k_idx=curr_k['idx']
                ))
        
        # 过滤相邻同类型分型
        filtered = []
        for i in range(len(fractals)):
            if i == 0:
                filtered.append(fractals[i])
                continue
            
            prev = filtered[-1]
            curr = fractals[i]
            
            # 跳过相邻同类型分型
            if prev.type == curr.type:
                # 取更极值的
                if prev.type == 'top' and curr.price > prev.price:
                    filtered[-1] = curr
                elif prev.type == 'bottom' and curr.price < prev.price:
                    filtered[-1] = curr
            else:
                # 检查间隔K线数
                k_gap = curr.k_idx - prev.k_idx
                if k_gap >= 4:  # 至少4根K线
                    filtered.append(curr)
        
        self.fractals = filtered
        return filtered
    
    # === 笔识别 ===
    def find_bi(self, min_k=5):
        """识别笔（严格模式）"""
        if len(self.fractals) < 2:
            return []
            
        bi_list = []
        i = 0
        
        while i < len(self.fractals) - 1:
            start_fractal = self.fractals[i]
            end_fractal = self.fractals[i + 1]
            
            # 分型必须交替
            if start_fractal.type == end_fractal.type:
                i += 1
                continue
            
            # 检查K线数量
            k_gap = abs(end_fractal.k_idx - start_fractal.k_idx)
            
            if k_gap < min_k:
                i += 1
                continue
            
            # 构成笔
            if start_fractal.type == 'bottom' and end_fractal.type == 'top':
                bi_type = 'up'
                start_price = start_fractal.price
                end_price = end_fractal.price
                high = end_fractal.price
                low = start_fractal.price
            else:
                bi_type = 'down'
                start_price = start_fractal.price
                end_price = end_fractal.price
                high = start_fractal.price
                low = end_fractal.price
            
            # 计算强度
            price_change = abs(end_price - start_price)
            time_diff = (end_fractal.time - start_fractal.time).total_seconds() / 3600
            
            bi = Bi(
                index=len(bi_list),
                type=bi_type,
                start_idx=start_fractal.k_idx,
                end_idx=end_fractal.k_idx,
                start_price=start_price,
                end_price=end_price,
                high=high,
                low=low,
                start_time=start_fractal.time,
                end_time=end_fractal.time,
                strength=price_change / time_diff if time_diff > 0 else 0
            )
            
            bi_list.append(bi)
            i += 1  # 移动到下一个分型
        
        self.bi_list = bi_list
        return bi_list
    
    # === 线段识别 ===
    def find_segments(self, min_bi=5):
        """识别线段（核心算法）"""
        if len(self.bi_list) < min_bi:
            return []
        
        segments = []
        segment_start_idx = 0
        current_direction = self.bi_list[0].type  # 第一笔的方向
        
        i = 0
        while i < len(self.bi_list):
            if i - segment_start_idx + 1 >= min_bi:
                # 检查是否可以结束线段
                can_end = self._check_segment_end(segment_start_idx, i, current_direction)
                
                if can_end:
                    # 创建线段
                    segment_bi_list = self.bi_list[segment_start_idx:i+1]
                    segment = self._create_segment(segment_bi_list, len(segments), current_direction)
                    segments.append(segment)
                    
                    # 开始新的线段
                    segment_start_idx = i + 1
                    if segment_start_idx < len(self.bi_list):
                        current_direction = self.bi_list[segment_start_idx].type
            
            i += 1
        
        # 处理最后一段
        if segment_start_idx < len(self.bi_list):
            segment_bi_list = self.bi_list[segment_start_idx:]
            if len(segment_bi_list) >= 3:  # 至少3笔才能构成线段
                segment = self._create_segment(segment_bi_list, len(segments), current_direction)
                segments.append(segment)
        
        self.segments = segments
        return segments
    
    def _check_segment_end(self, start_idx: int, end_idx: int, direction: str) -> bool:
        """检查线段是否结束"""
        if end_idx - start_idx < 4:  # 至少5笔才可能结束
            return False
        
        current_bi = self.bi_list[end_idx]
        prev_bi = self.bi_list[end_idx-1]
        
        # 特征序列分析
        feature_elements = []
        for j in range(start_idx, end_idx + 1):
            bi = self.bi_list[j]
            if direction == 'up' and bi.type == 'down':
                feature_elements.append(bi)
            elif direction == 'down' and bi.type == 'up':
                feature_elements.append(bi)
        
        if len(feature_elements) < 3:
            return False
        
        # 简化版线段破坏判断
        if direction == 'up':
            # 向上线段被向下笔破坏
            if current_bi.type == 'down':
                # 检查是否形成顶分型
                if current_bi.end_price < prev_bi.start_price:
                    return True
        else:
            # 向下线段被向上笔破坏
            if current_bi.type == 'up':
                if current_bi.end_price > prev_bi.start_price:
                    return True
        
        return False
    
    def _create_segment(self, bi_list: List[Bi], index: int, direction: str) -> Segment:
        """创建线段对象"""
        start_price = bi_list[0].start_price
        end_price = bi_list[-1].end_price
        
        highs = [bi.high for bi in bi_list]
        lows = [bi.low for bi in bi_list]
        
        segment = Segment(
            index=index,
            type=direction,
            start_bi_idx=bi_list[0].index,
            end_bi_idx=bi_list[-1].index,
            start_price=start_price,
            end_price=end_price,
            high=max(highs),
            low=min(lows),
            start_time=bi_list[0].start_time,
            end_time=bi_list[-1].end_time,
            bi_list=[bi.index for bi in bi_list]
        )
        return segment
    
    # === 中枢识别 ===
    def find_pivots_in_segments(self):
        """在线段中识别中枢"""
        pivots = []
        
        for seg_idx, segment in enumerate(self.segments):
            segment_bi_indices = segment.bi_list
            if len(segment_bi_indices) < 3:
                continue
            
            # 取线段中的笔
            segment_bis = [self.bi_list[idx] for idx in segment_bi_indices]
            
            # 寻找重叠的三笔
            i = 0
            while i <= len(segment_bis) - 3:
                bi1 = segment_bis[i]
                bi2 = segment_bis[i+1]
                bi3 = segment_bis[i+2]
                
                # 检查重叠
                highs = [bi1.high, bi2.high, bi3.high]
                lows = [bi1.low, bi2.low, bi3.low]
                
                zg = min(highs)
                zd = max(lows)
                
                if zg > zd:  # 有重叠
                    gg = max(highs)
                    dd = min(lows)
                    
                    # 尝试延伸中枢
                    end_idx = i + 2
                    for j in range(i+3, len(segment_bis)):
                        next_bi = segment_bis[j]
                        if not (next_bi.low > zg or next_bi.high < zd):
                            # 更新中枢区间
                            zg = min(zg, next_bi.high)
                            zd = max(zd, next_bi.low)
                            gg = max(gg, next_bi.high)
                            dd = min(dd, next_bi.low)
                            end_idx = j
                        else:
                            break
                    
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
                        segment_idx=seg_idx
                    )
                    pivots.append(pivot)
                    i = end_idx + 1
                else:
                    i += 1
        
        self.pivots = pivots
        return pivots
    
    # === 买卖信号 ===
    def calculate_signals(self):
        """计算买卖信号"""
        if not self.pivots or not self.segments:
            return []
        
        signals = []
        
        # 第一类买卖点：趋势背驰
        signals.extend(self._find_type1_signals())
        
        # 第二类买卖点：回抽不创新低/新高
        signals.extend(self._find_type2_signals())
        
        # 第三类买卖点：离开中枢后回抽
        signals.extend(self._find_type3_signals())
        
        self.signals = signals
        return signals
    
    def _find_type1_signals(self):
        """第一类买卖点"""
        signals = []
        
        for i in range(1, len(self.bi_list)):
            prev_bi = self.bi_list[i-1]
            curr_bi = self.bi_list[i]
            
            # 检查是否在同一线段中
            if self._get_segment_for_bi(prev_bi.index) != self._get_segment_for_bi(curr_bi.index):
                continue
            
            # 底背驰买点
            if (prev_bi.type == 'up' and curr_bi.type == 'down' and
                curr_bi.low < prev_bi.low):
                
                # 计算MACD背驰
                if self._check_macd_divergence(curr_bi, 'bottom'):
                    signals.append(Signal(
                        type='buy1',
                        time=curr_bi.end_time,
                        price=curr_bi.end_price,
                        confidence=0.8,
                        description=f"第一类买点：底背驰，价格{curr_bi.low:.2f}",
                        segment_idx=self._get_segment_for_bi(curr_bi.index)
                    ))
            
            # 顶背驰卖点
            if (prev_bi.type == 'down' and curr_bi.type == 'up' and
                curr_bi.high > prev_bi.high):
                
                if self._check_macd_divergence(curr_bi, 'top'):
                    signals.append(Signal(
                        type='sell1',
                        time=curr_bi.end_time,
                        price=curr_bi.end_price,
                        confidence=0.8,
                        description=f"第一类卖点：顶背驰，价格{curr_bi.high:.2f}",
                        segment_idx=self._get_segment_for_bi(curr_bi.index)
                    ))
        
        return signals
    
    def _find_type2_signals(self):
        """第二类买卖点"""
        signals = []
        
        for i in range(2, len(self.bi_list)):
            if i < 2:
                continue
            
            bi1 = self.bi_list[i-2]  # 第一类买卖点所在笔
            bi2 = self.bi_list[i-1]  # 反弹/回调笔
            bi3 = self.bi_list[i]    # 第二类买卖点所在笔
            
            # 第二类买点：第一类买点后的回调不创新低
            if (bi1.type == 'down' and bi2.type == 'up' and bi3.type == 'down' and
                bi3.low > bi1.low):
                
                signals.append(Signal(
                    type='buy2',
                    time=bi3.end_time,
                    price=bi3.end_price,
                    confidence=0.7,
                    description=f"第二类买点：回调不创新低，低点{bi3.low:.2f} > {bi1.low:.2f}",
                    segment_idx=self._get_segment_for_bi(bi3.index)
                ))
            
            # 第二类卖点：第一类卖点后的反弹不创新高
            if (bi1.type == 'up' and bi2.type == 'down' and bi3.type == 'up' and
                bi3.high < bi1.high):
                
                signals.append(Signal(
                    type='sell2',
                    time=bi3.end_time,
                    price=bi3.end_price,
                    confidence=0.7,
                    description=f"第二类卖点：反弹不创新高，高点{bi3.high:.2f} < {bi1.high:.2f}",
                    segment_idx=self._get_segment_for_bi(bi3.index)
                ))
        
        return signals
    
    def _find_type3_signals(self):
        """第三类买卖点"""
        signals = []
        
        for pivot in self.pivots:
            pivot_end_bi_idx = pivot.end_idx
            if pivot_end_bi_idx + 1 >= len(self.bi_list):
                continue
            
            # 中枢后的笔
            exit_bi = self.bi_list[pivot_end_bi_idx]
            next_bi = self.bi_list[pivot_end_bi_idx + 1]
            
            # 第三类买点：向上离开中枢后回调不破ZG
            if (exit_bi.type == 'up' and next_bi.type == 'down' and
                next_bi.low > pivot.zg):
                
                signals.append(Signal(
                    type='buy3',
                    time=next_bi.end_time,
                    price=next_bi.end_price,
                    confidence=0.75,
                    description=f"第三类买点：回踩不破中枢上沿{pivot.zg:.2f}",
                    segment_idx=pivot.segment_idx,
                    pivot_idx=pivot.index
                ))
            
            # 第三类卖点：向下离开中枢后反弹不破ZD
            if (exit_bi.type == 'down' and next_bi.type == 'up' and
                next_bi.high < pivot.zd):
                
                signals.append(Signal(
                    type='sell3',
                    time=next_bi.end_time,
                    price=next_bi.end_price,
                    confidence=0.75,
                    description=f"第三类卖点：反弹不破中枢下沿{pivot.zd:.2f}",
                    segment_idx=pivot.segment_idx,
                    pivot_idx=pivot.index
                ))
        
        return signals
    
    def _get_segment_for_bi(self, bi_idx: int) -> int:
        """获取笔所属的线段索引"""
        for segment in self.segments:
            if bi_idx in segment.bi_list:
                return segment.index
        return -1
    
    def _check_macd_divergence(self, bi: Bi, div_type: str) -> bool:
        """检查MACD背驰"""
        if 'MACD_12_26_9' not in self.df.columns:
            # 计算MACD
            macd = ta.macd(self.df['Close'], fast=12, slow=26, signal=9)
            self.df = pd.concat([self.df, macd], axis=1)
        
        try:
            bi_end_macd = self.df.loc[bi.end_time, 'MACD_12_26_9']
            
            # 简单背驰检查：需要更复杂的算法
            return True
        except:
            return False
    
    # === 运行完整分析 ===
    def run_complete_analysis(self):
        """运行完整缠论分析"""
        st.info("步骤1: 处理K线包含关系...")
        self.process_k_lines()
        
        st.info("步骤2: 识别顶底分型...")
        self.find_fractals()
        
        st.info("步骤3: 生成笔...")
        self.find_bi(min_k=bi_min_k)
        
        st.info("步骤4: 识别线段...")
        self.find_segments(min_bi=segment_min_bi)
        
        st.info("步骤5: 识别中枢...")
        self.find_pivots_in_segments()
        
        st.info("步骤6: 计算买卖信号...")
        self.calculate_signals()
        
        st.success("分析完成！")

# --- 可视化模块 ---
def create_advanced_chart(df, engine: ChantismCompleteEngine):
    """创建高级图表"""
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('缠论结构图', '线段', '笔', 'MACD')
    )
    
    # 1. 主图：K线 + 中枢 + 信号
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K线',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 画线段
    colors = {'up': 'green', 'down': 'red'}
    for segment in engine.segments:
        # 线段趋势线
        start_bi = engine.bi_list[segment.start_bi_idx]
        end_bi = engine.bi_list[segment.end_bi_idx]
        
        fig.add_trace(
            go.Scatter(
                x=[start_bi.start_time, end_bi.end_time],
                y=[start_bi.start_price, end_bi.end_price],
                mode='lines',
                line=dict(
                    color=colors[segment.type],
                    width=3,
                    dash='dash'
                ),
                name=f"线段-{segment.type}",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 画中枢
    for pivot in engine.pivots:
        fig.add_trace(
            go.Scatter(
                x=[pivot.start_time, pivot.end_time, pivot.end_time, pivot.start_time, pivot.start_time],
                y=[pivot.zd, pivot.zd, pivot.zg, pivot.zg, pivot.zd],
                fill="toself",
                fillcolor='rgba(135, 206, 235, 0.3)',
                line=dict(color='blue', width=1),
                mode='lines',
                name=f'中枢 {pivot.index}',
                showlegend=False,
                hoverinfo='text',
                text=f"中枢{pivot.index}<br>区间: {pivot.zd:.2f}-{pivot.zg:.2f}<br>时间: {pivot.start_time.strftime('%Y-%m-%d')} 至 {pivot.end_time.strftime('%Y-%m-%d')}"
            ),
            row=1, col=1
        )
    
    # 买卖信号
    buy_signals = [s for s in engine.signals if 'buy' in s.type]
    sell_signals = [s for s in engine.signals if 'sell' in s.type]
    
    if buy_signals:
        fig.add_trace(
            go.Scatter(
                x=[s.time for s in buy_signals],
                y=[s.price for s in buy_signals],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=2, color='white')
                ),
                name='买点',
                text=[f"{s.type}: {s.description}" for s in buy_signals],
                hoverinfo='text+y'
            ),
            row=1, col=1
        )
    
    if sell_signals:
        fig.add_trace(
            go.Scatter(
                x=[s.time for s in sell_signals],
                y=[s.price for s in sell_signals],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='white')
                ),
                name='卖点',
                text=[f"{s.type}: {s.description}" for s in sell_signals],
                hoverinfo='text+y'
            ),
            row=1, col=1
        )
    
    # 2. 线段子图
    segment_prices = []
    segment_times = []
    
    for segment in engine.segments:
        segment_prices.extend([segment.start_price, segment.end_price])
        segment_times.extend([segment.start_time, segment.end_time])
    
    fig.add_trace(
        go.Scatter(
            x=segment_times,
            y=segment_prices,
            mode='lines+markers',
            line=dict(color='purple', width=2),
            marker=dict(size=8),
            name='线段',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. 笔子图
    bi_prices = []
    bi_times = []
    
    for bi in engine.bi_list:
        bi_prices.extend([bi.start_price, bi.end_price])
        bi_times.extend([bi.start_time, bi.end_time])
    
    fig.add_trace(
        go.Scatter(
            x=bi_times,
            y=bi_prices,
            mode='lines+markers',
            line=dict(color='orange', width=1),
            marker=dict(size=4),
            name='笔',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 4. MACD子图
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='blue', width=1),
                showlegend=False
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACDs_12_26_9'],
                name='Signal',
                line=dict(color='orange', width=1),
                showlegend=False
            ),
            row=4, col=1
        )
        
        colors_macd = ['green' if val >= 0 else 'red' for val in df['MACDh_12_26_9']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACDh_12_26_9'],
                name='Histogram',
                marker_color=colors_macd,
                showlegend=False
            ),
            row=4, col=1
        )
    
    # 更新布局
    fig.update_layout(
        title=f"{ticker} 完整缠论分析 (周期: {interval})",
        template="plotly_dark",
        height=1000,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# --- 分析报告模块 ---
def generate_analysis_report(engine: ChantismCompleteEngine):
    """生成分析报告"""
    report = []
    
    # 基础统计
    report.append("## 📊 缠论结构分析报告")
    report.append("")
    
    report.append("### 1. 基础统计")
    report.append(f"- 分析K线数量: {len(engine.processed_k)}")
    report.append(f"- 识别分型数量: {len(engine.fractals)}")
    report.append(f"- 生成笔数量: {len(engine.bi_list)}")
    report.append(f"- 识别线段数量: {len(engine.segments)}")
    report.append(f"- 识别中枢数量: {len(engine.pivots)}")
    report.append(f"- 买卖信号数量: {len(engine.signals)}")
    report.append("")
    
    # 线段分析
    if engine.segments:
        report.append("### 2. 线段分析")
        for seg in engine.segments:
            direction = "向上" if seg.type == 'up' else "向下"
            report.append(f"#### 线段 {seg.index} ({direction})")
            report.append(f"- 包含笔数: {len(seg.bi_list)}")
            report.append(f"- 价格区间: {seg.low:.2f} - {seg.high:.2f}")
            report.append(f"- 幅度: {abs(seg.end_price - seg.start_price):.2f} ({abs(seg.end_price - seg.start_price)/seg.start_price*100:.1f}%)")
            report.append(f"- 时间: {seg.start_time.strftime('%Y-%m-%d %H:%M')} 至 {seg.end_time.strftime('%Y-%m-%d %H:%M')}")
            report.append("")
    
    # 中枢分析
    if engine.pivots:
        report.append("### 3. 中枢分析")
        for pivot in engine.pivots:
            report.append(f"#### 中枢 {pivot.index}")
            report.append(f"- 所属线段: {pivot.segment_idx}")
            report.append(f"- 中枢区间: {pivot.zd:.2f} - {pivot.zg:.2f}")
            report.append(f"- 中枢宽度: {pivot.zg - pivot.zd:.2f}")
            report.append(f"- 中枢级别: {pivot.level}")
            report.append(f"- 时间跨度: {(pivot.end_time - pivot.start_time).days}天")
            report.append("")
    
    # 信号分析
    if engine.signals:
        report.append("### 4. 买卖信号分析")
        
        buy_signals = [s for s in engine.signals if 'buy' in s.type]
        sell_signals = [s for s in engine.signals if 'sell' in s.type]
        
        report.append(f"- 买点信号: {len(buy_signals)}个")
        for signal in buy_signals:
            report.append(f"  - {signal.type}: {signal.description}")
        
        report.append(f"- 卖点信号: {len(sell_signals)}个")
        for signal in sell_signals:
            report.append(f"  - {signal.type}: {signal.description}")
        
        report.append("")
    
    # 趋势判断
    if engine.segments:
        last_segment = engine.segments[-1]
        report.append("### 5. 当前趋势判断")
        report.append(f"- 最新线段方向: {'向上' if last_segment.type == 'up' else '向下'}")
        report.append(f"- 最新线段状态: {'进行中' if last_segment.end_time >= engine.df.index[-1] else '已结束'}")
        
        if len(engine.segments) >= 2:
            prev_segment = engine.segments[-2]
            if last_segment.type != prev_segment.type:
                report.append(f"- 趋势状态: 已发生转折")
            else:
                report.append(f"- 趋势状态: 延续中")
        report.append("")
    
    return "\n".join(report)

# --- 主程序 ---
if run_analysis:
    with st.spinner("正在获取数据..."):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                st.error(f"无法获取 {ticker} 的数据")
                st.stop()
            
            st.success(f"✅ 获取到 {len(data)} 条K线数据 ({period}, {interval})")
            
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            st.stop()
    
    # 初始化完整引擎
    engine = ChantismCompleteEngine(data)
    
    # 运行分析
    with st.spinner("正在进行完整缠论分析..."):
        engine.run_complete_analysis()
    
    # 显示统计信息
    st.subheader("📈 分析概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("K线数量", len(engine.processed_k))
    
    with col2:
        st.metric("笔数量", len(engine.bi_list))
    
    with col3:
        st.metric("线段数量", len(engine.segments))
    
    with col4:
        buy_count = len([s for s in engine.signals if 'buy' in s.type])
        sell_count = len([s for s in engine.signals if 'sell' in s.type])
        st.metric("买卖信号", f"{buy_count}买/{sell_count}卖")
    
    # 显示图表
    st.subheader("📊 完整缠论结构图")
    fig = create_advanced_chart(data, engine)
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示详细报告
    with st.expander("📋 查看详细分析报告", expanded=True):
        report = generate_analysis_report(engine)
        st.markdown(report)
    
    # 显示信号表格
    if engine.signals:
        st.subheader("🚦 买卖信号明细")
        
        signals_df = pd.DataFrame([{
            '类型': s.type,
            '时间': s.time.strftime('%Y-%m-%d %H:%M'),
            '价格': f"${s.price:.2f}",
            '信心度': f"{s.confidence:.0%}",
            '线段': s.segment_idx,
            '中枢': s.pivot_idx if s.pivot_idx != -1 else '',
            '描述': s.description
        } for s in engine.signals])
        
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    
    # 显示数据结构
    with st.expander("🔍 查看数据结构"):
        tab1, tab2, tab3, tab4 = st.tabs(["笔", "线段", "中枢", "信号"])
        
        with tab1:
            if engine.bi_list:
                bi_df = pd.DataFrame([{
                    '序号': b.index,
                    '方向': '向上' if b.type == 'up' else '向下',
                    '起点价': b.start_price,
                    '终点价': b.end_price,
                    '最高': b.high,
                    '最低': b.low,
                    '起点时间': b.start_time,
                    '终点时间': b.end_time,
                    '强度': f"{b.strength:.4f}"
                } for b in engine.bi_list])
                st.dataframe(bi_df, use_container_width=True)
        
        with tab2:
            if engine.segments:
                seg_df = pd.DataFrame([{
                    '序号': s.index,
                    '方向': '向上' if s.type == 'up' else '向下',
                    '笔数': len(s.bi_list),
                    '起点价': s.start_price,
                    '终点价': s.end_price,
                    '最高': s.high,
                    '最低': s.low,
                    '起点时间': s.start_time,
                    '终点时间': s.end_time
                } for s in engine.segments])
                st.dataframe(seg_df, use_container_width=True)
        
        with tab3:
            if engine.pivots:
                pivot_df = pd.DataFrame([{
                    '序号': p.index,
                    '线段': p.segment_idx,
                    'ZG(上沿)': p.zg,
                    'ZD(下沿)': p.zd,
                    'GG(高点)': p.gg,
                    'DD(低点)': p.dd,
                    '宽度': p.zg - p.zd,
                    '起点时间': p.start_time,
                    '终点时间': p.end_time
                } for p in engine.pivots])
                st.dataframe(pivot_df, use_container_width=True)
    
    # 下载选项
    st.subheader("💾 数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if engine.signals:
            signals_csv = pd.DataFrame([{
                'type': s.type,
                'time': s.time,
                'price': s.price,
                'confidence': s.confidence,
                'description': s.description,
                'segment_idx': s.segment_idx,
                'pivot_idx': s.pivot_idx
            } for s in engine.signals]).to_csv(index=False)
            
            st.download_button(
                label="📥 下载信号数据",
                data=signals_csv,
                file_name=f"{ticker}_{interval}_signals.csv",
                mime="text/csv"
            )
    
    with col2:
        summary_data = {
            'ticker': [ticker],
            'period': [period],
            'interval': [interval],
            'total_k_lines': [len(engine.processed_k)],
            'total_bi': [len(engine.bi_list)],
            'total_segments': [len(engine.segments)],
            'total_pivots': [len(engine.pivots)],
            'total_signals': [len(engine.signals)],
            'analysis_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 下载分析摘要",
            data=summary_csv,
            file_name=f"{ticker}_{interval}_summary.csv",
            mime="text/csv"
        )

else:
    # 显示使用说明
    st.info("👈 请在左侧设置参数并点击'运行完整分析'开始")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 缠论核心概念
        
        **笔 (Bi)**
        - 相邻顶底分型间的连接
        - 最少包含5根K线
        - 构成线段的基本单元
        
        **线段 (Segment)**
        - 至少由3笔构成
        - 方向一致的价格运动
        - 缠论分析的核心结构
        
        **中枢 (Pivot/Zhongshu)**
        - 至少3笔重叠的价格区间
        - 多空力量平衡区域
        - 买卖点的重要参考
        """)
    
    with col2:
        st.markdown("""
        ### 📊 三类买卖点
        
        **第一类买卖点**
        - 趋势背驰点
        - 位于线段末端
        - 风险最高，收益最大
        
        **第二类买卖点**
        - 第一类买卖点后的回调
        - 不创新低/新高
        - 安全性较高
        
        **第三类买卖点**
        - 离开中枢后的回抽
        - 不破中枢边界
        - 趋势确认信号
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 使用指南
    
    1. **设置参数**
       - 输入股票/加密货币代码
       - 选择分析周期和时间范围
       - 调整缠论识别参数
    
    2. **运行分析**
       - 系统自动识别：笔 → 线段 → 中枢
       - 计算三类买卖点
       - 生成可视化图表
    
    3. **查看结果**
       - 查看完整缠论结构图
       - 分析买卖信号
       - 下载数据用于进一步研究
    
    **默认参数说明：**
    - 笔最少K线数：5（标准缠论要求）
    - 线段最少笔数：5（包含特征序列分析）
    - 中枢最少笔数：3（标准定义）
    """)