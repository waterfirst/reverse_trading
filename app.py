import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product  # 이 줄을 추가합니다
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta  # datetime 모듈을 import 추가

# 기존 함수들은 그대로 유지
def get_multi_asset_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        if ticker.endswith('=X'):
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']
        else:
            df = yf.download(ticker, start=start_date, end=end_date)['Close']
        df.index = df.index.tz_localize(None)
        data[ticker] = df
    combined_data = pd.concat(data.values(), axis=1, keys=data.keys())
    combined_data.index.name = 'Date'
    return combined_data

def preprocess_data(data):
    data = data.dropna()
    return data / data.iloc[0]

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_correlation(data, window=30):
    return data.pct_change().rolling(window=window).corr()

def advanced_reverse_trading_strategy(data, rsi_period=14, rsi_overbought=70, rsi_oversold=30, corr_threshold=0.7):
    signals = pd.DataFrame(0, index=data.index, columns=data.columns)
    for asset in data.columns:
        rsi = calculate_rsi(data[asset], rsi_period)
        signals.loc[rsi < rsi_oversold, asset] = 1  # 매수 신호
        signals.loc[rsi > rsi_overbought, asset] = -1  # 매도 신호
    
    correlations = calculate_correlation(data)
    for date in signals.index:
        corr_matrix = correlations.loc[date]
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns):
                if i < j:
                    corr = corr_matrix.loc[asset1, asset2]
                    if abs(corr) > corr_threshold:
                        if corr > 0:
                            signals.loc[date, asset2] = -signals.loc[date, asset1]
                        else:
                            signals.loc[date, asset2] = signals.loc[date, asset1]
    
    return signals

def backtest_multi_asset_strategy(data, signals):
    returns = data.pct_change()
    strategy_returns = (signals.shift(1) * returns).sum(axis=1)
    cumulative_returns = (1 + returns).cumprod()
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    return cumulative_returns, cumulative_strategy_returns, sharpe_ratio

def evaluate_params(args):
    data, rsi_period, rsi_overbought, rsi_oversold, corr_threshold = args
    signals = advanced_reverse_trading_strategy(data, rsi_period, rsi_overbought, rsi_oversold, corr_threshold)
    _, _, sharpe_ratio = backtest_multi_asset_strategy(data, signals)
    return (rsi_period, rsi_overbought, rsi_oversold, corr_threshold), sharpe_ratio

def optimize_strategy(data, rsi_periods, rsi_overboughts, rsi_oversolds, corr_thresholds):
    with ProcessPoolExecutor() as executor:
        args = ((data, rsi_period, rsi_overbought, rsi_oversold, corr_threshold)
                for rsi_period, rsi_overbought, rsi_oversold, corr_threshold 
                in product(rsi_periods, rsi_overboughts, rsi_oversolds, corr_thresholds))
        results = list(executor.map(evaluate_params, args))
    
    best_params, best_sharpe = max(results, key=lambda x: x[1])
    return best_params

# Streamlit 앱 시작
st.title('Multi-Asset Reverse Trading Strategy App')

# 사이드바에 사용자 입력 추가
st.sidebar.header('Parameters')
tickers = st.sidebar.text_input('Tickers (comma-separated)', "^GSPC,^KS11,KRW=X,JPY=X").split(',')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', datetime.now().date())

optimize = st.sidebar.checkbox('Optimize Strategy Parameters')

if optimize:
    st.sidebar.subheader('Optimization Ranges')
    rsi_period_range = st.sidebar.slider('RSI Period Range', 5, 30, (10, 20), step=5)
    rsi_overbought_range = st.sidebar.slider('RSI Overbought Range', 50, 90, (65, 75), step=5)
    rsi_oversold_range = st.sidebar.slider('RSI Oversold Range', 10, 50, (25, 35), step=5)
    corr_threshold_range = st.sidebar.slider('Correlation Threshold Range', 0.0, 1.0, (0.6, 0.8), step=0.1)
else:
    rsi_period = st.sidebar.slider('RSI Period', 5, 30, 14)
    rsi_overbought = st.sidebar.slider('RSI Overbought', 50, 90, 70)
    rsi_oversold = st.sidebar.slider('RSI Oversold', 10, 50, 30)
    corr_threshold = st.sidebar.slider('Correlation Threshold', 0.0, 1.0, 0.7)

# 데이터 가져오기 및 전처리
if st.button('Run Strategy'):
    with st.spinner('Fetching and processing data...'):
        multi_asset_data = get_multi_asset_data(tickers, start_date, end_date)
        processed_data = preprocess_data(multi_asset_data)

    # 최적화 또는 전략 적용
    if optimize:
        with st.spinner('Optimizing strategy parameters...'):
            rsi_periods = range(rsi_period_range[0], rsi_period_range[1]+1, 5)
            rsi_overboughts = range(rsi_overbought_range[0], rsi_overbought_range[1]+1, 5)
            rsi_oversolds = range(rsi_oversold_range[0], rsi_oversold_range[1]+1, 5)
            corr_thresholds = np.arange(corr_threshold_range[0], corr_threshold_range[1]+0.1, 0.1)
            
            best_params = optimize_strategy(processed_data, rsi_periods, rsi_overboughts, rsi_oversolds, corr_thresholds)
            st.write(f"Optimized parameters: RSI Period = {best_params[0]}, RSI Overbought = {best_params[1]}, RSI Oversold = {best_params[2]}, Correlation Threshold = {best_params[3]:.2f}")
            
            signals = advanced_reverse_trading_strategy(processed_data, *best_params)
    else:
        with st.spinner('Applying strategy...'):
            signals = advanced_reverse_trading_strategy(processed_data, rsi_period, rsi_overbought, rsi_oversold, corr_threshold)

    cumulative_returns, cumulative_strategy_returns, sharpe_ratio = backtest_multi_asset_strategy(processed_data, signals)

    # 결과 시각화
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Asset Performance', 'Strategy Performance'))

    for asset in tickers:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[asset], name=f'{asset} Buy & Hold', mode='lines'), row=1, col=1)
        
        # 매수/매도 신호 추가
        buy_signals = signals.index[signals[asset] == 1]
        sell_signals = signals.index[signals[asset] == -1]
        
        fig.add_trace(go.Scatter(x=buy_signals, y=cumulative_returns.loc[buy_signals, asset],
                                 mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name=f'{asset} Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals, y=cumulative_returns.loc[sell_signals, asset],
                                 mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name=f'{asset} Sell Signal'), row=1, col=1)

    fig.add_trace(go.Scatter(x=cumulative_strategy_returns.index, y=cumulative_strategy_returns, name='Strategy Returns', mode='lines'), row=2, col=1)

    fig.update_layout(height=800, title_text="Strategy Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

    # 결과 출력
    st.subheader('Performance Summary')
    for asset in tickers:
        st.write(f"{asset} Buy and Hold 수익률: {cumulative_returns[asset].iloc[-1]:.2f}")
    st.write(f"Multi-Asset Reverse Trading 전략 수익률: {cumulative_strategy_returns.iloc[-1]:.2f}")
    st.write(f"전략 Sharpe Ratio: {sharpe_ratio:.2f}")

# 앱 사용 설명
st.sidebar.markdown("""
## How to use this app:
1. Enter ticker symbols separated by commas
2. Select date range
3. Choose whether to optimize strategy parameters
4. Adjust strategy parameters or optimization ranges
5. Click 'Run Strategy' button
6. View results, performance chart, and buy/sell signals
""")

