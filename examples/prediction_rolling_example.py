import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosPredictor, KronosTokenizer


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0] :]
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(
        close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(
        volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def rolling_prediction(
    predictor,
    df,
    lookback,
    total_pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
):
    """
    实现滚动预测，每次预测1个数据点，然后使用真实值更新输入上下文。

    Args:
        predictor: KronosPredictor实例
        df: 包含历史数据和实际值的DataFrame
        lookback: 用于预测的历史数据长度
        total_pred_len: 总共要预测的数据点数量
        T: 采样温度
        top_p: 核采样阈值
        sample_count: 每个预测的样本数量
        verbose: 是否显示详细信息

    Returns:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    # 用于存储所有预测结果
    all_preds = []

    # 初始化上下文窗口
    context_window = df.iloc[:lookback].copy()
    context_timestamps = df.iloc[:lookback]["timestamps"]
    right_pred_count = 0
    right_pred_count_v2 = 0
    print("开始滚动预测...")
    pred_diff_percert = []
    for i in range(total_pred_len):
        next_timestamp = pd.Series(
            df.iloc[lookback + i : lookback + i + 10]["timestamps"],
            name="timestamps",
        )
        next_timestamp = pd.to_datetime(next_timestamp)
        # 当前收盘
        current_close = context_window.iloc[-1]["close"]
        # 每次只预测1个数据点
        pred_df = predictor.predict(
            df=context_window[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=context_timestamps,
            y_timestamp=next_timestamp,
            pred_len=len(next_timestamp),  # 每次只预测1个数据点
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose and i % 10 == 0,  # 每10步打印一次进度
        )

        # 保存预测结果
        all_preds.append(pred_df.iloc[0])
        # 预测收盘
        pred_colse = pred_df.iloc[0]["close"]
        pred_close_v2=pred_df["close"].mean()
        # 使用真实值更新上下文窗口（而不是预测值）
        # 获取下一个实际数据点
        actual_next_point = df.iloc[lookback + i].copy()
        actual_close = actual_next_point["close"]
        actual_close_v2 = df.iloc[lookback + i : lookback + i + 10]["close"].mean()
        # 将实际值添加到上下文窗口
        context_window = pd.concat(
            [context_window.iloc[1:], pd.DataFrame([actual_next_point])]
        )

        # 更新时间戳
        context_timestamps = pd.concat([context_timestamps.iloc[1:], next_timestamp])
        # 计算涨跌方向：预测 vs 真实
        pred_return = 1 if pred_colse > current_close else -1
        pred_return_v2 = 1 if pred_close_v2 > current_close else -1
        actual_return = 1 if actual_close > current_close else -1
        actual_return_v2 = 1 if actual_close_v2 > current_close else -1

        direction_match = pred_return == actual_return
        right_pred_count += 1 if direction_match else 0
        
        direction_match_v2 = pred_return_v2 == actual_return_v2
        right_pred_count_v2 += 1 if direction_match_v2 else 0

        pred_diff_percert.append(abs(pred_colse - actual_close) / actual_close)
        if verbose and i % 10 == 0:
            print(f"已完成 {i + 1}/{total_pred_len} 步预测")
    # 将所有预测结果合并为一个DataFrame
    pred_df = pd.DataFrame(all_preds)
    pred_df.index = df.loc[lookback : lookback + total_pred_len - 1, "timestamps"]
    print(f"预测准确率: {right_pred_count / total_pred_len:.4f}")
    
    print(f"预测准确率_v2: {right_pred_count_v2 / total_pred_len:.4f}")
    print(
        f"预测误差百分比: {np.mean(pred_diff_percert):.4f}  max: {np.max(pred_diff_percert):.4f}  min: {np.min(pred_diff_percert):.4f}"
    )
    return pred_df


# 1. 加载模型和分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. 实例化预测器
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

# 3. 准备数据
df = pd.read_csv("./examples/data/HK_ali_09988_kline_5min_all.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])

lookback = 400  # 历史数据长度
total_pred_len = 120  # 总共预测120个数据点

# 4. 执行滚动预测
pred_df = rolling_prediction(
    predictor=predictor,
    df=df,
    lookback=lookback,
    total_pred_len=total_pred_len,
    T=1,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# 5. 可视化结果
print("\n预测数据头部:")
print(pred_df.head())

# 合并历史数据和预测结果用于绘图
# kline_df = df.loc[:lookback + total_pred_len - 1]

# 可视化
# plot_prediction(kline_df, pred_df)
