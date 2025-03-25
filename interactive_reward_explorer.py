# interactive_reward_explorer.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import os

def load_rewards(file_path):
    """从pickle文件加载奖励数据"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['rewards']

def convert_to_dataframe(rewards_data):
    """将奖励字典转换为pandas DataFrame以便分析"""
    # 初始化字典来保存所有奖励
    all_rewards = {}
    
    # 找出最大回合数
    max_episodes = max(len(rewards) for rewards in rewards_data.values())
    
    # 创建以回合为索引的DataFrame
    for agent_id, rewards in rewards_data.items():
        # 如果需要，将奖励补齐到最大长度
        padded_rewards = rewards + [np.nan] * (max_episodes - len(rewards))
        all_rewards[agent_id] = padded_rewards
    
    # 创建DataFrame
    df = pd.DataFrame(all_rewards)
    df.index.name = '回合'
    df.index += 1  # 回合从1开始，而非0
    
    return df

def analyze_rewards_interactive(file_path):
    """提供奖励数据的交互式分析"""
    try:
        # 加载数据
        rewards_data = load_rewards(file_path)
        
        # 转换为DataFrame
        df = convert_to_dataframe(rewards_data)
        
        # 显示基本统计信息
        print("=== 奖励数据概览 ===")
        print(f"总回合数: {len(df)}")
        print(f"智能体: {', '.join(df.columns)}")
        
        # 显示前几行
        print("\n=== 前5个回合 ===")
        display(df.head())
        
        # 显示后几行
        print("\n=== 后5个回合 ===")
        display(df.tail())
        
        # 基本统计信息
        print("\n=== 奖励统计 ===")
        display(df.describe())
        
        # 滚动窗口统计
        window_size = min(100, len(df) // 10)
        rolling = df.rolling(window=window_size).mean()
        
        # 绘制交互式图表
        plt.figure(figsize=(14, 8))
        
        # 半透明原始数据
        for column in df.columns:
            plt.plot(df.index, df[column], alpha=0.3, label=f"{column} (原始)")
        
        # 平滑数据
        for column in rolling.columns:
            plt.plot(rolling.index, rolling[column], 
                    linewidth=2, label=f"{column} (平滑)")
        
        plt.title(f"奖励轨迹 (窗口大小: {window_size})")
        plt.xlabel("回合")
        plt.ylabel("奖励")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        output_dir = os.path.dirname(file_path)
        plt.savefig(os.path.join(output_dir, "interactive_reward_analysis.png"))
        plt.show()
        
        return df, rolling
        
    except Exception as e:
        print(f"分析错误: {e}")
        return None, None

# 使用示例
# df, rolling = analyze_rewards_interactive("D:/fromtlx/maddpg_pe_1_rvo_v3/results/simple_tag_v3/20250228-170047/avoid_test1/rewards.pkl")