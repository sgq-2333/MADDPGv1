# secure_pickle_reader.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_rewards_safely(file_path):
    """从pickle文件安全加载奖励数据，最小化安全暴露"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict) or 'rewards' not in data:
            raise ValueError("无效的奖励数据结构")
            
        return data['rewards']
    except Exception as e:
        print(f"加载奖励文件错误: {e}")
        return None

def analyze_rewards(rewards_data):
    """执行全面的奖励轨迹分析"""
    if rewards_data is None:
        return
    
    # 提取智能体ID
    agent_ids = list(rewards_data.keys())
    print(f"发现{len(agent_ids)}个智能体的奖励数据: {agent_ids}")
    
    # 计算每个智能体的奖励统计信息
    for agent_id in agent_ids:
        rewards = rewards_data[agent_id]
        episodes = len(rewards)
        
        # 计算滚动统计数据
        window = min(100, episodes // 10)
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        print(f"\n智能体: {agent_id}")
        print(f"回合总数: {episodes}")
        print(f"平均奖励: {np.mean(rewards):.4f}")
        print(f"奖励范围: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
        print(f"奖励标准差: {np.std(rewards):.4f}")
        print(f"最终平均值(最后{window}个回合): {np.mean(rewards[-window:]):.4f}")
        
    return rewards_data

def visualize_rewards(rewards_data, save_path=None):
    """生成全面的奖励可视化"""
    if rewards_data is None:
        return
        
    agent_ids = list(rewards_data.keys())
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for agent_id in agent_ids:
        rewards = rewards_data[agent_id]
        episodes = range(1, len(rewards) + 1)
        
        # 绘制原始奖励
        ax.plot(episodes, rewards, alpha=0.3, label=f"{agent_id} (原始)")
        
        # 绘制平滑奖励
        window = min(100, len(rewards) // 10)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smoothed_episodes = range(window, len(rewards) + 1)
            ax.plot(smoothed_episodes, smoothed, 
                   label=f"{agent_id} (平滑, 窗口={window})")
    
    ax.set_xlabel('huihe')
    ax.set_ylabel('cumulative reward')
    ax.set_title('Agent reward trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"可视化已保存至 {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 替换为您的实际路径
    rewards_path = "D:/fromtlx/maddpg_pe_1_rvo_v3/results/simple_tag_v3/20250304-164625/avoid_test1/rewards.pkl"
    output_dir = os.path.dirname(rewards_path)
    
    # 加载并分析奖励
    rewards_data = load_rewards_safely(rewards_path)
    analyze_rewards(rewards_data)
    
    # 生成可视化
    visualize_rewards(rewards_data, os.path.join(output_dir, "reward_analysis.png"))