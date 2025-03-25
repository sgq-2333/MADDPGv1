import argparse
import math
import os
import logging
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from Env_Config_target import simple_tag_v3
from pettingzoo.mpe.simple_tag.simple_tag import Scenario

from MADDPG.MADDPG import MADDPG

def setup_logger(filename):
    """配置双日志处理器（当前实现只有文件日志）
    
    文件日志 - 记录完整的训练过程细节
    （注：当前代码中控制台日志处理器已被注释/移除）
    """
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置基础日志级别

    # 文件日志处理器配置
    file_handler = logging.FileHandler(filename, mode='w')  # 覆盖写入模式
    file_handler.setLevel(logging.INFO)  # 设置文件日志级别
    # 定义日志格式：时间戳 - 日志级别 - 消息内容
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)  # 应用格式
    
    logger.addHandler(file_handler)  # 添加处理器到根日志器

    return logger  # 返回配置好的日志器

def get_env(env_name, ep_len=250, **kwargs):
    """创建环境并获取智能体维度信息"""
    # 环境初始化
    print(f"Creating environment: {env_name}, episode length: {ep_len}")
    new_env = None
    if env_name == 'simple_tag_v3':
        # 根据是否包含render_mode参数创建不同配置的环境
        new_env = simple_tag_v3.parallel_env(
            max_cycles=ep_len) if 'render_mode' not in kwargs else simple_tag_v3.parallel_env(
            max_cycles=ep_len, render_mode=kwargs['render_mode'])
    
    new_env.reset()  # 重置环境获取初始状态
    
    # 构建维度信息字典
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        # 获取观测空间维度
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        # 获取动作空间维度
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        print(f"Agent {agent_id} - Obs dim: {_dim_info[agent_id][0]}, Act dim: {_dim_info[agent_id][1]}")

    return new_env, _dim_info

def test(episode, file):
    # 测试流程初始化
    print(f"\nGenerating visualization for episode {episode}")
    # 创建带渲染功能的环境
    env, dim_info = get_env(args.env_name, args.episode_length, render_mode="rgb_array")
    world = env.aec_env.env.env.world  # 获取底层世界对象
    
    # 初始化MADDPG算法并加载预训练模型
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, result_dir, args.device)
    maddpg = maddpg.load(dim_info, file, args.device)
    
    # 环境重置与初始化
    obs = env.reset()
    obs = obs[0]  # 获取初始观察值（假设单智能体场景）
    frame_list = []  # 存储动画帧
    is_episode_done = False
    test_steps = 0

    # 字体设置（优先使用arial，失败则用默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # 主测试循环
    while not is_episode_done and test_steps < args.episode_length:
        test_steps += 1
        
        # 1. 生成智能体动作
        actions = maddpg.select_action(obs)# 神经网络决策
        
        # 2. 执行环境步进
        next_obs, reward, done, truncations, info = env.step(actions)
        
        # 3. 画面渲染与标注
        frame = env.render()
        if frame is not None:
            # 转换numpy数组为PIL图像
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            
            # 添加步数标注
            text = f'Step: {test_steps}'
            # 计算文本边界框
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 设置标注位置（右上角）
            margin = 10
            x = frame.width - text_width - margin * 2
            y = margin
            
            # 绘制背景框
            draw.rectangle([x, y, x + text_width + margin*2, y + text_height + margin], fill='white', outline='black')
            
            # 添加文本
            draw.text((x + margin, y), text, font=font, fill='black')
            frame_list.append(frame)

        # 调用环境终止条件检测函数
        isdone2, flag = is_done(world)  # 返回 (是否终止, 终止类型标识)

        # 环境终止处理
        if isdone2:
            is_episode_done = True  # 标记回合结束
            if flag == 1:  # 根据终止类型打印不同信息
                print(f"Episode {episode} ended: Defender successfully intercepted")
            else:
                print(f"Episode {episode} ended: Invader reached target")

        # 步数限制处理        
        elif test_steps >= args.episode_length:  # 超过预设的最大步数
            print(f"Episode {episode} ended: Maximum steps ({args.episode_length}) reached")

        # 更新观察值
        obs = next_obs  # 将下一时刻的观察值赋给当前观察，准备下一次循环

 
        

    # 检查是否存在有效帧序列
    if frame_list:
        # 构造GIF保存路径（在结果目录中按episode编号命名）
        gif_path = os.path.join(result_dir, f'episode_{episode}.gif')
        print(f"Saving GIF to {gif_path}")
        
        # 使用PIL库保存GIF动画
        frame_list[0].save(  # 以第一帧为基准
            gif_path,
            save_all=True,       # 保存所有帧
            append_images=frame_list[1:],  # 追加剩余帧
            duration=100,        # 每帧持续时间（毫秒）
            loop=0               # 循环次数（0=无限循环）
        )
        print(f"GIF saved successfully with {len(frame_list)} frames")
    else:
        # 无帧捕获时的错误处理
        print("Warning: No frames were captured for GIF")


def is_done(world):
    # 区分追击者（adversary）和逃避者（普通agent）

    eva_agents = [agent for agent in world.agents if not agent.adversary]    # 使用列表推导式过滤非对抗型智能体（逃避者）
    pur_agents = [agent for agent in world.agents if agent.adversary]    # 筛选对抗型智能体（追击者）
    pur = pur_agents[0]  # 取第一个追击者
    eva = eva_agents[0]  # 取第一个逃避者
    selected_target = world.selected_target  # 获取环境中的目标点
    target = selected_target[0]              # 取第一个目标点
    flag2=0  # 终止标志（0: 入侵者成功，1: 防御者成功）

    # 计算追击者与逃避者的欧氏距离
    dist1 = np.sqrt(np.sum(np.square(pur.state.p_pos - eva.state.p_pos)))
    
    # 获取追击者速度向量和追击方向
    pur_vel = pur.state.p_vel  # 追击者速度向量
    direction = eva.state.p_pos - pur.state.p_pos  # 追击方向向量
    
    # 计算速度向量与追击方向的夹角余弦值
    theta_cos = vec_angle_cos(direction, pur_vel)
    theta = np.arccos(theta_cos)  # 转换为弧度

    # 终止条件1：成功拦截（距离阈值+角度限制）
    if dist1 <= pur.size + eva.size + 0.08 and 0 <= theta < 30* math.pi/180:
        flag2=1
        return True,flag2  # 防御者成功拦截
    
    # 终止条件2：到达目标点
    dist = np.sqrt(np.sum(np.square(eva.state.p_pos - target.state.p_pos)))
    if dist < eva.size + target.size + 0.025:
        flag2 = 0
        return True,flag2  # 入侵者到达目标
    
    return False,flag2  # 未触发终止条件

def vec_angle_cos(vector1, vector2):
    # 零向量保护机制
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0
    # 计算点积（向量投影乘积）
    dot_product = np.dot(vector1, vector2)
    # 计算向量模长
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    # 余弦值计算公式：cosθ = 点积 / (模长乘积)
    cos_val = dot_product / (magnitude1 * magnitude2)

    return cos_val




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境配置
    parser.add_argument('--env_name', type=str, default='simple_tag_v3', help='name of the env', choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])# 选择训练环境（默认追捕环境）
    parser.add_argument('--episode_num', type=int, default=5000, help='total episode num during training procedure')# 总训练回合数（5000个episode）
    # 训练流程控制
    parser.add_argument('--save_interval', type=int, default=50, help='episode interval between saving models')# 模型保存间隔（每50个episode保存一次）
    parser.add_argument('--episode_length', type=int, default=250, help='steps per episode')# 单个episode最大步数（250步）
    parser.add_argument('--learn_interval', type=int, default=10, help='steps interval between learning time')# 学习间隔（每10步更新一次网络）
    parser.add_argument('--random_steps', type=int, default=1000, help='random steps before the agent start to learn') # 初始随机探索步数（前1000步不学习）
    # 算法超参数
    parser.add_argument('--tau', type=float, default=0.01, help='soft update parameter')# 目标网络软更新系数（0.01）
    parser.add_argument('--gamma', type=float, default=0.97, help='discount factor')# 未来奖励折扣因子（0.97）
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')# 经验回放池容量（1,000,000条经验）
    parser.add_argument('--batch_size', type=int, default=256, help='batch-size of replay buffer')# 采样批次大小（128）
    # 优化器参数
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='learning rate of actor') # Actor网络学习率（0.0001）
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')# Critic网络学习率（0.001）
    # 训练设备选择（优先使用CUDA）

    parser.add_argument('--device', type=str, default='cuda', help='device to use', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # 1. 实验结果存储体系初始化
    env_dir = os.path.join('./results', args.env_name)# 创建环境专属目录
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    # 2. 实验版本管理（时间戳 + 固定测试名称）
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) # 生成精确到秒的时间标识
    result_dir = os.path.join(env_dir, timestamp, 'avoid_test1')# 三级目录结构
    print(f'Create result folder: {result_dir}')
    os.makedirs(result_dir)

    # 3. 日志系统初始化（记录训练过程细节）
    logger = setup_logger(os.path.join(result_dir, 'maddpg.log'))# 创建日志文件
    logger.info("Training started") # 记录启动事件

    # 4. 训练参数持久化（用于实验复现
    with open(os.path.join(result_dir, 'training_parameters.txt'), 'w') as f:
        f.write('Training parameters:\n')
        msg = f'this is a train for 3 targets and there is one target is true'
        for arg in vars(args):# 遍历所有命令行参数
            f.write(f'{arg}: {getattr(args, arg)}\n')# 保存参数键值对
        f.write(f'\n')

    # 5. 算法与环境初始化（MADDPG核心组件）
    env, dim_info = get_env(args.env_name, args.episode_length)# 创建并行环境
    world = env.aec_env.env.env.world
    maddpg = MADDPG(
    dim_info,              # 智能体观测/动作空间维度 
    args.buffer_capacity, # 经验回放池容量（1e6）
    args.batch_size,      # 采样批次大小（256）
    args.actor_lr,        # Actor网络学习率（0.0001）
    args.critic_lr,       # Critic网络学习率（0.001）
    result_dir,           # 结果保存路径
    args.device           # 训练设备（GPU/CPU）
    )

    # 6. 训练指标初始化
    # 全局训练计数器
    step = 0                     # 总训练步数（跨所有episode）

    # 奖励跟踪系统
    episodes_reward = []         # 当前间隔周期内的奖励缓存（用于模型保存判断）
    all_episodes_reward = []     # 全局奖励记录（用于最终统计分析）
    last_episodes_reward_ave = 0  # 最近平均奖励（用于最佳模型判断）

    # 对抗结果统计
    defender_success_count = 0  # 防御者成功拦截次数
    invader_success_count = 0   # 入侵者到达目标次数
    timeout_count = 0           # 超时未决出胜负次数

    # 效能分析数据
    defender_success_steps = []  # 记录每次成功拦截的步数（用于分析拦截效率）
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}# 创建奖励存储矩阵：agents数量 × episode_num维度（5000）

    print("Starting training...")
    
    # 主训练循环结构
    for episode in range(args.episode_num): # 遍历每个训练回合
        # 1. 回合初始化
        obs = env.reset()# 重置环境状态
        obs = obs[0]
        episode_steps = 0# 当前回合步数清零
        agent_reward = {agent_id: 0 for agent_id in env.agents}# 初始化每个智能体的奖励为0

        # 2. 单回合循环
        while env.agents and episode_steps < args.episode_length:
            step += 1
            episode_steps += 1

            # 3. 动作决策模块
            if step < args.random_steps:
                actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}# 随机探索阶段
            else:
                actions = maddpg.select_action(obs)# 策略网络决策阶段

            # 4. 环境交互与学习
            next_obs, reward, done, truncations, info = env.step(actions) # 执行动作
            
            # 环境终止条件检测（调用自定义判断函数）
            isdone, flag2 = is_done(world) # 返回两个值：是否终止、终止类型标识

            # 5. 奖励处理与记录
  
            if isdone:
                if flag2 == 1:  # 防御者成功拦截
                    defender_success_count += 1  # 防御成功计数器+1
                    defender_success_steps.append(episode_steps)  # 记录成功时步数
                    reward["adversary_0"] += 200  # 给防御者额外奖励（强化正确行为）
                    logger.info(f"Episode {episode}: Defender success - Steps: {episode_steps}, "
                              f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")  # 记录防御成功日志（含详细奖励信息）
                else:  # 入侵者成功到达
                    invader_success_count += 1   # 入侵成功计数器+1
                    reward["adversary_0"] -= 100  # 惩罚防御者（抑制失败行为） 
                    reward["agent_0"] += 200     # 奖励入侵者（强化成功策略）
                    logger.info(f"Episode {episode}: Invader success - "
                              f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")  # 记录入侵成功日志
                done = {key: True for key in done}  # 强制标记所有智能体为终止状态
                break  # 提前终止当前回合

            # 经验存储模块（用于后续神经网络训练）
            maddpg.add(obs, actions, reward, next_obs, done)  # 将当前transition存入经验池
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r  # 累加智能体当前回合的总奖励

            # 策略更新模块（满足条件时进行学习）
            if step >= args.random_steps and step % args.learn_interval == 0:
                maddpg.learn(args.batch_size, args.gamma)  # 从经验池采样并更新网络
                maddpg.update_target(args.tau)  # 目标网络软更新（tau=0.01）

            obs = next_obs  # 将下一时刻的观察值传递到下一个时间步

            
        # 超时处理逻辑
        if episode_steps >= args.episode_length:# 达到最大步数限制
            timeout_count += 1 # 超时计数器递增
            logger.info(f"Episode {episode}: Timeout - "
                       f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")# 记录超时日志（含双方奖励值）

        # 奖励记录系统
        for agent_id, r in agent_reward.items():
            episode_rewards[agent_id][episode] = r # 存储每个agent的回合奖励
            if args.env_name == 'simple_tag_v3' and 'adversary' in agent_id:# 特定环境的特殊处理
                episodes_reward.append(r)# 添加到当前统计窗口
                all_episodes_reward.append(r)# 添加到全局统计池

        # 模型保存机制
        if (episode + 1) % args.save_interval == 0:  # 每N个episode保存一次
            episodes_reward_ave = np.mean(episodes_reward)  # 计算当前窗口期平均奖励
            model = episode + 1  # 生成模型版本号（基于episode数）
            maddpg.save(episode_rewards, model=model)  # 保存当前模型
            
            # 最佳模型判断逻辑
            if episodes_reward_ave > last_episodes_reward_ave:  # 优于历史最佳
                maddpg.save(episode_rewards, "best")  # 保存为最佳模型
                last_episodes_reward_ave = episodes_reward_ave  # 更新最佳基准值
            episodes_reward = []  # 重置统计窗口

        # 可视化生成模块
        if (episode + 1) % 100 == 0:  # 每100个episode生成测试动画
            model_path = os.path.join(result_dir, f'model_{episode + 1}.pt')  # 构造模型文件路径
            test(episode, model_path)  # 调用测试函数生成GIF
 

    # 控制台统计输出模块
    print("\nTraining complete! Final statistics:")  # 输出训练完成标识
    print(f"Total Episodes: {args.episode_num}")  # 显示总训练回合数
    # 输出防御者/入侵者成功率及超时率（带百分比）
    print(f"Defender Successes: {defender_success_count} ({defender_success_count/args.episode_num*100:.1f}%)")
    print(f"Invader Successes: {invader_success_count} ({invader_success_count/args.episode_num*100:.1f}%)")
    print(f"Timeouts: {timeout_count} ({timeout_count/args.episode_num*100:.1f}%)")

    # 结果文件存储模块
    max_reward = max(all_episodes_reward)  # 全局最大奖励值
    min_reward = min(all_episodes_reward)  # 全局最小奖励值
    normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in all_episodes_reward]# 归一化奖励值（0-1区间）
    reward_variance = np.var(normalized_rewards)  # 奖励方差（稳定性指标）
    ave_reward = np.average(all_episodes_reward[-600:])  # 最后600回合平均（收敛性指标）

    # 生成训练结果文件
    with open(result_dir + '/train_results.txt', 'w') as f:
        f.write(f'Reward Variance: {reward_variance}\n')  # 奖励波动性
        f.write(f'Average Reward: {ave_reward}\n')       # 后期平均表现
        f.write('\nEpisode Outcomes:\n')                # 事件统计标题
        f.write(f'Defender Success Count: {defender_success_count}\n')  # 防御成功次数
        f.write(f'Invader Success Count: {invader_success_count}\n')     # 入侵成功次数
        f.write(f'Timeout Count: {timeout_count}\n')     # 超时事件次数

    def get_running_reward(arr: np.ndarray, window=100):
        """计算指定窗口大小的滑动平均奖励"""
        running_reward = np.zeros_like(arr)  # 创建与输入数组相同形状的全零数组
        # 前window-1个元素的特殊处理（逐步扩大窗口）
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])  # 用前i+1个元素的均值填充
        # 完整窗口期的处理
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])  # 取最近window个元素的均值
        return running_reward
    # Generate training reward plot


    # 训练奖励曲线生成
    fig, ax = plt.subplots()  # 创建画布和坐标轴
    x = range(1, args.episode_num + 1)  # 生成x轴数据（1~总回合数）

    # 遍历所有智能体的奖励数据
    for agent_id, reward in episode_rewards.items():
        # 绘制原始奖励曲线（半透明细线）
        ax.plot(x, reward, alpha=0.3, linewidth=0.7, label=f'{agent_id} (raw)')  
        # 叠加滑动平均曲线（实线）
        ax.plot(x, get_running_reward(reward), linewidth=1.5, label=f'{agent_id} (smoothed)')

    ax.legend()  # 显示图例
    ax.set_xlabel('Episode')  # x轴标签
    ax.set_ylabel('Reward')   # y轴标签
    title = f'Training Result of MADDPG Solve {args.env_name}'  # 图表标题
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, 'training_rewards.png'))  # 保存为PNG

    # 防御者效能分析图（仅在成功时生成）
    if defender_success_steps:  # 检查是否有成功记录
        fig, ax = plt.subplots()  # 新建画布
        success_episodes = range(1, len(defender_success_steps) + 1)  # x轴为成功次数序号
        
        # 绘制散点+趋势线（反映拦截效率变化）
        ax.scatter(success_episodes, defender_success_steps, s=10, alpha=0.5, label='single performance')
        ax.plot(success_episodes, defender_success_steps, color='orange', linewidth=1, label='Efficiency Trends')
        
        ax.set_xlabel('Successful Defense Episode')
        ax.set_ylabel('Steps to Success')
        ax.set_title('Steps Taken in Successful Defense Episodes')
        ax.legend()
        plt.savefig(os.path.join(result_dir, 'defender_success_steps.png'))

    print("\nResults saved successfully")