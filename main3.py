import argparse
import math
import os
import logging
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from Env_Config_target import simple_tag_v3
from MADDPG.MADDPG import MADDPG

def setup_logger(filename):
    """Set up logger with specified format for both file and console output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(filename, mode='w')
    file_formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_env(env_name, ep_len=250, **kwargs):
    """Create environment and get observation and action dimension."""
    new_env = None
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(
            max_cycles=ep_len) if 'render_mode' not in kwargs else simple_tag_v3.parallel_env(
            max_cycles=ep_len, render_mode=kwargs['render_mode'])

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info

def test(episode, file):
    env, dim_info = get_env(args.env_name, args.episode_length, render_mode="rgb_array")
    world = env.aec_env.env.env.world
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr, result_dir, args.device)
    maddpg = maddpg.load(dim_info, file, args.device)
    obs = env.reset()
    obs = obs[0]
    frame_list = []
    is_episode_done = False
    test_steps = 0
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    while not is_episode_done and test_steps < args.episode_length:
        test_steps += 1
        actions = maddpg.select_action(obs)
        next_obs, reward, done, truncations, info = env.step(actions)
        
        frame = env.render()
        if frame is not None:
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            
            text = f'Step: {test_steps}'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            margin = 10
            x = frame.width - text_width - margin * 2
            y = margin
            
            draw.rectangle([x, y, x + text_width + margin * 2, y + text_height + margin],
                          fill='white', outline='black')
            draw.text((x + margin, y), text, font=font, fill='black')
            
            frame_list.append(frame)
        
        isdone2, flag = is_done(world)
        if isdone2:
            is_episode_done = True
            
        obs = next_obs

    if frame_list:
        gif_path = os.path.join(result_dir, f'episode_{episode}.gif')
        frame_list[0].save(
            gif_path,
            save_all=True,
            append_images=frame_list[1:],
            duration=100,
            loop=0
        )

def is_done(world):
    eva_agents = [agent for agent in world.agents if not agent.adversary]
    pur_agents = [agent for agent in world.agents if agent.adversary]
    pur = pur_agents[0]
    eva = eva_agents[0]
    selected_target = world.selected_target
    target = selected_target[0]
    flag2 = 0

    dist1 = np.sqrt(np.sum(np.square(pur.state.p_pos - eva.state.p_pos)))
    pur_vel = pur.state.p_vel
    direction = eva.state.p_pos - pur.state.p_pos
    theta_cos = vec_angle_cos(direction, pur_vel)
    theta = np.arccos(theta_cos)

    if dist1 <= pur.size + eva.size + 0.08 and 0 <= theta < 30 * math.pi/180:
        flag2 = 1
        return True, flag2
    dist = np.sqrt(np.sum(np.square(eva.state.p_pos - target.state.p_pos)))
    if dist < eva.size + target.size + 0.025:
        flag2 = 0
        return True, flag2
    return False, flag2

def vec_angle_cos(vector1, vector2):
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_val = dot_product / (magnitude1 * magnitude2)
    return cos_val


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_tag_v3', 
                       help='name of the env',
                       choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('--episode_num', type=int, default=10000, 
                       help='total episode num during training procedure')
    parser.add_argument('--save_interval', type=int, default=50, 
                       help='episode interval between saving models')
    parser.add_argument('--episode_length', type=int, default=500, 
                       help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=10, 
                       help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1000, 
                       help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, 
                       help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, 
                       help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), 
                       help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, 
                       help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, 
                       help='learning rate of critic')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='device to use', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # 创建结果保存目录
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    result_dir = os.path.join(env_dir, timestamp, 'avoid_test1')
    print(f'Create result folder: {result_dir}')
    os.makedirs(result_dir)

    # 设置日志记录器
    logger = setup_logger(os.path.join(result_dir, 'maddpg.log'))
    logger.info("Training started")

    # 保存训练参数
    with open(os.path.join(result_dir, 'training_parameters.txt'), 'w') as f:
        f.write('Training parameters:\n')
        msg = f'this is a train for 3 targets and there is one target is true'
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
        f.write(f'\n')

    # 初始化环境和智能体
    env, dim_info = get_env(args.env_name, args.episode_length)
    world = env.aec_env.env.env.world
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr, result_dir, args.device)

    # 初始化训练变量
    step = 0
    episodes_reward = []
    all_episodes_reward = []
    last_episodes_reward_ave = 0
    defender_success_count = 0
    invader_success_count = 0
    timeout_count = 0
    defender_success_steps = []
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}

    # 主训练循环
    for episode in range(args.episode_num):
        # 记录episode开始
        logger.info(f"Beginning episode {episode + 1}/{args.episode_num}")
        
        obs = env.reset()
        obs = obs[0]
        episode_steps = 0
        agent_reward = {agent_id: 0 for agent_id in env.agents}

        while env.agents and episode_steps < args.episode_length:
            step += 1
            episode_steps += 1
            
            # 每50步记录进度
            if episode_steps % 50 == 0:
                logger.info(f"Episode {episode + 1}, Step {episode_steps}/{args.episode_length}")

            # 选择动作
            if step < args.random_steps:
                actions = {agent_id: env.action_space(agent_id).sample() 
                          for agent_id in env.agents}
            else:
                actions = maddpg.select_action(obs)

            # 执行动作
            next_obs, reward, done, truncations, info = env.step(actions)
            
            # 检查是否完成
            isdone, flag = is_done(world)
            if isdone:
                if flag == 1:
                    defender_success_count += 1
                    defender_success_steps.append(episode_steps)
                    reward["adversary_0"] += 200
                    logger.info(f"Episode {episode + 1}: Defender succeeded in {episode_steps} steps")
                else:
                    invader_success_count += 1
                    reward["adversary_0"] -= 100
                    reward["agent_0"] += 200
                    logger.info(f"Episode {episode + 1}: Invader succeeded")
                done = {key: True for key in done}
                break

            # 更新经验和状态
            maddpg.add(obs, actions, reward, next_obs, done)
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # 学习更新
            if step >= args.random_steps and step % args.learn_interval == 0:
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # 处理超时情况
        if episode_steps >= args.episode_length:
            timeout_count += 1
            logger.info(f"Episode {episode + 1} timed out")

        # 记录奖励
        for agent_id, r in agent_reward.items():
            episode_rewards[agent_id][episode] = r
            if args.env_name == 'simple_tag_v3' and 'adversary' in agent_id:
                episodes_reward.append(r)
                all_episodes_reward.append(r)

        # 记录episode总结
        logger.info(f"""Episode {episode + 1} summary:
        - Steps taken: {episode_steps}
        - Defender successes so far: {defender_success_count}
        - Invader successes so far: {invader_success_count}
        - Timeouts so far: {timeout_count}
        - Current rewards: {agent_reward}
        """)



        # 定期保存模型和生成可视化
        if (episode + 1) % args.save_interval == 0:
            episodes_reward_ave = np.mean(episodes_reward)
            model = episode + 1
            
            # 保存模型
            maddpg.save(episode_rewards, model=model)
            
            # 记录模型保存信息
            if episodes_reward_ave > last_episodes_reward_ave:
                maddpg.save(episode_rewards, "best")
                last_episodes_reward_ave = episodes_reward_ave
            episodes_reward = []
            
        # 定期生成训练过程的可视化
        if (episode + 1) % 100 == 0:
            model_path = os.path.join(result_dir, f'model_{episode + 1}.pt')
            test(episode, model_path)

    print("\nTraining completed!")
    
    # 计算并保存最终统计信息
    max_reward = max(all_episodes_reward)
    min_reward = min(all_episodes_reward)
    normalized_rewards = [(r - min_reward) / (max_reward - min_reward) 
                         for r in all_episodes_reward]
    reward_variance = np.var(normalized_rewards)
    ave_reward = np.average(all_episodes_reward[-600:])
    
    # 将统计结果写入文件
    with open(result_dir + '/train_results.txt', 'w') as f:
        f.write(f'Reward Variance: {reward_variance}\n')
        f.write(f'Average Reward: {ave_reward}\n')
        f.write('\nEpisode Outcomes:\n')
        f.write(f'Defender Success Count: {defender_success_count}\n')
        f.write(f'Invader Success Count: {invader_success_count}\n')
        f.write(f'Timeout Count: {timeout_count}\n')

    def get_running_reward(arr: np.ndarray, window=100):
        """计算滑动平均奖励值"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward

    # 生成训练奖励曲线
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=f'{agent_id} (raw)')
        ax.plot(x, get_running_reward(reward), label=f'{agent_id} (smoothed)')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    title = f'Training Result of MADDPG Solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, 'training_rewards.png'))
    
    # 生成防守者成功步数分析图
    if defender_success_steps:
        fig, ax = plt.subplots()
        success_episodes = range(1, len(defender_success_steps) + 1)
        ax.plot(success_episodes, defender_success_steps)
        ax.set_xlabel('Successful Defense Episode')
        ax.set_ylabel('Steps to Success')
        ax.set_title('Steps Taken in Successful Defense Episodes')
        plt.savefig(os.path.join(result_dir, 'defender_success_steps.png'))

    print("Results saved successfully")