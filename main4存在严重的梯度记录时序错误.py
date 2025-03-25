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
from torch.utils.tensorboard import SummaryWriter  # ← 新增此行



def setup_logger(filename):
    """Set up separate handlers for file and console logging.
    
    File handler records detailed training progress.
    Console handler only shows important summary information.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler - detailed logging
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

def get_env(env_name, ep_len=250, **kwargs):
    """create environment and get observation and action dimension of each agent in this environment"""
    print(f"Creating environment: {env_name}, episode length: {ep_len}")
    new_env = None
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(
            max_cycles=ep_len) if 'render_mode' not in kwargs else simple_tag_v3.parallel_env(
            max_cycles=ep_len, render_mode=kwargs['render_mode'])

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        print(f"Agent {agent_id} - Obs dim: {_dim_info[agent_id][0]}, Act dim: {_dim_info[agent_id][1]}")

    return new_env, _dim_info


def test(episode, file):
    print(f"\nGenerating visualization for episode {episode}")
    env, dim_info = get_env(args.env_name, args.episode_length, render_mode="rgb_array")
    world = env.aec_env.env.env.world
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,args.actor_lr, args.critic_lr, result_dir, args.device)
    maddpg = maddpg.load(dim_info, file, args.device)
    obs = env.reset()
    obs = obs[0]
    frame_list = []
    is_episode_done = False
    test_steps = 0
    
    # 设置字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    while not is_episode_done and test_steps < args.episode_length:
        test_steps += 1
        actions = maddpg.select_action(obs)
        next_obs, reward, done, truncations, info = env.step(actions)
        
        # 获取当前帧并添加步数信息
        frame = env.render()
        if frame is not None:
            # 将numpy数组转换为PIL Image
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            
            # 添加步数文本
            text = f'Step: {test_steps}'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 设置文本位置和背景
            margin = 10
            x = frame.width - text_width - margin * 2
            y = margin
            
            # 绘制背景矩形
            draw.rectangle([x, y, x + text_width + margin * 2, y + text_height + margin], fill='white', outline='black')
            
            # 绘制文本
            draw.text((x + margin, y), text, font=font, fill='black')
            
            frame_list.append(frame)
        
        isdone2, flag = is_done(world)
     
        if isdone2:
            is_episode_done = True
            if flag == 1:
                print(f"Episode {episode} ended: Defender successfully intercepted")
            else:
                print(f"Episode {episode} ended: Invader reached target")
        elif test_steps >= args.episode_length:
            print(f"Episode {episode} ended: Maximum steps ({args.episode_length}) reached")
            
        obs = next_obs


 
        
    if frame_list:
        gif_path = os.path.join(result_dir, f'episode_{episode}.gif')
        print(f"Saving GIF to {gif_path}")
        frame_list[0].save(
            gif_path,
            save_all=True,
            append_images=frame_list[1:],
            duration=100,
            loop=0
        )
        print(f"GIF saved successfully with {len(frame_list)} frames")
    else:
        print("Warning: No frames were captured for GIF")



def is_done(world):
    # 1. 获取追击者（pur）和逃避者（eva）的智能体对象
    # - 筛选非敌对智能体（逃避者）
    eva_agents = [agent for agent in world.agents if not agent.adversary]
    # - 筛选敌对智能体（追击者）
    pur_agents = [agent for agent in world.agents if agent.adversary]
    
    # 2. 提取第一个追击者和逃避者（假设场景中各只有1个）
    pur = pur_agents[0]  # 追击者（防御方）
    eva = eva_agents[0]  # 逃避者（入侵方）
    
    # 3. 获取场景中的目标点
    selected_target = world.selected_target
    target = selected_target[0]  # 选择第一个目标点
    
    flag2=0  # 终止状态标志（1=防御成功，0=入侵成功）
    
    # 4. 计算追击者与逃避者的距离
    dist1 = np.sqrt(np.sum(np.square(pur.state.p_pos - eva.state.p_pos)))
    
    # 5. 计算追击者的速度方向与逃避者的相对位置夹角
    pur_vel = pur.state.p_vel  # 追击者的速度向量
    direction = eva.state.p_pos - pur.state.p_pos  # 逃避者的相对位置向量
    theta_cos = vec_angle_cos(direction,pur_vel)  # 计算余弦值
    theta = np.arccos(theta_cos)  # 转换为弧度值
    
    # 6. 判断防御成功条件（同时满足两个条件）：
    #    a) 距离足够近（考虑物理尺寸）
    #    b) 追击方向与逃避者的夹角小于30度
    if dist1 <= pur.size + eva.size + 0.08 and 0 <= theta < 30* math.pi/180:
        flag2=1
        return True,flag2  # 返回终止信号和防御成功标志
    
    # 7. 判断入侵成功条件：
    #    逃避者到达目标点（考虑物理尺寸）
    dist = np.sqrt(np.sum(np.square(eva.state.p_pos - target.state.p_pos)))
    if dist < eva.size + target.size + 0.025:
        flag2 = 0
        return True,flag2  # 返回终止信号和入侵成功标志
    
    return False,flag2  # 未达到终止条件
def vec_angle_cos(vector1, vector2):
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_val = dot_product / (magnitude1 * magnitude2)
    return cos_val




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_tag_v3', help='name of the env', choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('--episode_num', type=int, default=5000, help='total episode num during training procedure')
    parser.add_argument('--save_interval', type=int, default=50, help='episode interval between saving models')
    parser.add_argument('--episode_length', type=int, default=200, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=10, help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1000, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.01, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.97, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('--device', type=str, default='cuda', help='device to use', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # Create result directory
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    result_dir = os.path.join(env_dir, timestamp, 'avoid_test1')
    print(f'Create result folder: {result_dir}')
    os.makedirs(result_dir)
    
    
    
    # 初始化TensorBoard Writer  ← 新增这个代码块
    writer = SummaryWriter(log_dir=os.path.join(result_dir, 'tensorboard_logs')) 
    
    
    
    # Initialize logger
    logger = setup_logger(os.path.join(result_dir, 'maddpg.log'))
    logger.info("Training started")

    # Save training parameters
    with open(os.path.join(result_dir, 'training_parameters.txt'), 'w') as f:
        f.write('Training parameters:\n')
        msg = f'this is a train for 3 targets and there is one target is true'
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
        f.write(f'\n')

    # Initialize environment and agent
    env, dim_info = get_env(args.env_name, args.episode_length)
    world = env.aec_env.env.env.world
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, result_dir, args.device)

    # Initialize training variables
    step = 0
    episodes_reward = []
    all_episodes_reward = []
    last_episodes_reward_ave = 0
    defender_success_count = 0
    invader_success_count = 0
    timeout_count = 0
    defender_success_steps = []
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}

    print("Starting training...")
    
    # Main training loop
    for episode in range(args.episode_num):

       # 记录当前参数值  ← 新增这个代码块
        writer.add_scalar('Params/actor_lr', args.actor_lr, episode)
        writer.add_scalar('Params/critic_lr', args.critic_lr, episode) 
        writer.add_scalar('Params/tau', args.tau, episode)


        obs = env.reset()
        obs = obs[0]
        episode_steps = 0
        agent_reward = {agent_id: 0 for agent_id in env.agents}
        
        while env.agents and episode_steps < args.episode_length:
            step += 1
            episode_steps += 1

            # Select actions
            if step < args.random_steps:
                actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                actions = maddpg.select_action(obs)

            # Environment step
            next_obs, reward, done, truncations, info = env.step(actions)
            



            # Check episode termination
            isdone, flag2 = is_done(world)
            if isdone:
                if flag2 == 1:
                    defender_success_count += 1
                    defender_success_steps.append(episode_steps)
                    reward["adversary_0"] += 200
                    logger.info(f"Episode {episode}: Defender success - Steps: {episode_steps}, "
                              f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")
                else:
                    invader_success_count += 1
                    reward["adversary_0"] -= 100
                    reward["agent_0"] += 200
                    logger.info(f"Episode {episode}: Invader success - "
                              f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")
                done = {key: True for key in done}
                break

            # Update experience and rewards
            maddpg.add(obs, actions, reward, next_obs, done)
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Learning update
            if step >= args.random_steps and step % args.learn_interval == 0:
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)
            # 记录梯度信息  ← 新增
                for name, param in maddpg.agents['adversary_0'].actor.named_parameters():
                    if param.grad is not None:  # 添加空值检查
                        writer.add_histogram(f'Actor/{name}_grad', param.grad, episode)
            
            obs = next_obs
            
        # Handle timeout case
        if episode_steps >= args.episode_length:
            timeout_count += 1
            logger.info(f"Episode {episode}: Timeout - "
                       f"Rewards: [D: {agent_reward['adversary_0']:.2f}, I: {agent_reward['agent_0']:.2f}]")

        # Record rewards
        for agent_id, r in agent_reward.items():
            episode_rewards[agent_id][episode] = r
            if args.env_name == 'simple_tag_v3' and 'adversary' in agent_id:
                episodes_reward.append(r)
                all_episodes_reward.append(r)

        # Save model and evaluate
        if (episode + 1) % args.save_interval == 0:
            episodes_reward_ave = np.mean(episodes_reward)
            model = episode + 1
            maddpg.save(episode_rewards, model=model)
            
            if episodes_reward_ave > last_episodes_reward_ave:
                maddpg.save(episode_rewards, "best")
                last_episodes_reward_ave = episodes_reward_ave
            episodes_reward = []
            
        # Generate visualization
        if (episode + 1) % 100 == 0:
            model_path = os.path.join(result_dir, f'model_{episode + 1}.pt')
            test(episode, model_path)

    print("\nResults saved successfully")
    writer.close()  # ← 新增此行
    # Print final statistics
    print("\nTraining complete! Final statistics:")
    print(f"Total Episodes: {args.episode_num}")
    print(f"Defender Successes: {defender_success_count} ({defender_success_count/args.episode_num*100:.1f}%)")
    print(f"Invader Successes: {invader_success_count} ({invader_success_count/args.episode_num*100:.1f}%)")
    print(f"Timeouts: {timeout_count} ({timeout_count/args.episode_num*100:.1f}%)")

    # Save final statistics
    max_reward = max(all_episodes_reward)
    min_reward = min(all_episodes_reward)
    normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in all_episodes_reward]
    reward_variance = np.var(normalized_rewards)
    ave_reward = np.average(all_episodes_reward[-600:])
    
    with open(result_dir + '/train_results.txt', 'w') as f:
        f.write(f'Reward Variance: {reward_variance}\n')
        f.write(f'Average Reward: {ave_reward}\n')
        f.write('\nEpisode Outcomes:\n')
        f.write(f'Defender Success Count: {defender_success_count}\n')
        f.write(f'Invader Success Count: {invader_success_count}\n')
        f.write(f'Timeout Count: {timeout_count}\n')

    def get_running_reward(arr: np.ndarray, window=100):
        """Calculate running reward with specified window size"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward

    # Generate training reward plot
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
    
    # Generate defender performance plot
    if defender_success_steps:
        fig, ax = plt.subplots()
        success_episodes = range(1, len(defender_success_steps) + 1)
        ax.plot(success_episodes, defender_success_steps)
        ax.set_xlabel('Successful Defense Episode')
        ax.set_ylabel('Steps to Success')
        ax.set_title('Steps Taken in Successful Defense Episodes')
        plt.savefig(os.path.join(result_dir, 'defender_success_steps.png'))

    print("\nResults saved successfully")
