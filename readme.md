‖ Multi-Agent Defensive System (MADS) ‖
<div align="center">
<h2>多智能体防御拦截系统 | 基于MADDPG的集成解决方案</h2>
</div>
█ 系统概述 █
MADS是一个基于多智能体深度确定性策略梯度(MADDPG)算法的防御拦截系统，专注于解决复杂环境中的追踪-逃避任务。系统采用竞争型多智能体框架，由防御者(Adversary)和入侵者(Agent)在含障碍物的环境中进行对抗，防御者需精确拦截入侵者，入侵者则试图到达目标区域。
本系统实现了中央化训练、去中央化执行的学习范式，配备完整的训练评估工具链，适用于各类多智能体防御场景。
Show Image
█ 系统架构 █
▌代码结构
CopyProject/
├── Env_Config_target/             # 环境组件层
│   ├── core.py                    # 核心实体定义
│   ├── env_config.py              # 环境配置管理
│   ├── scenario.py                # 场景抽象基类
│   ├── simple_env.py              # 通用环境实现
│   ├── simple_tag_new.py          # 标签追踪环境实现
│   └── simple_tag_v3.py           # 环境接口导出
│
├── MADDPG/                        # 算法组件层
│   ├── Agent.py                   # 智能体类与网络结构
│   ├── Buffer.py                  # 经验回放缓冲区
│   └── MADDPG.py                  # 多智能体学习框架
│ 
└── main5.py                       # 主控执行层
▌核心类交互图
classDiagram
    class World {
        +agents[]
        +landmarks[]
        +borders[]
        +target[]
        +selected_target[]
        +step()
        +apply_action_force()
        +integrate_state()
    }
    
    class Entity {
        +name
        +size
        +movable
        +collide
        +state
        +color
    }
    
    class Agent {
        +movable
        +silent
        +action
        +u_range
        +target_selected
    }
    
    class Scenario {
        +make_world()
        +reset_world()
        +reward()
        +observation()
    }
    
    class SimpleEnv {
        +world
        +scenario
        +observe()
        +step()
        +reset()
    }
    
    Entity <|-- Agent
    Entity <|-- Landmark
    Entity <|-- Target
    Entity <|-- Selected_Target
    Entity <|-- Border
    
    World *-- Agent
    World *-- Landmark
    World *-- Target
    World *-- Selected_Target
    World *-- Border
    
    SimpleEnv o-- World
    SimpleEnv o-- Scenario
    
    BaseScenario <|-- Scenario

观测关系图
graph LR
    World((World)) -->|landmarks| Obstacle[障碍物]
    World -->|agents| Agent[逃避者]
    World -->|agents| Adversary[追捕者]
    World -->|target| Target[目标点]
    World -->|selected_target| SelectedTarget[选定目标]
    
    Agent -->|观测| Obstacle
    Agent -->|观测| Adversary
    Agent -->|观测| SelectedTarget
    
    Adversary -->|观测| Obstacle
    Adversary -->|观测| Agent
    Adversary -->|无法直接观测| SelectedTarget




▌智能体交互关系图
mermaidCopygraph TD
    A[环境观测] --> B{MADDPG协调器}
    B --> C1[防御者智能体]
    B --> C2[入侵者智能体]
    
    C1 --> D1[Actor策略网络]
    C1 --> E1[Critic价值网络]
    C2 --> D2[Actor策略网络]
    C2 --> E2[Critic价值网络]
    
    D1 --> F1[防御者动作]
    D2 --> F2[入侵者动作]
    
    F1 --> G[联合动作a]
    F2 --> G
    
    H[全局状态s] --> E1
    H --> E2
    
    G --> E1
    G --> E2
█ 系统功能 █
▌环境组件

实体定义：Agent(智能体)、Landmark(障碍物)、Target(目标点)、Border(边界)
世界物理：碰撞检测、动力学更新、状态整合
观测构建：相对位置表示、速度信息、全局/局部观测

▌算法组件

策略网络：多层感知机(MLP)架构、离散动作Gumbel-Softmax处理
价值网络：中央化状态-动作评估、批标准化
经验回放：独立智能体缓冲区、批量采样
学习机制：梯度裁剪、目标网络软更新

▌主控流程

训练循环：多智能体交互、经验存储、网络更新
评估系统：奖励跟踪、终止条件统计、效能度量
可视化：动画生成、学习曲线绘制

█ 训练流程 █
▌MADDPG训练时序图
mermaid
Copy
sequenceDiagram
    participant E as 环境(Environment)
    participant M as MADDPG框架
    participant A as 智能体网络(Agent)
    participant B as 经验缓冲区(Buffer)
    
    Note over E,B: 初始化阶段
    E->>E: 环境初始化(reset)
    M->>M: 构建智能体与缓冲区字典
    
    loop 训练循环(episode_num=5000)
        E->>E: 重置环境状态(reset)
        
        loop 单回合步骤(max=250)
            alt 随机探索阶段(random_steps<1000)
                M->>E: 请求随机动作
                E->>M: 返回随机动作
            else 策略驱动阶段
                M->>A: 请求策略动作(select_action)
                A->>M: 返回策略网络预测动作
            end
            
            M->>E: 执行动作(step)
            E->>M: 返回下一状态、奖励、终止标志
            
            M->>M: 检查终止条件(is_done)
            
            alt 终止条件满足
                M->>M: 终止回合循环
            else 继续回合
                M->>B: 存储转换样本(add)
                
                alt 学习条件满足(step>random_steps && step%learn_interval==0)
                    M->>B: 请求批量样本(sample)
                    B->>M: 返回经验批次
                    M->>A: 更新critic网络
                    M->>A: 更新actor网络
                    M->>A: 软更新目标网络(update_target)
                end
            end
        end
        
        alt 模型保存条件满足(episode%save_interval==0)
            M->>M: 保存当前模型
            M->>M: 判断是否为最佳模型
        end
        
        alt 可视化条件满足(episode%100==0)
            M->>M: 生成测试动画(test)
        end
    end
█ 核心算法 █
▌MADDPG算法原理
MADDPG算法基于Actor-Critic架构，采用中央化训练、去中央化执行范式：
步骤1: 环境交互与经验采集
    ↳ 实现位置: main5.py::170-194行
    ↳ 经验转化: (s_t, a_t, r_t, s_{t+1}, done) → Buffer
    
步骤2: 批量经验采样处理
    ↳ 实现位置: MADDPG.py::sample()方法(36-61行)
    ↳ 关键步骤: 随机索引生成 → 提取样本 → 转换为张量 → 设备转移
    
步骤3: Critic网络更新
    ↳ 实现位置: MADDPG.py::learn()方法(86-98行)
    ↳ 核心公式: L_Q = E[(Q(s,a) - y)^2], y = r + γQ'(s',a')
    
步骤4: Actor网络更新
    ↳ 实现位置: MADDPG.py::learn()方法(100-107行)
    ↳ 核心公式: L_μ = -E[Q(s,μ(s))] + λE[||μ(s)||^2]
    
步骤5: 目标网络软更新
    ↳ 实现位置: MADDPG.py::update_target()方法(109-118行)
    ↳ 核心公式: θ' ← τθ + (1-τ)θ', τ=0.01
Actor网络：输入个体观测，输出动作分布
Critic网络：输入全局观测和全局动作，输出价值估计
目标网络：提供稳定的价值评估目标

损失函数：
mathCopy\text{Critic损失函数} = \mathcal{L}(\theta^Q) = \mathbb{E}[(Q_i(x, a_1, \ldots, a_N) - y_i)^2]
其中目标值：
mathCopyy_i = r_i + \gamma Q_i^{target}(x', a_1', \ldots, a_N')
mathCopy\text{Actor损失函数} = \mathcal{L}(\theta^{\mu}) = -\mathbb{E}[Q_i(x, a_1, \ldots, \mu_i(o_i), \ldots, a_N)]
mathCopy\text{策略正则化} = \mathcal{L}_{reg} = \mathbb{E}[\|\mu_i(o_i)\|^2]
▌核心类交互图
mermaidCopy
classDiagram
    class MADDPG {
        +agents: Dict[str, Agent]
        +buffers: Dict[str, Buffer]
        +add(obs, action, reward, next_obs, done)
        +sample(batch_size)
        +select_action(obs)
        +learn(batch_size, gamma)
        +update_target(tau)
        +save(reward, model)
        +load(dim_info, file, device)
    }
    
    class Agent {
        +actor: MLPNetwork
        +critic: MLPNetwork
        +target_actor: MLPNetwork
        +target_critic: MLPNetwork
        +action(obs, model_out)
        +target_action(obs)
        +critic_value(state_list, act_list)
        +target_critic_value(state_list, act_list)
        +update_actor(loss)
        +update_critic(loss)
    }
    
    class Buffer {
        +capacity: int
        +obs: numpy.ndarray
        +action: numpy.ndarray
        +reward: numpy.ndarray
        +next_obs: numpy.ndarray
        +done: numpy.ndarray
        +add(obs, action, reward, next_obs, done)
        +sample(indices)
    }
    
    class MLPNetwork {
        +net: Sequential
        +forward(x)
    }
    
    MADDPG o-- Agent : 包含多个
    MADDPG o-- Buffer : 为每个智能体配置
    Agent *-- MLPNetwork : 策略网络
    Agent *-- MLPNetwork : 价值网络




▌离散动作处理
使用Gumbel-Softmax技术实现可微分离散采样：
pythonCopy@staticmethod
def gumbel_softmax(logits, tau=1.0, eps=1e-20):
    epsilon = torch.rand_like(logits)
    logits += -torch.log(-torch.log(epsilon + eps) + eps)
    return F.softmax(logits / tau, dim=-1)
█ 网络架构 █
▌MLPNetwork结构
CopyInput Layer [obs_dim/global_obs_dim] → 
    Linear(in, 64) → ReLU → 
    Linear(64, 64) → ReLU → 
    Linear(64, out_dim) → 
Output [act_dim/1]
pythonCopyclass MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
█ 奖励设计 █
▌防御者奖励机制
Copy防御者奖励 = -20 * dist_{防御者-入侵者} + 
             10 * cos(方向_{入侵者-防御者}, 速度_{防御者}) +
             拦截奖励(+200) + 
             入侵惩罚(-100) +
             碰撞惩罚(-30)
▌入侵者奖励机制
Copy入侵者奖励 = -15 * dist_{入侵者-目标} + 
             cos(方向_{目标-入侵者}, 速度_{入侵者}) +
             成功奖励(+200) +
             碰撞惩罚(-30)
█ 终止条件 █
▌防御成功判定
pythonCopy# 终止条件1: 防御者成功拦截
if dist1 <= pur.size + eva.size + 0.08 and 0 <= theta < 30* math.pi/180:
    flag2 = 1  # 防御成功标记
    return True, flag2
▌入侵成功判定
pythonCopy# 终止条件2: 入侵者到达目标
dist = np.sqrt(np.sum(np.square(eva.state.p_pos - target.state.p_pos)))
if dist < eva.size + target.size + 0.025:
    flag2 = 0  # 入侵成功标记
    return True, flag2
█ 使用指南 █
▌环境配置
shellCopy# 依赖安装
pip install numpy torch pygame gymnasium pettingzoo matplotlib pillow
▌运行训练
shellCopy# 使用默认参数启动训练
python main5.py

# 自定义训练参数
python main5.py --episode_num 10000 --batch_size 512 --gamma 0.99
▌核心参数
参数名称默认值说明episode_num5000训练回合数episode_length250单回合最大步数batch_size256批量采样大小buffer_capacity1000000经验池容量actor_lr0.0001Actor学习率critic_lr0.001Critic学习率gamma0.97折扣因子tau0.01软更新系数random_steps1000随机探索步数learn_interval10学习间隔
█ 分析工具 █
▌性能评估工具
系统自动生成以下分析图表：

训练奖励曲线：显示防御者与入侵者奖励变化趋势
防御效能分析：拦截成功步数分布与趋势
训练动画：每100回合生成拦截过程GIF动画

▌统计分析输出
CopyTraining complete! Final statistics:
Total Episodes: 5000
Defender Successes: 3820 (76.4%)
Invader Successes: 980 (19.6%)
Timeouts: 200 (4.0%)

Reward Variance: 0.0824
Average Reward: 156.72
█ 系统扩展 █
▌自定义环境配置
修改env_config.py中的参数可调整：

场景规模(scale)
障碍物数量(obstacle_num)
目标点数量(target_num)
智能体速度与加速度(max_speed, max_acc)

▌算法替换接口
系统设计允许更换强化学习算法，只需实现以下接口：

select_action(obs)
add(obs, action, reward, next_obs, done)
learn(batch_size, gamma)
update_target(tau)
█ 3. 算法执行链解构 █
▌MADDPG训练流程分解
步骤1：环境交互采样 [S_t, A_t, R_t, S_{t+1}]
    ↳ 实现位置：MADDPG.py::add()
    ↳ 数据流向：环境交互样本 → 各智能体Buffer
    
步骤2：经验批量采样 [Batch(S_t, A_t, R_t, S_{t+1})]
    ↳ 实现位置：MADDPG.py::sample(), Buffer.py::sample()
    ↳ 采样处理：随机索引 → 转换为Tensor → 设备转移
    
步骤3：Critic网络更新 [θ^Q ← θ^Q - α_c∇L_c]
    ↳ 实现位置：MADDPG.py::learn() → Agent.py::update_critic()
    ↳ 计算流程：当前状态价值估计 → 目标价值计算 → MSE损失 → 参数更新
    
步骤4：Actor网络更新 [θ^μ ← θ^μ - α_a∇L_a]
    ↳ 实现位置：MADDPG.py::learn() → Agent.py::update_actor()
    ↳ 计算流程：当前策略动作采样 → 动作价值评估 → 策略梯度更新
    
步骤5：目标网络软更新 [θ^{target} ← τθ + (1-τ)θ^{target}]
    ↳ 实现位置：MADDPG.py::update_target()
    ↳ 更新方式：软更新策略 → 稳定训练过程
█ 高级功能 █
▌多场景训练支持
pythonCopy# 在main5.py中修改场景初始化参数
parser.add_argument('--init-scenario', type=str, default="random", 
                    help="init scenario", choices=["random", "fixed"])
parser.add_argument('--init-agent', type=str, default="random", 
                    help="init agent mode", choices=["random", "fixed"])
parser.add_argument('--scale', type=str, default="small", 
                    help="scale of env", choices=["small", "large"])
▌训练继续加载
pythonCopy# 从已有模型继续训练
maddpg = MADDPG.load(dim_info, "results/simple_tag_v3/best_model.pt", args.device)
█ 系统优化方向 █
<Systematic Optimization>
[架构优化]
✔️ 模块化设计: 环境与算法组件解耦
✔️ 评估体系: 全面的性能监控与可视化
[优化方向]
⚠️ 超参数管理: 实现集中化配置管理
⚠️ 探索策略: 引入结构化探索机制
⚠️ 训练效率: 实现分布式训练支持
[功能扩展建议]

自适应奖励缩放: 动态调整奖励权重
分层强化学习: 分解为导航与拦截子任务
对抗性样本训练: 增强策略鲁棒性
优先经验回放: 提升样本效率
</Systematic Optimization>

