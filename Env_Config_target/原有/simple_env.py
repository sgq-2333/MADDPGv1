import os
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from Env_Config_target.core import Agent, Action
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env

class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(self, scenario, world, max_cycles, render_mode=None, continuous_actions=False, local_ratio=None):
        super().__init__()

        # 渲染相关初始化
        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # 物理模拟参数
        self.contact_force = 1e1
        self.contact_margin = 1e-2
        self.damping = 0.85
        self.dt = 0.1

        self.renderOn = False
       # 修改这里：使用np_random替代_seed
        self.np_random = np.random.RandomState()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        # 初始化智能体和实体
        self.scripted_agents = [agent for agent in self.world.agents if agent.action_callback is not None]
        self.entities = self.world.entities

        # 确保所有智能体的action都被正确初始化
        for agent in self.world.agents:
            if agent.action is None:
                agent.action = Action()
            agent.action.u = np.zeros(self.world.dim_p)
            agent.action.c = np.zeros(self.world.dim_c)

        # 初始化世界
        self.scenario.reset_world(self.world, self.np_random)
        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0
        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        """Returns the observation space for the specified agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Returns the action space for the specified agent."""
        return self.action_spaces[agent]

    
            
            




    @property
    def num_agents(self):
        return len(self.agents)

    def apply_action_force(self, p_force):
        """应用智能体的动作力"""
        for i, agent in enumerate(self.world.agents):
            if agent.movable:
                if agent.action.u is None:
                    agent.action.u = np.zeros(self.world.dim_p)
                
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

    def apply_environment_force(self, p_force):
        """应用环境力（如碰撞）"""
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def get_collision_force(self, entity_a, entity_b):
        """Calculate collision force between two entities with zero division protection."""
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        if entity_a is entity_b:
            return [None, None]

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        
         # 更强的安全检查
        if dist < 1e-3:  # 增大最小距离阈值
        # print(f"Warning: Very small distance ({dist}) between entities")
            dist = 1e-3
            delta_pos = np.array([1e-3, 1e-3])  # 设置一个默认的微小偏移
        
        dist_min = entity_a.size + entity_b.size

        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        
        # 限制力的大小
        force = np.clip(
            self.contact_force * delta_pos / dist * penetration,
            -1e3, 1e3  # 限制力的范围
        )

        # Debug输出
        if np.any(np.isnan(force)) or np.any(np.isinf(force)):
            print(f"Warning: Invalid force calculated: {force}")
            print(f"Distance: {dist}, Delta pos: {delta_pos}")
            print(f"Penetration: {penetration}")
            force = np.zeros_like(force)  # 如果出现无效值，使用零力

        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    def integrate_state(self, p_force):
        """integrate physical state"""
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue

                

            # 验证力
        if p_force[i] is not None:
            if np.any(np.isnan(p_force[i])) or np.any(np.isinf(p_force[i])):
                print(f"Warning: Invalid force for entity {i}: {p_force[i]}")
                p_force[i] = np.zeros_like(p_force[i])


            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            
            if p_force[i] is not None:
                acceleration = (p_force[i] / entity.mass) * self.dt
                # 限制加速度
                acceleration = np.clip(acceleration, -1e2, 1e2)
                entity.state.p_vel += acceleration

            # 速度限制
            if entity.max_speed is not None:
                speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
                if speed > entity.max_speed:
                    entity.state.p_vel = (entity.state.p_vel / speed) * entity.max_speed

            # 验证结果
            if np.any(np.isnan(entity.state.p_pos)) or np.any(np.isnan(entity.state.p_vel)):
                print(f"Warning: Invalid state for entity {i}")
                print(f"Position: {entity.state.p_pos}")
                print(f"Velocity: {entity.state.p_vel}")
                # 重置到安全状态
                entity.state.p_pos = np.zeros_like(entity.state.p_pos)
                entity.state.p_vel = np.zeros_like(entity.state.p_vel)


    def get_agent_target(self, agent):
        """获取防守者的目标位置"""
        if not agent.adversary:
            return None
            
        # 获取逃避者位置作为防守者的目标
        evaders = [a for a in self.world.agents if not a.adversary]
        if evaders:
            return evaders[0].state.p_pos
        return None

    def get_closest_evader(self, pursuer_pos):
        """获取最近的逃避者"""
        evaders = [a for a in self.world.agents if not a.adversary]
        if not evaders:
            return None
            
        min_dist = float('inf')
        closest_evader = None
        
        for evader in evaders:
            dist = np.sqrt(np.sum(np.square(
                pursuer_pos - evader.state.p_pos)))
            if dist < min_dist:
                min_dist = dist
                closest_evader = evader
                
        return closest_evader

    def get_evader_direction(self, pursuer):
        """获取逃避者相对方向"""
        closest_evader = self.get_closest_evader(pursuer.state.p_pos)
        if closest_evader is None:
            return np.zeros(self.world.dim_p)
            
        return closest_evader.state.p_pos - pursuer.state.p_pos

                    
    def _execute_world_step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        
        # 确保所有智能体的action.u都被初始化
        for agent in self.world.agents:
            if agent.action.u is None:
                agent.action.u = np.zeros(self.world.dim_p)
            if agent.action.c is None:
                agent.action.c = np.zeros(self.world.dim_c)

        

        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        
        # integrate physical state
        self.integrate_state(p_force)

        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def seed(self, seed=None):
            """Seeds the environment."""
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)  # 使用新的seed方法
            
        self.scenario.reset_world(self.world, self.np_random)
        # 重置所有智能体的action
        for agent in self.world.agents:
            agent.action.u = np.zeros(self.world.dim_p)
            agent.action.c = np.zeros(self.world.dim_c)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        #print(f"Step: Agent {self.agent_selection}, Action: {action}")
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            #print("Executing world step...")  # 添加调试信息
            self._execute_world_step()
            self.steps += 1
            #print(f"Step completed. Total steps: {self.steps}")  # 添加调试信息
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()    
        
        
        

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = 1.2

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= -1  # this makes the display mimic the old pyglet setup
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            
            if 'border' in entity.name:
                rect_size = entity.size * 350
                rect = pygame.Rect(x - rect_size, y - rect_size, 1.0*rect_size, 1.0*rect_size)
                pygame.draw.rect(self.screen, entity.color * 200, rect)
                
            elif 'landmark' in entity.name:
                rect_size1 = entity.size * 350
                rect = pygame.Rect(x - rect_size1, y - rect_size1, 1.0 * rect_size1, 1.0 * rect_size1)
                pygame.draw.rect(self.screen, entity.color * 200, rect)

            elif 'target' in entity.name:
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 250
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), entity.size * 250, 1
                )

            elif 'select' in entity.name:
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 250
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), entity.size * 250, 1
                )

            elif 'adversary' or 'agent' in entity.name:
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 500
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), entity.size * 500, 1)
                
                if isinstance(entity, Agent):
                    if np.linalg.norm(entity.state.p_vel) > 0:
                        vel_direction = entity.state.p_vel / np.linalg.norm(entity.state.p_vel)
                        vel_pos = (entity.state.p_pos)
                        vel_start = (x,y)
                        x_e = vel_pos[0] + vel_direction[0]*0.065
                        y_e = vel_pos[1] + vel_direction[1]*0.065
                        vel_end_0 = (x_e,y_e)
                        x_e = (x_e / cam_range) * self.width // 2 * 0.9
                        y_e *= -1
                        y_e = (y_e / cam_range) * self.height // 2 * 0.9
                        x_e += self.width // 2
                        y_e += self.height // 2
                        vel_end = (x_e, y_e)

                        pygame.draw.aaline(self.screen, (0, 0, 0, 128), vel_start, vel_end, 2)

                        # 计算箭头两个斜边的端点
                        arrow_length = 10
                        arrow_angle = np.pi / 6
                        x1_end = vel_end_0[0] - arrow_length * np.cos(np.arctan2(vel_direction[1], vel_direction[0]) - arrow_angle)*0.003
                        y1_end = vel_end_0[1] - arrow_length * np.sin(np.arctan2(vel_direction[1], vel_direction[0]) - arrow_angle)*0.003
                        x2_end = vel_end_0[0] - arrow_length * np.cos(np.arctan2(vel_direction[1], vel_direction[0]) + arrow_angle)*0.003
                        y2_end = vel_end_0[1] - arrow_length * np.sin(np.arctan2(vel_direction[1], vel_direction[0]) + arrow_angle)*0.003

                        x1_end = (x1_end / cam_range) * self.width // 2 * 0.9
                        y1_end *= -1
                        y1_end = (y1_end / cam_range) * self.height // 2 * 0.9
                        x1_end += self.width // 2
                        y1_end += self.height // 2
                        arrow1_end = (x1_end,y1_end)

                        x2_end = (x2_end / cam_range) * self.width // 2 * 0.9
                        y2_end *= -1
                        y2_end = (y2_end / cam_range) * self.height // 2 * 0.9
                        x2_end += self.width // 2
                        y2_end += self.height // 2
                        arrow2_end = (x2_end,y2_end)

                        pygame.draw.aaline(self.screen, (0, 0, 0, 128), vel_end, arrow1_end, 2)
                        pygame.draw.aaline(self.screen, (0, 0, 0, 128), vel_end, arrow2_end, 2)

            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False