# noqa
"""
# Simple Tag

```{figure} mpe_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_tag_v3`                 |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
import numpy as np
from gymnasium.utils import EzPickle
import random
from Env_Config_target.core import Agent, Landmark, World, Border,Target,Selected_Target
from Env_Config_target.scenario import BaseScenario
from Env_Config_target.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from Env_Config_target.env_config import setup_logger, get_args, EnvConfig
import argparse

class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            num_good=1,
            num_adversaries=1,
            num_obstacles=5,
            num_select=1,
            num_targets=3,
            max_cycles=250,
            num_borders=80,  # (20 * 4) 4条边，每边20个border
            continuous_actions=False,
            render_mode=None,
            cfgs=EnvConfig(),
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_targets=num_targets,
            max_cycles=max_cycles,
            num_select=num_select,
            num_borders=num_borders,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles,num_targets,num_select,num_borders,cfgs)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=1, num_obstacles=5,num_targets=3,num_select=1,num_borders=80,cfgs=EnvConfig()):
        # def make_world(self, num_good=1, num_adversaries=1, num_obstacles=2,num_targets=3,num_borders=80,num_select=1):
        self.cfgs = cfgs
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_target = num_targets
        world.cfgs=cfgs
        num_select = num_select

        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        num_border = num_borders
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = cfgs.pursuer_size if agent.adversary else cfgs.evader_size
            # agent.size = 0.5 / 250 if agent.adversary else 0.5 / 250
            # agent.accel = 20 / 250 if agent.adversary else 20 / 250
            # agent.max_speed = 10 / 250 if agent.adversary else 10 / 2500
            agent.accel = cfgs.pursuer_max_acc if agent.adversary else cfgs.evader_max_acc
            agent.max_speed = (
                cfgs.pursuer_max_speed if agent.adversary else cfgs.evader_max_speed
            )

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            # landmark.size = 15 / 250
            landmark.boundary = False

        # add target
        world.target = [Target() for i in range(num_target)]
        for i, target in enumerate(world.target):
            target.name = 'target %d' % i

            target.collide = True
            target.movable = False
            target.size = 4 / 50
            target.boundary = False
            # target.shape = [[-0.025, -0.025], [0.025, -0.025],
            #                [0.025, 0.025], [-0.025, 0.025]]

        world.selected_target = [Selected_Target() for i in range(num_select)]
        for i, selected_target in enumerate(world.selected_target):
            selected_target.name = 'select %d' % i
            selected_target.collide =True
            selected_target.movable =False
            selected_target.size = 4 / 50
            selected_target.boundary = False

        # add borders
        world.borders = [Border() for i in range(num_border)]
        for i, border in enumerate(world.borders):
            border.name = 'border %d' % i
            border.collide = True
            border.movable = False
            border.size = 0.03  # 边界大小
            border.boundary = True
            # 改变边界厚度border.shape 此处设为方形
            border.shape = [[-0.05, -0.05], [0.05, -0.05],
                            [0.05, 0.05], [-0.05, 0.05]]

        return world


    def reset_world(self, world, np_random):
        # random properties for agents
        # for i, agent in enumerate(world.agents):
        #     agent.color = (
        #         np.array([0.85, 0.35, 0.35])  # 设置为红色 这是入侵无人机
        #         if not agent.adversary
        #         else np.array([0.35, 0.35, 0.85])  # 设置为蓝色 这是追捕无人机
        #     )
        #     # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.8, 0.8, 0.8])
        # for i, border in enumerate(world.borders):
        #     border.color = np.array([0.25, 0.25, 0.25])  # 边界颜色
        #
        # for i, target in enumerate(world.target):
        #     target.color = np.array([0.35, 0.85, 0.35])  # 红色
        # # set random initial states
        #
        # self.cfgs.config_entities()
        # agent_init_pos = self.cfgs.agent_init_pos
        # obstacle_pos = self.cfgs.obstacle_pos
        # obstacle_size = self.cfgs.obstacle_size
        # # print(obstacle_size)
        # target_pos = self.cfgs.target_pos
        # # print(target_pos)
        #
        # for i, agent in enumerate(world.agents):
        #     agent.state.p_pos = agent_init_pos[i]
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     if not landmark.boundary:
        #         # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        #         landmark.size = obstacle_size[i]
        #         landmark.state.p_pos = obstacle_pos[i]
        #         landmark.state.p_vel = np.zeros(world.dim_p)
        # for i, target in enumerate(world.target):
        #     # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        #     target.state.p_pos = target_pos[i]
        #     target.state.p_vel = np.zeros(world.dim_p)
        #
        # for i,selct in enumerate(world.selected_target):
        #     selcted = random.choice(world.target)
        #     selct.state.p_pos = selcted.state.p_pos
        #     selct.state.p_vel = np.zeros(world.dim_p)
        #     selct.color = (np.array([0.85, 0.35, 0.35]))
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.85, 0.35, 0.35])  # 设置为红色 这是入侵无人机
                if not agent.adversary
                else np.array([0.35, 0.35, 0.85])  # 设置为蓝色 这是追捕无人机
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.8, 0.8, 0.8])
        for i, border in enumerate(world.borders):
            border.color = np.array([0.25, 0.25, 0.25])  # 边界颜色

        for i, target in enumerate(world.target):
            target.color = np.array([0.35, 0.85, 0.35])  # 红色
        # set random initial states

        self.cfgs.config_entities()
        eva_init_pos = self.cfgs.eva_init_pos
        pur_init_pos = self.cfgs.pur_init_pos
        select_pos = self.cfgs.select_target
        obstacle_pos = self.cfgs.obstacle_pos
        obstacle_size = self.cfgs.obstacle_size
        # print(obstacle_size)
        target_pos = self.cfgs.target_pos
        # print(target_pos)

        for i, agent in enumerate(world.agents):
            if agent.name == "adversary_0":
                agent.state.p_pos = pur_init_pos[0]
            else:agent.state.p_pos = eva_init_pos[0]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.size = obstacle_size[i]
                landmark.state.p_pos = obstacle_pos[i]
                landmark.state.p_vel = np.zeros(world.dim_p)
        for i, target in enumerate(world.target):
            # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            target.state.p_pos = target_pos[i]
            target.state.p_vel = np.zeros(world.dim_p)

        for i,selct in enumerate(world.selected_target):
            # selcted = random.choice(world.target)
            # selct.state.p_pos = selcted.state.p_pos
            selct.state.p_pos = select_pos[i]
            selct.state.p_vel = np.zeros(world.dim_p)
            selct.color = (np.array([0.85, 0.35, 0.35]))

        # 每条边20个border， 计算好大概位置，依次为每条边的border生成位置坐标
        pos = []
        x = -1
        y = -1.0
        # bottom
        for count in range(20):
            pos.append([x, y])
            x += 0.1

        x = 1.0
        y = -1
        # right
        for count in range(20):
            pos.append([x, y])
            y += 0.1

        x = 1
        y = 1.0
        # top
        for count in range(20):
            pos.append([x, y])
            x -= 0.1

        x = -1.0
        y = 1
        # left
        for count in range(20):
            pos.append([x, y])
            y -= 0.1

        for i, border in enumerate(world.borders):
            border.state.p_pos = np.asarray(pos[i])  # 将设好的坐标传到border的位置坐标
            border.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size/2 + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):

        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if (
                shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for i, target in enumerate(world.selected_target):
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))

                rew -= 10 * dist
                ag_vel = agent.state.p_vel

                direction = target.state.p_pos - agent.state.p_pos

                theta_cos2 = self.vec_angle_cos(direction, ag_vel)
                rew += theta_cos2

                if dist <= agent.size + target.size + 0.025:
                    rew += 10
            for adv in adversaries:
                dist_agent = np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
                if dist_agent <= 0.1:  # 如果逃避者与追捕者之间的距离小于0.09*50
                    # 计算逃避者与追捕者之间的角度，并给予奖励
                    ag_vel = agent.state.p_vel

                    direction = adv.state.p_pos - agent.state.p_pos

                    theta_cos = self.vec_angle_cos(direction, ag_vel)

                    # test3测这个
                    # if theta_cos >= 0:
                    #     rew -= 2 * theta_cos
                    # else:
                    #     rew -= 0.1 * theta_cos
                    #############test2先测这个
                    rew -= 1.2 * theta_cos

                rew += 0.002 * dist_agent

            # for adv in adversaries:
            #     if dist_agent <= agent.size + adv.size + 0.021:
            #         rew -= 5

        if agent.collide:
            # for a in adversaries:
            #     if self.is_collision(a, agent):
            #         rew -= 10
            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    if self.is_collision(landmark, agent):
                        rew -= 60
            for i, border in enumerate(world.borders):
                if self.is_collision(border, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        #     dists = [
        #         np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
        #         for i, target in enumerate(world.selected_target)
        #     ]
        #     # for i, target in enumerate(world.selected_target):
        #     #     dist = np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
        #     #
        #     rew -= 2.2 * np.max(dists)
        #     for i, target in enumerate(world.selected_target):
        #         if np.max(dists) <= agent.size + target.size + 0.01:
        #             rew += 20
        #     # dist_agent =
        #     #     np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        #     #     for adv in adversaries
        #
        #     for adv in adversaries:
        #         dist_agent = np.sqrt(
        #             np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
        #         )
        #         if dist_agent <= 0.05:  # 如果逃避者与追捕者之间的距离小于0.04*100
        #             # 计算逃避者与追捕者之间的角度，并给予奖励
        #             ag_vel = agent.state.p_vel
        #
        #             direction = adv.state.p_pos - agent.state.p_pos
        #             theta_cos = self.vec_angle_cos(direction, ag_vel)
        #             # test3测这个
        #             if theta_cos >= 0:
        #                 # rew -= 0.8 * theta_cos
        #                 rew -= 0.3 * theta_cos
        #             else:
        #                 rew += 0
        #                 #############test2先测这个
        #                 # rew -= 0.2 * theta_cos
        #         rew += 0.02 * dist_agent
        #         # for adv in adversaries:
        #         if dist_agent <= agent.size + adv.size + 0.02:
        #             rew -= 15
        #
        #
        # if agent.collide:
        #     # for a in adversaries:
        #     #     if self.is_collision(a, agent):
        #     #         rew -= 10
        #     for i, landmark in enumerate(world.landmarks):
        #         if not landmark.boundary:
        #             if self.is_collision(landmark, agent):
        #                 rew -= 30
        #     # for i, border in enumerate(world.borders):
        #     #     if self.is_collision(border, agent):
        #     #         rew -= 10
        #
        # # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        #
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew


    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
                shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for a in agents:
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                rew = -5 * dist
                if dist <= a.size + agent.size + 0.08:
                    rew += 10

        if agent.collide:
            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    for adv in adversaries:
                        if self.is_collision(landmark, adv):
                            rew -= 60
        # for ag in agents:
        #     for i, target in enumerate(world.selected_target):
        #         dist = np.sqrt(np.sum(np.square(ag.state.p_pos - target.state.p_pos)))
        #         if dist <= ag.size + target.size:
        #             rew -= 15

        # agent 和 adversary 位置连线的矢量
        for ag in agents:
            # 获取位置矢量
            adv_vel = agent.state.p_vel

            direction = ag.state.p_pos - agent.state.p_pos
            theta_cos = self.vec_angle_cos(direction, adv_vel)

            # 添加到奖励中，这里的惩罚系数可以根据具体情况调整
            rew += 5*theta_cos
        return rew

    def vec_angle_cos(self, vector1, vector2):
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cos_val = dot_product / (magnitude1 * magnitude2)
        return cos_val

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                # entity_pos.append(entity.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        target_pos = []
        for i, target in enumerate(world.selected_target):
            if agent.name == 'agent_0':
                target_pos.append(target.state.p_pos - agent.state.p_pos)
                # target_pos.append(target.state.p_pos)

        # target_pos = select_target.state.p_pos-agent.state.p_pos
        # for target in world.target:
        #     target_pos.append(target.state.p_pos-agent.state.p_pos)
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # other_pos.append(other.state.p_pos)

            other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + target_pos
            + other_pos
            + other_vel
        )
