#环境奖励函数设置，里面有让逃逸者策略更强的奖励项，可以接入到自己的环境中



import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from utils import common as util

from env_matd3._utils.core import Agent, Landmark, World
from env_matd3._utils.scenario import BaseScenario
from env_matd3._utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_evasion=1,
        num_pursuit=3,
        num_obstacles=2,
        max_cycles=250,
        continuous_actions=False,
        render_mode=None,
        cfgs=None,
    ):
        print("init: ", num_evasion, "evader, ", num_pursuit, "pursuer, ", num_obstacles, "obstacle")

        EzPickle.__init__(
            self,
            num_good=num_evasion,
            num_adversaries=num_pursuit,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(cfgs)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "pursuit_evasion"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, cfgs):
        self.cfgs = cfgs

        num_evasion = cfgs.args.evader_num
        num_pursuit = cfgs.args.pursuer_num
        num_obstacles = cfgs.args.obstacle_num

        world = World()
        # set any world properties first
        world.dim_c = 2
        world.cfgs = cfgs
        world.approach_dist = self.cfgs.approach_dist
        world.surround_dist = self.cfgs.surround_dist
        world.enclose_dist = self.cfgs.enclose_dist
        world.safe_dist_drone = self.cfgs.safe_dist_drone
        self.num_evasion_agents = num_evasion
        self.num_pursuit_agents = num_pursuit
        num_agents = num_pursuit + num_evasion
        num_landmarks = num_obstacles

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.pursuit = True if i < num_pursuit else False
            base_name = "pursuit" if agent.pursuit else "evaision"
            base_index = i if i < num_pursuit else i - num_pursuit
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            # agent.size = 0.075 if agent.adversary else 0.05
            # agent.accel = 3.0 if agent.adversary else 4.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.size = cfgs.pursuer_size if agent.pursuit else cfgs.evader_size
            agent.view_range = cfgs.pursuer_view_range if agent.pursuit else cfgs.evader_view_range
            agent.view_num = cfgs.pursuer_view_ray_num if agent.pursuit else cfgs.evader_view_ray_num
            agent.view_res = []
            agent.pre_vel = np.array([0, 0])
            agent.found_eva = False
            agent.know_eva_pos = False
            # 记录agent的最大奖励的目标位置
            agent.target_pos = np.zeros(world.dim_p)
            # 添加对探索过区域的记录
            agent.explored_areas = []
            agent.accel = cfgs.pursuer_max_acc if agent.pursuit else cfgs.evader_max_acc
            agent.max_speed = cfgs.pursuer_max_speed if agent.pursuit else cfgs.evader_max_speed
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            # landmark.size = 0.2
            landmark.boundary = False
        # world.step() entity.state.p_pos += entity.state.p_vel * self.dt
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if agent.pursuit else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        # self.cfgs.gen_random_entities()
        self.cfgs.config_entities()
        agent_init_pos = self.cfgs.agent_init_pos
        obstacle_pos = self.cfgs.obstacle_init_pos
        obstacle_size = self.cfgs.obstacle_size

        for i, agent in enumerate(world.agents):
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = agent_init_pos[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.view_res = []
            agent.pre_vel = np.array([0, 0])
            agent.target_pos = np.array(agent.state.p_pos)
            agent.explored_areas = []
            agent.found_eva = False
            agent.know_eva_pos = False
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.size = obstacle_size[i]
                landmark.state.p_pos = obstacle_pos[i]
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.pursuit:
            collisions = 0
            for a in self.evasion_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + self.cfgs.safe_dist_drone
        return True if dist < dist_min else False

    def is_pur_close(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + self.cfgs.away_dist_from_pur
        return True if dist < dist_min else False

    # return all evasion agents
    def evasion_agents(self, world):
        return [agent for agent in world.agents if not agent.pursuit]

    # return all pursuit agents
    def pursuit_agents(self, world):
        return [agent for agent in world.agents if agent.pursuit]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.pursuit_reward(agent, world) if agent.pursuit else self.evasion_reward(agent, world)
        return main_reward

    def bound(self, x, soft_bound):
        soft_bound_1 = soft_bound
        soft_bound_2 = soft_bound_1 + 0.1
        if x < soft_bound_1:
            return 0
        if x < soft_bound_2:
            return (x - soft_bound_1) * 10
        # return min(np.exp(2 * (x - soft_bound_2)), 10)
        return min(np.exp(20 * (x - soft_bound_2)), 30)

    def evasion_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        pursuers = self.pursuit_agents(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            pur_dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) for adv in pursuers]
            rew += np.sum(pur_dists) - np.max(pur_dists)

        # if agent.collide:
        #     for a in pursuers:
        #         if self.is_pur_close(a, agent):
        #             rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        soft_bound = self.cfgs.soft_bound_1
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= self.bound(x, soft_bound)

        # NOTE: 这里设置逃逸目标撞到obstacle的惩罚更大，避免为了躲避追捕者而直接进入obstacle
        # FIXME: 设置env的termination和truncation
        avoid_obstacle_rew = self.avoid_obstacle_reward(agent, world)
        rew += 5 * avoid_obstacle_rew

        hide_reward, found_penalize = self.hide_from_pur(agent, pursuers, world)  # hide reward, found punishment
        rew += 0.45 * found_penalize  # 加了这一项被发现的惩罚以后逃逸者有点太聪明了，先调小系数
        # 逃逸者远离追捕者的奖励
        rew += 0.35 * self.run_away_from_pur(agent, pursuers, world)

        # 考虑利用建筑遮挡躲避追捕者的奖励，前提是自己不能撞到建筑
        if avoid_obstacle_rew > -5:
            rew += hide_reward
        return rew
# 2. 方向性逃离奖励 - 引导逃避者离开追捕者质心
    def run_away_from_pur(self, agent, pursurers, world):
        # 计算追捕者的质心
        pursuers_pos = np.array([p.state.p_pos for p in pursurers])
        pursuers_center = np.mean(pursuers_pos, axis=0)
        # 计算逃逸者的速度，是否在追捕者质心到自己的位置的连线方向上
        eva_vel = agent.state.p_vel
        direction = agent.state.p_pos - pursuers_center
        theta_cos = util.vec_angle_cos(direction, eva_vel)
        return 10 * theta_cos
# 1. 视线阻挡躲避奖励 - 利用障碍物遮挡视线
    def hide_from_pur(self, agent, pursuers, world):
        # 逃逸者躲避追捕者的奖励
        # 如果利用建筑躲避追捕者，可以获得奖励; 如果被发现，得到惩罚

        obstacles = world.landmarks
        hides = []
        founds = []
        for pur in pursuers:
            hide = False
            found = False
            for obs in obstacles:
                if util.is_line_intersecting_circle(agent.state.p_pos, pur.state.p_pos, obs.state.p_pos, obs.size):
                    hide = True
                    break
                elif self.is_pur_close(agent, pur):
                    found = True
            hides.append(hide)
            founds.append(found)

        rew1 = 5 * sum(hides)         # 利用障碍物遮挡奖励
        rew2 = -7.5 * sum(founds)     # 被发现惩罚
        return rew1, rew2

    def pursuit_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        eva_agents = self.evasion_agents(world)
        eva = eva_agents[0]
        pur_agents = self.pursuit_agents(world)

        # 限制追捕者的范围，但是比逃逸目标更大
        has_soft_bound = True
        if has_soft_bound:
            soft_bound = self.cfgs.soft_bound_2 * 1.2  # 扩大purser的活动范围
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= self.bound(x, soft_bound)  # bound 可以取消最大30的限制

        if self.is_collision(agent, eva):  # 和逃逸目标冲突的惩罚
            rew -= 10
        else:
            rew += self.enclose_task_reward(agent, pur_agents, eva_agents)

        rew += 2 * self.avoid_obstacle_reward(agent, world)  # 原来是2，会离障碍物太远不敢前进
        rew += self.avoid_other_pursuer_reward(agent, world)
        rew += 0.5 * self.find_eva_reward(agent, eva, world)
        # rew += self.communication_reward(agent, pur_agents, world.landmarks)
        return rew

    # 看不到逃逸目标的惩罚
    def find_eva_reward(self, agent, eva, world):
        # 逃逸者躲避追捕者的奖励
        # 如果利用建筑躲避追捕者，可以获得奖励; 如果被发现，得到惩罚

        obstacles = world.landmarks

        found = True
        near = False
        for obs in obstacles:
            if util.is_line_intersecting_circle(agent.state.p_pos, eva.state.p_pos, obs.state.p_pos, obs.size):
                found = False
                break
            elif self.is_pur_close(agent, eva):
                near = True

        rew1 = 2.5 if found else -5
        rew2 = 7.5 if found and near else 0
        return rew1 + rew2

    # 考虑和障碍物的碰撞惩罚
    def avoid_obstacle_reward(self, agent, world):
        rew = []
        for landmark in world.landmarks:
            if landmark.boundary:
                continue
            delta_pos = landmark.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            actual_dist = dist - agent.size - landmark.size

            # 除了碰撞距离的因素，还应该考虑速度和障碍物的夹角
            safe_dist = self.cfgs.safe_dist_obstacle
            # safe_dist = 0
            rew_coff = -15
            if actual_dist > safe_dist:
                rew.append(0)
            elif actual_dist <= 0:
                rew.append(rew_coff)
                break
            else:
                rew1 = rew_coff if actual_dist <= 0 else rew_coff * pow(1 - actual_dist / safe_dist, 2)
                # rew.append(-20 if actual_dist <= 0 else -20*pow(1-actual_dist/safe_dist,2)) #原来是-10

                # 避让角度的计算
                theta = util.vec_angle(delta_pos, agent.state.p_vel)
                theta_tau1 = np.arcsin(landmark.size / dist)
                theta_tau2 = np.arctan2(agent.size, dist * np.cos(theta_tau1))
                theta_tau = theta_tau1 + theta_tau2

                if 0 <= theta <= theta_tau:
                    theta_ratio = 2 - theta / theta_tau
                else:
                    theta_ratio = 1
                # elif theta_tau < theta <= np.pi / 2:
                #     theta_ratio = 1
                # else:
                #     theta_ratio = theta - np.pi / 2

                rew.append(rew1 * theta_ratio)
                # rew.append(-20) #原来是-10
        # 返回一个最小的，避免累加值太大，覆盖其他项奖励的作用
        return np.min(rew)

    # 考虑和其他追捕智能体的碰撞惩罚
    def avoid_other_pursuer_reward(self, agent, world):
        rew = []
        safe_dist = self.cfgs.safe_dist_drone
        for other in self.pursuit_agents(world):
            if other is agent:
                continue
            dist = util.dist(agent, other)
            actual_dist = dist - agent.size - other.size
            rew_coff = -10
            if actual_dist > safe_dist:
                rew.append(0)
            elif actual_dist <= 0:
                rew.append(rew_coff)
            else:
                rew1 = rew_coff if actual_dist <= 0 else rew_coff * pow(1 - actual_dist / safe_dist, 2)
                # rew.append(-10 if actual_dist <= 0 else -10 * pow(1 - actual_dist / safe_dist, 2))

                # 考虑无人机之间的相对速度大小、角度，进行避让的惩罚
                rel_v = agent.state.p_vel - other.state.p_vel
                rel_p = other.state.p_pos - agent.state.p_pos
                # 避让角度的计算
                theta = util.vec_angle(rel_p, rel_v)
                theta_tau = 2 * np.arcsin(other.size / dist)

                if 0 <= theta < theta_tau:
                    theta_ratio = 2 - theta / theta_tau
                else:
                    theta_ratio = 1
                # elif theta_tau <= theta <= np.pi / 2:
                #     theta_ratio = 1
                # else:
                #     theta_ratio = theta - np.pi / 2
                rew.append(theta_ratio * rew1)

        # 返回一个最小的，避免累加值太大，覆盖其他项奖励的作用
        return np.min(rew)

    # 避免和eva的碰撞
    def avoid_eva_reward(self, agent, world):
        rew = []
        safe_dist = self.cfgs.safe_dist_drone
        for eva in self.evasion_agents(world):
            dist = util.dist(agent, eva)
            actual_dist = dist - agent.size - eva.size
            if actual_dist > safe_dist:
                rew.append(0)
            elif actual_dist <= 0:
                rew.append(-20)
            else:
                rew_coff = -10
                rew1 = rew_coff if actual_dist <= 0 else rew_coff * pow(1 - actual_dist / safe_dist, 2)
                # rew.append(-10 if actual_dist <= 0 else -10 * pow(1 - actual_dist / safe_dist, 2))

                # 考虑无人机之间的相对速度大小、角度，进行避让的惩罚
                rel_v = agent.state.p_vel - eva.state.p_vel
                rel_p = eva.state.p_pos - agent.state.p_pos
                # 避让角度的计算
                theta = util.vec_angle(rel_p, rel_v)
                theta_tau1 = np.arcsin(eva.size / dist)
                theta_tau2 = np.arctan2(agent.size, dist * np.cos(theta_tau1))
                theta_tau = theta_tau1 + theta_tau2

                if 0 <= theta < theta_tau:
                    theta_ratio = 2 - theta / theta_tau
                elif theta_tau <= theta <= np.pi / 2:
                    theta_ratio = 1
                else:
                    theta_ratio = (theta - np.pi / 2) / (np.pi / 2) + 1
                rew.append(theta_ratio * rew1)

        return np.min(rew)

    def avoid_drone_reward(self, agent, world):
        rew = 0
        rew += self.avoid_other_pursuer_reward(agent, world)
        rew += self.avoid_eva_reward(agent, world)
        return rew

    def enclose_task_reward(self, agent, pur_agents, eva_agents):
        # 相对位置、相对速度
        rew = 0
        eva = eva_agents[0]

        encls_dist = self.cfgs.enclose_dist
        encls_dist_err = encls_dist / 4

        encls_idx, actual_theta = util.clockwise_traversal(pur_agents, agent, eva)
        # eva_pos = eva.state.p_pos
        agent_target_pos, agent_target_theta = util.gen_surround_loc(eva.state.p_pos, len(pur_agents), encls_dist, encls_idx)
        agent.target_pos = agent_target_pos

        pt_dist = util.dist_vec(agent.state.p_pos, agent_target_pos)  # /(2*np.sqrt(2))  # [0, 2*sqrt(2)] -> [0,1]
        rele_vel = np.linalg.norm(agent.state.p_vel - eva.state.p_vel)

        s = encls_dist_err
        d = pt_dist
        r1 = 15
        r2 = 50
        r3 = 5
        # rew = r1 * (1 - pow(d / s, 2)) if d < s else r2 * (1 - np.exp(d - s))
        rew = r1 * (1 - pow(d / s, 2)) if d < s else -r2 * abs(d - s)
        rew -= r3 * rele_vel

        return rew

    def termination(self, agent, world):
        print("check agent termination")

    def truncation(self, agent, world):
        print("check agent truncation")

    def observation_bak(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_size = []
        for entity in world.landmarks:
            if not entity.boundary:
                # obs_pos = entity.state.p_pos - agent.state.p_pos  # 目前用的绝对坐标，不应该减去agent求相对
                obs_pos = entity.state.p_pos
                obs_pos_size = np.append(obs_pos, entity.size)
                entity_pos_size.append(obs_pos_size)
        # communication of all other agents
        comm = []
        other_pos_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            # other_pos.append(other.state.p_pos - agent.state.p_pos)  # 目前用的绝对坐标，不应该减去agent求相对
            # other_vel.append(other.state.p_vel - agent.state.p_vel)  # 目前用的绝对坐标，不应该减去agent求相对
            a_pos_vel = np.append(other.state.p_pos, other.state.p_vel)
            other_pos_vel.append(a_pos_vel)
            # if not other.pursuit:
            #     other_vel.append(other.state.p_vel)

        obs = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + other_pos_vel + entity_pos_size)
        # print(obs.shape, obs)
        return obs

    def observation(self, agent, world):
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel
        agent_state = np.append(agent_pos, agent_vel)
        eva = self.evasion_agents(world)[0]
        agent_state = np.append(agent_state, [eva.state.p_pos, eva.state.p_vel])

        ray_obs = []
        view_num = agent.view_num
        # 检测是否在obs内部
        for obstacle in world.landmarks:
            if not obstacle.boundary:
                if np.linalg.norm(agent_pos - obstacle.state.p_pos) < obstacle.size:
                    ray_obs = [-1, 0, 0, 1, 0] * view_num
                    # TODO: 拼接其他的状态观测
                    agent.view_res = []
                    agent.pre_vel = agent_vel
                    return np.append(agent_state, ray_obs)

        view_res = []
        entity_pos_size = []
        for entity in world.agents + world.landmarks:
            if entity is not agent:
                entity_pos_size.append((entity.state.p_pos, entity.size, entity.name[0]))  # [((x, y), r, type), ...]  type: ['p', 'e', 'l']

        theta_v = np.arctan2(agent_vel[1], agent_vel[0])
        delta_theta = 2 * np.pi / view_num
        theta_list = (np.array(list(range(view_num))) - view_num // 2) * delta_theta
        type_dict = {"e": [1, 0, 0, 0], "p": [0, 1, 0, 0], "l": [0, 0, 1, 0]}
        for theta_i in theta_list:
            ray_theta_rad_i = theta_v + theta_i
            # TODO: 优化大量循环计算
            ray_view_i = util.intersection_in_detect_radius_list(agent_pos, ray_theta_rad_i, agent.view_range, entity_pos_size)
            if ray_view_i:
                (entity_x, entity_y), entity_dist, entity_type = ray_view_i
                ray_obs.extend([entity_dist, *type_dict[entity_type]])
                view_res.append((theta_i, (entity_x, entity_y), entity_dist, entity_type))
            else:
                ray_obs.extend([agent.view_range, 0, 0, 0, 1])
                view_res.append((theta_i, (None, None), agent.view_range, None))
        agent.view_res = view_res
        agent.pre_vel = agent_vel
        return np.append(agent_state, ray_obs)
