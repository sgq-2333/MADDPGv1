import logging
import numpy as np
import argparse
import random

def setup_logger(filename):
    """set up logger with filename."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode="w")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s--%(levelname)s--%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser()

    # 场景每轮初始化配置
    parser.add_argument("--init-scenario", type=str, default="random", help="init scenario", choices=["random", "fixed"])
    parser.add_argument("--init-agent", type=str, default="random", help="init agent mode", choices=["random", "fixed"])
    parser.add_argument("--scale", type=str, default="small", help="scale of env", choices=["small", "large"])
    parser.add_argument("--init-target", type=str, default="random", help="init target",
                        choices=["random", "fixed"])
    parser.add_argument("--num-good", type=int, default=1, help="number of pursuer")
    parser.add_argument("--evader-num", type=int, default=1, help="number of evader")
    parser.add_argument("--obstacle-num", type=int, default=5, help="number of obstacle")
    parser.add_argument("--target-num", type=int, default=3, help="number of targets")

    args = parser.parse_args()
    return args


class EnvConfig:
    def __init__(self) -> None:
        self.args = get_args()

        self.max_goal_step_length = 15

        # 维度
        self.world_dim = 2

        # 场地大小 单位m (单边全长，两倍半边长度)
        # self.side_full_size = 50

        #! side_full_size是一边全长！最早的场景设置是side_full_size为100，坐标范围为+-50！
        self.side_full_size = 200 if self.args.scale == "large" else 100  # 30 # 16和原本的simple_tag_v3基本一致
        self.size = self.side_full_size / 2

        self.soft_bound_1 = 1.25  # 逃逸目标的软边界 1.3
        self.soft_bound_2 = 1.4  # 追捕者的软边界

        # 追捕者和逃逸目标大小 单位m (半径)
        self.actual_pursuer_size = 0.5
        self.actual_evader_size = 0.5
        # 追捕者之间的通信半径（暂不考虑信噪比）
        self.actual_pursuer_communication_radius = 30
        # 追捕者和逃逸目标的视野范围 单位m (半径)
        self.actual_pursuer_view_range = 10
        self.actual_evader_view_range = 7.5


        # 逃逸目标距离追捕者的距离阈值，小于此距离开始获得惩罚
        self.actual_away_dist_from_pur = 10

        # TODO: 这里先设置成速度、加速度一样
        # 最大速度 单位m/s
        self.actual_pursuer_max_speed = 20  # 5
        self.actual_evader_max_speed = 20  # 5
        # 加速度 单位m/s^2
        self.actual_pursuer_max_acc = 10  # 4
        self.actual_evader_max_acc = 10  # 4

        # 障碍物半径 单位m
        self.actual_obstacle_size_choices = [5, 8]
        self.actual_obstacle_size = []
        # self.actual_obstacle_size = 5
        # # self.actual_obstacle_size = self.side_full_size / 20
        # # self.actual_obstacle_size = min(5, max(1, self.actual_obstacle_size))
        self.actual_target_size_choices = 15
        self.actual_target_size = []

        # 安全距离


        self.actual_safe_distance_from_obstacle = 1    # 静态障碍物安全距离
        self.actual_safe_distance_from_drone = 0.5     # 动态智能体安全距离




        # self.config_entities()
        self.existing_entities = []

    def gen_random_entities(self, type):
        if type == "obstacle":
            obs_num = 5
            for i in range(obs_num):
                while True:
                    x = np.random.uniform(-self.size+10, self.size-10)
                    y = np.random.uniform(-self.size+10, self.size-10)
                    r = np.random.choice(self.actual_obstacle_size_choices)
                    safe_dist = 1
                    if self.is_safe([x, y], r + safe_dist):
                        self.existing_entities.append([x, y, r])
                        self.actual_obstacle_size.append(r)
                        self.obstacle_pos.append(np.array([x, y]) / self.size)
                        break
        elif type == "target":
            tar_num = 3
            for i in range(tar_num):
                while True:
                    x = np.random.uniform(-self.size+30, self.size-30)
                    y = np.random.uniform(-self.size+30, self.size-30)
                    r = 4
                    safe_dist = 1
                    if self.is_safe([x, y], r + safe_dist):
                        self.existing_entities.append([x, y, r])
                        self.actual_obstacle_size.append(r)
                        self.target_pos.append(np.array([x, y]) / self.size)
                        break

        elif type == "eva":
            eva_num = 1
            for i in range(eva_num):
                while True:
                    radius = np.random.uniform(0.8 * self.size, 0.9 * self.size)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x, y = self.polar_to_cartesian(radius, angle)
                    r = 0.5
                    if self.is_safe([x, y], r):
                        self.existing_entities.append([x, y, r])
                        self.eva_init_pos.append(np.array([x, y]) / self.size)
                        break
        elif type == "pur":
            pur_num = 1
            for i in range(pur_num):
                while True:
                    x_e,y_e = self.eva_init_pos[0]
                    x_t,y_t = self.select_target[0]
                    angle0 = np.arctan2(x_e-x_t,y_e-y_t)
                    radius = np.random.uniform(0, 0.3 * self.size)
                    angle = np.random.uniform(angle0- np.pi/2, angle0 + np.pi/2)
                    x, y = self.polar_to_cartesian(radius, angle)
                    # x = np.random.uniform(-self.size, self.size)
                    # y = np.random.uniform(-self.size, self.size)
                    r = 0.5
                    if self.is_safe([x, y], r):
                        self.existing_entities.append([x, y, r])
                        self.pur_init_pos.append(np.array([x, y]) / self.size)
                        break

            # print(self.agent_init_pos)

    def polar_to_cartesian(self,radius, angle):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return x, y


    # 仅用于初始化位置时判断是否安全
    def is_safe(self, pos, r):
        for entity in self.existing_entities:
            if (
                np.linalg.norm(np.array(pos) - np.array(entity[:2]))
                < r + entity[2] + self.actual_safe_distance_from_obstacle
            ):
                return False
        return True

    # 缩放位置
    def scale_array(self, array):
        scale_factor = self.size / 50
        scaled_array = [[scale_factor * x for x in row] for row in array]
        return scaled_array

    def config_entities(self):

        self.existing_entities = []
        self.eva_init_pos = []
        self.pur_init_pos = []
        self.obstacle_pos = []
        self.target_pos = []
        self.select_target = []
        self.actual_obstacle_size = []

        # 配置固定障碍物和智能体位置，以size = 50为基准，也就是边长100，在+-50范围内配置
        fix_size = 50

        config_scenario = self.args.init_scenario
        config_agent = self.args.init_agent
        config_target = self.args.init_target

        # 配置环境中的障碍物
        if config_scenario == "random":
            self.gen_random_entities("obstacle")
        elif config_scenario == "fixed":
            # 障碍物信息 (x,y,radius)
            obstacle_info = [
                [-0.65, 0.35, 15],
                [0.55, 0.45, 10],
                [0.05, -0.65, 15]
            ]
            for i in range(3):
                x, y, r = obstacle_info[i]
                self.existing_entities.append([x, y, r])
                self.actual_obstacle_size.append(r)
                self.obstacle_pos.append(np.array([x, y]))
        if config_target == "random":
            self.gen_random_entities("target")
        elif config_target == "fixed":
            # 障碍物信息 (x,y,radius)
            target_info = [
                [0, 0, 15],
                [0.4,-0.6,15],
                [-0.3,0.3,10]
            ]
            for i in range(3):
                x, y, r = target_info[i]
                print(x,y,r)
                self.existing_entities.append([x, y, r])
                self.target_pos.append(np.array([x, y]))
        pos = random.choice(self.target_pos)
        self.select_target.append(pos)

        # if config_eva == "random":
        self.gen_random_entities("eva")
        self.gen_random_entities("pur")
        # elif config_agent == "fixed":
            # 追捕者和逃逸目标初始位置
            # pursuer_init_pos = np.array([0.0, 0.0])  # 将对抗方的位置固定在原点
            # x = np.random.choice([-0.2, 0.2])
            # y = np.random.choice([-0.2, 0.2])
            # pursuer_init_pos = [np.array([x, y])]
            # for i in range(1):
            #     x, y = pursuer_init_pos[i]
            #     self.existing_entities.append([x, y, 0.5])
            #     self.agent_init_pos.append(np.array([x, y]))
            #
            # x = np.random.choice([-0.9, -0.6]) if np.random.rand() < 0.5 else np.random.choice([0.6, 0.9])
            # y = np.random.choice([-0.9, -0.6]) if np.random.rand() < 0.5 else np.random.choice([0.6, 0.9])
            # evader_init_pos = [np.array([x, y])]
            # for i in range(1):
            #     x, y = evader_init_pos[i]
            #     self.existing_entities.append([x, y, 0.5])
            #     self.agent_init_pos.append(np.array([x, y]))


    @property
    def pursuer_size(self):
        return self.actual_pursuer_size / self.size

    @property
    def evader_size(self):
        return self.actual_evader_size / self.size

    @property
    def pursuer_comm_r(self):
        return self.actual_pursuer_communication_radius / self.size

    @property
    def pursuer_view_range(self):
        return self.actual_pursuer_view_range / self.size

    @property
    def evader_view_range(self):
        return self.actual_evader_view_range / self.size

    @property
    def away_dist_from_pur(self):
        return self.actual_away_dist_from_pur / self.size

    @property
    def pursuer_max_speed(self):
        return self.actual_pursuer_max_speed / self.size

    @property
    def evader_max_speed(self):
        return self.actual_evader_max_speed / self.size

    @property
    def pursuer_max_acc(self):
        return self.actual_pursuer_max_acc / self.size

    @property
    def evader_max_acc(self):
        return self.actual_evader_max_acc / self.size

    @property
    def safe_dist_obstacle(self):
        return self.actual_safe_distance_from_obstacle / self.size

    @property
    def safe_dist_drone(self):
        return self.actual_safe_distance_from_drone / self.size

    @property
    def obstacle_num(self):
        return len(self.obstacle_pos)

    @property
    def pursuer_num(self):
        return len(self.pursuer_init_pos)

    @property
    def evader_num(self):
        return len(self.evader_init_pos)

    @property
    def obstacle_size(self):
        return np.array(self.actual_obstacle_size) / self.size

    @property
    def normalized_obstacle_safety_radius(self):
        return self.obstacle_safety_radius / self.size
    def generate_agent(self, name):
        def rand_pos():
            return np.random.uniform(-1, +1, self.world_dim)

        [is_pursuer, i] = name.split("_")
        i = int(i)
        if is_pursuer == "pursuit":
            return np.array(self.pursuer_init_pos[i]) / self.size if i < self.pursuer_num else rand_pos()
        return np.array(self.evader_init_pos[i]) / self.size if i < self.evader_num else rand_pos()