# coding: utf-8
import json
import math
from queue import Queue

import numpy as np
from collections import namedtuple
import torch

import matplotlib.pyplot as plt
import random

from ConstData import ConstData
from util.map_load import *
from visualization import angle_to_radian, Visualization
from parking_entity.parkinggroup import ParkingGroup

# 主屏幕设置
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 500
dest_dir = 'dest-new/'
# l0: 后轮到车尾， l1: 前轮到车头， l2: 前轮到后轮, w: 车宽
Car = namedtuple('Car', ['l0', 'l1', 'l2', 'w'])
'''
State： （未定）
(v,delta,x_i,y_i,img)
'''
State = namedtuple('State', ['x', 'y', 'v', 'phi', 'delta', 'img'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ParkingEnv1(object):

    def __init__(self, car, l_slot, w_slot, max_angle, mode, position_penalty,
                 final_angle_penalty, in_slot_reward=10000, collision_penalty=10000
                 , angle_change_penalty=10, speed_penalty=25,
                 reverse_penalty=50, dt=0.02, max_dv=1., max_ddelta=0.2, max_speed=2.,
                 xbound=15, ybound=15, total_dest_size=1,
                 close_to_reward=100, visualize=True, parking_group=None, **init_config):
        self.car = car  # 汽车的参数，类型是上面定义的namedtuple Car
        self.car.dt = dt
        self.car.max_speed = max_speed
        self.l_slot = l_slot  # 停车位长度
        self.w_slot = w_slot  # 停车位宽度
        self.max_angle = max_angle  # 方向盘最大转角
        self.mode = mode  # 车位类型 reverse倒车入库 parallel侧方停车
        self.position_penalty = position_penalty  # 位置惩罚，最终位置离原点(停车位中心点)越远，惩罚越大
        self.final_angle_penalty = final_angle_penalty  # 车辆最终朝向的惩罚项
        self.in_slot_reward = in_slot_reward  # 停车成功(车辆整体停在停车位内)奖励
        self.collision_penalty = collision_penalty  # 碰撞惩罚
        self.angle_change_penalty = angle_change_penalty  # 方向盘转角的变化率惩罚，转角变化率越大，惩罚项越大，用于保证车辆行驶的横向平顺性
        self.speed_penalty = speed_penalty  # 速度变化率(即加速度)惩罚，速度变化率越大，惩罚项越大，用于保证车辆行驶的纵向平顺性
        self.reverse_penalty = reverse_penalty  # 中途停车惩罚
        self.close_to_reward = 0  # 车辆靠近终点的奖励，如果不需要这一项，可以设置为0
        self.dt = dt  # 两个状态之间的时间差
        self.max_dv = max_dv  # 最大速度变化率(即最大加速度)
        self.max_ddelta = max_ddelta  # 最大方向盘转角变化率
        self.state = None  # 当前状态，形式为27元组
        self.corners = None  # 车身的四个角
        self.max_speed = max_speed  # 最大车速
        self.x_bound = xbound  # 地图的x方向边界
        self.y_bound = ybound  # 地图的y方向边界
        self._visualize = visualize  # 是否可视化，布尔值

        # self.img_id = 'map/1.png'  # 保存当前场景，为图片id
        self.total_dest_size = total_dest_size  # dest文件夹下一共有多少文件
        # TODO:
        self.timeout = 100  # 多少step以后算超时,暂时没加
        self.angle = None
        self.target_x = 0
        self.target_y = 0
        self.step_num = 0  # 观察多少轮以后会成功/会撞车
        self.is_in_easy = True

        # [(0.20, 0.20, 45), (0.18, 0.19, 30)]
        self.near_q = [(0.18, 0.18, 30), (0.2, 0.2, 30),(0.5,0.5,30)]
        self.index = 0
        self.queue = Queue()
        for item in self.near_q:
            x_, y_, dis_rot_ = item
            if not self.check_in(x_, y_, dis_rot_):
                self.queue.put(item)

        self.line_width = 3  # 绘图时线段的宽度
        self.img_size = (6, 5)  # 场景大小(w,h)
        plt.rcParams['figure.figsize'] = self.img_size
        if parking_group is None:
            self.parking_group = load_dest_v2(3)
        else:
            self.parking_group = parking_group
        if self._visualize:
            self.vis = Visualization(self.car, self.parking_group)

    def set_car_reset_range(self, reset_x, reset_y):
        self.parking_group.reset_x = reset_x
        self.parking_group.reset_y = reset_y

    def get_car_reset_range(self):
        return self.parking_group.reset_x, self.parking_group.reset_y

    def add_car_reset_range(self):
        self.parking_group.add_reset_range()

    def dec_car_reset_range(self):
        self.parking_group.dec_reset_range()

    def load_dest(self):
        num = random.randint(1, self.total_dest_size)
        path = dest_dir + str(3) + '.json'
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        # 读取每一条json数据
        parkings = data['parkings']
        parking_group = ParkingGroup()
        for key, value in parkings.items():
            parking_group.append_by_list(value)
        return parking_group

    def reset(self):
        # 随机决定加载地图,1/10概率换地图
        num = random.randint(0, 10)

        """Randomly initialize the position of the car."""
        self.parking_group.reset_car(self.car)
        self.state = self.parking_group.get_relative_all_state(self.car)

        self.queue = Queue()
        for item in self.near_q:
            x_, y_, dis_rot_ = item
            if not self.check_in(x_, y_, dis_rot_):
                self.queue.put(item)

        return self.state

    def step(self, action):
        """Calculate the next state of the car based on the following equation
        and return the reward

        Returns next_state, reward, is_terminal, description
        """
        v = self.car.move_speed

        dis_car_ori_x = abs(self.car.now_pos[0] - self.parking_group.target.get_absolute_point()[0])
        dis_car_ori_y = abs(self.car.now_pos[1] - self.parking_group.target.get_absolute_point()[1])
        # rank_score_ori = self.parking_group.get_rank_score(self.car)
        ori_intersect_area = self.parking_group.get_intersect_area(self.car)
        rot_ori = self.car.now_rot

        self.step_num += 1
        # 车辆更新状态
        self.car.car_move_and_rotate()

        dv, ddelta = action
        # print(dv, ddelta)
        dv *= self.max_dv  # [-1, 1] -> [-max_dv, max_dv]
        ddelta *= self.max_ddelta  # [-1, 1] -> [-max_ddelta, max_ddelta]
        self.car.set_alpha(dv, ddelta)
        self.car.update_speed()

        stop = False
        if self.car.move_speed * v < 0:  # 速度反向，说明车辆停下来了
            self.car.move_speed = 0
            stop = True
        # self.parking_group.update_rank_score(self.car)

        self.state = self.parking_group.get_relative_all_state(self.car)

        # 速度变化率与方向盘转角变化率惩罚，保证平顺性
        reward = -np.abs(dv) * self.speed_penalty - \
                 np.abs(ddelta) * self.angle_change_penalty

        # 中间点奖励
        index = 0
        # todo: add
        # for item in self.near_q:
        #     x_, y_, dis_rot_ = item
        #     if self.check_in(x_, y_, dis_rot_):
        #         break
        #     index += 1
        # if self.index > index:
        #     reward += self.in_slot_reward / 2
        # elif self.index < index:
        #     reward -= 1.5 * self.in_slot_reward / 2
        # self.index = index
        # 时间惩罚
        reward -= 10
        #
        intersect_area = self.parking_group.get_intersect_area(self.car)

        # 靠近奖励,
        if (intersect_area - ori_intersect_area) > 0:
            reward += ((intersect_area - ori_intersect_area) / (ConstData.proportion * ConstData.proportion)) * 5000
            # reward += 5
        else:
            reward += ((intersect_area - ori_intersect_area) / (ConstData.proportion * ConstData.proportion)) * 6000
        # reward += (dis_car_ori - self.parking_group.dis_car_target(self.car)) * self.close_to_reward

        # if self.parking_group.target.direct == 0 or self.parking_group.target.direct == 2:
        dis_car_x = abs(self.car.now_pos[0] - self.parking_group.target.get_absolute_point()[0])
        if (dis_car_ori_x - dis_car_x) > 0:
            reward += (dis_car_ori_x - dis_car_x) * 5000 / ConstData.proportion
        else:
            reward += (dis_car_ori_x - dis_car_x) * 6000 / ConstData.proportion

        dis_car_y = abs(self.car.now_pos[1] - self.parking_group.target.get_absolute_point()[1])
        if (dis_car_ori_y - dis_car_y) > 0:
            reward += (dis_car_ori_y - dis_car_y) * 5000 / ConstData.proportion
        else:
            reward += (dis_car_ori_y - dis_car_y) * 6000 / ConstData.proportion

        # todo: add
        if (dis_rot(rot_ori) - dis_rot(self.car.now_rot)) > 0:
            reward += (dis_rot(rot_ori) - dis_rot(self.car.now_rot)) * 1500
        else:
            reward += (dis_rot(rot_ori) - dis_rot(self.car.now_rot)) * 1700

        # rank_score_now = self.parking_group.get_rank_score(self.car)
        # reward += ((rank_score_now - rank_score_ori) / (ConstData.proportion * ConstData.proportion)) * 10000
        # TODO:
        if self._visualize:
            self.visualize()

        is_in_easy = self.parking_group.is_in_easy_area(self.car)
        if not is_in_easy and self.is_in_easy:
            reward -= self.collision_penalty / 3
        elif is_in_easy and not self.is_in_easy:
            reward += self.collision_penalty / 3
        self.is_in_easy = is_in_easy

        is_collide = self.car.collide_list(self.parking_group.nearest_4(self.car.now_pos))
        if is_collide:
            # 碰撞位置离停车位越远，惩罚越大
            reward -= self.position_penalty * self.parking_group.dis_car_target(self.car) + self.collision_penalty

            # 如果汽车的一部分进入了停车位
            # intersect_area = self.parking_group.get_intersect_area(self.car)

            # 奖励，如果汽车进入车位的部分越多，奖励越大
            # reward += 0.6 * self.in_slot_reward * intersect_area / (self.car.car_width * self.car.car_height)

            temp = self.step_num
            self.step_num = 0

            return self.state, reward, True, 'collide', temp

        if stop:
            # the car stop
            is_car_in_slot = self.parking_group.is_car_in_slot(self.car)
            if is_car_in_slot:
                target_phi = self.parking_group.target.theta

                # 最终角度惩罚
                reward -= self.final_angle_penalty * calc_angle(angle_to_radian(self.car.now_rot),
                                                                target_phi) + self.position_penalty * self.parking_group.dis_car_target(
                    self.car)

                reward += self.in_slot_reward
                # 后来加的
                # reward += self.in_slot_reward

                # print('in_slot')
                temp = self.step_num
                self.step_num = 0

                return self.state, reward, True, 'in_slot', temp

            else:
                reward -= self.reverse_penalty
                return self.state, reward, False, 'stop', self.step_num

        return self.state, reward, False, 'parking', self.step_num

    def check_in(self, x, y, rot_dis):
        if dis_rot(self.car.now_rot) > rot_dis:
            return False
        x_range = x * SCREEN_WIDTH
        y_range = y * SCREEN_HEIGHT
        now_x = self.state[0] * ConstData.proportion
        now_y = self.state[1] * ConstData.proportion
        if abs(now_x) > x_range or abs(now_y) > y_range:
            return False
        return True

    # 可视化展示
    def visualize(self):
        self.vis.run()

    '''
    # 将当前图片转换为能够进行卷积的Tensor
    def get_current_scene(self):
        # 将plt转化为numpy数据
        canvas = FigureCanvasAgg(self.current_scene)
        # 获取图像尺寸
        w, h = canvas.get_width_height()
        # 解码string 得到argb图像
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        # 重构成w h 4(argb)图像
        buf.shape = (w, h, 4)
        # 转换为 RGBA
        buf = np.roll(buf, 3, axis=2)
        # 得到 Image RGBA图像对象
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        # (1,4,640,480)
        self.current_scene = util.PIL_to_tensor(image).to(device)
        # self.current_scene = torch.Tensor(self.current_scene).cuda()
    '''

    @property
    def dim_states(self):
        return 22

    @property
    def dim_actions(self):
        return 2


# 计算x1,x2的夹角（弧度）
def calc_angle(x1, x2):
    ans = abs(x1 - x2)
    if ans >= math.pi:
        ans -= math.pi
    return ans


def dis_rot(rot):
    return min(abs(90 - rot), abs(270 - rot))


if __name__ == '__main__':
    car = Car(3, 1, 1, 2)
    env = ParkingEnv1(car, 6, 3, np.pi / 4, 'parallel',
                      10, 10, 1000, 10, 1, 100)
