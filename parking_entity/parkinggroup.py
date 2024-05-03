import json
import math


from parking_entity.parking import Parking
from parking_entity.mycar import My_Car
import numpy as np
import pygame as py
import shapely

from ConstData import ConstData
from shapely.geometry import Polygon

from util.util import fill_list

# 数学常量
PI = math.pi

# 设置手动模式下速度，角速度等信息
move_speed = 1
rot_speed = 2

# 主屏幕设置
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 500

RED = (255, 0, 0)
BLACK = (0, 0, 0)


def get_linear_equation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return a, b, c


def cal_line_const_y(a, b, c, y):
    return -(b * y + c) / a


def cal_line_const_x(a, b, c, x):
    return -(a * x + c) / b


# up: 0, right: 1, bottom: 2, left: 3
def get_y_const_point(p1x, p1y, p2x, p2y, direct):
    a, b, c = get_linear_equation(p1x, p1y, p2x, p2y)
    li = []
    # y = 0
    if direct != 2:
        x = cal_line_const_y(a, b, c, 0)
        if 0 <= x <= 600:
            li.append((x, 0))
    # y= 500
    if direct != 0:
        x = cal_line_const_y(a, b, c, 500)
        if 0 <= x <= 600:
            li.append((x, 500))
    return li


# up: 0, right: 1, bottom: 2, left: 3
def get_x_const_point(p1x, p1y, p2x, p2y, direct):
    a, b, c = get_linear_equation(p1x, p1y, p2x, p2y)
    li = []
    # x = 0
    if direct != 1:
        y = cal_line_const_x(a, b, c, 0)
        if 0 < y < 500:
            li.append((0, y))
    # x = 600
    if direct != 3:
        y = cal_line_const_x(a, b, c, 600)
        if 0 < y < 500:
            li.append((600, y))
    return li


def get_interact_points(point_middle, point_list, direct):
    li = []
    li.extend(get_y_const_point(point_middle[0], point_middle[1], point_list[0][0], point_list[0][1], direct))
    li.extend(get_y_const_point(point_middle[0], point_middle[1], point_list[1][0], point_list[1][1], direct))
    li.extend(get_x_const_point(point_middle[0], point_middle[1], point_list[0][0], point_list[0][1], direct))
    li.extend(get_x_const_point(point_middle[0], point_middle[1], point_list[1][0], point_list[1][1], direct))
    return li


class ParkingGroup:
    def __init__(self):
        self.parking_group = []
        self.target = None
        self.easy_area = None
        self.parking_sprites = py.sprite.Group()
        self.reset_x = 0.01
        self.reset_y = 0.01
        self.rank_score = None
        self.rank_car = My_Car()

    def add_reset_range(self):
        self.reset_x += 0.001
        self.reset_y += 0.001
        if self.reset_x > 1:
            self.reset_x = 1
        if self.reset_y > 1:
            self.reset_y = 1

    def dec_reset_range(self):
        self.reset_x -= 0.002
        self.reset_y -= 0.002
        if self.reset_x < 0:
            self.reset_x = 1
        if self.reset_y < 0:
            self.reset_y = 0

    # parking_group: {
    #   target: [],
    #   parkings: {
    #      0 :[], 1 :[] , 2 :[]
    #   }
    # }
    def get_parking_group_info(self):
        parking_group = {}
        target = self.target.get_parking_info()
        parkings = {}
        index = 0
        for parking in self.parking_group:
            parkings[index] = parking.get_parking_info()
            index += 1
        parking_group['target'] = target
        parkings[index] = target
        parking_group['parkings'] = parkings
        parking_group_info = json.dumps(parking_group)
        return parking_group_info

    # (target, nearest)前五个为target，后4个为停车位（16格子）
    # 5 + 16 = 21
    def get_state(self, car):
        nearest_4 = self.get_nearest_4_list(car.now_pos)
        return tuple(nearest_4)

    # (target, nearest)前五个为target，后4个为停车位（16格子）
    # 5 + 16 = 21
    def get_relative_state(self, car):
        nearest_4 = self.get_relative_nearest_4_list(car.now_pos)
        return tuple(nearest_4)

    # 5 + 21 = 26个位置
    def get_all_state(self, car):
        car_state = car.get_state()
        parking_state = self.get_state(car)
        return tuple(car_state) + tuple(parking_state)

    def get_relative_all_state(self, car):
        rect = self.target.rect
        x = rect.centerx / ConstData.proportion
        y = rect.centery / ConstData.proportion
        car_x, car_y, v, rot, phi = car.get_state()
        parking_state = self.get_relative_state(car)
        return (car_x - x, car_y - y, v, rot, phi) + tuple(parking_state)

    def store_parking_group_info(self, index=2):
        parking_group_info = self.get_parking_group_info()
        with open('./dest/' + str(index) + '.json', 'w', encoding='utf-8') as json_file:
            json.dump(parking_group_info, json_file, ensure_ascii=False)

    def append(self, parking):
        if parking.is_target:
            self.target = parking
        else:
            self.parking_group.append(parking)
        self.parking_sprites.add(parking)
        if self.target is not None and self.easy_area is None:
            self.cal_easy_area()

    # 添加停车场信息
    def append_by_list(self, li):
        size = (li[2], li[3])
        pos = (li[0] - size[0] / 2, li[1] - size[1] / 2)
        if len(li) == 4:
            parking = Parking(size, False, pos)
        else:
            parking = Parking(size, True, pos, li[4])
        self.append(parking)

    def cal_easy_area(self):
        # corners = [topleft, topright, bottomright, bottomleft]
        absolute_corner = self.target.get_absolute_corner()
        direct = self.target.direct
        # up: 0, right: 1, bottom: 2, left: 3
        middle_point = (0, 0)
        other_point = [(0, 0), (0, 0)]
        index = 0
        if direct == 0:
            middle_point = ((absolute_corner[2][0] + absolute_corner[3][0]) / 2, absolute_corner[2][1])
            other_point = [absolute_corner[0], absolute_corner[1]]
            index = 0
        elif direct == 1:
            middle_point = (absolute_corner[0][0], (absolute_corner[0][1] + absolute_corner[4][1]) / 2)
            other_point = [absolute_corner[1], absolute_corner[2]]
            index = 1
        elif direct == 2:
            middle_point = ((absolute_corner[0][0] + absolute_corner[1][0]) / 2, absolute_corner[0][1])
            other_point = [absolute_corner[2], absolute_corner[3]]
            index = 2
        elif direct == 3:
            middle_point = (absolute_corner[1][0], (absolute_corner[1][1] + absolute_corner[2][1]) / 2)
            other_point = [absolute_corner[3], absolute_corner[0]]
            index = 3
        li = get_interact_points(middle_point, other_point, direct)
        pol_li = []
        for i in range(len(absolute_corner)):
            pol_li.append(absolute_corner[i])
            if i == index:
                pol_li.extend(li)
        self.easy_area = Polygon(pol_li)

    def is_car_in_slot(self, car):
        corner_list = car.get_car_absolute_corner()
        rect = self.target.rect
        top, left, bottom, right = rect.top, rect.left, rect.bottom, rect.right
        for pos in corner_list:
            if left <= pos[0] <= right and top <= pos[1] <= bottom:
                pass
            else:
                return False
        return True

    def is_car_in_slot_v2(self, car, parking):
        corner_list = car.get_car_absolute_corner()
        rect = parking.rect
        top, left, bottom, right = rect.top, rect.left, rect.bottom, rect.right
        for pos in corner_list:
            if left <= pos[0] <= right and top <= pos[1] <= bottom:
                pass
            else:
                return False
        return True

    def reset_car(self, car):
        # 0.26666666666666666, 0.05

        target_state = self.target.get_parking_info()
        target_x, target_y = target_state[0], target_state[1]
        x_bound = max(0, target_x - self.reset_x)
        x_top = min(1, target_x + self.reset_x)
        y_bound = max(0, target_y - self.reset_y)
        y_top = min(1, target_y + self.reset_y)
        while True:
            x = np.random.uniform(x_bound, x_top)
            y = np.random.uniform(y_bound, y_top)
            rot_random = np.random.uniform(0, 1)
            # if rot_random > 0.75:
            #     # rot = np.random.uniform(45, 60)
            #     rot = np.random.uniform(45, 90)
            # elif rot_random > 0.5:
            #     # rot = np.random.uniform(120, 135)
            #     rot = np.random.uniform(90, 135)
            # elif rot_random > 0.25:
            #     # rot = np.random.uniform(225, 240)
            #     rot = np.random.uniform(225, 270)
            # else:
            #     # rot = np.random.uniform(300, 335)
            #     rot = np.random.uniform(270, 335)
            rot = rot_random * 360
            car.set_car(x, y, rot)
            is_collide = car.collide_list(self.nearest_4(car.now_pos))
            flag_0 = False
            if not is_collide:
                flag_0 = True

            flag_1 = self.is_in_easy_area(car)

            flag = True
            for parking in self.parking_group:
                if parking == self.target:
                    continue
                if self.is_car_in_slot_v2(car, parking):
                    flag = False
                    break
            if flag and flag_0 and flag_1:
                break
        # self.rank_score = self.rank(car)

    def get_easy_intersect_area(self, car):
        car_rect = Polygon(car.get_car_absolute_corner())
        if car_rect.intersects(self.easy_area):  # 如果汽车的一部分进入了停车位
            try:
                intersect_area = car_rect.intersection(self.easy_area).area
            except shapely.geos.TopologicalError:
                intersect_area = 0
        else:
            intersect_area = 0
        return intersect_area

    def is_in_easy_area(self, car):
        corner_list = car.get_car_absolute_corner()
        for pos in corner_list:
            point = shapely.geometry.Point(pos)
            if self.easy_area.intersects(point):
                pass
            else:
                return False
        rot_flag = False
        if 45 <= car.now_rot <= 135 or 225 <= car.now_rot <= 315:
            rot_flag = True
        move_flag = False
        if -0.3 < car.move_speed < 0.3:
            move_flag = True
        if rot_flag and move_flag:
            return True
        return False

    def get_rank_score(self, car):
        if self.rank_score is None:
            self.rank_score = self.rank(car)
        return self.rank_score

    def update_rank_score(self, car):
        self.rank_score = self.rank(car)

    def rank(self, car):
        self.rank_car.set_car_state(car)
        i = 0
        while i < 200 and car.move_speed != 0:
            self.rank_car.car_move_and_rotate()
            self.rank_car.set_alpha(0, 0)
            self.rank_car.update_speed()

            is_collide = self.rank_car.collide_list(self.nearest_4(self.rank_car.now_pos))
            if is_collide:
                break
            i += 1
        return self.get_intersect_area(self.rank_car)

    def rank_by_elements(self, now_pos, now_rot, move_speed, phi):
        self.rank_car.set_car_state_by_elements(now_pos, now_rot, move_speed, phi)
        i = 0
        while i < 200 and move_speed != 0:
            self.rank_car.car_move_and_rotate()
            self.rank_car.set_alpha(0, 0)
            self.rank_car.update_speed()

            is_collide = self.rank_car.collide_list(self.nearest_4(self.rank_car.now_pos))
            if is_collide:
                break
            i += 1
        return self.get_intersect_area(self.rank_car)

    def get_intersect_area(self, car):
        car_rect = Polygon(car.get_car_absolute_corner())
        slot_rect = Polygon(self.target.get_absolute_corner())
        if car_rect.intersects(slot_rect):  # 如果汽车的一部分进入了停车位
            try:
                intersect_area = car_rect.intersection(slot_rect).area
            except shapely.geos.TopologicalError:
                intersect_area = 0
        else:
            intersect_area = 0
        return intersect_area

    def dis_car_target(self, car):
        car_state = car.get_state()
        target_state = self.target.get_parking_info()
        car_x, car_y = car_state[0], car_state[1]
        target_x, target_y = target_state[0] * SCREEN_WIDTH / ConstData.proportion \
            , target_state[1] * SCREEN_HEIGHT / ConstData.proportion
        return math.sqrt((car_x - target_x) * (car_x - target_x) + (car_y - target_y) * (car_y - target_y))

    def nearest_4(self, pos):
        if len(self.parking_group) == 0:
            return []
        self.parking_group.sort(key=lambda parking: (
                math.pow(pos[0] - parking.rect.center[0], 2) + math.pow(pos[1] - parking.rect.center[1], 2)))
        return self.parking_group[: min(4, len(self.parking_group))]

    def draw(self, screen):
        self.parking_sprites.draw(screen)

    # def get_nearest_4_list(self, pos):
    #     group = self.nearest_4(pos)
    #     rect = self.target.rect
    #     li = [rect.centerx / SCREEN_WIDTH, rect.centery / SCREEN_HEIGHT,
    #           rect.width / SCREEN_WIDTH, rect.height / SCREEN_HEIGHT, self.target.direct]
    #     for parking in group:
    #         li.append(parking.rect.centerx / SCREEN_WIDTH)
    #         li.append(parking.rect.centery / SCREEN_HEIGHT)
    #         li.append(parking.rect.width / SCREEN_WIDTH)
    #         li.append(parking.rect.height / SCREEN_HEIGHT)
    #     li = fill_list(li)
    #     return li

    def get_nearest_4_list(self, pos):
        group = self.nearest_4(pos)
        rect = self.target.rect
        li = [rect.centerx / ConstData.proportion, rect.centery / ConstData.proportion,
              rect.width / ConstData.proportion, rect.height / ConstData.proportion, self.target.direct]
        for parking in group:
            li.append(parking.rect.centerx / ConstData.proportion)
            li.append(parking.rect.centery / ConstData.proportion)
            li.append(parking.rect.width / ConstData.proportion)
            li.append(parking.rect.height / ConstData.proportion)
        li = fill_list(li)
        return li

    def get_relative_nearest_4_list(self, pos):
        group = self.nearest_4(pos)
        rect = self.target.rect
        x = rect.centerx / ConstData.proportion
        y = rect.centery / ConstData.proportion
        li = []
        # li = [0, 0,
        #       rect.width / ConstData.proportion, rect.height / ConstData.proportion, self.target.direct]
        for parking in group:
            li.append(parking.rect.centerx / ConstData.proportion - x)
            li.append(parking.rect.centery / ConstData.proportion - y)
            li.append(parking.rect.width / ConstData.proportion)
            li.append(parking.rect.height / ConstData.proportion)
        # li = fill_list(li)
        return li