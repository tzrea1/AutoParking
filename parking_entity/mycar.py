import math

import pygame as py

from ConstData import ConstData

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


def angle_to_radian(angle):
    return angle * PI / 180


# 把角度调整到360以内
def rot_to_circle(rot):
    while rot < 0:
        rot += 360
    return rot % 360


def tuple_add(first, second):
    return tuple(map(sum, zip(first, second)))


# line [xa, ya, xb, yb]
def sort_line(line):
    if line[0] > line[2]:
        line = [line[2], line[3], line[0], line[1]]
    elif line[0] == line[2] and line[1] > line[3]:
        line = [line[2], line[3], line[0], line[1]]
    return line


#   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
def line_intersect_sec(l1, l2):
    l1 = sort_line(l1)
    l2 = sort_line(l2)
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a * b < 0 and c * d < 0:
        return True
    elif (a * b == 0 and (a + b) != 0) and c * d < 0:
        return True
    elif (c * d == 0 and (c + d) != 0) and a * b < 0:
        return True
    elif (a * b == 0 and (a + b) != 0) and (c * d == 0 and (c + d) != 0):
        return True
    elif (a * b == 0 and (a + b) == 0) or (c * d == 0 and (c + d) == 0):
        if l1[0] != l1[2]:
            if (l2[0] <= l1[0] <= l2[2]) or (l2[0] <= l1[2] <= l2[2]) or (
                    l1[0] <= l2[0] <= l1[2]) or (l1[0] <= l2[2] <= l1[2]):
                return True
            else:
                return False
        else:
            if (l2[1] <= l1[1] <= l2[3]) or (l2[1] <= l1[3] <= l2[3]) or (
                    l1[1] <= l2[1] <= l1[3]) or (l1[1] <= l2[3] <= l1[3]):
                return True
            else:
                return False
    else:
        return False


#   l1 [(xa, ya), (xb, yb)]   l2 [(xa, ya), (xb, yb)]
def line_intersect(l1, l2):
    l1 = [l1[0][0], l1[0][1], l1[1][0], l1[1][1]]
    l2 = [l2[0][0], l2[0][1], l2[1][0], l2[1][1]]
    return line_intersect_sec(l1, l2)


def get_screen_boundary(width, height):
    return [[(0, 0), (width, 0)], [(0, 0), (0, height)],
            [(width, 0), (width, height)],
            [(0, height), (width, height)]]


class My_Car(py.sprite.Sprite):
    def __init__(self):
        py.sprite.Sprite.__init__(self)
        self.car_width, self.car_height, self.L = ConstData.get_car_width_height_L_image()
        self.l0 = 1 / 4 * self.car_width
        self.l1 = 1 / 4 * self.car_width
        self.image = py.Surface((self.car_width, self.car_height))
        self.image.set_colorkey(RED)
        self.image.fill('pink')
        self.now_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        # self.now_pos = (160, 50)
        self.rect = self.image.get_rect()
        self.rect.center = self.now_pos
        self.now_rot = 0
        self.move_speed = 0
        self.rot_speed = 0
        self.alpha_move_speed = 0
        self.alpha_phi_speed = 0
        self.dt = 0.1
        # [-45, 45]
        self.phi = 0
        self.max_speed = 2
        self.is_update_speed = False

        # 设置是否上下左右
        self.is_up = False
        self.is_down = False
        self.is_left = False
        self.is_right = False
        self.is_rotate_left = False
        self.is_rotate_right = False

        self.car_move_and_rotate()

    def set_car(self, x, y, rot):
        self.now_pos = (x * SCREEN_WIDTH, y * SCREEN_HEIGHT)
        self.rect = self.image.get_rect()
        self.rect.center = self.now_pos
        self.now_rot = rot
        self.move_speed = 0
        self.rot_speed = 0
        self.alpha_move_speed = 0
        self.alpha_phi_speed = 0
        # [-45, 45]
        self.phi = 0
        self.max_speed = 2
        self.car_move_and_rotate()

    def set_car_state(self, car):
        self.now_pos = car.now_pos
        self.rect = self.image.get_rect()
        self.rect.center = self.now_pos
        self.now_rot = car.now_rot
        self.move_speed = car.move_speed
        self.rot_speed = 0
        self.alpha_move_speed = 0
        self.alpha_phi_speed = 0
        # [-45, 45]
        self.phi = car.phi
        self.max_speed = 2
        self.car_move_and_rotate()

    def set_car_state_by_elements(self, now_pos, now_rot, move_speed, phi):
        self.now_pos = now_pos
        self.rect = self.image.get_rect()
        self.rect.center = self.now_pos
        self.now_rot = now_rot
        self.move_speed = move_speed
        self.rot_speed = 0
        self.alpha_move_speed = 0
        self.alpha_phi_speed = 0
        # [-45, 45]
        self.phi = phi
        self.max_speed = 2
        self.car_move_and_rotate()
    # def get_state(self):
    #     return self.now_pos[0] / SCREEN_WIDTH, self.now_pos[1] / SCREEN_HEIGHT \
    #         , self.move_speed, angle_to_radian(self.now_rot), angle_to_radian(self.phi)

    def get_state(self):
        return self.now_pos[0] / ConstData.proportion, self.now_pos[1] / ConstData.proportion \
            , self.move_speed, angle_to_radian(self.now_rot), angle_to_radian(self.phi)

    def set_alpha(self, alpha_move_speed, alpha_phi_speed):
        self.alpha_move_speed = alpha_move_speed
        self.alpha_phi_speed = math.degrees(alpha_phi_speed)

    # 上下移动，左转右转, 调整位置以及旋转姿态
    def car_move_and_rotate(self):
        height_move = self.move_speed * math.sin(angle_to_radian(self.now_rot)) * ConstData.proportion
        width_move = self.move_speed * math.cos(angle_to_radian(self.now_rot)) * ConstData.proportion

        pos = (self.now_pos[0] + width_move * self.dt, self.now_pos[1] - height_move * self.dt)
        rot = (self.now_rot + self.rot_speed * self.dt) % 360
        if rot < 0:
            rot += 360
        self.now_rot = rot
        self.now_pos = pos

    def update_speed(self):
        self.move_speed = self.move_speed + self.alpha_move_speed * self.dt
        self.phi = self.phi + self.alpha_phi_speed * self.dt
        if self.phi > 45:
            self.phi = 45
        if self.phi < -45:
            self.phi = -45
        rot_speed_temp = self.move_speed * ConstData.proportion * math.tan(angle_to_radian(self.phi)) / self.L
        self.rot_speed = math.degrees(rot_speed_temp)
        if self.move_speed > self.max_speed:
            self.move_speed = self.max_speed
        if self.move_speed < -self.max_speed:
            self.move_speed = -self.max_speed

    # 碰撞函数
    def collide(self, parking):
        corner_list = self.get_car_absolute_corner()
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        boundarys = parking.get_absolute_boundary()
        screen_boundary = get_screen_boundary(SCREEN_WIDTH, SCREEN_HEIGHT)
        for l in lines:
            l_temp = [corner_list[l[0]], corner_list[l[1]]]
            for boundary in boundarys:
                if line_intersect(l_temp, boundary):
                    # print("相撞")
                    return True
            for boundary in screen_boundary:
                if line_intersect(l_temp, boundary):
                    # print("相撞")
                    return True
        return False

    def collide_list(self, parking_list):
        for parking in parking_list:
            is_collide = self.collide(parking)
            if is_collide:
                return True
        return False

    def update(self):
        self.car_move_and_rotate()

    # 设置矩阵中心，且旋转
    def draw_car_with_rotate(self):
        # 旋转图片， 输入必须是弧度
        new_image_temp = py.transform.rotate(self.image, self.now_rot)
        rect_temp = new_image_temp.get_rect()
        # set the rotated rectangle to the old center
        rect_temp.center = self.now_pos
        # drawing the rotated rectangle to the screen
        self.rect = rect_temp

        new_image_temp = self.set_car_head_with_black_line(new_image_temp)
        return new_image_temp

    def judge_car_width_or_height(self):
        rot = self.now_rot
        while rot < 0:
            rot += 180
        rot = rot % 180
        if (rot != 0) and rot <= 90:
            return self.car_width, self.car_height
        else:
            return self.car_height, self.car_width

    @staticmethod
    def get_x_and_y(a, b, w, h):
        return (a * a * w - a * b * h) / (a * a - b * b), (a * a * h - a * b * w) / (a * a - b * b)

    # 获取矩形的宽高
    def get_rect_w_h(self):
        h = abs(self.car_width * math.sin(angle_to_radian(self.now_rot))) + abs(self.car_height * math.cos(
            angle_to_radian(self.now_rot)))
        w = abs(self.car_width * math.cos(angle_to_radian(self.now_rot))) + abs(self.car_height * math.sin(
            angle_to_radian(self.now_rot)))
        return w, h

    # 获取车的四个角的坐标
    def get_car_inner_corner(self):
        w, h = self.get_rect_w_h()
        a, b = self.judge_car_width_or_height()
        x, y = My_Car.get_x_and_y(a, b, w, h)
        corner_list = [(x, 0), (w, h - y), (w - x, h), (0, y)]
        return corner_list

    def get_absolute_point(self):
        w, h = self.get_rect_w_h()
        return self.now_pos[0] - w / 2, self.now_pos[1] - h / 2

    # 绝对坐标系的四个点
    def get_car_absolute_corner(self):
        corner_inner_list = self.get_car_inner_corner()
        corner_list = [tuple_add(item, self.get_absolute_point()) for item in corner_inner_list]
        return corner_list

    # 绘制车头的一根黑线
    def set_car_head_with_black_line(self, new_image_temp):
        self.image.set_at(self.image.get_rect().center, BLACK)
        start, end = self.get_head()
        py.draw.line(new_image_temp, BLACK, start, end, width=5)
        return new_image_temp

    def get_head(self):
        corner_list = self.get_car_inner_corner()
        rot_normal = rot_to_circle(self.now_rot)
        first, second = 0, 0
        # 根据角度决定黑线加在哪里
        if rot_normal != 0 and rot_normal <= 90:
            first, second = 0, 1
        elif 90 < rot_normal <= 180:
            first, second = 3, 0
            # first, second = 1, 2
        elif 180 < rot_normal <= 270:
            first, second = 2, 3
            # first, second = 0, 1
        elif 270 < rot_normal < 360 or rot_normal == 0:
            first, second = 1, 2
        return corner_list[first], corner_list[second]

    def get_absolute_head(self):
        start_point, end_point = self.get_head()
        start_point = tuple_add(start_point, self.get_absolute_point())
        end_point = tuple_add(end_point, self.get_absolute_point())
        return start_point, end_point

    # 这里采用了固定加速度的方式，因为固定加速度变化量实在太难操作了
    def change_by_alpha(self):
        if self.is_up:
            self.alpha_move_speed = 0.01
        elif self.is_down:
            self.alpha_move_speed = - 0.01
        else:
            self.alpha_move_speed = 0

        if self.is_rotate_left:
            self.alpha_phi_speed = 1
        elif self.is_rotate_right:
            self.alpha_phi_speed = - 1
        else:
            self.alpha_phi_speed = 0
        # if not self.is_update_speed:
        #     self.update_speed()
        return self.alpha_move_speed, self.alpha_phi_speed

    def change_by_position(self):
        if self.is_up or self.is_down:
            self.move_speed = move_speed
        else:
            self.move_speed = 0
        if self.is_rotate_left or self.is_rotate_right:
            self.rot_speed = rot_speed
        else:
            self.rot_speed = 0

        if self.is_down is True:
            self.move_speed = - move_speed

        if self.is_rotate_right is True:
            self.rot_speed = - rot_speed
