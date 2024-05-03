import json
import math

import pygame as py

from parking_entity.mycar import My_Car
from parking_entity.parkinggroup import ParkingGroup
from util.map_load import *

# 数学常量
PI = math.pi

# 主屏幕设置
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 500
FPS = 60

# 颜色设置
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (0, 0, 255)

# 设置是否上下左右
is_up = False
is_down = False
is_left = False
is_right = False
is_rotate_left = False
is_rotate_right = False

# 初始话并创建屏幕，设置fps
py.init()


def angle_to_radian(angle):
    return angle * PI / 180


def key_operate(event, car):
    global is_up
    global is_down
    global is_left
    global is_right
    global is_rotate_left
    global is_rotate_right
    if event.type == py.KEYDOWN:
        if event.key == py.K_w:
            # 向上
            is_up = True
            car.is_up = is_up
        if event.key == py.K_s:
            # 向下
            is_down = True
            car.is_down = is_down
        if event.key == py.K_a:
            # 左转
            is_rotate_left = True
            car.is_rotate_left = is_rotate_left
        if event.key == py.K_d:
            # 右转
            is_rotate_right = True
            car.is_rotate_right = is_rotate_right
    if event.type == py.KEYUP:
        # print('键盘弹起', chr(event.key))
        if event.key == py.K_w:
            # 向上
            is_up = False
            car.is_up = is_up
        if event.key == py.K_s:
            # 向下
            is_down = False
            car.is_down = is_down
        if event.key == py.K_a:
            # 左转
            is_rotate_left = False
            car.is_rotate_left = is_rotate_left
        if event.key == py.K_d:
            # 右转
            is_rotate_right = False
            car.is_rotate_right = is_rotate_right


def operate_by_position(event, car):
    key_operate(event, car)
    car.change_by_position()


def operate_by_alpha(event, car):
    key_operate(event, car)
    dv, dphi = car.change_by_alpha()
    return dv, dphi


class Visualization:
    def __init__(self, car, parking_group):
        self.car = car
        self.parking_group = parking_group
        # 主屏幕设置
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 500
        self.FPS = 60
        # 初始话并创建屏幕，设置fps
        py.init()
        self.screen = py.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = py.time.Clock()

    def visualization(self):
        running = True
        while running:
            # set FPS
            self.clock.tick(FPS)
            # clear the screen every time before drawing new objects
            self.screen.fill(WHITE)
            self.create_screen_boundary()
            self.display_v_a(self.car)
            self.car.is_update_speed = False
            # check for the exit
            for event in py.event.get():
                if event.type == py.QUIT:
                    running = False
                # 加速度变化量固定
                operate_by_alpha(event, self.car)
                self.car.is_update_speed = True
            self.parking_group.draw(self.screen)
            self.car.update()
            new_image = self.car.draw_car_with_rotate()
            self.screen.blit(new_image, self.car.rect)
            is_collide = self.car.collide_list(self.parking_group.nearest_4(self.car.now_pos))
            if is_collide:
                self.write_collide()
                self.parking_group.reset_car(self.car)
            py.display.flip()

        py.quit()

    def move_like_env(self):
        running = True
        while running:
            # set FPS
            self.clock.tick(FPS)
            # clear the screen every time before drawing new objects
            self.screen.fill(WHITE)
            self.create_screen_boundary()
            self.display_v_a(self.car)
            self.car.is_update_speed = False
            # check for the exit
            for event in py.event.get():
                if event.type == py.QUIT:
                    running = False
                # 加速度变化量固定
                action = operate_by_alpha(event, self.car)
                self.car.is_update_speed = True

            self.car.car_move_and_rotate()
            dv, ddelta = action
            # print(dv, ddelta)
            self.car.set_alpha(dv, ddelta)
            self.car.update_speed()

            is_collide = self.car.collide_list(self.parking_group.nearest_4(self.car.now_pos))
            if is_collide:
                self.write_collide()
                self.parking_group.reset_car(self.car)
            # py.display.flip()
            self.run()

        py.quit()

    def run(self):
        # clear the screen every time before drawing new objects
        self.screen.fill(WHITE)
        self.create_screen_boundary()
        self.display_v_a(self.car)
        # check for the exit
        self.parking_group.draw(self.screen)
        new_image = self.car.draw_car_with_rotate()
        self.screen.blit(new_image, self.car.rect)
        # is_collide = self.car.collide_list(self.parking_group.nearest_4(self.car.now_pos))
        # if is_collide:
        #     self.write_collide()
        #     self.parking_group.reset_car(self.car)
        py.display.flip()

    def create_screen_boundary(self):
        screen_boundary = self.get_screen_boundary()
        for line in screen_boundary:
            py.draw.line(self.screen, BLACK, line[0], line[1], width=5)

    def get_screen_boundary(self):
        return [[(0, 0), (self.SCREEN_WIDTH, 0)], [(0, 0), (0, self.SCREEN_HEIGHT)],
                [(self.SCREEN_WIDTH, 0), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)],
                [(0, self.SCREEN_HEIGHT), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)]]

    def write_success(self):
        self.write_text('停车成功', (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 80)

    def write_collide(self):
        self.write_text('撞车了', (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 80)

    def write_text(self, text, pos, size=18):
        # 引入字体类型
        f = py.font.Font('C:/Windows/Fonts/simhei.ttf', size)
        # 生成文本信息，第一个参数文本内容；第二个参数，字体是否平滑；
        # 第三个参数，RGB模式的字体颜色；第四个参数，RGB模式字体背景颜色；
        text = f.render(text, True, (255, 0, 0), (0, 0, 0))
        # 获得显示对象的rect区域坐标
        textRect = text.get_rect()
        # 设置显示对象居中
        textRect.center = pos
        # 将准备好的文本信息，绘制到主屏幕 Screen 上。
        self.screen.blit(text, textRect)

    def display_v_a(self, car):
        self.write_text("%.2f" % car.move_speed + 'm/s', (50, 10))
        self.write_text("%.2f" % car.alpha_move_speed + 'm/s^2', (50, 30))
        self.write_text("%.2f" % car.rot_speed + '°/s', (50, 50))
        # write_text("%.2f" % car.alpha_rot_speed + '°/s^2', (50, 70))


def start_by_hand():
    car = My_Car()
    car.set_car(0.6, 0.52, 0)
    parkingGroup = load_dest_v2(3)
    vis = Visualization(car, parkingGroup)
    vis.move_like_env()


start_by_hand()
