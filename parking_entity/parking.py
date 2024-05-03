import math

import pygame as py


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


def tuple_add(first, second):
    return tuple(map(sum, zip(first, second)))


class Parking(py.sprite.Sprite):
    # size: (width, height)
    def __init__(self, size, is_target, pos, direct=None):
        py.sprite.Sprite.__init__(self)
        if 0 <= size[0] <= 1 and 0 <= size[1] <= 1:
            size = (size[0] * SCREEN_WIDTH, size[1] * SCREEN_HEIGHT)
        if 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1:
            pos = (pos[0] * SCREEN_WIDTH, pos[1] * SCREEN_HEIGHT)
        self.image = py.Surface(size)
        self.direct = None
        self.is_target = is_target
        if is_target:
            color = 'green'
            self.direct = direct
        else:
            color = 'pink'
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = pos
        self.line_list = []
        self.draw_boundary()
        if size[0] > size[1]:
            self.theta = 0
        else:
            self.theta = PI / 2

    def get_parking_info(self):
        if self.is_target:
            return [self.rect.centerx / SCREEN_WIDTH, self.rect.centery / SCREEN_HEIGHT,
                    self.rect.width / SCREEN_WIDTH, self.rect.height / SCREEN_HEIGHT, self.direct]
        else:
            return [self.rect.centerx / SCREEN_WIDTH, self.rect.centery / SCREEN_HEIGHT, self.rect.width / SCREEN_WIDTH,
                    self.rect.height / SCREEN_HEIGHT]

    def get_absolute_point(self):
        return self.rect[0], self.rect[1]

    # 上下左右顺序的边界线
    # up: 0, right: 1, bottom: 2, left: 3
    def get_boundary(self):
        line_list = []
        if self.direct is None or self.direct != 0:
            up = [self.image.get_rect().topleft, self.image.get_rect().topright]
            line_list.append(up)
        if self.direct is None or self.direct != 2:
            bottom = [self.image.get_rect().bottomleft, self.image.get_rect().bottomright]
            line_list.append(bottom)
        if self.direct is None or self.direct != 3:
            left = [self.image.get_rect().topleft, self.image.get_rect().bottomleft]
            line_list.append(left)
        if self.direct is None or self.direct != 1:
            right = [self.image.get_rect().topright, self.image.get_rect().bottomright]
            line_list.append(right)
        return line_list

    def get_absolute_corner(self):
        corners = [self.image.get_rect().topleft, self.image.get_rect().topright,
                   self.image.get_rect().bottomright, self.image.get_rect().bottomleft]
        absolute_corner = [tuple_add(item, self.get_absolute_point()) for item in corners]
        return absolute_corner

    # 上下左右顺序的边界线，绝对坐标系种
    def get_absolute_boundary(self):
        boundarys = self.get_boundary()
        line_list = []
        for boundary in boundarys:
            line_list.append([tuple_add(item, self.get_absolute_point()) for item in boundary])
        return line_list

    def draw_boundary(self):
        boundary = self.get_boundary()
        for line in boundary:
            now_line = py.draw.line(self.image, 'red', line[0], line[1], width=5)
            self.line_list.append(now_line)