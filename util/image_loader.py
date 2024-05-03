# coding: utf-8
import abc
import os
import queue

import cv2 as cv
import torchvision.transforms as transforms

# 图片加载
from util.LRUList import LRUList

before_dir = '../'


class Image_loader(metaclass=abc.ABCMeta):
    myself = None

    def __init__(self, max_len):
        self.max_len = max_len
        self.img_map = {}

    def ini_load_img(self):
        index = 0
        for root, dirs, files in os.walk('../map'):
            for file in files:
                if index >= self.max_len:
                    break
                img_id = before_dir + 'map/' + file
                self.load_img(img_id)
                index += 1
        # print('初始加载结束')

    @abc.abstractmethod
    def append(self, img_id):
        pass

    @abc.abstractmethod
    def del_one(self):
        pass

    @abc.abstractmethod
    def use(self, val):
        pass

    @classmethod
    @abc.abstractmethod
    def get_instance(cls, max_len):
        pass

    # 获取图像所对应的Tensor
    def get_img_tensor(self, img_id):
        # print(len(self.img_map))
        if img_id in self.img_map.keys():
            self.use(img_id)
            return self.img_map[img_id]
        else:
            self.load_img(img_id)
            self.use(img_id)
            return self.img_map[img_id]

    # 加载图片并转换为Tensor
    def load_img(self, img_id):
        if img_id in self.img_map.keys():
            return
        img = cv.imread(img_id)
        # print(img.shape)  # numpy数组格式为（H,W,C）
        transf = transforms.ToTensor()
        img_tensor = transf(img).reshape(1, img.shape[2], img.shape[0], img.shape[1])  # tensor数据格式是torch(C,H,W)
        # print(img_tensor.size())
        # print(img_tensor)
        self.append(img_id)
        self.img_map[img_id] = img_tensor.cuda()


class FIFO_Image_loader(Image_loader):
    def __init__(self, max_len):
        super().__init__(max_len)
        self.q = queue.Queue(maxsize=max_len)
        self.ini_load_img()

    def append(self, img_id):
        if self.q.full():
            self.del_one()
        # print('加入:' + img_id)
        self.q.put(img_id)

    def del_one(self):
        if self.q.qsize() > 0:
            d = self.q.get()
            # print('删除:' + d)
            self.img_map.pop(d)
            return d

    def use(self, val):
        pass

    @classmethod
    def get_instance(cls, max_len):
        if cls.myself is None:
            cls.myself = cls(max_len)
        return cls.myself


class LRU_Image_loader(Image_loader):
    def __init__(self, max_len):
        super().__init__(max_len)
        self.list = LRUList(maxsize=max_len)
        self.ini_load_img()

    def append(self, img_id):
        # print('加入:' + img_id)
        val = self.list.insert_rear(img_id)
        if val is not None:
            # print('删除:' + val)
            self.img_map.pop(val)

    def del_one(self):
        pass

    def use(self, val):
        self.list.use_val(val)

    @classmethod
    def get_instance(cls, max_len):
        if cls.myself is None:
            cls.myself = cls(max_len)
        return cls.myself


if __name__ == "__main__":
    img_loader = LRU_Image_loader.get_instance(2)
    print(id(img_loader))
    img_loader.get_img_tensor(before_dir + 'map/1-1.png')
