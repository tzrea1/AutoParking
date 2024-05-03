# https://www.icauto.com.cn/baike/66/664663.html
class ConstData:
    proportion = 8

    def __init__(self):
        pass

    @staticmethod
    def get_parallel_parking():
        return 6, 2.5

    @staticmethod
    def get_vertical_parking():
        return 2.5, 6

    @staticmethod
    def get_car_width_height_L():
        return 1.8, 4.8, 2.6

    @staticmethod
    def get_parallel_parking_image():
        return 6 * ConstData.proportion, 2.5 * ConstData.proportion

    @staticmethod
    def get_vertical_parking_image():
        return 2.5 * ConstData.proportion, 6 * ConstData.proportion

    @staticmethod
    def get_car_width_height_L_image():
        return 4.8 * ConstData.proportion, 1.8 * ConstData.proportion, 2.6 * ConstData.proportion
