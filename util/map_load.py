import json
from parking_entity.parkinggroup import ParkingGroup


# 加载老地图数据
def load_dest_v1(index):
    path = 'dest/' + str(index) + '.json'
    with open(path, "r", encoding='utf-8') as f:
        row_data = json.load(f)
        data = json.loads(row_data)
    # 读取每一条json数据
    parkings = data['parkings']
    parking_group = ParkingGroup()
    for key, value in parkings.items():
        parking_group.append_by_list(value)
    return parking_group


# 加载新地图数据
def load_dest_v2(index):
    path = 'dest-new/' + str(index) + '.json'
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    # 读取每一条json数据
    parkings = data['parkings']
    parking_group = ParkingGroup()
    for key, value in parkings.items():
        parking_group.append_by_list(value)
    return parking_group
