# coding=utf-8
# 链节点类
class ListNode:
    def __init__(self, val, pre ,next=None):
        self.val = val
        self.next = next
        self.pre = pre


class LRUList:
    def __init__(self, maxsize):
        self.head = None
        self.end = None
        self.len = 0
        self.max_len = maxsize
        # val-Node 对
        self.node_map = {}

    # 尾部插入元素
    def insert_rear(self, val):
        del_val = None
        if self.len >= self.max_len:
            del_val = self.pop_front()
        if self.head is None:
            node = ListNode(val, None)
            self.head = node
            self.end = node
        else:
            node = ListNode(val, self.end)
            self.end.next = node
            self.end = node
        self.node_map[val] = node
        self.len += 1
        return del_val

    # 把值为val的元素移动至链表末尾
    def use_val(self, val):
        node = self.node_map[val]
        node_val = node.val
        if node.pre is None:
            self.head = node.next
        else:
            node.pre.next = node.next
        if node.next is None:
            self.end = node.pre
        else:
            node.next.pre = node.pre
        node.pre = None
        node.next = None
        self.len -= 1
        self.insert_rear(node_val)

    # 删除队首元素
    def pop_front(self):
        self.len -= 1
        if self.len == 0:
            self.head = 0
            self.end = 0
            self.node_map = {}
        self.node_map.pop(self.head.val)
        temp = self.head
        self.head = self.head.next
        self.head.pre = None
        temp.next = None
        temp.pre = None
        # print(temp.val)
        return temp.val

    def print_in_order(self):
        print('\n-------------------')
        print('顺序输出')
        cur = self.head
        while cur is not None:
            print(cur.val, end=' ')
            cur = cur.next
        print('\n-------------------')

    def print_in_reversed_order(self):
        print('\n-------------------')
        print('逆序输出')
        cur = self.end
        while cur is not None:
            print(cur.val, end=' ')
            cur = cur.pre
        print('\n-------------------')


if __name__ == "__main__":
    l = LRUList(2)
    l.insert_rear(1)
    l.insert_rear(2)
    l.insert_rear(3)
    l.insert_rear(4)
    l.print_in_order()


