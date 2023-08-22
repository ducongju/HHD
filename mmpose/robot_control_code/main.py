import keyboard
from UBTech import *


vid = 0x0525
pid = 0xA4AC
robot = UBTech(vid, pid)
while(True):
    if keyboard.is_pressed('q'):
        robot.disconnect()
        exit(0)
    else:
        id = input("输入舵机ID:")
        angle = input("请输入运行角度:")
        time_need = input("请输入运行时间:")
        timeout = input("请依输入下一帧等待时间:")

        robot.controlSigleServo(int(id), int(
            angle), int(time_need), int(timeout))
