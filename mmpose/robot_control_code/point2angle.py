import numpy as np

'''
    v1 = [0, 1]
    v2 = [1, 0]
    angle = math.atan2(v1, v2)
    print(angle)
    angle = -int(angle*180/math.pi)
    if angle < 0:
        angle += 360
    print(angle)
'''


def vector2angle(v1, v2):

    v1_len = np.sqrt(np.dot(v1, v1))
    v2_len = np.sqrt(np.dot(v2, v2))
    cos_angle = np.dot(v1, v2)/(v1_len*v2_len)
    angle = np.degrees(np.arccos(cos_angle))

    return int(angle)


def plane2angle(p1, p2, p3, p4, p5, p6):

    p1v1 = p1-p2
    p1v2 = p1-p3
    p2v1 = p4-p5
    p2v2 = p4-p6
    n1 = np.cross(p1v1, p1v2)
    n2 = np.cross(p2v1, p2v2)
    angle = vector2angle(n1, n2)

    if angle > 90:
        angle = 180-angle

    return angle


def pv2angle(p1, p2, p3, p4, p5):

    p1v1 = p1-p2
    p1v2 = p1-p3
    n1 = np.cross(p1v1, p1v2)
    v3 = p4-p5

    angle = vector2angle(n1, v3)
    if angle >= 90:
        angle -= 90
    else:
        angle = 90-angle
    return angle


v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

print(vector2angle(v1, v2))
