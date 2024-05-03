import streamlit as st
import copy

import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import numpy as np
duration = 2  # 持续时间

st.write("""
# 3D 狗子轨迹显示
""")

S = st.slider('请输入步长:', 0, 100)
H = st.slider('请输入步高:', 0, 100)
fps = st.slider('请输入一个周期的时长:', 1, 100)


class BEZIER:
    def __init__(self, BezierX2=[0, 0], BezierY2=[0, 0], BezierX6=[0, 0, 0, 0, 0, 0], BezierY6=[0, 0, 0, 0, 0, 0],
                 BezierX12=[], BezierY12=[], H=H, L=S,
                 start_x=0, start_y=174, BezierLen=6):
        self.BezierX2 = BezierX2
        self.BezierY2 = BezierY2
        self.BezierX6 = BezierX6
        self.BezierY6 = BezierY6
        self.BezierX12 = BezierX12
        self.BezierY12 = BezierY12
        self.H = H
        self.L = L
        self.start_x = start_x
        self.start_y = start_y
        self.BezierLen = BezierLen


Bezier = [BEZIER() for _ in range(2)]


class VECTOR2D:
    def __init__(self, x=[0, 0, 0, 0, 0, 0], y=[0, 0, 0, 0, 0, 0]):
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)


points = [VECTOR2D() for _ in range(2)]


def TrotControlClass_updateBezierParam():
    # 2阶贝塞尔规划，着地相直线
    Bezier[0].BezierX2 = [-Bezier[0].L / 2, Bezier[0].L / 2]
    Bezier[0].BezierY2 = [0, 0]
    for i in range(2):
        Bezier[0].BezierX2[i] += Bezier[0].start_x
        Bezier[0].BezierY2[i] += Bezier[0].start_y
    if Bezier[0].BezierLen == 6:
        Bezier[0].BezierX6[0] = (-Bezier[0].L / 2)
        Bezier[0].BezierX6[1] = (-Bezier[0].L / 2) * 0.6
        Bezier[0].BezierX6[2] = (-Bezier[0].L / 2) * 0.2
        Bezier[0].BezierX6[3] = (Bezier[0].L / 2) * 0.2
        Bezier[0].BezierX6[4] = (Bezier[0].L / 2) * 0.6
        Bezier[0].BezierX6[5] = (Bezier[0].L / 2)
        Bezier[0].BezierY6[0] = -0
        Bezier[0].BezierY6[1] = -H * 0.1
        Bezier[0].BezierY6[2] = -H * 0.2
        Bezier[0].BezierY6[3] = -H * 0.2
        Bezier[0].BezierY6[4] = -H * 0.1
        Bezier[0].BezierY6[5] = -0
        for i in range(6):
            Bezier[0].BezierX6[i] += Bezier[0].start_x
            Bezier[0].BezierY6[i] += Bezier[0].start_y
    elif Bezier[0].BezierLen == 12:
        dL = Bezier[0].L / 11
        Bezier[0].BezierX12[0] = Bezier[0].L / 2
        Bezier[0].BezierX12[1] = Bezier[0].L / 2 + dL
        Bezier[0].BezierX12[2] = Bezier[0].L / 2 + 2 * dL
        Bezier[0].BezierX12[3] = Bezier[0].L / 2 + 3 * dL
        Bezier[0].BezierX12[4] = Bezier[0].L / 2 + 2 * dL
        Bezier[0].BezierX12[5] = Bezier[0].L / 2
        Bezier[0].BezierX12[6] = -Bezier[0].L / 2
        Bezier[0].BezierX12[7] = -Bezier[0].L / 2 - 2 * dL
        Bezier[0].BezierX12[8] = -Bezier[0].L / 2 - 3 * dL
        Bezier[0].BezierX12[9] = -Bezier[0].L / 2 - 2 * dL
        Bezier[0].BezierX12[10] = -Bezier[0].L / 2 - dL
        Bezier[0].BezierX12[11] = -Bezier[0].L / 2
        Bezier[0].BezierY12[0] = 0
        Bezier[0].BezierY12[1] = 0
        Bezier[0].BezierY12[2] = -H / 3
        Bezier[0].BezierY12[3] = -H
        Bezier[0].BezierY12[4] = -H
        Bezier[0].BezierY12[5] = -H * 6 / 5
        Bezier[0].BezierY12[6] = -H * 6 / 5
        Bezier[0].BezierY12[7] = -H
        Bezier[0].BezierY12[8] = -H
        Bezier[0].BezierY12[9] = -H / 3
        Bezier[0].BezierY12[10] = 0
        Bezier[0].BezierY12[11] = 0
        for i in range(12):
            Bezier[0].BezierX12[i] += Bezier[0].start_x
            Bezier[0].BezierY12[i] += Bezier[0].start_y


def get_BezierPoints_rank6(Bezier):
    for i in range(6):
        points[0].x[i] = Bezier.BezierX6[i]
        points[0].y[i] = Bezier.BezierY6[i]


def get_BezierPoints_rank2(Bezier):
    for i in range(2):
        points[1].x[i] = Bezier.BezierX2[i]
        points[1].y[i] = Bezier.BezierY2[i]


def getBezier_chang(points, rank, t, noneliner):
    if t > 1:
        t = 1
    if t < 0:
        t = 0
    point = copy.deepcopy(points)

    t = t ** noneliner

    for r in range(1, rank):
        for i in range(rank - r):
            point.x[i] = (point.x[i] * (1 - t) + point.x[i + 1] * t)
            point.y[i] = (point.y[i] * (1 - t) + point.y[i + 1] * t)

    return point.x[0], -point.y[0]


TrotControlClass_updateBezierParam()
get_BezierPoints_rank6(Bezier[0])
get_BezierPoints_rank2(Bezier[0])


def get_point(i):
    Set_Point1 = [0, 0, 0]
    i = int(i)
    temp_t = i / fps
    if temp_t >= 2:
        temp_t -= 2
    if temp_t <= 1:
        Set_Point1[0], Set_Point1[1] = getBezier_chang(points[0], 6, temp_t, 1)
        return Set_Point1[0], Set_Point1[1]
    elif temp_t > 1:
        temp_t = temp_t - 1
        Set_Point1[0], Set_Point1[1] = getBezier_chang(points[1], 2, temp_t, 1)
        return -Set_Point1[0], Set_Point1[1]


fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 1, 1, projection="3d")
ax1.set_xlim(-335, 135)
ax1.set_ylim(-225, 225)
ax1.set_zlim(-200, 250)
ax1.grid(False)
ax1.fill()

x = lambda i: [0, np.cos(np.arctan2(get_point(i)[1], get_point(i)[0]) - np.arccos(
    (100 * 100 + np.sqrt(get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) * np.sqrt(
        get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) - 200 * 200) / (
            2 * 100 * np.sqrt(get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1])))) * 100,
               get_point(i)[0],
               np.cos(np.arctan2(get_point(i)[1], get_point(i)[0]) + np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) * np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1])))) * 100, 0, 0,
               np.cos(np.arctan2(get_point(i + fps)[1], get_point(i + fps)[0]) - np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1])))) * 100,
               get_point(i + fps)[0],
               np.cos(np.arctan2(get_point(i + fps)[1], get_point(i + fps)[0]) + np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1])))) * 100, 0]
z = lambda i: [0, np.sin(np.arctan2(get_point(i)[1], get_point(i)[0]) - np.arccos(
    (100 * 100 + np.sqrt(get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) * np.sqrt(
        get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) - 200 * 200) / (
            2 * 100 * np.sqrt(get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1])))) * 100,
               get_point(i)[1],
               np.sin(np.arctan2(get_point(i)[1], get_point(i)[0]) + np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) * np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i)[0] * get_point(i)[0] + get_point(i)[1] * get_point(i)[1])))) * 100, 0, 0,
               np.sin(np.arctan2(get_point(i + fps)[1], get_point(i + fps)[0]) - np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1])))) * 100,
               get_point(i + fps)[1],
               np.sin(np.arctan2(get_point(i + fps)[1], get_point(i + fps)[0]) + np.arccos(
                   (100 * 100 + np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1]) - 200 * 200) / (
                           2 * 100 * np.sqrt(
                       get_point(i + fps)[0] * get_point(i + fps)[0] + get_point(i + fps)[1] * get_point(i + fps)[
                           1])))) * 100, 0]

y = lambda i: [-100, -100, -100, -100, -100, 100, 100, 100, 100, 100]
# ax1.plot(x, y, z, label="3d curve")  # 其余参数同2d图
# ax1.legend()
# plt.show()
ax1.set_title("what are you doing?")
line1, = ax1.plot(x(0), y(0), z(0), lw=3, color='#93d5dc')
list_0 = x(0) - np.array([250, 250, 250, 250, 250, 250, 250, 250, 250, 250])
line4, = ax1.plot(list_0, y(0), z(0), lw=3, color='#93d5dc')
line5, = ax1.plot([0, 0, -250, -250, 0], [-110, 110, 110, -110, -110], [0, 0, 0, 0, 0], lw=3, color='#93d5dc')

t1 = np.linspace(0, 1, 1000)

Set_Point2 = np.zeros((len(t1), 3))
for i in range(len(t1)):
    temp_t = i / 500
    if temp_t <= 1:
        Set_Point2[i, 0], Set_Point2[i, 1] = getBezier_chang(points[0], 6, temp_t, 1)
        Set_Point2[i, 2] = -100
    elif temp_t > 1:
        temp_t = temp_t - 1
        Set_Point2[i, 0], Set_Point2[i, 1] = getBezier_chang(points[1], 2, temp_t, 1)
        Set_Point2[i, 2] = -100

x2 = Set_Point2[:, 0]
y2 = Set_Point2[:, 2]
z2 = Set_Point2[:, 1]
line2, = ax1.plot(x2, y2, z2, lw=1)
line3, = ax1.plot(x2, -y2, z2, lw=1)


def make_frame_mpl(t):
    temp_m = (t * fps) // 1
    line1.set_ydata(y(temp_m))  # 更新曲面
    line1.set_xdata(x(temp_m))  # 更新曲面
    line1.set_3d_properties(z(temp_m))  # 更新曲面
    line4.set_ydata(y(temp_m + fps))  # 更新曲面
    line4.set_xdata(x(temp_m + fps) - np.array([250, 250, 250, 250, 250, 250, 250, 250, 250, 250]))  # 更新曲面
    line4.set_3d_properties(z(temp_m + fps))  # 更新曲面
    return mplfig_to_npimage(fig)  # 图形的RGB图像


duration = 2
animation = mpy.VideoClip(make_frame_mpl, duration=duration)
animation.write_gif("3D狗子轨迹.gif", fps=fps)

# 加载GIF图片
gif_path = "3D狗子轨迹.gif"
gif = open(gif_path, "rb").read()

# 在Streamlit中显示GIF图片
st.image(gif)
