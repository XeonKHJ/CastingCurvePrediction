from pickletools import optimize
from liner_param_model import LinerParamModel
from param_model import ParamModel
import torch
import torch.nn as nn

# 最好不要动这里面的算法

B = 1250  # 连铸坯宽度
W = 230  # 连铸坯厚度
L = 1  # 结晶器内液面高度
c2h = 1  # c2(h)：流量系数
A = 11313  # 下水口侧孔面积
Ht = 10  # 计算水头高
H1t = 1  # 中间包液面高度
H2 = 1300  # 下水口水头高度
H3 = 2  # 下侧孔淹没高度，需要计算
h = 1  # 塞棒高度
sensor_to_dummy_bar_height = 350
def get_input():
    file_path = "dataset/2021-04-07-17-54-00-strand-1.csv"
    file = open(file_path)
    lines = file.readlines()
    hs = list()
    ts = list()
    ls = list()

    phs = list()
    pts = list()

    is_header_passed = False
    is_lv_detected = False  # 在结晶器中的钢液是否能被检测到
    ready_to_start = False

    for line in lines:
        if is_header_passed:
            nums = line.split(',')
            current_l = float(nums[1])
            if is_lv_detected:
                hs.append(float(nums[0]))
                ls.append((float(nums[1]) + sensor_to_dummy_bar_height))
                ts.append(0.5)
            if ready_to_start and not is_lv_detected:
                pre_lv_act = float(nums[1]) + sensor_to_dummy_bar_height
                is_lv_detected = True
            if current_l > 2:
                ready_to_start = True
            else:
                phs.append(float(nums[0]))
                pts.append(0.5)
        else:
            is_header_passed = True
    return (hs, ls, ts, phs, pts, pre_lv_act)


def steelTypeRelatedParams(steelType="dont't know"):
    return {0.2184, 2.0283}


def calculate_h1(t):
    H1t = 651+(42/19)*(t)
    return H1t

def calculate_c2(h):
    c2param1, c2param2 = steelTypeRelatedParams()
    c2h = c2param1*(h)-c2param2  # c值，action是-15至15，先加15
    return c2h


def stp_pos_flow(h_act, lv_act, t, dt=0.5):
    H1t = calculate_h1(t)  # H1：中间包液位高度，t的函数
    g = 9.8                 # 重力
    c2h = calculate_c2(h_act)  # C2：和钢种有关的系数

    # 引锭头顶部距离结晶器底部高度350+结晶器液位高度（距离引锭头）283
    if lv_act < 283:
        H3 = 0
    else:
        H3 = lv_act - 283  # H3下侧出口淹没高度
    Ht = H1t+H2-H3
    dL = (pow(2 * g * Ht, 0.5) * c2h * A * dt) / (B * W)
    return dL

def calculate_lv_acts(hs, ts, init_lv_act = 0.0, previous_time = 0):
    sampleRate = 2  # 采样率是2Hz。
    lv_acts = list()
    lv_act = init_lv_act
    for stage in range(hs.__len__()):
        stopTimeSpan = ts[stage]
        if stage > 0:
            previous_time += ts[stage-1]
        else:
            previous_time = 0
        for time in range(int(stopTimeSpan / 0.5)):
            # print(previousTime + time / 2)
            dlv_act = stp_pos_flow(hs[stage], lv_act,
                                   previous_time + time / 2, 1 / sampleRate)
            lv_act += dlv_act
            lv_acts.append(lv_act[0].item())
    # for c in lv_acts:
    #     print(c)
    return lv_acts

if __name__ == '__main__':
    hs, ls, ts, phs, pts, pre_lv_act = get_input()

    # 处理hs，phs代表tesnor hs。
    ths = torch.tensor(hs)  # 代表处理过后的hs，1维。Shape为(时长)
    # 将时长变成数据集数量
    thstob = ths.reshape([-1, 1])
    # reshape后的tensor为3维。(数据集数量, 时长, 特征数)
    ths = ths.reshape([-1, hs.__len__(), 1])
    tphs = torch.tensor(phs).reshape([-1, phs.__len__(), 1])

    # 处理ls
    tls_act = torch.tensor(ls)
    tls_act = tls_act.reshape([ -1, ls.__len__(), 1])

    output_acts = calculate_lv_acts(ths[0], ts, init_lv_act = sensor_to_dummy_bar_height, previous_time=phs.__len__()*0.5)
    print("STP_POS,LV_ACT,LV_PRED")
    for i in range(output_acts.__len__()):
        print(str(hs[i]) + ',' + str(ls[i]) + ',' + str(output_acts[i]))
    
