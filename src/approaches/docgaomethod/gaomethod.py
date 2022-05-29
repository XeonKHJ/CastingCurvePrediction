from pickletools import optimize
from param_model import ParamModel
import torch
import torch.nn as nn

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
    for line in lines:
        if is_header_passed:
            nums = line.split(',')
            current_l = float(nums[1])
            if current_l > 2:
                is_lv_detected = True
            if is_lv_detected:
                hs.append(float(nums[0]))
                ls.append((float(nums[1]) + 330))
                ts.append(0.5)
            else:
                phs.append(float(nums[0]))
                pts.append(0.5)
        else:
            is_header_passed = True
    return (hs, ls, ts, phs, pts)


def steelTypeRelatedParams(steelType="dont't know"):
    return {0.2184, 2.0283}

# TODO 能用机器学习的方式拟合H1


def calculate_h1(a, b, t):
    # H1t = 651+(42/19)*(t)
    H1t = a+(b)*(t)
    return H1t

# TODO 能用机器学习的方式拟合


def calculate_c2(a, b, h):
    # c2param1, c2param2 = steelTypeRelatedParams()
    # c2h = c2param1*(h)-c2param2  # c值，action是-15至15，先加15
    c2h =  a+ b * h
    return c2h


def stp_pos_flow(h_act, lv_act, t, dt=0.5, params=[0,0,0,0]):
    H1t = calculate_h1(params[0], params[1], t)  # H1：中间包液位高度，t的函数
    g = 9.8                 # 重力
    c2h = calculate_c2(params[2], params[3], h_act)  # C2：和钢种有关的系数

    # 引锭头顶部距离结晶器底部高度350+结晶器液位高度（距离引锭头）283
    if lv_act < 633:
        H3 = 0
    else:
        H3 = lv_act-633  # H3下侧出口淹没高度
    Ht = H1t+H2-H3
    dL = (pow(2 * g * Ht, 0.5) * c2h * A * dt) / (B * W)
    return dL

def calculate_lv_acts_tensor(hs, ts, params, batch_size, batch_first = True, previousTime = 0):
    sampleRate = 2  # 采样率是2Hz。
    # 维度为（时间，数据集数量，特征数）
    tlvs = torch.zeros([ ts.__len__(), batch_size, 1])
    lv = torch.zeros([tlvs.shape[1], 1])
    sample_count = 0
    for stage in range(ts.__len__()):
        stopTimeSpan = ts[stage]
        if stage > 0:
            previousTime += ts[stage-1]
        for time in range(int(stopTimeSpan / 0.5)):
            lv += stp_pos_flow(hs[stage], lv, previousTime + time / 2, 1 / sampleRate, params)
            tlvs[sample_count] = lv
            sample_count += 1
    if batch_first:
        tlvs = tlvs.reshape([tlvs.shape[1], tlvs.shape[0], -1])
    return tlvs

def calculate_lv_acts(hs, ts, params):
    sampleRate = 2  # 采样率是2Hz。
    lv_acts = list()
    lv_act = 0.0
    for stage in range(hs.__len__()):
        stopTimeSpan = ts[stage]
        if stage > 0:
            previousTime += ts[stage-1]
        else:
            previousTime = 0
        for time in range(int(stopTimeSpan / 0.5)):
            # print(previousTime + time / 2)
            lv_act += stp_pos_flow(hs[stage], lv_act,
                                   previousTime + time / 2, 1 / sampleRate, params)
            lv_acts.append(lv_act)
    # for c in lv_acts:
    #     print(c)
    return lv_acts


if __name__ == '__main__':
    hs, ls, ts, phs, pts = get_input()
    pm = ParamModel()
    lpm = l

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(pm.parameters(), lr=1e-2)

    # 处理hs，phs代表tesnor hs。
    ths = torch.tensor(hs)  # 代表处理过后的hs，1维。Shape为(时长)
    # reshape后的tensor为3维。(数据集数量, 时长, 特征数)
    ths = ths.reshape([-1, hs.__len__(), 1])
    tphs = torch.tensor(phs).reshape([-1, phs.__len__(), 1])

    # 处理ls
    tls_act = torch.tensor(ls)
    tls_act = tls_act.reshape([ -1, ls.__len__(), 1])

    output_act = calculate_lv_acts_tensor(ths[0], ts, [651, 42/19, -2.0283, 0.2184], 1, previousTime=phs.__len__()*0.5)

    epoch = 0
    while True:
        epoch += 1
        output = pm(ths)
        tls_pred = calculate_lv_acts_tensor(ths[0], ts, output, ths.shape[0], previousTime=phs.__len__()*0.5)
        tls_pred = tls_pred / 500
        loss = loss_function(tls_pred, tls_act /500)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if epoch > 120:
        #     tphs_output = pm(tphs)
        #     tpls_act = calculate_lv_acts_tensor(pts, tphs_output, 1)

