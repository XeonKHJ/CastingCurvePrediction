from cast_env import A

B = 1 # 连铸坯宽度
W = 1 # 连铸坯厚度
L = 1 # 结晶器内液面高度
c2h = 1 # c2(h)：流量系数
A = 1 # 下水口侧孔面积
Ht = 10 #计算水头高
H1t = 1 # 中间包液面高度
H2 = 1 # 下水口水头高度
H3 = 2 # 下侧孔淹没高度
h = 1 # 塞棒高度


def stp_pos_flow(h, lv_act, t):
    H1 = 651+(42/19)*(t)  # H1中间包液位高度，时间函数，时间步为半秒，要除2
    
    # 引锭头顶部距离结晶器底部高度350+结晶器液位高度（距离引锭头）283
    if lv_act < 633:
        H3 = 0
    else:
        H3 = lv_act-633  # H3下侧出口淹没高度
    c2 = 0.2184*(h+15)-2.0283  # c值，action是-15至15，先加15
    dL = (pow(20, 0.5) * c2*A*pow((H1+H2-H3), 0.5)*0.5) / (B * W)
    return dL


if __name__ == '__main__':
    hs = [100, 200, 300, 400, 500]
    t = [10, 20, 10, 20, 10]
    lv_acts = list()
    lv_act = 0
    for stage in range(5):
        sampleCount = t[stage]
        for time in range(sampleCount):
            if stage > 0:
                previousTime = t[stage-1]
            else:
                previousTime = 0
            lv_act += stp_pos_flow(hs[stage], lv_act, previousTime + time)
            lv_acts.append(lv_act)