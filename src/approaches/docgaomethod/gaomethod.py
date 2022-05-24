from cast_env import A

B = 1 # 连铸坯宽度
W = 1 # 连铸坯厚度
L = 1 # 结晶器内液面高度
c2h = 1 # c2(h)：流量系数
A = 1 # 下水口侧孔面积
Ht = 10 #计算水头高
H1t = 1 # 中间包页面高度
H2 = 1 # 下水口水头高度
H3 = 2 # 下侧孔淹没高度


def stp_pos_flow(action, lv_act, count):
    H1 = 651+(42/19)*(steps/2)  # H1中间包液位高度，时间函数，时间步为半秒，要除2
    if lv_act < 633:
        H3 = 0
    else:
        H3 = lv_act-633  # H3下侧出口淹没高度
    c2 = 0.2184*(action+15)-2.0283  # c值，action是-15至15，先加15
    dH = pow(20, 0.5) * c2*A*pow((H1+H2-H3), 0.5)*0.5
    dH = dH / mode_area  # 结晶器液位高度变化值
    
    # 特征方程
    # B * W * (dL / dt) = c2(h)A * sqrt(2gH(t))
    
    return dH


if __name__ == '__main__':
    print("hello")
