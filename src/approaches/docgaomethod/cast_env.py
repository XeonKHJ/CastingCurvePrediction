import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path

MAX_STP_POS = 15 # 塞棒最大开度
MIN_STP_POS =0 # 塞棒最小开度
INIT_LV = 350  # 引锭头位置，距结晶器底面距离
ZERO_LV = 700  # 液位监测0点位，距结晶器底面距离
STRAT_LV = 764  # 拉矫机起步液位，距结晶器底面距离
WORK_LV = 800  # 稳定浇铸液位，距结晶器底面距离
MAX_LV = 900  # 结晶器最高液位，距结晶器底面距离
H2=1300 # 下水口高度,H1和H3与时间相关
A=11313 # 下侧孔面积


class CastEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,mode_len=1,mode_wid=1,steel_type=1,out_time=75):

    self.steel_type = steel_type #环境参数，不用管
    self.mode_area=1100*200 #结晶器横截面积
    self.out_time=out_time*2 #出苗时间/s
    self.target_lv=self.set_lv() #结晶器目标液位生成
    self.act_lv=np.zeros(self.out_time+1,dtype=np.float32) #实际液位
    self.stp_pos=np.zeros(self.out_time+1,dtype=np.float32) #塞棒位置
    self.action_space = spaces.Box(low=-MAX_STP_POS, high=MAX_STP_POS, shape=(1,),dtype=np.float32) # 动作空间
    self.observation_space = spaces.Box(low=INIT_LV, high=MAX_LV,shape=(2,),dtype=np.float32)  # 状态空间
    self.steps=0
    self.viewer = None #可视化，关闭
    high = np.array([MAX_STP_POS], dtype=np.float32)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
 #设置塞棒开度对应流量
  def stp_pos_flow(self,action,lv_act):
    H1=651+(42/19)*(self.steps/2) # H1中间包液位高度，时间函数，时间步为半秒，要除2
    if lv_act<633:
      H3=0
    else:
      H3=lv_act-633 # H3下侧出口淹没高度
    c2=0.2184*(action+15)-2.0283 # c值，action是-15至15，先加15
    dH=pow(20,0.5)*c2*A*pow((H1+H2-H3),0.5)*0.5
    dH=dH/self.mode_area # 结晶器液位高度变化值
    return dH
 #设置目标液位
  def set_lv(self):
    time=self.out_time-(14+6+40+40) #前四个开度固定时间，第五个用剩余时间
    lv1 = np.linspace(INIT_LV,380,14) #开度1起始液位和时间步（每步0.5s）
    lv2 = np.linspace(380,400,6) #开度2起始液位和时间步（每步0.5s）
    lv3 = np.linspace(400,600,40) #开度3起始液位和时间步（每步0.5s）
    lv4 = np.linspace(600,664,40) #开度4起始液位和时间步（每步0.5s）
    lv5 = np.linspace(664, 764,time+1) #开度5起始液位和时间步（每步0.5s）
    target_lv=np.concatenate([lv1,lv2,lv3,lv4,lv5])
    return target_lv

  def step(self, action):
    lv_set,lv_act=self.state
    #self.stp_pos[self.steps]=action+15
    #self.act_lv[self.steps+1]=self.stp_pos_flow(action)/self.mode_area+self.act_lv[self.steps]
    self.steps+=1
    #new_lv_act=self.stp_pos_flow(action,lv_act)/self.mode_area+lv_act
    new_lv_act=lv_act+self.stp_pos_flow(action,lv_act) # 实际液位更新：原液位+液位变化值
    new_lv_set = self.target_lv[self.steps] # 目标液位更新，数组后移
    self.state=np.array([new_lv_set,new_lv_act],dtype=np.float32)

    diff_lv=abs(lv_set-lv_act)
    cost=0.1
    if diff_lv>=10:
      cost=-1
    elif diff_lv>=5:
      cost=-0.1
    elif diff_lv<=1:
      cost=1
    done=False
    return self._get_obs(), cost, done, {}

  def reset(self):
    self.steps=0
    self.act_lv.fill(0)
    self.act_lv[0]=INIT_LV
    self.state=np.array([self.target_lv[0],self.act_lv[0]])
    return self._get_obs()

  def _get_obs(self):
    lv_set,lv_act=self.state
    return np.array([lv_set,lv_act], dtype=np.float32)

  def render(self, mode='human'):
    if self.viewer is None:
      from gym.envs.classic_control import rendering

      self.viewer = rendering.Viewer(500, 500)
      self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
      rod = rendering.make_capsule(1, 0.2)
      rod.set_color(0.8, 0.3, 0.3)
      self.pole_transform = rendering.Transform()
      rod.add_attr(self.pole_transform)
      self.viewer.add_geom(rod)
      axle = rendering.make_circle(0.05)
      axle.set_color(0, 0, 0)
      self.viewer.add_geom(axle)
      fname = path.join(path.dirname(__file__), "assets/clockwise.png")
      self.img = rendering.Image(fname, 1.0, 1.0)
      self.imgtrans = rendering.Transform()
      self.img.add_attr(self.imgtrans)

    self.viewer.add_onetime(self.img)
    self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
    if self.last_u is not None:
      self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

    return self.viewer.render(return_rgb_array=mode == "rgb_array")
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

