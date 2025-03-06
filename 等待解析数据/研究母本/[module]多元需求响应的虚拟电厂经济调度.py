"""
Description:本文档建模为只报量建模,考虑多灵活性负荷资源
Author: Huahsin Sun
Date-Version :
2024.10.18 Version 2.0
2024.11.7 version 2.1 modularize the code
说明:
2024.10.22
这个文档预设所有元件尺度单位都是mwh,如果需要建模ess,tcr和dl为kwh,需要修改部分代码,加入缩放因子,详见中国移动项目代码
"""
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from docplex.mp.context import Context

# Set up the default context for the optimization model
context = Context.make_default_context()
context.cpus = 4




class data_info():
    def __init__(self):
        # data preparation
        self.power_load = [ 3.84, 3.92, 3.76, 3.44, 3.28, 3.36, 3.6, 4.64, 15, 6, 5.76,
                       5.84, 5.76, 5.6, 5.44, 5.2, 4.96, 4.8, 4.72, 4.72, 4.8, 4.96,
                       5.2, 4.8 ]
        # price
        self.price = [
            40.4411764700000, 38.9705882400000, 36.0294117600000, 33.3823529400000, 30.5882352900000,
            28.9705882400000, 32.9411764700000, 20.4411764700000, 43.5294117600000, 48.5294117600000,
            54.4117647100000, 55.0000000000000, 45.5882352900000, 46.3235294100000, 51.4705882400000,
            51.7647058800000, 53.8235294100000, 51.4705882400000, 44.8529411800000, 48.5294117600000,
            47.5000000000000, 48.0882352900000, 47.6470588200000, 43.2352941200000
        ]

        # pv
        self.pv = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06825511, 0.53766546, 0.89338147, 1.5766065, 1.60779783,
               2.0, 1.93636582, 1.60038508, 1.31850782, 0.69622142, 0.46180505, 0.17915764, 0.01482551,
               0.0, 0.0, 0.0, 0.0, 0.0 ]

        # ambient temperature (TCR-external) 外部
        self.temperature = [
            16, 15, 15, 14, 13, 15, 17, 18, 19, 22, 23, 24, 25, 26, 26,
            26, 26, 24, 23, 22, 21, 21, 20, 19
        ]

        self.wind_power = np.roll(self.pv, 5)  # 利用pv的数据滚动模拟风力数据





class dg():
    """DG"""
    def __init__(self,pmin,pmax,esr,cdr,cost):
        self.pmin = pmin
        self.pmax = pmax
        self.esr = esr
        self.cdr = cdr
        self.cost = cost
class pv()
    def __init__(self,pmin,pmax,cost):
        self.pmin = pmin
        self.pmax = pmax
        self.cost = cost
class wind()
    def __init__(self,pmin,pmax,cost):
        self.pmin = pmin
        self.pmax = pmax
        self.cost = cost
class ess():
    """Energy Storage System"""
    def __init__(self,ess0,ess_max,pmin,pmax,esr,cdr,ess_cost,safe_ratio=0.05):
        self.ess0 = ess0
        self.ess_max = ess_max
        self.pmin = pmin
        self.pmax = pmax
        self.esr = esr
        self.cdr = cdr
        self.ess_cost = ess_cost
class TCR():
    



class Powerplant:
    time_line = 96
    def __init__(self):
        self.M = 1e9
        self.data = data_info() # 数据接口
    def add_gene(self,pmin,pmax,esr,cdr,cost):
        self.gene_list = []
        self.gene_list.append(gene(pmin,pmax,esr,cdr,cost))
    def add_estore(self,ess0,ess_max,pmin,pmax,esr,cdr,ess_cost,safe_ratio=0.05):
        self.estore=[]
        self.estore.append(estore(ess0,ess_max,pmin,pmax,esr,cdr,ess_cost,safe_ratio=0.05))



    def create_mdl(self):
        mdl = Model("只报量模块化模拟")








