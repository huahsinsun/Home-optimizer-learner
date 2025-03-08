
# -*- Programmed by Python 3.8 based on Pycharm -*- #
"""
Created on Wed Jun 28 12:24:34 2022
@author: Dr. Zhongkai YI
Explanation: (1) To formulate a general bidding method for a general VPP (with energy storage, renewable energy generation, and load demand).
             (2) Including two modes: "Quantity only mode" (仅报量模式 in Chinese) and "Price-Quantity mode" (报量报价模式 in Chinese)
"""

import numpy as np
import random
from datetime import datetime as dt
# import gurobipy as gp
# from gurobipy import GRB
# from gurobipy import quicksum
import copy
import docplex.mp.model as cpx
import argparse
import matplotlib.pyplot as plt


# --------- VPP parameters and functions defination --------- #
class VPP_function():   # Initialize the VPP environment, formulate the VPP functions

    def __init__(self,Energy_price_forecast,Load_forecast,RES_forecast,Time_slots): # Environment and parameters initialization
        # 输入参数
        self.Energy_price_forecast = Energy_price_forecast      # 能量市场电价预测
        self.Load_forecast = Load_forecast                      # 负荷预测
        self.RES_forecast = RES_forecast                        # 可再生能源预测
        self.Time_slots = Time_slots                            # 时间点数（市场竞标分割时段总数）

        # 储能设备相关参数
        self.ESS_energy_capacity = [5,3]                        # 储能能量容量
        self.ESS_energy_min = [0,0]                        # 储能最小剩余能量
        self.ESS_P_capacity = [1,0.5]                           # 储能有功容量
        self.ESS_energy_0 = [2.0,1.8]                           # 储能初始能量
        self.ESS_P_cost_coeff = [1.0,1.0]                       # 储能单位功率成本
        self.ESS_number = len(self.ESS_energy_capacity)         # 储能数量
        self.ESS_energy_reserve = 1.2                           # 储能的总能量预留容量：为了抵消日前/日内偏差
        self.ESS_Target_final_energy = [2.5,1.5]                # 储能的一天后的期望剩余能量（这个可以不作为硬性约束）
        self.ESS_Target_final_energy_weight = 200.0                # 储能的一天后的期望剩余能量（这个可以不作为硬性约束）

        # 虚拟电厂相关参数
        self.VPP_P_max = np.dot(5.0,np.ones(self.Time_slots))          # 虚拟电厂各时段上限
        self.VPP_P_min = np.dot(-5.0,np.ones(self.Time_slots))          # 虚拟电厂各时段下限
        self.Price_bid_min_step = 5.0                                    # 虚拟电厂最小的价格阶越
        self.Quantity_bid_min_step = 0.2                               # 虚拟电厂最小的电量阶越
        self.Price_bid_min = 0.0                                # 报价的最小值
        self.Price_bid_max = 100                                # 报价的最大值
        self.Quantity_min = np.dot(-5.0,np.ones(self.Time_slots))                      # 报量的最小范围
        self.Quantity_max = np.dot(5.0,np.ones(self.Time_slots))                       # 报量的最大范围
        self.Bid_segment_number = 8      # 投标分段数（3-10段）
        self.Segment_split_mode = 1      # 投标段分解方式（仅在两阶段竞标函数VPP_bid_curve_Quantity_price_two_step_algorithm中有用） = 0 代表均匀分解； = 1代表不均匀分解

        self.Market_clearing_quantity_curve_real = np.zeros(self.Time_slots)                 # 实际的出清曲线
        self.Energy_price_real = np.zeros(self.Time_slots)                              # 实际的能量价格曲线

        # 辅助参数
        self.Big_M = 1e2   # 辅助参数 Big-M

        # 输出参数
        self.ESS_energy_var = np.zeros([self.ESS_number,self.Time_slots])        # 储能能量状态
        self.ESS_Pin_var = np.zeros([self.ESS_number,self.Time_slots])           # 储能有功注入功率
        self.ESS_Pout_var = np.zeros([self.ESS_number,self.Time_slots])          # 储能有功输出功率
        self.VPP_P_overall_expect = np.zeros(self.Time_slots)                           # 虚拟电厂有功出力(优化中的期望值)
        self.VPP_P_overall_real = np.zeros(self.Time_slots)                           # 虚拟电厂有功出力(优化中的期望值)
        self.RES_P = np.zeros(self.Time_slots)                                   # 可再生能源有功出力
        self.Quantity_bid_seg = np.zeros([self.Bid_segment_number,self.Time_slots])      # 竞标电价曲线
        self.Price_bid_curve = np.zeros([self.Bid_segment_number,self.Time_slots])       # 竞标电量分块
        self.Quantity_clear_seg = np.zeros([self.Bid_segment_number,self.Time_slots])      # 电量分块出清曲线
        self.Laimuta_addition_var = np.zeros([self.Bid_segment_number,self.Time_slots])    # 辅助变量，0、1变量

    def Market_clear_simulation(self,price_uncertain_ratio = 0.0):  # Market clearing environment Simulation

       

        Energy_price_real = copy.deepcopy(self.Energy_price_forecast)
        Market_clearing_quantity_curve_real = np.zeros(self.Time_slots)
        Market_clearing_quantity_curve_segment = np.zeros([self.Bid_segment_number,self.Time_slots])
        for time_t in range(self.Time_slots):
            Energy_price_real[time_t] = self.Energy_price_forecast[time_t] * (1 + price_uncertain_ratio * np.random.randn())

        for time_t in range(self.Time_slots):
            for Seg_i in range(self.Bid_segment_number):
                if self.Price_bid_curve[Seg_i,time_t] < Energy_price_real[time_t]:
                    Market_clearing_quantity_curve_segment[Seg_i,time_t] = self.Quantity_bid_seg[Seg_i,time_t]
                else:
                    Market_clearing_quantity_curve_segment[Seg_i, time_t] = 0
            Market_clearing_quantity_curve_real[time_t] = np.sum(Market_clearing_quantity_curve_segment[:,time_t]) + self.Quantity_min[time_t]

        self.Market_clearing_quantity_curve_real = copy.deepcopy(Market_clearing_quantity_curve_real)
        self.Energy_price_real = copy.deepcopy(Energy_price_real)
        return Market_clearing_quantity_curve_real,Energy_price_real

    def VPP_economic_dispatch(self):

        # 构建gurobi模型
        model = gp.Model("lp")

        # 储能约束范围设置
        ESS_energy_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_energy_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_ub = np.ones([self.ESS_number,self.Time_slots])

        for ESS_i in range(self.ESS_number):
            for time_t in range(self.Time_slots):
                ESS_energy_capacity_ub[ESS_i,time_t] = self.ESS_energy_capacity[ESS_i]
                ESS_energy_capacity_lb[ESS_i,time_t] = self.ESS_energy_min[ESS_i]
                ESS_Pin_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]
                ESS_Pout_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]

        # 变量定义
        ESS_energy_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_energy_capacity_lb,ub=ESS_energy_capacity_ub, name="ESS_energy(i,t)")
        ESS_Pin_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pin_capacity_lb,ub=ESS_Pin_capacity_ub, name="ESS_Pin(i,t)")
        ESS_Pout_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pout_capacity_lb,ub=ESS_Pout_capacity_ub, name="ESS_Pout(i,t)")
        VPP_P_overall_real = model.addVars(self.Time_slots,lb = self.VPP_P_min, ub = self.VPP_P_max, name="VPP power output(t))")
        RES_P = model.addVars(self.Time_slots,lb = 0, ub = GRB.INFINITY, name="VPP power output(t))")

        # 目标函数定义
        obj = 0
        obj_Revenue_from_market = 0
        obj_ESS_cost = 0
        obj_Target_final_energy = 0
        obj_market_clearing_energy_deviation = 0

        for time_t in range(self.Time_slots):
            obj_Revenue_from_market += (self.Energy_price_real[time_t] * VPP_P_overall_real[time_t])
            for ESS_i in range(self.ESS_number):
                obj_ESS_cost += self.ESS_P_cost_coeff[ESS_i] * (ESS_Pin_var[ESS_i, time_t] + ESS_Pout_var[ESS_i, time_t])
        for ESS_i in range(self.ESS_number):
            obj_Target_final_energy += self.ESS_Target_final_energy_weight * (ESS_energy_var[ESS_i,(self.Time_slots-1)] - self.ESS_Target_final_energy[ESS_i])**2

        for time_t in range(self.Time_slots):
            obj_market_clearing_energy_deviation += 100*(self.Market_clearing_quantity_curve_real[time_t] - VPP_P_overall_real[time_t])**2


        obj = - obj_Revenue_from_market + obj_ESS_cost + obj_Target_final_energy + obj_market_clearing_energy_deviation
        model.setObjective(obj)

        # 约束定义
        for time_t in range(self.Time_slots):
            # 功率平衡约束
            # dg+res+vpp=load for global
            model.addConstr(VPP_P_overall_real[time_t] == quicksum(ESS_Pout_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(ESS_Pin_var[ESS_i, time_t] for ESS_i in range(self.ESS_number))  - self.Load_forecast[time_t] + RES_P[time_t])

            # 可再生能源发电约束(可弃风)
            model.addConstr(RES_P[time_t] <= self.RES_forecast[time_t])

            # 储能模型约束
            for ESS_i in range(self.ESS_number):
                # 储能模型约束
                if time_t == 0:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + self.ESS_energy_0[ESS_i])
                else:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + ESS_energy_var[ESS_i, time_t - 1])
                # # 储能的能量预留约束
                # model.addConstr(quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(self.ESS_energy_min[ESS_i] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)
                # model.addConstr(quicksum(self.ESS_energy_capacity[ESS_i] for ESS_i in range(self.ESS_number)) - quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)

        # 模型求解
        model.optimize()

        # 变量录入
        for time_t in range(self.Time_slots):
            self.RES_P[time_t] = RES_P[time_t].X
            self.VPP_P_overall_real[time_t] = VPP_P_overall_real[time_t].X
            for ESS_i in range(self.ESS_number):
                self.ESS_energy_var[ESS_i,time_t] = ESS_energy_var[ESS_i,time_t].X
                self.ESS_Pin_var[ESS_i,time_t] = ESS_Pin_var[ESS_i,time_t].X
                self.ESS_Pout_var[ESS_i,time_t] = ESS_Pout_var[ESS_i,time_t].X

        return 0


    def VPP_bidding_Quantity_only_mode(self):

        # 构建gurobi模型
        model = gp.Model("lp")

        # 储能约束范围设置
        ESS_energy_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_energy_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_ub = np.ones([self.ESS_number,self.Time_slots])

        for ESS_i in range(self.ESS_number):
            for time_t in range(self.Time_slots):
                ESS_energy_capacity_ub[ESS_i,time_t] = self.ESS_energy_capacity[ESS_i]
                ESS_energy_capacity_lb[ESS_i,time_t] = self.ESS_energy_min[ESS_i]
                ESS_Pin_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]
                ESS_Pout_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]

        # 变量定义
        ESS_energy_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_energy_capacity_lb,ub=ESS_energy_capacity_ub, name="ESS_energy(i,t)")
        ESS_Pin_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pin_capacity_lb,ub=ESS_Pin_capacity_ub, name="ESS_Pin(i,t)")
        ESS_Pout_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pout_capacity_lb,ub=ESS_Pout_capacity_ub, name="ESS_Pout(i,t)")
        VPP_P_overall_expect = model.addVars(self.Time_slots,lb = self.VPP_P_min, ub = self.VPP_P_max, name="VPP power output(t))")
        RES_P = model.addVars(self.Time_slots,lb = 0, ub = GRB.INFINITY, name="VPP power output(t))")

        # 目标函数定义
        obj = 0
        obj_Revenue_from_market = 0
        obj_ESS_cost = 0
        obj_Target_final_energy = 0
        for time_t in range(self.Time_slots):
            obj_Revenue_from_market += (self.Energy_price_forecast[time_t] * VPP_P_overall_expect[time_t])
            for ESS_i in range(self.ESS_number):
                obj_ESS_cost += self.ESS_P_cost_coeff[ESS_i] * (ESS_Pin_var[ESS_i, time_t] + ESS_Pout_var[ESS_i, time_t])
        for ESS_i in range(self.ESS_number):
            obj_Target_final_energy += self.ESS_Target_final_energy_weight * (ESS_energy_var[ESS_i,(self.Time_slots-1)] - self.ESS_Target_final_energy[ESS_i])**2

        obj = - obj_Revenue_from_market + obj_ESS_cost + obj_Target_final_energy
        model.setObjective(obj)

        # 约束定义
        for time_t in range(self.Time_slots):
            # 功率平衡约束
            model.addConstr(VPP_P_overall_expect[time_t] == quicksum(ESS_Pout_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(ESS_Pin_var[ESS_i, time_t] for ESS_i in range(self.ESS_number))  - self.Load_forecast[time_t] + RES_P[time_t])

            # 可再生能源发电约束(可弃风)
            model.addConstr(RES_P[time_t] <= self.RES_forecast[time_t])

            # 储能模型约束
            for ESS_i in range(self.ESS_number):
                # 储能模型约束
                if time_t == 0:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + self.ESS_energy_0[ESS_i])
                else:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + ESS_energy_var[ESS_i, time_t - 1])
                # 储能的能量预留约束
                model.addConstr(quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(self.ESS_energy_min[ESS_i] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)
                model.addConstr(quicksum(self.ESS_energy_capacity[ESS_i] for ESS_i in range(self.ESS_number)) - quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)

        # 模型求解
        model.optimize()

        # 变量录入
        for time_t in range(self.Time_slots):
            self.RES_P[time_t] = RES_P[time_t].X
            self.VPP_P_overall_expect[time_t] = VPP_P_overall_expect[time_t].X
            for ESS_i in range(self.ESS_number):
                self.ESS_energy_var[ESS_i,time_t] = ESS_energy_var[ESS_i,time_t].X
                self.ESS_Pin_var[ESS_i,time_t] = ESS_Pin_var[ESS_i,time_t].X
                self.ESS_Pout_var[ESS_i,time_t] = ESS_Pout_var[ESS_i,time_t].X

        return 0

    def VPP_bid_curve_Quantity_price_two_step_algorithm(self):
        #竞标价格-数量曲线，分成两步进行：第一步，先基于线性规划计算出来竞标期望出清值，
        # 然后在出清值附近，对竞标曲线离散化（越靠近出清功率，电价段变化越大，功率段越小）
        self.VPP_bidding_Quantity_only_mode()

        Price_curve = np.zeros([self.Bid_segment_number,self.Time_slots])
        Quantity_curve = np.zeros([self.Bid_segment_number,self.Time_slots])
        if self.Segment_split_mode == 0:  # 均匀分解模式

            for time_t in range(self.Time_slots):

                # 计算各段拐点：
                # 能量拐点
                Quantity_curve[0,time_t] = copy.deepcopy(self.Quantity_min[time_t])
                Quantity_curve[int(self.Bid_segment_number/2), time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t])
                for seg_i in range(1,int(self.Bid_segment_number/2)):
                    Quantity_curve[seg_i,time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t]) - (int(self.Bid_segment_number/2) - seg_i)*(Quantity_curve[int(self.Bid_segment_number/2), time_t] - self.Quantity_min[time_t]) / int(self.Bid_segment_number / 2)

                for seg_i in range(int(self.Bid_segment_number / 2)+1,self.Bid_segment_number):
                    Quantity_curve[seg_i, time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t]) + (seg_i - int(self.Bid_segment_number/2))*(self.Quantity_max[time_t] - Quantity_curve[int(self.Bid_segment_number / 2), time_t]) / (self.Bid_segment_number - int(self.Bid_segment_number / 2))

                for seg_i in range(self.Bid_segment_number-1):
                    self.Quantity_bid_seg[seg_i,time_t] = Quantity_curve[seg_i+1, time_t] - Quantity_curve[seg_i, time_t]
                self.Quantity_bid_seg[-1, time_t] = copy.deepcopy(self.Quantity_max[time_t]) - Quantity_curve[-1, time_t]

                # 电价拐点
                Price_curve[int(self.Bid_segment_number/2), time_t] = copy.deepcopy(self.Energy_price_forecast[time_t])
                for seg_i in range(0,int(self.Bid_segment_number/2)):
                    Price_step_ratio = 0.1
                    Price_curve[seg_i,time_t] = self.Energy_price_forecast[time_t] * (1 - (int(self.Bid_segment_number/2) - seg_i)*Price_step_ratio)
                for seg_i in range(int(self.Bid_segment_number / 2) + 1, self.Bid_segment_number):
                    Price_curve[seg_i, time_t] = self.Energy_price_forecast[time_t] * (1 + (seg_i - int(self.Bid_segment_number/2))*Price_step_ratio)
                self.Price_bid_curve = Price_curve

        else:        # 非均匀分解模式（越靠近出清功率，电价段变化越大，功率段越小）

            Deta_quantity_positive = [0.05,0.1,0.15,0.2,0.2,0.4]
            Deta_price_positive = [8.0, 5.0, 4.0, 2.0, 1.0, 1.0]

            for time_t in range(self.Time_slots):

                # 计算各段拐点：
                # 能量拐点
                Quantity_curve[0,time_t] = copy.deepcopy(self.Quantity_min[time_t])
                Quantity_curve[int(self.Bid_segment_number/2), time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t])
                for seg_i in range(1,int(self.Bid_segment_number/2)):
                    Quantity_curve[seg_i,time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t]) - np.sum(Deta_quantity_positive[0:(int(self.Bid_segment_number/2) - seg_i)])
                    Quantity_curve[seg_i, time_t] = np.clip(Quantity_curve[seg_i,time_t],self.Quantity_min[time_t],self.Quantity_max[time_t])

                for seg_i in range(int(self.Bid_segment_number / 2)+1,self.Bid_segment_number):
                    Quantity_curve[seg_i, time_t] = copy.deepcopy(self.VPP_P_overall_expect[time_t]) + np.sum(Deta_quantity_positive[0:(seg_i - int(self.Bid_segment_number/2))])
                    Quantity_curve[seg_i, time_t] = np.clip(Quantity_curve[seg_i, time_t],self.Quantity_min[time_t],self.Quantity_max[time_t])

                for seg_i in range(self.Bid_segment_number-1):
                    self.Quantity_bid_seg[seg_i,time_t] = Quantity_curve[seg_i+1, time_t] - Quantity_curve[seg_i, time_t]
                self.Quantity_bid_seg[-1, time_t] = copy.deepcopy(self.Quantity_max[time_t]) - Quantity_curve[-1, time_t]

                # 电价拐点
                Price_curve[int(self.Bid_segment_number/2), time_t] = copy.deepcopy(self.Energy_price_forecast[time_t])
                for seg_i in range(0,int(self.Bid_segment_number/2)):
                    Price_step_ratio = 0.1
                    Price_curve[seg_i,time_t] = self.Energy_price_forecast[time_t] - np.sum(Deta_price_positive[0:(int(self.Bid_segment_number/2) - seg_i)])
                for seg_i in range(int(self.Bid_segment_number / 2) + 1, self.Bid_segment_number):
                    Price_curve[seg_i, time_t] = self.Energy_price_forecast[time_t] + np.sum(Deta_price_positive[0:(seg_i - int(self.Bid_segment_number/2))])
                self.Price_bid_curve = Price_curve

        return 0

    def VPP_bid_curve_Quantity_price(self): #直接优化价格-数量曲线

        Price_curve = self.Price_bid_curve
        Quantity_curve = np.zeros([self.Bid_segment_number,self.Time_slots])
        for time_t in range(self.Time_slots):
            for Seg_i in range(self.Bid_segment_number):
                Quantity_curve[Seg_i, time_t] = copy.deepcopy(self.Quantity_min[time_t])
                for Seg_i_j in range(Seg_i):
                    Quantity_curve[Seg_i,time_t] += copy.deepcopy(self.Quantity_bid_seg[Seg_i_j,time_t])

        cols = 2  # 定义每行的子图数量
        rows = int(np.ceil(self.Time_slots / cols))  # 计算需要的行数
        # 创建多个子图
        fig, axs = plt.subplots(rows, cols)
        axs = axs.flatten()  # 将子图数组变为一维，便于迭代
        # 遍历每个时间段，绘制子图
        for time_t in range(self.Time_slots):
            if time_t < len(axs):  # 确保不会在没有子图的情况下绘图
                ax = axs[time_t]
                # 设置x轴和y轴的界限
                ax.set_xlim([self.Quantity_min[time_t], self.Quantity_max[time_t]])
                ax.set_ylim([np.min(Price_curve), np.max(Price_curve)])
                # 绘制阶梯图，Quantity_curve 和 Price_curve1 必须是二维数组
                ax.step(Quantity_curve[:, time_t], Price_curve[:, time_t], label=f'Time slot {time_t + 1}')
                # 添加子图标题
                ax.set_title(f'Time slot {time_t + 1}')
                # 添加图例
                ax.legend()
        # 如果子图数量不是完全匹配的，隐藏多余的子图
        for i in range(self.Time_slots, len(axs)):
            fig.delaxes(axs[i])
        # 调整子图间距
        fig.tight_layout()
        # 显示图形
        plt.show()

    def VPP_bidding_Price_Quantity_mode(self):

        # 构建gurobi模型
        model = gp.Model("lp")

        # 储能约束范围设置
        ESS_energy_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_lb = np.zeros([self.ESS_number,self.Time_slots])
        ESS_energy_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pin_capacity_ub = np.ones([self.ESS_number,self.Time_slots])
        ESS_Pout_capacity_ub = np.ones([self.ESS_number,self.Time_slots])

        for ESS_i in range(self.ESS_number):
            for time_t in range(self.Time_slots):
                ESS_energy_capacity_ub[ESS_i,time_t] = self.ESS_energy_capacity[ESS_i]
                ESS_energy_capacity_lb[ESS_i,time_t] = self.ESS_energy_min[ESS_i]
                ESS_Pin_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]
                ESS_Pout_capacity_ub[ESS_i,time_t] = self.ESS_P_capacity[ESS_i]

        # 变量定义
        ESS_energy_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_energy_capacity_lb,ub=ESS_energy_capacity_ub, name="ESS_energy(i,t)")
        ESS_Pin_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pin_capacity_lb,ub=ESS_Pin_capacity_ub, name="ESS_Pin(i,t)")
        ESS_Pout_var = model.addVars(self.ESS_number,self.Time_slots,lb=ESS_Pout_capacity_lb,ub=ESS_Pout_capacity_ub, name="ESS_Pout(i,t)")
        VPP_P_overall_expect = model.addVars(self.Time_slots,lb = self.VPP_P_min, ub = self.VPP_P_max, name="VPP power output(t))")
        RES_P = model.addVars(self.Time_slots,lb = 0, ub = GRB.INFINITY, name="VPP power output(t))")

        Quantity_bid_seg = model.addVars(self.Bid_segment_number,self.Time_slots, lb = 0, ub = GRB.INFINITY, name="Quantity_bid_curve") # 报量曲线（分段，大于0，且相加等于虚拟电厂的可调范围）
        Price_bid_curve = model.addVars(self.Bid_segment_number,self.Time_slots,lb = self.Price_bid_min, ub = self.Price_bid_max, name="Price_bid_curve") # 报价曲线（单调递增）
        Quantity_clear_seg = model.addVars(self.Bid_segment_number,self.Time_slots, lb = 0, ub = GRB.INFINITY, name="Quantity_clear_seg") # 辅助变量laimuta
        Laimuta_addition_var = model.addVars(self.Bid_segment_number,self.Time_slots,vtype=GRB.BINARY, name="Laimuta_addition_var") # 辅助变量laimuta

        # 目标函数定义
        obj = 0
        obj_Revenue_from_market = 0
        obj_ESS_cost = 0
        obj_Target_final_energy = 0
        for time_t in range(self.Time_slots):
            obj_Revenue_from_market += (self.Energy_price_forecast[time_t] * VPP_P_overall_expect[time_t])
            for ESS_i in range(self.ESS_number):
                obj_ESS_cost += self.ESS_P_cost_coeff[ESS_i] * (ESS_Pin_var[ESS_i, time_t] + ESS_Pout_var[ESS_i, time_t])
        for ESS_i in range(self.ESS_number):
            obj_Target_final_energy += self.ESS_Target_final_energy_weight * (ESS_energy_var[ESS_i,(self.Time_slots-1)] - self.ESS_Target_final_energy[ESS_i])**2

        obj = - obj_Revenue_from_market + obj_ESS_cost + obj_Target_final_energy
        model.setObjective(obj)

        # 约束定义

        # 竞标电量，电价范围限制:
        for time_t in range(self.Time_slots):
            model.addConstr(quicksum(Quantity_bid_seg[Seg_i,time_t] for Seg_i in range(self.Bid_segment_number)) == (self.Quantity_max[time_t]-self.Quantity_min[time_t]) )     # 保证各报量能量块相加 等于 总范围约束
            model.addConstr(quicksum(Quantity_clear_seg[Seg_i,time_t] for Seg_i in range(self.Bid_segment_number) ) + self.Quantity_min[time_t] == VPP_P_overall_expect[time_t])       # 实际总出清功率 = 各段出清功率之和 + 竞标下限
            for Seg_i in range(self.Bid_segment_number):
                model.addConstr(Quantity_bid_seg[Seg_i,time_t] >= self.Quantity_bid_min_step)   # 保证报量曲线阶跃最小值
                model.addConstr(Quantity_clear_seg[Seg_i,time_t] <= Quantity_bid_seg[Seg_i,time_t])  # 各段出清值小于竞标范围

            for Seg_i in range(self.Bid_segment_number-1):
                model.addConstr(Price_bid_curve[Seg_i+1,time_t] >= Price_bid_curve[Seg_i,time_t])                               # 保证报价曲线的单调性
                model.addConstr(Price_bid_curve[Seg_i+1,time_t] - Price_bid_curve[Seg_i,time_t] >= self.Price_bid_min_step)     # 保证报价曲线阶跃最小值

        # 出清行为模拟：（根据预测电价，预估出清功率）
        for time_t in range(self.Time_slots):
            for Seg_i in range(self.Bid_segment_number):
                model.addConstr( self.Energy_price_forecast[time_t] - Price_bid_curve[Seg_i, time_t] <= self.Big_M * Laimuta_addition_var[Seg_i, time_t] )
                model.addConstr( - self.Big_M * (1-Laimuta_addition_var[Seg_i, time_t])  <= self.Energy_price_forecast[time_t] - Price_bid_curve[Seg_i, time_t] )
                model.addConstr( Quantity_bid_seg[Seg_i,time_t] - Quantity_clear_seg[Seg_i,time_t] <= self.Big_M * (1-Laimuta_addition_var[Seg_i,time_t]))
                model.addConstr( -self.Big_M * (1-Laimuta_addition_var[Seg_i,time_t]) <= Quantity_bid_seg[Seg_i,time_t] - Quantity_clear_seg[Seg_i,time_t] )
                model.addConstr( Quantity_clear_seg[Seg_i,time_t] <= self.Big_M * Laimuta_addition_var[Seg_i,time_t])
                model.addConstr( -self.Big_M * Laimuta_addition_var[Seg_i,time_t] <= Quantity_clear_seg[Seg_i,time_t] )

                # 辅助变量约束：确定出清值

        for time_t in range(self.Time_slots):
            # 功率平衡约束
            model.addConstr(VPP_P_overall_expect[time_t] == quicksum(ESS_Pout_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(ESS_Pin_var[ESS_i, time_t] for ESS_i in range(self.ESS_number))  - self.Load_forecast[time_t] + RES_P[time_t])

            # 可再生能源发电约束(可弃风)
            model.addConstr(RES_P[time_t] <= self.RES_forecast[time_t])

            # 储能模型约束
            for ESS_i in range(self.ESS_number):
                # 储能模型约束
                if time_t == 0:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + self.ESS_energy_0[ESS_i])
                else:
                    model.addConstr(ESS_energy_var[ESS_i, time_t] == ESS_Pin_var[ESS_i, time_t] - ESS_Pout_var[ESS_i, time_t] + ESS_energy_var[ESS_i, time_t - 1])
                # 储能的能量预留约束
                model.addConstr(quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) - quicksum(self.ESS_energy_min[ESS_i] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)
                model.addConstr(quicksum(self.ESS_energy_capacity[ESS_i] for ESS_i in range(self.ESS_number)) - quicksum(ESS_energy_var[ESS_i, time_t] for ESS_i in range(self.ESS_number)) >= self.ESS_energy_reserve)

        # 模型求解
        model.optimize()

        # 变量录入
        for time_t in range(self.Time_slots):
            self.RES_P[time_t] = RES_P[time_t].X
            self.VPP_P_overall_expect[time_t] = VPP_P_overall_expect[time_t].X
            for ESS_i in range(self.ESS_number):
                self.ESS_energy_var[ESS_i,time_t] = ESS_energy_var[ESS_i,time_t].X
                self.ESS_Pin_var[ESS_i,time_t] = ESS_Pin_var[ESS_i,time_t].X
                self.ESS_Pout_var[ESS_i,time_t] = ESS_Pout_var[ESS_i,time_t].X
            for Seg_i in range(self.Bid_segment_number):
                self.Quantity_bid_seg[Seg_i,time_t] = Quantity_bid_seg[Seg_i,time_t].X
                self.Price_bid_curve[Seg_i,time_t] = Price_bid_curve[Seg_i,time_t].X
                self.Quantity_clear_seg[Seg_i,time_t] = Quantity_clear_seg[Seg_i,time_t].X
                self.Laimuta_addition_var[Seg_i,time_t] = Laimuta_addition_var[Seg_i,time_t].X

        return 0

    def Result_plot(self):

        price_uncertain_ratio = 0.2
        self.Market_clear_simulation(price_uncertain_ratio)
        VPP_P_deviation_between_expect_and_real = self.VPP_P_overall_expect - self.Market_clearing_quantity_curve_real

        x_label_1 = np.zeros(self.Time_slots, dtype=float)
        for t in range(self.Time_slots):
            x_label_1[t] = t

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(3, 2, 1)
        plt.ion()
        plt.axis([0, self.Time_slots, 0.5*min(self.Energy_price_forecast[0:self.Time_slots]), 1.5*max(self.Energy_price_forecast[0:self.Time_slots])])
        ax1.plot(x_label_1, self.Energy_price_forecast[0:self.Time_slots])
        ax1.plot(x_label_1, self.Energy_price_real,linestyle=':')
        plt.title('Energy_price_forecast and real')
        ax1 = fig1.add_subplot(3, 2, 2)
        plt.ion()
        plt.axis([0, self.Time_slots, -1.0, 1.0])
        ax1.plot(x_label_1, self.RES_forecast[0:self.Time_slots],color='r')
        ax1.plot(x_label_1, self.Load_forecast[0:self.Time_slots])
        ax1.plot(x_label_1, self.RES_P,color='g',linestyle=':')
        plt.title('RES_forecast,load demand, and RES generation')

        ax1 = fig1.add_subplot(3, 2, 3)
        plt.ion()
        plt.axis([0, self.Time_slots, min(self.VPP_P_overall_expect), max(self.VPP_P_overall_expect)])
        ax1.plot(x_label_1, self.VPP_P_overall_expect,color='r')
        ax1.plot(x_label_1, self.VPP_P_overall_real,color='g')
        ax1.plot(x_label_1, self.Market_clearing_quantity_curve_real,color='b',linestyle=':')
        plt.title('VPP_P_overall_expect and Market_clearing_quantity_curve_real')

        ax1 = fig1.add_subplot(3, 2, 4)
        plt.ion()
        plt.axis([0, self.Time_slots, 0, max(self.ESS_energy_capacity)*1.1])
        for ESS_i in range(self.ESS_number):
            ax1.plot(x_label_1, np.dot(self.ESS_energy_capacity[ESS_i], np.ones(self.Time_slots)), linestyle=':')
            ax1.plot(x_label_1, self.ESS_energy_var[ESS_i,:])
        plt.title('ESS_energy_var')

        ax1 = fig1.add_subplot(3, 2, 6)
        plt.ion()
        plt.axis([0, self.Time_slots, -max(self.ESS_P_capacity), max(self.ESS_P_capacity)])
        for ESS_i in range(self.ESS_number):
            ax1.plot(x_label_1, self.ESS_Pin_var[ESS_i,:])
            ax1.plot(x_label_1, -self.ESS_Pout_var[ESS_i,:],linestyle=':')
        plt.title('ESS_Pin_Pout_var')

        self.VPP_bid_curve_Quantity_price()





def Initialization_data():  # load the initialization simulation_cap_data, including the price, load, renewable energy, overall time slots, etc.

    # Energy_price_forecast = [16.07,14.96,14.28,13.26,13.21,14.08,15,16.93,18.6,20.41,22.11,24.5,26.57,28.07,29.97,30.66,35.5,35.58,29.54,26.73,23.91,22.16,19.93,17.93]
    # Load_forecast = [0.605585931, 0.607214238, 0.608842546, 0.656974725, 0.709757292, 0.790594894,0.877666948,0.913087467, 0.92913866, 0.931251234, 0.925398882, 0.918255032, 0.910034934, 0.909848113,0.919301226, 0.934087481, 0.958206733, 0.974063106, 0.967885132, 0.94853486, 0.869909243,0.792083167, 0.722252488, 0.65242181]
    # Load_forecast = np.dot(0.5, Load_forecast)
    # RES_forecast = [0,0,0,0,0,0,0.0341275572,0.2688327316,0.4466907341,0.7883032491,0.803898917,1.0,0.9681829122,0.8001925391,0.659253911,0.34811071,0.2309025271,0.0895788207,0.0074127557,0,0,0,0,0]

    Energy_price_forecast = [16.07,13.26,20.41,26.57,30.66,19.93,]
    Load_forecast = [0.605585931, 0.877666948, 0.910034934, 0.934087481, 0.967885132, 0.65242181]
    Load_forecast = np.dot(0.5, Load_forecast)
    RES_forecast = [0,0,0.7883032491,0.9681829122,0.34811071,0]

    Time_slots = len(Energy_price_forecast)

    return Energy_price_forecast,Load_forecast,RES_forecast,Time_slots


parser = argparse.ArgumentParser(description="")
parser.add_argument("-mode", type=str, default="Quantity only mode", help="Quantity only mode, or, Price-Quantity mode ")

if __name__ == "__main__":
    # ---------------------------- Run the algorithm ---------------------------- #
    Energy_price_forecast,Load_forecast,RES_forecast,Time_slots = Initialization_data()     # load the simulation_cap_data
    VPP_env = VPP_function(Energy_price_forecast,Load_forecast,RES_forecast,Time_slots)                    # Initialize the VPP environment

    args = parser.parse_args()
    args.mode = "Price-Quantity mode"
    if args.mode == "Prssice-Quantity mode":
         VPP_env.VPP_bidding_Quantity_only_mode()
        VPP_env.VPP_bid_curve_Quantity_price_two_step_algorithm()
    else:
        VPP_env.VPP_bidding_Price_Quantity_mode()

    price_uncertain_ratio = 0.2
    VPP_env.Market_clear_simulation(price_uncertain_ratio)
    VPP_env.VPP_economic_dispatch()

    VPP_env.Result_plot()

    debug_point = 0

    # try:
    #
    #     # Create a new model
    #     m = gp.Model("mip1")
    #
    #     # Create variables
    #     # x = m.addVar(vtype=GRB.BINARY, name="x")
    #     x = m.addVar(ub=6.5,name="x")
    #     y = m.addVar(vtype=GRB.BINARY, name="y")
    #     # y = m.addVar(ub=5.5, name="y")
    #     z = m.addVar(vtype=GRB.BINARY, name="z")
    #     # z = m.addVar(ub=3.5, name="z")
    #
    #     # Set objective
    #     m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
    #
    #     # Add constraint: x + 2 y + 3 z <= 4
    #     m.addConstr(x + 2 * y + 3 * z <= 20, "c0")
    #
    #     # Add constraint: x + y >= 1
    #     m.addConstr(x + y >= 1, "c1")
    #
    #     # Optimize model
    #     m.optimize()
    #
    #     for v in m.getVars():
    #         print('%s %g' % (v.VarName, v.X))
    #
    #     print('Obj: %g' % m.ObjVal)
    #
    # except gp.GurobiError as e:
    #     print('Error code ' + str(e.errno) + ': ' + str(e))
    #
    # except AttributeError:
    #     print('Encountered an attribute error')
