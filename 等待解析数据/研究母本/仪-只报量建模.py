'''
只报量建模
'''

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 加载数据
P_PV_pu24 = np.array([
    0, 0, 0, 0, 0, 0, 0.0341275571600482, 0.268832731648616, 0.446690734055355,
    0.788303249097473, 0.803898916967509, 1, 0.968182912154031, 0.800192539109507,
    0.659253910950662, 0.348110709987966, 0.230902527075812, 0.0895788206979543,
    0.00741275571600482, 0, 0, 0, 0, 0
])
E_Price24 = np.array([
    40.4411764700000, 38.9705882400000, 36.0294117600000, 33.3823529400000, 30.5882352900000,
    28.9705882400000, 32.9411764700000, 20.4411764700000, 43.5294117600000, 48.5294117600000,
    54.4117647100000, 55.0000000000000, 45.5882352900000, 46.3235294100000, 51.4705882400000,
    51.7647058800000, 53.8235294100000, 51.4705882400000, 44.8529411800000, 48.5294117600000,
    47.5000000000000, 48.0882352900000, 47.6470588200000, 43.2352941200000
])

# 光伏出力翻倍
P_PV_pu24 = P_PV_pu24 * 2

# 储能参数
battery_0 = 7.5  # 初始储能状态
battery_max = 15  # 最大储能状态
battery_p_max = 3  # 储能系统最大充放电功率
Laimuta = 0.995  # 储能存储效率
Yita = 0.95  # 充放电效率
ESS_cost_coeff = 5  # 储能成本系数

time_line = 24  # 优化时间范围(24小时)

# 分布式发电参数
DG_P_max = 2  # 分布式发电最大出力
DG_P_min = 1  # 分布式发电最小出力
DG_a = 2.4  # 分布式发电成本系数 a
DG_b = 35  # 分布式发电成本系数 b
DG_c = 3  # 分布式发电成本系数 c

Penalty_coeffi = 10  # 惩罚系数

rand_ratio = 0.2  # 随机系数

Operation_flag = 6  # 操作模式标志

# 根据操作模式配置参数
if Operation_flag == 1:
    DG_P_max = 0
    DG_P_min = 0
    battery_p_max = 0
elif Operation_flag == 2:
    P_PV_pu24 = np.zeros(24)
    battery_p_max = 0
elif Operation_flag == 3:
    P_PV_pu24 = np.zeros(24)
    DG_P_max = 0
    DG_P_min = 0
elif Operation_flag == 4:
    DG_P_max = 0
    DG_P_min = 0
elif Operation_flag == 5:
    battery_p_max = 0

# 虚拟电厂（VPP）投标优化
mdl = gp.Model("VPP_bidding_Optimization")

# 定义变量
PV_P_output = mdl.addVars(time_line, name="PV_P_output")
DG_P_output = mdl.addVars(time_line, name="DG_P_output")
ESS_P_output = mdl.addVars(time_line, name="ESS_P_output")
ESS_P_input = mdl.addVars(time_line, name="ESS_P_input")
SOC_state = mdl.addVars(time_line, name="SOC_state")
Agent_output = mdl.addVars(time_line, name="Agent_output")

# 约束条件
for k in range(time_line):
    mdl.addConstr(PV_P_output[k] >= 0)
    mdl.addConstr(PV_P_output[k] <= P_PV_pu24[k])
    mdl.addConstr(ESS_P_output[k] >= 0)
    mdl.addConstr(ESS_P_output[k] <= battery_p_max)
    mdl.addConstr(ESS_P_input[k] >= 0)
    mdl.addConstr(ESS_P_input[k] <= battery_p_max)
    mdl.addConstr(SOC_state[k] >= 0)
    mdl.addConstr(SOC_state[k] <= battery_max)
    mdl.addConstr(DG_P_output[k] >= DG_P_min)
    mdl.addConstr(DG_P_output[k] <= DG_P_max)
    mdl.addConstr(Agent_output[k] == PV_P_output[k] + DG_P_output[k] + ESS_P_output[k] * Yita - ESS_P_input[k])

# 储能状态初始条件
mdl.addConstr(SOC_state[0] == battery_0 * Laimuta + ESS_P_input[0] * Yita - ESS_P_output[0])
for k in range(1, time_line):
    mdl.addConstr(SOC_state[k] == SOC_state[k-1] * Laimuta + ESS_P_input[k] * Yita - ESS_P_output[k])

# 如果考虑PV+DG+ESS
if Operation_flag >= 4:
    for k in range(time_line):
        mdl.addConstr(P_PV_pu24[k] * rand_ratio <= DG_P_max + battery_p_max - (DG_P_output[k] + ESS_P_output[k]))
        mdl.addConstr(P_PV_pu24[k] * rand_ratio <= DG_P_output[k] - DG_P_min + battery_p_max - ESS_P_input[k])

# 目标函数
Objective = gp.quicksum(-E_Price24[k] * Agent_output[k]
                        + DG_a * DG_P_output[k] * DG_P_output[k] + DG_b * DG_P_output[k]
                        + ESS_cost_coeff * (ESS_P_output[k] + ESS_P_input[k]) for k in range(time_line))

# 最小化目标函数
mdl.setObjective(Objective, GRB.MINIMIZE)

# 求解模型
mdl.optimize()

if mdl.status == GRB.OPTIMAL:
    PV_P_output = [PV_P_output[k].X for k in range(time_line)]
    DG_P_output = [DG_P_output[k].X for k in range(time_line)]
    ESS_P_output = [ESS_P_output[k].X for k in range(time_line)]
    ESS_P_input = [ESS_P_input[k].X for k in range(time_line)]
    SOC_state = [SOC_state[k].X for k in range(time_line)]
    Agent_output_bid = [Agent_output[k].X for k in range(time_line)]
    VPP_profits = -mdl.objVal
    print("期望收益: ", VPP_profits)
else:
    print("未找到解决方案")

# 考虑不确定性的真实收益计算
P_PV_max_real = [P_PV_pu24[t] * (1 + rand_ratio * (np.random.rand() - 0.5) / 0.5) for t in range(time_line)]

mdl_real = gp.Model("VPP_real_Optimization")

# 定义变量
PV_P_output_real = mdl_real.addVars(time_line, name="PV_P_output_real")
DG_P_output_real = mdl_real.addVars(time_line, name="DG_P_output_real")
ESS_P_output_real = mdl_real.addVars(time_line, name="ESS_P_output_real")
ESS_P_input_real = mdl_real.addVars(time_line, name="ESS_P_input_real")
SOC_state_real = mdl_real.addVars(time_line, name="SOC_state_real")
Agent_output_real = mdl_real.addVars(time_line, name="Agent_output_real")

# 约束条件
for k in range(time_line):
    mdl_real.addConstr(PV_P_output_real[k] >= 0)
    mdl_real.addConstr(PV_P_output_real[k] <= P_PV_max_real[k])
    mdl_real.addConstr(ESS_P_output_real[k] >= 0)
    mdl_real.addConstr(ESS_P_output_real[k] <= battery_p_max)
    mdl_real.addConstr(ESS_P_input_real[k] >= 0)
    mdl_real.addConstr(ESS_P_input_real[k] <= battery_p_max)
    mdl_real.addConstr(SOC_state_real[k] >= 0)
    mdl_real.addConstr(SOC_state_real[k] <= battery_max)
    mdl_real.addConstr(DG_P_output_real[k] >= DG_P_min)
    mdl_real.addConstr(DG_P_output_real[k] <= DG_P_max)
    mdl_real.addConstr(Agent_output_real[k] == PV_P_output_real[k] + DG_P_output_real[k] + ESS_P_output_real[k] * Yita - ESS_P_input_real[k])

# 储能状态初始条件
mdl_real.addConstr(SOC_state_real[0] == battery_0 * Laimuta + ESS_P_input_real[0] * Yita - ESS_P_output_real[0])
for k in range(1, time_line):
    mdl_real.addConstr(SOC_state_real[k] == SOC_state_real[k-1] * Laimuta + ESS_P_input_real[k] * Yita - ESS_P_output_real[k])

# 使用条件约束代替abs()
Penalty_terms = []

for k in range(time_line):
    deviation = Agent_output_bid[k] - Agent_output_real[k]
    penalty = mdl_real.addVar(name=f'penalty_{k}')
    mdl_real.addConstr(penalty >= deviation)
    mdl_real.addConstr(penalty >= -deviation)
    Penalty_terms.append(Penalty_coeffi * E_Price24[k] * penalty)

# 真实优化的目标函数
Objective_real = gp.quicksum(-E_Price24[k] * Agent_output_real[k]
                              + DG_a * DG_P_output_real[k] * DG_P_output_real[k] + DG_b * DG_P_output_real[k]
                              + ESS_cost_coeff * (ESS_P_output_real[k] + ESS_P_input_real[k])
                              + Penalty_terms[k] for k in range(time_line))

# 最小化目标函数
mdl_real.setObjective(Objective_real, GRB.MINIMIZE)

# 求解模型
mdl_real.optimize()

if mdl_real.status == GRB.OPTIMAL:
    PV_P_output_real = [PV_P_output_real[k].X for k in range(time_line)]
    DG_P_output_real = [DG_P_output_real[k].X for k in range(time_line)]
    ESS_P_output_real = [ESS_P_output_real[k].X for k in range(time_line)]
    ESS_P_input_real = [ESS_P_input_real[k].X for k in range(time_line)]
    SOC_state_real = [SOC_state_real[k].X for k in range(time_line)]
    Agent_output_real = [Agent_output_real[k].X for k in range(time_line)]
    VPP_real_profits = -mdl_real.objVal
    print("真实收益: ", VPP_real_profits)
else:
    print("未找到真实优化的解决方案")