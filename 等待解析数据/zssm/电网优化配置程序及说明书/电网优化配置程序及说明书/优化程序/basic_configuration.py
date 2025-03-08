"""
Description:本文档建模为电网优化配置建模方法，考虑风光储电源优化配置
Date-Version : 2024.11.22 Version 1.0
说明:
2024.11.22
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
import warnings

warnings.filterwarnings("ignore")
# distribution network structure description
class Struct:
    pass


ppc = Struct()
ppc.baseMVA = 100
ppc.bus = np.array([[1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1, 1],
                    [2, 1, 100, 60, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [3, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [4, 1, 120, 80, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [5, 1, 60, 30, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [6, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [7, 1, 200, 100, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [8, 1, 200, 100, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [9, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [10, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [11, 1, 45, 30, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [12, 1, 60, 35, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [13, 1, 60, 35, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [14, 1, 120, 80, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [15, 1, 60, 10, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [16, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [17, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [18, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [19, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [20, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [21, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [22, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [23, 1, 90, 50, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [24, 1, 420, 200, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [25, 1, 420, 200, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [26, 1, 60, 25, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [27, 1, 60, 25, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [28, 1, 60, 20, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [29, 1, 120, 70, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [30, 1, 200, 600, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [31, 1, 150, 70, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [32, 1, 210, 100, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
                    [33, 1, 60, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9]])
ppc.branch = np.array([[1, 2, 0.0922, 0.0470, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [2, 3, 0.4930, 0.2511, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [3, 4, 0.3660, 0.1864, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [4, 5, 0.3811, 0.1941, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [5, 6, 0.8190, 0.7070, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [6, 7, 0.1872, 0.6188, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [7, 8, 0.7114, 0.2351, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [8, 9, 1.0300, 0.7400, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [9, 10, 1.0440, 0.7400, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [10, 11, 0.1966, 0.0650, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [11, 12, 0.3744, 0.1238, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [12, 13, 1.4680, 1.1550, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [13, 14, 0.5416, 0.7129, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [14, 15, 0.5910, 0.5260, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [15, 16, 0.7463, 0.5450, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [16, 17, 1.2890, 1.7210, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [17, 18, 0.7320, 0.5740, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [2, 19, 0.1640, 0.1565, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [19, 20, 1.5042, 1.3554, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [20, 21, 0.4095, 0.4784, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [21, 22, 0.7089, 0.9373, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [3, 23, 0.4512, 0.3083, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [23, 24, 0.8980, 0.7091, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [24, 25, 0.8960, 0.7011, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [6, 26, 0.2030, 0.1034, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [26, 27, 0.2842, 0.1447, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [27, 28, 1.0590, 0.9337, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [28, 29, 0.8042, 0.7006, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [29, 30, 0.5075, 0.2585, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [30, 31, 0.9744, 0.9630, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [31, 32, 0.3105, 0.3619, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       [32, 33, 0.3410, 0.5302, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                       ])

BASE_KV = 10 -1
BR_R = 3 - 1  # minus 1 for the beginning index is 0 in python but 1 in MATLAB
BR_X = 4 - 1
PD = 3 - 1
QD = 4 - 1
baseMVA = ppc.baseMVA
Vbase = ppc.bus[0, BASE_KV] * 1000
Sbase = ppc.baseMVA * 1e6
ppc.branch[:, [BR_R, BR_X]] = ppc.branch[:, [BR_R, BR_X]] / (Vbase ** 2 / Sbase)
ppc.bus[:, [PD, QD]] = ppc.bus[:, [PD, QD]] / 1e3
bus = ppc.bus
branch = ppc.branch
Bus_Number = bus.shape[0]
Branch_Number = branch.shape[0]

branch_r_x = np.zeros((branch.shape[0], 4))
branch_r_x[:, [0, 1]] = branch[:, [0, 1]]
branch_r_x[:, 2] = branch[:, 2] * 1
branch_r_x[:, 3] = branch[:, 3] * 1
Rij_matrix = np.full((Bus_Number, Bus_Number), np.inf)
Xij_matrix = np.full((Bus_Number, Bus_Number), np.inf)
node_head_matrix = (branch[:, 0] - 1).astype(int)
node_tail_matrix = (branch[:, 1] - 1).astype(int)

for i in range(Branch_Number):
    Rij_matrix[branch_r_x[i, 0].astype(int) - 1, branch_r_x[i, 1].astype(int) - 1] = branch_r_x[i, 2]
    Rij_matrix[branch_r_x[i, 1].astype(int) - 1, branch_r_x[i, 0].astype(int) - 1] = branch_r_x[i, 2]
    Xij_matrix[branch_r_x[i, 0].astype(int) - 1, branch_r_x[i, 1].astype(int) - 1] = branch_r_x[i, 3]
    Xij_matrix[branch_r_x[i, 1].astype(int) - 1, branch_r_x[i, 0].astype(int) - 1] = branch_r_x[i, 3]

Rij_matrix = Rij_matrix / 1
Xij_matrix = Xij_matrix / 1

Y_po, Y_pv = np.zeros((Bus_Number, Bus_Number)), np.zeros((Bus_Number, Bus_Number))
Y_qo, Y_qv = np.zeros((Bus_Number, Bus_Number)), np.zeros((Bus_Number, Bus_Number))

# Y_po/Y_pv/Y_qo/Y_qv--K_ij for the PF model
for i in range(Bus_Number):
    for j in range(Bus_Number):
        if j != i:
            if Rij_matrix[i, j] < 10000:  # less than Inf
                k = Rij_matrix[i, j] ** 2 + Xij_matrix[i, j] ** 2
                Y_po[i, j] = -Xij_matrix[i, j] / k
                Y_pv[i, j] = -Rij_matrix[i, j] / k
                Y_qo[i, j] = +Rij_matrix[i, j] / k
                Y_qv[i, j] = -Xij_matrix[i, j] / k

for i in range(Bus_Number):
    Y_po[i, i] = -np.sum(Y_po[i, :])
    Y_pv[i, i] = -np.sum(Y_pv[i, :])
    Y_qo[i, i] = -np.sum(Y_qo[i, :])
    Y_qv[i, i] = -np.sum(Y_qv[i, :])

# Parameter for DG
N_DG_units = 2  # number of the DG
DG_con_SN = [5, 5]
DG_node_Num = [16 - 1, 24 - 1]
Pcon_max = [DG_con_SN[0], DG_con_SN[1]]
Qcon_max = [DG_con_SN[0], DG_con_SN[1]]
Qcon_min = [-DG_con_SN[0], -DG_con_SN[1]]
Pcon_min = [x * 0.3 for x in Pcon_max]
a_con = [3.0, 4.2]
b_con = [38.0, 44.5]
c_con = [3.4, 3.0]
Pcon_ramp_max = [x * 0.5 for x in Pcon_max]  # Pcon_max .* 0.5
DN_Buy_P_limit = 5
DN_Buy_Q_limit = 5
Reverse_P_max = 0.1
Bus_V_min = 0.90
Bus_V_max = 1.10
Pij_limit = 5 * np.ones((Bus_Number, Bus_Number))
Qij_limit = 5 * np.ones((Bus_Number, Bus_Number))
seg_num = 8
e_matrix = np.array(
    [[1, - 2.413], [1, - 0.414], [1, 0.414], [1, 0.414], [1, 2.413], [-1, - 2.413], [-1, - 0.414], [-1, 0.414],
     [-1, 2.413]])
f_matrix = [2.413, 1, 1, 2.413, 2.413, 1, 1, 2.413]

# data preparation
# data input
power_load = pd.read_excel('负荷数据.xlsx', header=0, index_col=0)
pv = pd.read_excel('光伏数据.xlsx', header=0, index_col=0)
wt = pd.read_excel('风机数据.xlsx', header=0, index_col=0)
price = pd.read_excel('价格数据.xlsx', header=0, index_col=0)
probability = pd.read_excel('概率数据.xlsx', header=0, index_col=0)
# data transform
power_load = np.array(power_load)
pv = np.array(pv)
wt = np.array(wt)
price = np.array(price)
probability = np.array(probability)
Num_Days = probability.shape[0]
# line up the data in a row
power_load = list(chain.from_iterable(power_load))
pv = list(chain.from_iterable(pv))
wt = list(chain.from_iterable(wt))
price = list(chain.from_iterable(price))
probability = list(chain.from_iterable(probability))
# load deal
load_factor = 1
N_time = 24
Total_time = Num_Days * N_time
Bus_load_P_MW24 = np.zeros((Bus_Number, Total_time))
Bus_load_Q_MW24 = np.zeros((Bus_Number, Total_time))
for i in range(Bus_Number):
    for t in range(Total_time):
        Bus_load_P_MW24[i, t] = bus[i, 2] * load_factor * power_load[t]
        Bus_load_Q_MW24[i, t] = bus[i, 3] * load_factor * power_load[t]
# technical and economic parameters
deltat_t = 1
ESS_E_P = 5
CAP_ESSP_max = 0.2 * np.ones((1, Bus_Number))
CAP_ESSE_max = ESS_E_P * CAP_ESSP_max
CAP_WT_max = 5 * np.ones((1, Bus_Number))
CAP_PV_max = 5 * np.ones((1, Bus_Number))
cost_ESSP = 45*1e3
cost_ESSE = 180*1e3
cost_WT = 1180*1e3
cost_PV = 1060*1e3
life_ESSP = 15
life_ESSE = 20
life_WT = 25
life_PV = 25
cost_cur = 10
discount_rate = 0.025
self_dis = 1e-4
SOC_0 = 0.5
SOC_min = 0.1
SOC_max = 0.9
cost_ch = 0.5
cost_dis = 0.5
# equivalence annual cost and other parameters
EA_ESSP = discount_rate*(1+discount_rate)**life_ESSP/((1+discount_rate)**life_ESSP-1)
EA_ESSE = discount_rate*(1+discount_rate)**life_ESSE/((1+discount_rate)**life_ESSE-1)
EA_WT = discount_rate*(1+discount_rate)**life_WT/((1+discount_rate)**life_WT-1)
EA_PV = discount_rate*(1+discount_rate)**life_PV/((1+discount_rate)**life_PV-1)

class DN_configuration:

    def __init__(self, num_nodes=3):  #number of nodes requiring power configuration
        # number of nodes to be configured
        self.num_nodes = num_nodes
        # distribution network structure description
        self.Bus_Number = Bus_Number
        self.Branch_Number = Branch_Number
        self.baseMVA = baseMVA
        self.node_head_matrix = node_head_matrix
        self.node_tail_matrix = node_tail_matrix
        self.Y_po = Y_po
        self.Y_pv = Y_pv
        self.Y_qo = Y_qo
        self.Y_qv = Y_qv
        self.N_DG_units = N_DG_units
        self.DG_con_SN = DG_con_SN
        self.DG_node_Num = DG_node_Num
        self.Pcon_max = Pcon_max
        self.Pcon_min = Pcon_min
        self.Qcon_max = Qcon_max
        self.Qcon_min = Qcon_min
        self.a_con = a_con
        self.b_con = b_con
        self.c_con = c_con
        self.Pcon_ramp_max = Pcon_ramp_max
        self.DN_Buy_P_limit = DN_Buy_P_limit
        self.DN_Buy_Q_limit = DN_Buy_Q_limit
        self.Reverse_P_max = Reverse_P_max
        self.Bus_V_min = Bus_V_min
        self.Bus_V_max = Bus_V_max
        self.Pij_limit = Pij_limit
        self.Qij_limit = Qij_limit
        self.seg_num = seg_num
        self.e_matrix = e_matrix
        self.f_matrix = f_matrix
        # wind, solar and load data description
        self.Bus_load_P_MW24 = Bus_load_P_MW24
        self.Bus_load_Q_MW24 = Bus_load_Q_MW24
        self.pv = pv
        self.wt = wt
        self.price = price
        self.probability = probability
        # technical and economic parameters
        self.Num_Days = Num_Days
        self.N_time = N_time
        self.Total_time = Total_time
        self.deltat_t = deltat_t
        self.ESS_E_P = ESS_E_P
        self.cost_ESSP = cost_ESSP
        self.cost_ESSE = cost_ESSE
        self.cost_WT = cost_WT
        self.cost_PV = cost_PV
        self.cost_cur = cost_cur
        self.num_nodes = num_nodes
        self.CAP_ESSP_max = CAP_ESSP_max
        self.CAP_ESSE_max = CAP_ESSE_max
        self.CAP_WT_max = CAP_WT_max
        self.CAP_PV_max = CAP_PV_max
        self.self_dis = self_dis
        self.SOC_0 = SOC_0
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.cost_ch = cost_ch
        self.cost_dis = cost_dis
        self.EA_ESSP = EA_ESSP
        self.EA_ESSE = EA_ESSE
        self.EA_WT = EA_WT
        self.EA_PV = EA_PV

    def create_model(self):
        mdl = gp.Model('电网优化配置模型')
        # variable definition
        # planning variables
        WT_Plan = mdl.addVars(1, self.Bus_Number, lb=0, name='节点风机容量')
        PV_Plan = mdl.addVars(1, self.Bus_Number, lb=0, name='节点光伏容量')
        E_Plan = mdl.addVars(1, self.Bus_Number, lb=0, name='节点储能电量容量')
        P_Plan = mdl.addVars(1, self.Bus_Number, lb=0, name='节点储能功率容量')
        u_Plan = mdl.addVars(1, self.Bus_Number, lb=0, vtype=GRB.BINARY, name='节点配置状态')
        # operational variables
        DN_Buy_P = mdl.addVars(1, self.Total_time, lb=-GRB.INFINITY, name='上级电网注入有功')
        DN_input_Q = mdl.addVars(1, self.Total_time, lb=-GRB.INFINITY, name='上级电网注入无功')
        DG_P = mdl.addVars(self.N_DG_units, self.Total_time, lb=0, name='发电机有功')
        DG_Q = mdl.addVars(self.N_DG_units, self.Total_time, lb=0, name='发电机无功')
        Bus_Fai = mdl.addVars(self.Bus_Number, self.Total_time, lb=-GRB.INFINITY, name='节点相角')
        Bus_V = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点电压')
        Bus_P_injection = mdl.addVars(self.Bus_Number, self.Total_time, lb=-GRB.INFINITY, name='节点有功注入')
        Bus_Q_injection = mdl.addVars(self.Bus_Number, self.Total_time, lb=-GRB.INFINITY, name='节点无功注入')
        PWT = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点风机出力')
        PPV = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点光伏出力')
        Pdis = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点储能放电功率')
        Pch = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点储能充电功率')
        E_state = mdl.addVars(self.Bus_Number, self.Total_time, lb=0, name='节点储存能量')
        P_ij = mdl.addVars(self.Branch_Number, self.Total_time, lb=-GRB.INFINITY, name='线路有功')
        Q_ij = mdl.addVars(self.Branch_Number, self.Total_time, lb=-GRB.INFINITY, name='线路无功')

        print('*' * 10, 'Adding Constraints', '*' * 10)


        # planning constraints
        mdl.addConstr(gp.quicksum(u_Plan[0, i] for i in range(self.Bus_Number)) == self.num_nodes)
        for i in range(self.Bus_Number):
            mdl.addConstr(WT_Plan[0, i] <= self.CAP_WT_max[0, i] * u_Plan[0, i])
            mdl.addConstr(PV_Plan[0, i] <= self.CAP_PV_max[0, i] * u_Plan[0, i])
            mdl.addConstr(E_Plan[0, i] <= self.CAP_ESSE_max[0, i] * u_Plan[0, i])
            mdl.addConstr(P_Plan[0, i] <= self.CAP_ESSP_max[0, i] * u_Plan[0, i])
            mdl.addConstr(E_Plan[0, i] == self.ESS_E_P * P_Plan[0, i])

        # operational constraints
        # 1-DG
        # output
        for i in range(self.N_DG_units):
            for t in range(self.Total_time):
                mdl.addConstr(DG_P[i, j] <= self.Pcon_max[i])
                mdl.addConstr(DG_P[i, j] >= self.Pcon_min[i])
                mdl.addConstr(DG_Q[i, j] <= self.Qcon_max[i])
                mdl.addConstr(DG_Q[i, j] >= self.Qcon_min[i])
                for k in range(self.seg_num):
                    mdl.addConstr(self.e_matrix[k, 0] * DG_P[i, j] + self.e_matrix[k, 1] * DG_Q[i, j] <= self.f_matrix[k] * self.DG_con_SN[i])
        # ramping
        for i in range(self.N_DG_units):
            for d in range(self.Num_Days):
                for t in range(1, self.N_time):
                    mdl.addConstr(DG_P[i, d*self.N_time+t] - DG_P[i, d*self.N_time+t-1] <= self.Pcon_ramp_max[i])
                    mdl.addConstr(DG_P[i, d*self.N_time+t] - DG_P[i, d*self.N_time+t-1] >= -self.Pcon_ramp_max[i])
        # purchase and sale of electricity from up power grid
        for t in range(self.Total_time):
             mdl.addConstr(DN_Buy_P[0, t] <= self.DN_Buy_P_limit)
             mdl.addConstr(DN_Buy_P[0, t] >= -self.Reverse_P_max*self.DN_Buy_P_limit)
             mdl.addConstr(DN_input_Q[0, t] <= self.DN_Buy_Q_limit)
             mdl.addConstr(DN_input_Q[0, t] >= -self.DN_Buy_Q_limit)
        # 2-Wind and solar output power constraints
        for i in range(self.Bus_Number):
            for t in range(self.Total_time):
                mdl.addConstr(PWT[i, t] <= WT_Plan[0, i] * self.wt[t])
                mdl.addConstr(PPV[i, t] <= PV_Plan[0, i] * self.pv[t])
        # 3-Energy storage constraints
        # power
        for i in range(self.Bus_Number):
            for t in range(self.Total_time):
                mdl.addConstr(Pch[i, t] <= P_Plan[0, i])
                mdl.addConstr(Pdis[i, t] <= P_Plan[0, i])
        # stored energy
        for i in range(self.Bus_Number):
            for d in range(self.Num_Days):
                mdl.addConstr(E_state[i, (d+1) * self.N_time - 1] == self.SOC_0 * E_Plan[0, i])
                mdl.addConstr(E_state[i, d * self.N_time] == self.SOC_0 * E_Plan[0, i] * (1 - self.self_dis) + Pch[i, d * self.N_time] - Pdis[i, d * self.N_time])
                for t in range(1, self.N_time):
                    mdl.addConstr(E_state[i, t] <= self.SOC_max * E_Plan[0, i])
                    mdl.addConstr(E_state[i, t] >= self.SOC_min * E_Plan[0, i])
                    mdl.addConstr(E_state[i, d * self.N_time + t] == E_state[i, d * self.N_time + t - 1] * (1 - self.self_dis) + Pch[i, d * self.N_time + t] - Pdis[i, d * self.N_time + t])
        # 4-power flow constraints
        # power injection
        for i in range(self.Bus_Number):
            for t in range(self.Total_time):
                if i == 0:
                    mdl.addConstr((PWT[i, t] + PPV[i, t] + Pdis[i, t] - Pch[i, t] + DN_Buy_P[i, t]) / self.baseMVA - self.Bus_load_P_MW24[i, t] / self.baseMVA == Bus_P_injection[i, t])
                    mdl.addConstr(DN_input_Q[i, t] / self.baseMVA - self.Bus_load_Q_MW24[i, t] / self.baseMVA == Bus_Q_injection[i, t])
                elif i in self.DG_node_Num:
                    idx_temp = self.DG_node_Num.index(i)
                    mdl.addConstr((PWT[i, t] + PPV[i, t] + Pdis[i, t] - Pch[i, t] + DG_P[idx_temp, t]) / self.baseMVA - self.Bus_load_P_MW24[i, t] / self.baseMVA == Bus_P_injection[i, t])
                    mdl.addConstr(DG_Q[idx_temp, t] / self.baseMVA - self.Bus_load_Q_MW24[i, t] / self.baseMVA == Bus_Q_injection[i, t])
                else:
                    mdl.addConstr((PWT[i, t] + PPV[i, t] + Pdis[i, t] - Pch[i, t]) / self.baseMVA - self.Bus_load_P_MW24[i, t] / self.baseMVA == Bus_P_injection[i, t])
                    mdl.addConstr(- self.Bus_load_Q_MW24[i, t] / self.baseMVA == Bus_Q_injection[i, t])
        # power flow
        for i in range(self.Bus_Number):
            for t in range(self.Total_time):
                mdl.addConstr(
                    Bus_P_injection[i, t] == gp.quicksum(
                        self.Y_po[i, j] * Bus_Fai[j, t] for j in range(self.Bus_Number)) + gp.quicksum(
                        self.Y_pv[i, j] * Bus_V[j, t] for j in range(self.Bus_Number)),
                    name='P_balance'
                )
                mdl.addConstr(
                    Bus_Q_injection[i, t] == gp.quicksum(
                        self.Y_qo[i, j] * Bus_Fai[j, t] for j in range(self.Bus_Number)) + gp.quicksum(
                        self.Y_qv[i, j] * Bus_V[j, t] for j in range(self.Bus_Number)),
                    name='Q_balance'
                )
        # branch power flow
        for i_br in range(self.Branch_Number):
            for t in range(self.Total_time):
                node_head = self.node_head_matrix[i_br]
                node_tail = self.node_tail_matrix[i_br]
                mdl.addConstr(P_ij[i_br, t] / self.baseMVA == self.Y_po[node_head, node_tail] * (
                        Bus_Fai[node_head, t] - Bus_Fai[node_tail, t]) + self.Y_pv[node_head, node_tail] * (
                                    Bus_V[node_head, t] - Bus_V[node_tail, t]))
                mdl.addConstr(Q_ij[i_br, t] / self.baseMVA == self.Y_qo[node_head, node_tail] * (
                        Bus_Fai[node_head, t] - Bus_Fai[node_tail, t]) + self.Y_qv[node_head, node_tail] * (
                                    Bus_V[node_head, t] - Bus_V[node_tail, t]))
                mdl.addConstr(P_ij[i_br, t] >= -self.Pij_limit[node_head, node_tail])
                mdl.addConstr(P_ij[i_br, t] <= self.Pij_limit[node_head, node_tail])
                mdl.addConstr(Q_ij[i_br, t] <= self.Qij_limit[node_head, node_tail])
                mdl.addConstr(Q_ij[i_br, t] >= -self.Qij_limit[node_head, node_tail])
        # node voltage
        # balance node
        for t in range(self.Total_time):
            mdl.addConstr(Bus_Fai[0, t] == 0)
            mdl.addConstr(Bus_V[0, t] == 1)
        # other nodes
        for i in range(1, self.Bus_Number):
            for t in range(self.Total_time):
                mdl.addConstr(Bus_V[i, t] >= self.Bus_V_min)
                mdl.addConstr(Bus_V[i, t] <= self.Bus_V_max)

        print('*' * 10, 'Objective function', '*' * 10)

        mdl.update()

        # objective function
        # planning cost
        obj_plan = gp.quicksum(
            self.cost_WT * self.EA_WT * WT_Plan[0, i] +
            self.cost_PV * self.EA_PV * PV_Plan[0, i] +
            self.cost_ESSE * self.EA_ESSE * E_Plan[0, i] +
            self.cost_ESSP * self.EA_ESSP * P_Plan[0, i] for i in range(self.Bus_Number))
        # operational cost
        obj_ope = 0
        obj_temp_list = []
        for d in range(self.Num_Days):
            obj_temp = 0
            obj_temp = gp.quicksum(
                        self.a_con[i] * DG_P[i, t] * DG_P[i, t] +
                        self.b_con[i] * DG_P[i, t] +
                        self.c_con[i] for i in range(self.N_DG_units) for t in range(d * self.N_time, (d + 1) * self.N_time)
                        ) + gp.quicksum(
                        self.cost_cur * (WT_Plan[0, i] * wt[t] - PWT[i, t] + PV_Plan[0, i] * pv[t] - PPV[i, t]) +
                        self.cost_ch * Pch[i, t] + self.cost_dis * Pdis[i, t]
                        for i in range(self.Bus_Number) for t in range(d * self.N_time, (d + 1) * self.N_time)) + gp.quicksum(
                        self.price[t] * DN_Buy_P[0, t] for t in range(d * self.N_time, (d + 1) * self.N_time))
            obj_temp_list.append(obj_temp * self.probability[d] * 365)
        obj_ope = gp.quicksum(obj_temp_list[i] for i in range(self.Num_Days))


        # total cost
        obj = obj_plan + obj_ope

        mdl.setObjective(obj, GRB.MINIMIZE)
        mdl.optimize()

        if mdl.status == GRB.OPTIMAL:
            print(f'minimum cost: {mdl.objVal} dollar')
            wt_output = [WT_Plan[0, i].X for i in range(self.Bus_Number)]
            pv_output = [PV_Plan[0, i].X for i in range(self.Bus_Number)]
            esse_output = [E_Plan[0, i].X for i in range(self.Bus_Number)]
            essp_output = [P_Plan[0, i].X for i in range(self.Bus_Number)]
            # planning cost calculation
            obj_plan = gp.quicksum(
                self.cost_WT * self.EA_WT * WT_Plan[0, i].X +
                self.cost_PV * self.EA_PV * PV_Plan[0, i].X +
                self.cost_ESSE * self.EA_ESSE * E_Plan[0, i].X +
                self.cost_ESSP * self.EA_ESSP * P_Plan[0, i].X for i in range(self.Bus_Number))
            # operational cost calculation
            obj_ope = 0
            obj_temp_list = []
            for d in range(self.Num_Days):
                obj_temp = 0
                obj_temp = sum(
                    self.a_con[i] * DG_P[i, t].X * DG_P[i, t].X +
                    self.b_con[i] * DG_P[i, t].X +
                    self.c_con[i] for i in range(self.N_DG_units) for t in range(d * self.N_time, (d + 1) * self.N_time)
                ) + sum(
                    self.cost_cur * (WT_Plan[0, i].X * wt[t] - PWT[i, t].X + PV_Plan[0, i].X * pv[t] - PPV[i, t].X) +
                    self.cost_ch * Pch[i, t].X + self.cost_dis * Pdis[i, t].X
                    for i in range(self.Bus_Number) for t in range(d * self.N_time, (d + 1) * self.N_time)) + sum(
                        self.price[t] * DN_Buy_P[0, t].X for t in range(d * self.N_time, (d + 1) * self.N_time))
                obj_temp_list.append(obj_temp * self.probability[d] * 365)
            obj_ope = sum(obj_temp_list[i] for i in range(self.Num_Days))
            obj = mdl.objVal
            record_cost = [obj, obj_plan, obj_ope]
            # figure save
            lable_device = ['WT', 'PV', 'ESSE', 'ESSP']
            TOTAL_PLAN = [sum(wt_output), sum(pv_output), sum(esse_output), sum(essp_output)]
            fig = plt.bar(lable_device, TOTAL_PLAN, width=0.8, bottom=[0, 0, 0, 0], alpha=0.8)
            plt.bar_label(fig, label_type='edge')
            plt.xlabel('type')
            plt.ylabel('capacity/MW(MWh)')
            plt.title('planning results')
            plt.savefig('vis/total_plan.svg')
            plt.close()
            # data save
            record_cost = pd.DataFrame(record_cost)
            wt_output = pd.DataFrame(wt_output)
            pv_output = pd.DataFrame(pv_output)
            esse_output = pd.DataFrame(esse_output)
            essp_output = pd.DataFrame(essp_output)
            record_cost.index = ['年总成本', '年规划成本', '年运行成本']
            wt_output.index = range(1, self.Bus_Number + 1)
            pv_output.index = range(1, self.Bus_Number + 1)
            esse_output.index = range(1, self.Bus_Number + 1)
            essp_output.index = range(1, self.Bus_Number + 1)
            record_cost.columns = ['费用/万美元']
            wt_output.columns = ['风机/MW']
            pv_output.columns = ['光伏/MW']
            esse_output.columns = ['储能电量容量/MWh']
            essp_output.columns = ['储能功率容量/MW']
            # write
            record_cost.to_excel("vis//结果_效果.xlsx", sheet_name="费用", index=True)
            with pd.ExcelWriter("vis//结果_效果.xlsx", mode='a', if_sheet_exists='overlay') as writer:
                wt_output.to_excel(writer, sheet_name="风机", index=True)
                pv_output.to_excel(writer, sheet_name="光伏", index=True)
                esse_output.to_excel(writer, sheet_name="储能电量", index=True)
                essp_output.to_excel(writer, sheet_name="储能功率", index=True)
        else:
            print("No optimal solution found")
object = DN_configuration()
object.create_model()
