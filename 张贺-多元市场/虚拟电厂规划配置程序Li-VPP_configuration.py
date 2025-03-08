import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 电网优化配置建模类
class VPP_Configuration:
    def __init__(self, DG_info, WT_info, PV_info, ESS_info):
        # 各设备数量
        self.N_DG_units = DG_info["N_DG_units"]
        self.N_WT_units = WT_info["N_WT_units"]
        self.N_PV_units = PV_info["N_PV_units"]
        self.N_ESS_units = ESS_info["N_ESS_units"]
        # DG信息
        self.DG_Cap_total = sum(DG_info["DG_Cap"])
        self.DG_Cap = DG_info["DG_Cap"]
        self.DG_Output_min = DG_info["DG_Output_min"]
        self.DG_Ramp_up = DG_info["DG_Ramp_up"]
        self.DG_Ramp_down = DG_info["DG_Ramp_down"]
        self.DG_off = DG_info["DG_off"]
        self.DG_on = DG_info["DG_on"]
        self.DG_a = DG_info["DG_a"]
        self.DG_b = DG_info["DG_b"]
        self.DG_c = DG_info["DG_c"]
        self.DG_loc = DG_info["DG_loc"]
        # WT信息
        self.WT_Cap_total = sum(WT_info["WT_Cap"])
        self.WT_Cap = WT_info["WT_Cap"]
        self.WT_Output_min = WT_info["WT_Output_min"]
        self.WT_loc = WT_info["WT_loc"]
        #PV信息
        self.PV_Cap_total = sum(PV_info["PV_Cap"])
        self.PV_Cap = PV_info["PV_Cap"]
        self.PV_Output_min = PV_info["PV_Output_min"]
        self.PV_loc = PV_info["PV_loc"]
        #BES信息
        self.ESS_PCap_total = sum(ESS_info["ESS_PCap"])
        self.ESS_ECap_total = sum(ESS_info["ESS_ECap"])
        self.ESS_PCap = ESS_info["ESS_PCap"]
        self.ESS_ECap = ESS_info["ESS_ECap"]
        self.ESS_Prated = ESS_info["ESS_PCap"]
        self.ESS_SOC_Cap = ESS_info["ESS_ECap"]
        self.ESS_SOC_initial = ESS_info["ESS_SOC_initial"]
        self.ESS_SOC_min = ESS_info["ESS_SOC_min"]
        self.ESS_SOC_max = ESS_info["ESS_SOC_max"]
        self.ESS_cost = ESS_info["ESS_cost"]
        self.ESS_diss = ESS_info["ESS_diss"]
        self.ESS_yita = ESS_info["ESS_yita"]
        self.ESS_loc = ESS_info["ESS_loc"]
    #待建设设备定义
    def Info_To_Built(self, WT_info_cons, PV_info_cons, ESS_info_cons):
        # WT
        self.N_WT_Blt = WT_info_Blt["N_WT_Blt"]
        self.WT_Cap_max_Blt = WT_info_Blt["WT_Cap_max_Blt"]
        self.WT_unit_cost = WT_info_Blt["WT_unit_cost"]
        self.WT_Cap_Blt = WT_info_Blt["WT_Cap_Blt"]
        self.WT_Output_min_Blt = WT_info_Blt["WT_Output_min_Blt"]
        self.WT_loc_Blt = WT_info_Blt["WT_loc_Blt"]
        #PV
        self.N_PV_Blt = PV_info_Blt["N_PV_Blt"]
        self.PV_Cap_max_Blt = PV_info_Blt["PV_Cap_max_Blt"]
        self.PV_unit_cost = PV_info_Blt["PV_unit_cost"]
        self.PV_Cap_Blt = PV_info_Blt["PV_Cap_Blt"]
        self.PV_Output_min_Blt = PV_info_Blt["PV_Output_min_Blt"]
        self.PV_loc_Blt = PV_info_Blt["PV_loc_Blt"]
        #BES
        self.N_ESS_Blt = ESS_info_Blt["N_ESS_Blt"]
        self.ESSP_Cap_max_Blt = ESS_info_Blt["ESSP_Cap_max_Blt"]
        self.ESSE_Cap_max_Blt = ESS_info_Blt["ESSE_Cap_max_Blt"]
        self.ESS_unit_cost = ESS_info_Blt["ESS_unit_cost"]
        self.ESS_Prated_Blt = ESS_info_Blt["ESS_PCap_Blt"]
        self.ESS_SOC_Cap_Blt = ESS_info_Blt["ESS_ECap_Blt"]
        self.ESS_SOC_initial_Blt = ESS_info_Blt["ESS_SOC_initial_Blt"]
        self.ESS_SOC_min_Blt = ESS_info_Blt["ESS_SOC_min_Blt"]
        self.ESS_SOC_max_Blt = ESS_info_Blt["ESS_SOC_max_Blt"]
        self.ESS_cost_Blt = ESS_info_Blt["ESS_cost_Blt"]
        self.ESS_diss_Blt = ESS_info_Blt["ESS_diss_Blt"]
        self.ESS_yita_Blt = ESS_info_Blt["ESS_yita_Blt"]
        self.ESS_loc_Blt = ESS_info_Blt["ESS_loc_Blt"]
    #场景集输入
    def set_scenarios(self, N_scenarios, probability_scenarios, load_scenarios, pv_scenarios, wt_scenarios, price_scenarios):
        self.N_scenarios = N_scenarios
        self.probability_scenarios = probability_scenarios
        self.load_scenarios = load_scenarios
        self.pv_scenarios = pv_scenarios
        self.wt_scenarios = wt_scenarios
        self.price_scenarios = price_scenarios
    #技术经济参数
    def Technical_Economic(self, r, Life, Time, Grid_Interactive, big_M):
        self.r = r
        self.Life = Life
        self.Time = Time
        self.Grid_Interactive = Grid_Interactive
        self.big_M = big_M
        self.EAC = (1 + r) ** Life/((1 + r) ** Life - 1)
    #优化模型构建
    def optimize(self):
        # 创建模型
        model = gp.Model("VPP_Configuration")
        ##变量定义
        # 待建设变量
        WT_state_Blt = model.addVars(self.N_WT_Blt, lb=0, vtype=GRB.BINARY, name="WT_Built")
        PV_state_Blt = model.addVars(self.N_PV_Blt, lb=0, vtype=GRB.BINARY, name="PV_Built")
        BES_state_Blt = model.addVars(self.N_ESS_Blt, vtype=GRB.BINARY, name="BES_Built")
        # 现存设备输出功率
        DG_output = model.addVars(self.N_DG_units, self.N_scenarios, self.Time, lb=0, name="DG_output")
        WT_output = model.addVars(self.N_WT_units, self.N_scenarios, self.Time, lb=0, name="WT_output")
        PV_output = model.addVars(self.N_PV_units, self.N_scenarios, self.Time, lb=0, name="PV_output")
        ESSP_dis = model.addVars(self.N_ESS_units, self.N_scenarios, self.Time, lb=0, name="ESSP_output")
        ESSP_ch = model.addVars(self.N_ESS_units, self.N_scenarios, self.Time, lb=0, name="ESSP_output")
        ESS_state = model.addVars(self.N_ESS_units, self.N_scenarios, self.Time, vtype=GRB.BINARY, name="ESSP_output")
        SOCE_state = model.addVars(self.N_ESS_units, self.N_scenarios, self.Time, lb=0, name="ESSE_state")
        ## 待建设设备输出功率
        WT_output_Blt = model.addVars(self.N_WT_Blt, self.N_scenarios, self.Time, lb=0, name="WT_output")
        PV_output_Blt = model.addVars(self.N_PV_Blt, self.N_scenarios, self.Time, lb=0, name="PV_output")
        ESSP_dis_Blt = model.addVars(self.N_ESS_Blt, self.N_scenarios, self.Time, lb=0, name="ESSP_output")
        ESSP_ch_Blt = model.addVars(self.N_ESS_Blt, self.N_scenarios, self.Time, lb=0, name="ESSP_output")
        ESS_state_Blt = model.addVars(self.N_ESS_Blt, self.N_scenarios, self.Time, vtype=GRB.BINARY, name="ESSP_output")
        SOCE_state_Blt = model.addVars(self.N_ESS_Blt, self.N_scenarios, self.Time, lb=0, name="ESSE_state")
        ## 与电网交互功率
        Grid_output = model.addVars(self.N_scenarios, self.Time, lb=0, name="DG_output")
        ##约束构建
        #设备输出功率约束
        #与电网交互功率
        model.addConstrs(Grid_output[j, t] >= - self.Grid_Interactive for t in range(self.Time) for j in range(self.N_scenarios))
        model.addConstrs(Grid_output[j, t] <= self.Grid_Interactive for t in range(self.Time) for j in range(self.N_scenarios))
        #已存在设备功率约束
        model.addConstrs(DG_output[i, j, t] >= self.DG_Output_min[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_DG_units))
        model.addConstrs(DG_output[i, j, t] <= self.DG_Cap[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_DG_units))
        model.addConstrs(DG_output[i, j, t] - DG_output[i, j, t - 1] >= -self.DG_Ramp_down[i] for t in range(1, self.Time) for j in range(self.N_scenarios) for i in range(self.N_DG_units))
        model.addConstrs(DG_output[i, j, t] - DG_output[i, j, t - 1] <= self.DG_Ramp_up[i] for t in range(1, self.Time) for j in range(self.N_scenarios) for i in range(self.N_DG_units))
        model.addConstrs(WT_output[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_WT_units))
        model.addConstrs(WT_output[i, j, t] <= self.WT_Cap[i] * self.wt_scenarios[j, t] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_WT_units))
        model.addConstrs(PV_output[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_PV_units))
        model.addConstrs(PV_output[i, j, t] <= self.PV_Cap[i] * self.pv_scenarios[j, t] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_PV_units))
        model.addConstrs(ESSP_dis[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(ESSP_dis[i, j, t] <= self.ESS_PCap[i] * ESS_state[i, j, t] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(ESSP_ch[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(ESSP_ch[i, j, t] <= self.ESS_PCap[i] * (1 - ESS_state[i, j, t]) for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        #待建设设备功率约束
        model.addConstrs(WT_output_Blt[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_WT_Blt))
        model.addConstrs(WT_output_Blt[i, j, t] <= self.WT_Cap_Blt[i] * self.wt_scenarios[j, t] * WT_state_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_WT_Blt))
        model.addConstrs(PV_output_Blt[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_PV_Blt))
        model.addConstrs(PV_output_Blt[i, j, t] <= self.PV_Cap_Blt[i] * self.pv_scenarios[j, t] * PV_state_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_PV_Blt))
        model.addConstrs(ESSP_dis_Blt[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(ESSP_dis_Blt[i, j, t] <= self.ESS_Prated_Blt[i] * BES_state_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(ESSP_dis_Blt[i, j, t] <= self.big_M * ESS_state_Blt[i, j, t] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(ESSP_ch_Blt[i, j, t] >= 0 for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(ESSP_ch_Blt[i, j, t] <= self.ESS_Prated_Blt[i] * BES_state_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(ESSP_ch_Blt[i, j, t] <= self.big_M * (1 - ESS_state_Blt[i, j, t]) for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        #功率平衡约束
        model.addConstrs(self.load_scenarios[j, t] - Grid_output[j, t] == gp.quicksum(DG_output[i, j, t] for i in range(self.N_DG_units)) + \
                         gp.quicksum(WT_output[i, j, t] for i in range(self.N_WT_units)) + \
                         gp.quicksum(PV_output[i, j, t] for i in range(self.N_PV_units)) + \
                         gp.quicksum(ESSP_dis[i, j, t] for i in range(self.N_ESS_units)) - \
                         gp.quicksum(ESSP_ch[i, j, t] for i in range(self.N_ESS_units)) + \
                         gp.quicksum(WT_output_Blt[i, j, t] for i in range(self.N_WT_Blt)) + \
                         gp.quicksum(PV_output_Blt[i, j, t] for i in range(self.N_PV_Blt)) + \
                         gp.quicksum(ESSP_dis_Blt[i, j, t] for i in range(self.N_ESS_Blt)) - \
                         gp.quicksum(ESSP_ch_Blt[i, j, t] for i in range(self.N_ESS_Blt)) for t in range(self.Time) for j in range(self.N_scenarios))
        #储能设备相关约束
        model.addConstrs(SOCE_state[i, j, t] >= self.ESS_SOC_min[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(SOCE_state[i, j, t] <= self.ESS_SOC_max[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(SOCE_state_Blt[i, j, t] >= self.ESS_SOC_min_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        model.addConstrs(SOCE_state_Blt[i, j, t] <= self.ESS_SOC_max_Blt[i] for t in range(self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        #第一时段能量变化约束
        model.addConstrs(SOCE_state[i, j, 1] == self.ESS_SOC_initial[i] * self.ESS_diss[i] + \
                         ESSP_ch[i, j, 1] * self.ESS_yita[i] - ESSP_dis[i, j, 1] / self.ESS_yita[i] \
                         for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(SOCE_state_Blt[i, j, 1] == self.ESS_SOC_initial_Blt[i] * self.ESS_diss_Blt[i] + \
                         ESSP_ch_Blt[i, j, 1] * self.ESS_yita_Blt[i] - ESSP_dis_Blt[i, j, 1] / self.ESS_yita_Blt[i] \
                         for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))
        #后续时段能量变化约束
        model.addConstrs(SOCE_state[i, j, t] == SOCE_state[i, j, t - 1] * self.ESS_diss[i] + \
                         ESSP_ch[i, j, t] * self.ESS_yita[i] - ESSP_dis[i, j, t] / self.ESS_yita[i] \
                         for t in range(1, self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_units))
        model.addConstrs(SOCE_state_Blt[i, j, t] == SOCE_state_Blt[i, j, t - 1] * self.ESS_diss_Blt[i] + \
                         ESSP_ch_Blt[i, j, t] * self.ESS_yita_Blt[i] - ESSP_dis_Blt[i, j, t] / self.ESS_yita_Blt[i] \
                         for t in range(1, self.Time) for j in range(self.N_scenarios) for i in range(self.N_ESS_Blt))

        ##目标函数构建
        objective_built = self.EAC * (sum(self.WT_unit_cost[i] * self.WT_Cap_Blt[i] * WT_state_Blt[i] for i in range(self.N_WT_Blt)) + \
                               sum(self.PV_unit_cost[i] * self.PV_Cap_Blt[i] * PV_state_Blt[i] for i in range(self.N_PV_Blt)) + \
                               sum(self.ESS_unit_cost[i] * self.ESS_Prated_Blt[i] * BES_state_Blt[i] for i in range(self.N_ESS_Blt)))
        objective_operation = 365 * sum(self.probability_scenarios[j] * sum(self.price_scenarios[j, t] * Grid_output[j, t] + \
                                                                   sum(self.ESS_cost_Blt[i] * (ESSP_ch_Blt[i, j, t] + ESSP_dis_Blt[i, j, t]) for i in range(self.N_ESS_units)) + \
                                                                   sum(self.ESS_cost[i] * (ESSP_ch[i, j, t] + ESSP_dis[i, j, t]) for i in range(self.N_ESS_units)) + \
                                                                   sum(self.DG_a[i] * DG_output[i, j, t]**2 + self.DG_b[i] * DG_output[i, j, t] + self.DG_c[i] for i in range(self.N_DG_units)) \
                                                                       for t in range(self.Time)) for j in range(self.N_scenarios))
        objective = objective_built + objective_operation
        model.setObjective(objective, GRB.MINIMIZE)
        # 优化模型求解
        model.optimize()

        if model.status == GRB.OPTIMAL:
            self.WT_Cap_Result = self.WT_Cap_total
            self.PV_Cap_Result = self.PV_Cap_total
            self.ESSP_Cap_Result = self.ESS_PCap_total
            self.ESSE_Cap_Result = self.ESS_ECap_total
            self.Load_Result = np.zeros((self.N_scenarios, self.Time))
            self.PV_Result = np.zeros((self.N_scenarios, self.Time))
            self.WT_Result = np.zeros((self.N_scenarios, self.Time))
            self.DG_Result = np.zeros((self.N_scenarios, self.Time))
            self.ESS_Result = np.zeros((self.N_scenarios, self.Time))
            self.Grid_Result = np.zeros((self.N_scenarios, self.Time))

            #各设备容量计算
            for i in range(self.N_WT_Blt):
                self.WT_Cap_Result = self.WT_Cap_Result + self.WT_Cap_Blt[i] * WT_state_Blt[i].x
            for i in range(self.N_PV_Blt):
                self.PV_Cap_Result = self.PV_Cap_Result + self.PV_Cap_Blt[i] * PV_state_Blt[i].x
            for i in range(self.N_ESS_Blt):
                self.ESSP_Cap_Result = self.ESSP_Cap_Result + self.ESS_Prated_Blt[i] * BES_state_Blt[i].x
                self.ESSE_Cap_Result = self.ESSE_Cap_Result + self.ESS_SOC_Cap_Blt[i] * BES_state_Blt[i].x
            #所需结果
            for j in range(self.N_scenarios):
                for t in range(self.Time):
                    self.Load_Result[j, t] = self.load_scenarios[j, t]
                    self.PV_Result[j, t] = sum(PV_output[i, j, t].x for i in range(self.N_PV_units)) + sum(PV_output_Blt[i, j, t].x for i in range(self.N_PV_Blt))
                    self.WT_Result[j, t] = sum(WT_output[i, j, t].x for i in range(self.N_WT_units)) + sum(WT_output_Blt[i, j, t].x for i in range(self.N_WT_Blt))
                    self.DG_Result[j, t] = sum(DG_output[i, j, t].x for i in range(self.N_DG_units))
                    self.ESS_Result[j, t] = sum(ESSP_dis[i, j, t].x - ESSP_ch[i, j, t].x for i in range(self.N_ESS_units)) + sum(ESSP_dis_Blt[i, j, t].x - ESSP_ch_Blt[i, j, t].x for i in range(self.N_ESS_Blt))
                    self.Grid_Result[j, t] = Grid_output[j, t].x
            #所需规划成本参数
            self.built_cost = self.EAC * (
                        sum(self.WT_unit_cost[i] * self.WT_Cap_Blt[i] * WT_state_Blt[i].x for i in range(self.N_WT_Blt)) + \
                        sum(self.PV_unit_cost[i] * self.PV_Cap_Blt[i] * PV_state_Blt[i].x for i in range(self.N_PV_Blt)) + \
                        sum(self.ESS_unit_cost[i] * self.ESS_Prated_Blt[i] * BES_state_Blt[i].x for i in range(self.N_ESS_Blt)))
            self.operation_cost = 365 * sum(
                self.probability_scenarios[j] * sum(self.price_scenarios[j, t] * Grid_output[j, t].x + \
                                                    sum(self.ESS_cost_Blt[i] * (ESSP_ch_Blt[i, j, t].x + ESSP_dis_Blt[i, j, t].x) for i in range(self.N_ESS_units)) + \
                                                    sum(self.ESS_cost[i] * (ESSP_ch[i, j, t].x + ESSP_dis[i, j, t].x) for i in range(self.N_ESS_units)) + \
                                                    sum(self.DG_a[i] * DG_output[i, j, t].x ** 2 + self.DG_b[i] * DG_output[i, j, t].x + self.DG_c[i] for i in range(self.N_DG_units)) \
                                                    for t in range(self.Time)) for j in range(self.N_scenarios))

    #结果绘制
    #规划结果绘制
    def plot_results_planning(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.planning_results = [self.WT_Cap_Result, self.PV_Cap_Result, self.ESSP_Cap_Result, self.ESSE_Cap_Result]
        plt.bar(np.arange(1, 5), self.planning_results)
        x_labels = ["风机", "光伏", "储能功率容量", "储能电量容量"]
        plt.xticks(np.arange(1, 5), x_labels)
        plt.title(f"规划结果展示（单位MW）")
        plt.show()
    #运行结果绘制
    def plot_results_operation(self, scenario_idx):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar(np.arange(1, 25), self.Load_Result[scenario_idx - 1])
        plt.plot(np.arange(1, 25), self.PV_Result[scenario_idx - 1], label=f"光伏出力")
        plt.plot(np.arange(1, 25), self.WT_Result[scenario_idx - 1], label=f"风机出力")
        plt.plot(np.arange(1, 25), self.DG_Result[scenario_idx - 1], label=f"DG出力")
        plt.plot(np.arange(1, 25), self.ESS_Result[scenario_idx - 1], label=f"储能功率")
        plt.plot(np.arange(1, 25), self.Grid_Result[scenario_idx - 1], label=f"电网功率")
        plt.title(f"典型场景{scenario_idx}运行模拟结果")
        plt.legend()
        plt.show()

## 各类信息录入
#各类设备初始总量
DG_info = {
    "N_DG_units": 1,
    "DG_Cap": np.array([2]),
    "DG_Output_min": np.array([0.3]),
    "DG_Ramp_up": np.array([0.4]),
    "DG_Ramp_down": np.array([0.4]),
    "DG_off": np.array([0]),
    "DG_on": np.array([0]),
    "DG_a": np.array([54.5]) / 1e4,
    "DG_b": np.array([605.2]) / 1e4,
    "DG_c": np.array([34.1]) / 1e4,
    "DG_loc": np.array([2])
}
#DG基本信息录入
WT_info = {
    "N_WT_units": 2,
    "WT_Cap": np.array([1, 0.5]),
    "WT_Output_min": np.array([0, 0]),
    "WT_loc": np.array([3, 6])
}
#PV参数录入
PV_info = {
    "N_PV_units": 2,
    "PV_Cap": np.array([1, 0.5]),
    "PV_Output_min": np.array([0, 0]),
    "PV_loc": np.array([4, 2])
}
#ESS基本信息录入
ESS_info = {
    "N_ESS_units": 2,
    "ESS_PCap": np.array([0.1, 0.1]),
    "ESS_ECap": np.array([0.5, 0.5]),
    "ESS_SOC_initial": np.array([0.05, 0.05]),
    "ESS_SOC_min": np.array([0.01, 0.01]),
    "ESS_SOC_max": np.array([0.09, 0.09]),
    "ESS_cost": np.array([0.01, 0.01]),
    "ESS_diss": np.array([0.999, 0.999]),
    "ESS_yita": np.array([0.95, 0.95]),
    "ESS_loc": np.array([3, 5])
}
#待建设数据录入
#待建WT参数
WT_info_Blt = {
    "N_WT_Blt": 3,
    "WT_Cap_max_Blt": 2,
    "WT_unit_cost": np.array([1200, 1200, 1200]),
    "WT_Cap_Blt": np.array([1, 0.4, 0.6]),
    "WT_Output_min_Blt": np.array([0, 0, 0]),
    "WT_loc_Blt": np.array([1, 4, 2])
}
#待建PV参数
PV_info_Blt = {
    "N_PV_Blt": 2,
    "PV_Cap_max_Blt": 1.5,
    "PV_unit_cost": np.array([1100, 1100]),
    "PV_Cap_Blt": np.array([1, 0.5]),
    "PV_Output_min_Blt": np.array([0, 0]),
    "PV_loc_Blt": np.array([3, 2])
}
#待建ESS参数
ESS_info_Blt = {
    "N_ESS_Blt": 2,
    "ESSP_Cap_max_Blt": 0.3,
    "ESSE_Cap_max_Blt": 1.5,
    "ESS_unit_cost": np.array([150, 150]),
    "ESS_PCap_Blt": np.array([0.2, 0.1]),
    "ESS_ECap_Blt": np.array([1, 0.5]),
    "ESS_SOC_initial_Blt": np.array([0.1, 0.05]),
    "ESS_SOC_min_Blt": np.array([0.02, 0.01]),
    "ESS_SOC_max_Blt": np.array([0.18, 0.09]),
    "ESS_cost_Blt": np.array([0.01, 0.01]),
    "ESS_diss_Blt": np.array([0.999, 0.999]),
    "ESS_yita_Blt": np.array([0.95, 0.95]),
    "ESS_loc_Blt": np.array([4, 2])
}
#场景参数录入
N_scenarios = 4
probability_scenarios = np.array([0.230136986, 0.246575342, 0.24109589, 0.282191781])
load_scenarios = np.array([
    [2.047614039, 1.968599539, 1.927541085, 1.908803609, 1.954350155, 2.130348724, 2.526925251, 2.714983899, 2.593529821, 2.602174189, 2.622058184, 2.722372851, 2.492108344, 2.448248789, 2.507553639, 2.586737739, 2.843448192, 3.092486928, 3.043262684, 2.981971929, 2.889350356, 2.70965332, 2.454454485, 2.229784856],
    [2.075683804, 1.994502326, 1.959706845, 1.944014951, 1.973504807, 2.053740098, 2.26297268, 2.656187551, 2.76642927, 2.713014421, 2.661272166, 2.687947989, 2.541472955, 2.502471463, 2.57256222, 2.681795184, 2.943768344, 3.155453574, 3.037063112, 2.927230762, 2.832696598, 2.717884792, 2.497384168, 2.272687717],
    [2.155736709, 2.053956255, 2.00570662, 1.978531456, 2.035832788, 2.268043053, 2.59558481, 2.703509926, 2.68599629, 2.801578539, 2.907802287, 3.06173895, 2.8140764, 2.738079863, 2.783933516, 2.828151871, 2.97403563, 3.197825695, 3.05594566, 2.961075319, 2.986995237, 2.86674398, 2.593855548, 2.351422974],
    [2.036314987, 1.953877416, 1.915327736, 1.89472676, 1.94032069, 2.127295703, 2.537146507, 2.693083019, 2.580900192, 2.612255106, 2.657742486, 2.778072894, 2.531604973, 2.479381467, 2.537162038, 2.608972028, 2.855180895, 3.108242688, 3.037559824, 2.960942996, 2.879543133, 2.692611505, 2.444030895, 2.230071592]
])
pv_scenarios = np.array([
    [7.02381E-05, 7.00397E-05, 6.98696E-05, 7.00113E-05, 7.0068E-05, 0.002080754, 0.017792375, 0.091408305, 0.214087301, 0.334876786, 0.407567007, 0.446637922, 0.438059832, 0.395402323, 0.339488859, 0.241253854, 0.129306122, 0.038877098, 0.006224093, 0.000770833, 0.000440788, 5.61225E-05, 4.69388E-05, 4.65986E-05],
    [1.77249E-06, 1.90476E-06, 1.74603E-06, 1.90476E-06, 1.74603E-06, 1.79894E-06, 1.8254E-06, 0.004532222, 0.075684523, 0.278410926, 0.399801745, 0.474621057, 0.485614285, 0.452276428, 0.357488176, 0.176405503, 0.043332328, 0.001441138, 2.48677E-06, 2.1164E-06, 2.14286E-06, 2.14286E-06, 2.1164E-06, 2.06349E-06],
    [9.74026E-06, 9.74026E-06, 9.74026E-06, 9.74026E-06, 0.000270265, 0.015049567, 0.057947159, 0.144059172, 0.246354544, 0.325220969, 0.3818319, 0.412977462, 0.411347756, 0.393144722, 0.336310471, 0.268325568, 0.182442072, 0.092978166, 0.031013095, 0.005081683, 2.54329E-05, 2.5974E-05, 9.60498E-06, 9.60498E-06],
    [5.34905E-05, 5.34212E-05, 5.34212E-05, 5.34212E-05, 9.1031E-05, 0.005289159, 0.027844129, 0.120901041, 0.250371821, 0.358681532, 0.424152661, 0.445346045, 0.451449651, 0.409660402, 0.340210888, 0.248124942, 0.13194748, 0.038698775, 0.007258068, 0.000512043, 0.000117799, 9.68793E-05, 9.37818E-05, 7.28155E-05]
])
wt_scenarios = np.array([
    [0.335766853, 0.326020488, 0.323914728, 0.321359608, 0.3230353, 0.330795552, 0.336748546, 0.331223751, 0.317395366, 0.316728372, 0.320299913, 0.331688736, 0.327211235, 0.327446736, 0.332666168, 0.329409069, 0.32192974, 0.314729569, 0.307125533, 0.30109856, 0.297629171, 0.288391931, 0.280280852, 0.272795495],
    [0.296328838, 0.293920093, 0.290678948, 0.291834862, 0.290404272, 0.289949156, 0.289288291, 0.287494664, 0.276072642, 0.246143847, 0.219235927, 0.214452096, 0.215874331, 0.219383446, 0.227582356, 0.233467837, 0.24796141, 0.269382089, 0.286987133, 0.29834024, 0.305039214, 0.308990805, 0.30942342, 0.305223035],
    [0.137515105, 0.13378752, 0.129035996, 0.126437602, 0.125880838, 0.119122346, 0.105891672, 0.094617215, 0.090971625, 0.092988786, 0.097868454, 0.101298653, 0.105850109, 0.110193304, 0.111012779, 0.112927967, 0.11104227, 0.105678892, 0.108148899, 0.119287221, 0.132100259, 0.135795658, 0.139504817, 0.14090358],
    [0.164802477, 0.157542097, 0.152025416, 0.147470647, 0.145411255, 0.14203444, 0.135111412, 0.118179954, 0.099543759, 0.092343201, 0.102189082, 0.116301645, 0.123380908, 0.130279682, 0.134585535, 0.138402616, 0.138895948, 0.149459345, 0.167311087, 0.191471671, 0.208739063, 0.218658578, 0.223752305, 0.22519549]
])
price_scenarios = np.array([
    [0.2666, 0.2666, 0.2666, 0.2666, 0.2666, 0.538, 0.538, 0.888, 0.888, 0.888, 0.888, 0.888, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.888, 0.888, 0.888, 0.538, 0.2666, 0.2666],
    [0.2666, 0.2666, 0.2666, 0.2666, 0.2666, 0.538, 0.538, 0.888, 0.888, 0.888, 0.888, 0.888, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.888, 0.888, 0.888, 0.538, 0.2666, 0.2666],
    [0.2666, 0.2666, 0.2666, 0.2666, 0.2666, 0.538, 0.538, 0.888, 0.888, 0.888, 0.888, 0.888, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.888, 0.888, 0.888, 0.538, 0.2666, 0.2666],
    [0.2666, 0.2666, 0.2666, 0.2666, 0.2666, 0.538, 0.538, 0.888, 0.888, 0.888, 0.888, 0.888, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.888, 0.888, 0.888, 0.538, 0.2666, 0.2666]
]) / 10
#技术经济参数录入
r = 0.05
Life = 20
Time = 24
Grid_Interactive = 1
big_M = 1e4
#待展示场景
scenario_idx = 1 #所有场景中选择一个
########################################################################
##模型求解
# 初始化VPP参数
vpp = VPP_Configuration(DG_info, WT_info, PV_info, ESS_info)
# 待建设设备参数
vpp.Info_To_Built(WT_info_Blt, PV_info_Blt, ESS_info_Blt)
# 设置负荷场景
vpp.set_scenarios(N_scenarios, probability_scenarios, load_scenarios, pv_scenarios, wt_scenarios, price_scenarios)
# 技术经济参数
vpp.Technical_Economic(r, Life, Time, Grid_Interactive, big_M)
# 优化求解
vpp.optimize()
# 规划结果展示
vpp.plot_results_planning()
# 运行结果展示
vpp.plot_results_operation(scenario_idx)
########################################################################
