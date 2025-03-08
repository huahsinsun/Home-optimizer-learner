from gurobipy import *
import Sys_data_typical as Sys_data

# (No noise)Optimal 794638.689387829
# 0.268063952933279   0.272342358824407   0.083632654931722
## Construct model
# Model ( name="", env=defaultEnv )
M = Model("Grid_Cost")

Noise_degree = 0
NOISE_SE = 0
dim = 0
delta_P = 0
delta_V = 0

# 0.102470604589965*10+6
# Input = [0,0,0]
if dim == 1:
    wind = M.addVar(lb=6, ub=16, name='wind_capacity')
    # wind = M.addVar(lb=0.102470604589965*10+6, ub=0.102470604589965*10+6, name='wind_capacity')
    solar = M.addVar(lb=0, ub=0, name='solar_capacity')
    res = M.addVar(lb=0, ub=0, name='res_capacity')
if dim == 3:
    wind = M.addVar(lb=3, ub=13, name='wind_capacity')
    solar = M.addVar(lb=3, ub=13, name='solar_capacity')
    res = M.addVar(lb=0, ub=10, name='res_capacity')
elif dim == 2.1:
    wind = M.addVar(lb=3, ub=13, name='wind_capacity')
    solar = M.addVar(lb=1, ub=11, name='solar_capacity')
    res = M.addVar(lb=0, ub=0, name='res_capacity')
elif dim == 2.2:
    wind = M.addVar(lb=2, ub=12, name='wind_capacity')
    solar = M.addVar(lb=0, ub=0, name='solar_capacity')
    res = M.addVar(lb=2, ub=12, name='res_capacity')
elif dim == 2.3:
    wind = M.addVar(lb=0, ub=0, name='wind_capacity')
    solar = M.addVar(lb=9, ub=19, name='solar_capacity')
    res = M.addVar(lb=9, ub=19, name='res_capacity')
else:
    wind = M.addVar(lb=0, ub=0, name='wind_capacity')
    solar = M.addVar(lb=0, ub=0, name='solar_capacity')
    res = M.addVar(lb=0, ub=0, name='res_capacity')

scheme = 0
Y_po = Sys_data.Y_po
Y_pv = Sys_data.Y_pv
Y_qo = Sys_data.Y_qo
Y_qv = Sys_data.Y_qv
ESS = M.addVars(1, 3, lb=0, ub=0, name='ESS_capacity')
SVG = M.addVars(1, 3, lb=0, ub=0, name='SVG_capacity')

M.update()
if scheme == 1:  # the expansion of network
    Y_po = Y_po / 2
    Y_pv = Y_pv / 2
    Y_qo = Y_qo / 2
    Y_qv = Y_qv / 2
    expansion_cost = 100000  # to be determined
elif scheme == 2:  # install ESS
    ESS.setAttr('UB', [10] * 3)
elif scheme == 3:  # install SVG
    SVG.setAttr('UB', [10] * 3)

WT_cost, Solar_cost, RES_cost = Sys_data.WT_cost, Sys_data.Solar_cost, Sys_data.RES_cost

P_Wind_24 = Sys_data.P_Wind_24
P_PV_24 = Sys_data.P_PV_24
N_time = Sys_data.N_time * 4

Pij_limit = delta_P + Sys_data.Pij_limit

# if delta_P == -2.8:
#     wind, solar, res = (0.4 + Input[0]) * 10, (0.4 + Input[1]) * 10, (Input[2]) * 2
# elif delta_P == -2.9:
#     wind, solar, res = (0.4 + Input[0]) * 10, (0.4 + Input[1]) * 10, (Input[2] + 0.05) * 2

# if delta_V == -0.1
# elif delta_P == -3:
#     wind, solar, res = (0.5 + Input[0]) * 10, (0.5 + Input[1]) * 10, (Input[2]+0.2) * 2

Qij_limit = Sys_data.Qij_limit
Bus_V_max = delta_V + Sys_data.Bus_V_max
Bus_V_min = -delta_V + Sys_data.Bus_V_min

## Define variables
# addVar ( lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None )
# addVars ( *indices, lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="" )
# Out put P\Q of generators 1/2
Gen_P = M.addVars(1, N_time, lb=[0] * N_time,
                  ub=[10] * N_time, name="DG0_P")
Gen_Q = M.addVars(1, N_time, lb=[-10] * N_time,
                  ub=[10] * N_time, name="DG0_Q")
DG1_P = M.addVars(1, N_time, lb=[0] * N_time,
                  ub=[0] * N_time, name="DG1_P")
DG1_Q = M.addVars(1, N_time, lb=[0] * N_time,
                  ub=[0] * N_time, name="DG1_Q")
# Output power of Solar panel/Wind turbine (temporary ub\lb)
PV_P_output = M.addVars(1, N_time, lb=[0] * N_time, name="PV_P_output")
WF_P_output = M.addVars(1, N_time, lb=[0] * N_time, name="WF_P_output")
# Purchased power/Input Var power of VPP
DN_Buy_P = M.addVars(1, N_time, lb=[-Sys_data.DN_Buy_P_limit] * N_time,
                     ub=[Sys_data.DN_Buy_P_limit] * N_time, name="DN_Buy_P")
DN_input_Q = M.addVars(1, N_time, lb=[-Sys_data.DN_Buy_Q_limit] * N_time,
                       ub=[Sys_data.DN_Buy_Q_limit] * N_time, name="DN_input_Q")
# Bus phase/voltage/injected power/injected var
Bus_Fai = M.addVars(Sys_data.Bus_Number, N_time, lb=-float('inf'),
                    name="Bus_Fai")  # Bus Fai == 0 due the assumption of PF model
Bus_V = M.addVars(Sys_data.Bus_Number, N_time, name="Bus_V")
Bus_P_injection = M.addVars(Sys_data.Bus_Number, N_time, lb=-float('inf'), name="Bus_IN_P")
Bus_Q_injection = M.addVars(Sys_data.Bus_Number, N_time, lb=-float('inf'), name="Bus_IN_Q")
# Branch P/Q from node i to node j
P_ij = M.addVars(Sys_data.Branch_Number, N_time, lb=-float('inf'), name='P_ij')
Q_ij = M.addVars(Sys_data.Branch_Number, N_time, lb=-float('inf'), name="Q_ij")
# Reserve power
ESS_P_output = M.addVars(3, N_time, lb=[0] * N_time, ub=[Sys_data.ESS_P_max] * N_time,
                         name='ESS_P_output')  # Output power
ESS_P_input = M.addVars(3, N_time, lb=[0] * N_time, ub=[Sys_data.ESS_P_max] * N_time,
                        name='ESS_P_input')  # input power
SOC_state = M.addVars(3, N_time, ub=10 * N_time, name='SOC_state')

M.update()
## Add constrains
# addConstrs ( generator, name="" )
# Warning: A constraint can only have a single comparison operator.
# While 1 <= x + y <= 2 may look like a valid constraint, addConstr won't accept it.

# ESS constrains
for j in range(3):
    M.addConstr(
        SOC_state[j, 0] == SOC_state[j, N_time - 1],
        name=f'SOC_{j}')
    M.addConstrs(
        (SOC_state[j, i] == SOC_state[j, i - 1] * Sys_data.Laimuta + ESS_P_input[j, i] - ESS_P_output[
            j, i] for i in range(1, N_time)), name='SOC')
    M.addConstrs(SOC_state[j, i] <= ESS[j] for i in range(N_time))

# # DG constrains
# M.addConstrs(
#     (Sys_data.e_matrix[i, 0] * Gen_P[0, i] + Sys_data.e_matrix[i, 1] * Gen_Q[0, i] <= Sys_data.DG_con_SN[0] *
#      Sys_data.f_matrix[i] for i in range(8)), name='unknown0')
# M.addConstrs(
#     (Sys_data.e_matrix[i, 0] * DG1_P[0, i] + Sys_data.e_matrix[i, 1] * DG1_Q[0, i] <= Sys_data.DG_con_SN[1] *
#      Sys_data.f_matrix[i]
#      for i in range(8)), name='unknown1')

# RES constraints
M.addConstrs((PV_P_output[0, i] <= P_PV_24[i] * solar for i in range(N_time)), name='PV')
M.addConstrs((WF_P_output[0, i] <= P_Wind_24[i] * wind for i in range(N_time)), name='Wind')

# Power balance constraints
for t in range(N_time):
    for k_row in range(Sys_data.Bus_Number):
        if k_row == 0:
            M.addConstr(Gen_P[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA ==
                        Bus_P_injection[k_row, t])
            M.addConstr(
                Gen_Q[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA ==
                Bus_Q_injection[k_row, t])
        elif k_row == 9:
            M.addConstr(ESS_P_output[0, t] / Sys_data.baseMVA - ESS_P_input[0, t] / Sys_data.baseMVA -
                        Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA == Bus_P_injection[k_row, t])
            M.addConstr(- Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA == Bus_Q_injection[k_row, t])
        elif k_row == 21:
            M.addConstr(DG1_P[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA ==
                        Bus_P_injection[k_row, t])
            M.addConstr(DG1_Q[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA ==
                        Bus_Q_injection[k_row, t])
        elif k_row == 13:
            M.addConstr(
                PV_P_output[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA ==
                Bus_P_injection[
                    k_row, t])
            M.addConstr(- Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA == Bus_Q_injection[k_row, t])
        elif k_row == 19:
            M.addConstr(
                WF_P_output[0, t] / Sys_data.baseMVA - Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA ==
                Bus_P_injection[
                    k_row, t])
            M.addConstr(- Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA == Bus_Q_injection[k_row, t])
        else:
            M.addConstr(-Sys_data.Bus_load_P_MW24[k_row, t] / Sys_data.baseMVA == Bus_P_injection[k_row, t])
            M.addConstr(-Sys_data.Bus_load_Q_MW24[k_row, t] / Sys_data.baseMVA == Bus_Q_injection[k_row, t])

# Power flow
for i in range(Sys_data.Bus_Number):
    for t in range(N_time):
        M.addConstr(
            Bus_P_injection[i, t] == quicksum(
                Y_po[i, j] * Bus_Fai[j, t] for j in range(Sys_data.Bus_Number)) + quicksum(
                Y_pv[i, j] * Bus_V[j, t] for j in range(Sys_data.Bus_Number)),
            name='P_balance'
        )
        M.addConstr(
            Bus_Q_injection[i, t] == quicksum(
                Y_qo[i, j] * Bus_Fai[j, t] for j in range(Sys_data.Bus_Number)) + quicksum(
                Y_qv[i, j] * Bus_V[j, t] for j in range(Sys_data.Bus_Number)),
            name='Q_balance'
        )

for t in range(N_time):
    for i_br in range(Sys_data.Branch_Number):
        node_head = int(Sys_data.branch[i_br, 0]) - 1
        node_tail = int(Sys_data.branch[i_br, 1]) - 1
        M.addConstr(P_ij[i_br, t] / Sys_data.baseMVA == Y_po[node_head, node_tail] * (
                Bus_Fai[node_head, t] - Bus_Fai[node_tail, t]) + Y_pv[node_head, node_tail] * (
                            Bus_V[node_head, t] - Bus_V[node_tail, t]))
        M.addConstr(Q_ij[i_br, t] / Sys_data.baseMVA == Y_qo[node_head, node_tail] * (
                Bus_Fai[node_head, t] - Bus_Fai[node_tail, t]) + Y_qv[node_head, node_tail] * (
                            Bus_V[node_head, t] - Bus_V[node_tail, t]))
        M.addConstr(P_ij[i_br, t] >= -Pij_limit[node_head, node_tail])
        M.addConstr(P_ij[i_br, t] <= Pij_limit[node_head, node_tail])
        M.addConstr(Q_ij[i_br, t] <= Qij_limit[node_head, node_tail])
        M.addConstr(Q_ij[i_br, t] >= -Qij_limit[node_head, node_tail])

M.addConstrs((Bus_Fai[0, i] == 0 for i in range(N_time)), name='Fai')
M.addConstrs((Bus_V[0, i] == 1 for i in range(N_time)), name='V')
M.addConstrs(
    (Bus_V[k_row, i] <= Bus_V_max for k_row in range(Sys_data.Bus_Number) for i in range(N_time)),
    name='V_ub')
M.addConstrs(
    (Bus_V_min <= Bus_V[k_row, i] for k_row in range(Sys_data.Bus_Number) for i in range(N_time)),
    name='V_lb')

DN_Buy_P_p = M.addVars(1, N_time, lb=[0] * N_time,
                       ub=[Sys_data.DN_Buy_P_limit] * N_time, name="DN_Buy_P_p")
M.addConstrs(DN_Buy_P_p[0, i] >= DN_Buy_P[0, i] for i in range(N_time))

M.update()

## Objective function
# Create new variables for each term in the expression
term1 = quicksum(Sys_data.a_con[0] * Gen_P[0, t] ** 2 + Sys_data.b_con[0] * Gen_P[0, t] + Sys_data.c_con[0] for t in
                 range(N_time))
term2 = quicksum(Sys_data.a_con[1] * DG1_P[0, t] ** 2 + Sys_data.b_con[1] * DG1_P[0, t] + Sys_data.c_con[1] for t in
                 range(N_time))
term3 = quicksum(Sys_data.E_Price24[t] * DN_Buy_P_p[0, t] for t in range(N_time))
term5 = 100 * quicksum(
    P_PV_24[i] * solar - PV_P_output[0, i] + P_Wind_24[i] * wind - WF_P_output[0, i] for i in
    range(N_time))
term6 = 5 * quicksum(ESS_P_input[0, t] + ESS_P_output[0, t] for t in range(N_time))
term4 = WT_cost * wind + Solar_cost * solar + RES_cost * res

# Add the terms together to create the objective function
obj = term1 + term2 + term3 + term5 + term6
overall_cost = obj * 365 / 4 + term4
# Pass the new variable as the objective function
M.setObjective(overall_cost, GRB.MINIMIZE)

# Solve the model
M.setParam('outPutFlag', 1)  # 不输出求解日志
# M.setParam('BarHomogeneous', 1)

M.optimize()

wind_capcity = wind.X
solar_capcity = solar.X
res_capcity = res.X
term4 = WT_cost * wind_capcity + Solar_cost * solar_capcity + RES_cost * res_capcity

if M.status == 2 or M.status == 13:
    # overall_cost =
    result = [M.status, M.ObjVal / 1e6]
    print(result)
else:
    result = [M.status, 0]

print(result)
