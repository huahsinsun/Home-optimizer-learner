import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

N_time = 24

# wind_base = np.array([30, 30, 50, 60])  # MW
wind_base = np.array([30.0, 30.0, 30.0, 30.0])
WF_num = wind_base.size
line_base = 20
overplant_rate = 0.5
# WT_cost, Solar_cost, RES_cost = 60000, 20000, 5000
# AC_wind_turbine_cost = 1.051 * (1.374 * PWF^0.87/(Pt/Prated)+0.363*Prated^1.06) * 1e6
P_WF_rated = 30
# AC_line_cost = 0.1437*( A_p + B_p * exp(C_p * S_n / 100) + 2.4)*1e6  Euro / km
ACline_33_Ap, AC_line_33_Bp, AC_line_33_Cp = 0.411, 0.596, 4.1
ACline_220_Ap, AC_line_220_Bp, AC_line_220_Cp = 3.181, 0.11, 1.16

# DC_line_cost = 0.1437(A_p + B_p * P_cap +2.4 )* 1e6  Euro / km
DCline_40_Ap, DCline_40_Bp = -0.314, 0.0618
DCline_230_Ap, DCline_230_Bp = 0.079, 0.0120

Euro2dollar = 1.11  # transfer euro_2021 to dollar_2024


def HV_cable_cost(cap, len):
    if cap != 0:
        cost = Euro2dollar * 1.1452 * (DCline_230_Ap + DCline_230_Bp * cap + 2.4) * len * 1e6
    else:
        cost = 0
    return cost
def OWF_cost(cap):
    WF_cost = 1.051 * (1374 * cap ** 0.87 + 363 * cap ** 1.06) * 1e3
    AC_platform_cost = 1.0519 * (0.0738 * cap + 53.25) * 1e6
    Coverter_cost = 0.1437 * cap * 1e6
    return WF_cost+AC_platform_cost+Coverter_cost

line_len = [200, 200, 200, 200, 200, 200, 200, 200]
interest_rate = 0.1
life_time = 25
CRF_25 = (interest_rate * (1 + interest_rate) ** life_time) / ((1 + interest_rate) ** life_time - 1)
CRF_15 = (interest_rate * (1 + interest_rate) ** 15) / ((1 + interest_rate) ** 15 - 1)
# AC_plat_form_cost = 1.0519 * (0.0738 * PWF+53.25) * 1e6  Euro / MW
# DC_plat_form_cost = 1.0519 * (0.125 * PWF+165) * 1e6  Euro / MW
LHV = 33 * 1e-3  # MWh/kg
eff_H = 0.5
H_price = 6  # EURO/kg
# Coverter_cost = 0.1437 * Pwf
BESS_capcity_cost = 460000  # MWh/EUro
BESS_power_cost = 1500  # MW/Euro


class Struct:
    pass


mpc = Struct()
mpc.baseMVA, baseMVA = 100, 100
Vbase = 220e3
mpc.bus = np.array([
    [1, 3, 0, 0, 0, 0, 1, 1.06, 0, 0, 1, 1.06, 0.94],
    [2, 2, 21.7, 12.7, 0, 0, 1, 1.045, -4.98, 0, 1, 1.06, 0.94],
    [3, 2, 94.2, 19, 0, 0, 1, 1.01, -12.72, 0, 1, 1.06, 0.94],
    [4, 1, 47.8, -3.9, 0, 0, 1, 1.019, -10.33, 0, 1, 1.06, 0.94],
    [5, 1, 7.6, 1.6, 0, 0, 1, 1.02, -8.78, 0, 1, 1.06, 0.94],
    [6, 2, 11.2, 7.5, 0, 0, 1, 1.07, -14.22, 0, 1, 1.06, 0.94],
    [7, 1, 0, 0, 0, 0, 1, 1.062, -13.37, 0, 1, 1.06, 0.94],
    [8, 2, 0, 0, 0, 0, 1, 1.09, -13.36, 0, 1, 1.06, 0.94],
    [9, 1, 29.5, 16.6, 0, 19, 1, 1.056, -14.94, 0, 1, 1.06, 0.94],
    [10, 1, 9, 5.8, 0, 0, 1, 1.051, -15.1, 0, 1, 1.06, 0.94],
    [11, 1, 3.5, 1.8, 0, 0, 1, 1.057, -14.79, 0, 1, 1.06, 0.94],
    [12, 1, 6.1, 1.6, 0, 0, 1, 1.055, -15.07, 0, 1, 1.06, 0.94],
    [13, 1, 13.5, 5.8, 0, 0, 1, 1.05, -15.16, 0, 1, 1.06, 0.94],
    [14, 1, 14.9, 5, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],
    [15, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # wf 1 with AC
    [16, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # wf 2 with AC
    # [17, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # LFAC bus connects bus 4
    # [18, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # LFAC bus connects bus 9
    # [19, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # DC bus connects bus 12
    # [20, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # DC bus connects bus 13
    # [21, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # LFAC bus connects bus 3 and bus 22(wf1)
    # [22, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # wf 3
    # [23, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # DC bus connects bus 4 and bus 24(wf2)
    # [24, 1, 0, 0, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],  # wf 4
])
AC_bus_number = mpc.bus.shape[0]

AC_WF_bus = {15, 16}

MMC_Number = 5
M3C_Number = 3
mpc.branch = np.array([
    [1, 2, 0.01938, 0.05917, 0.0528, 9900, 0, 0, 0, 0, 1, -360, 360],
    [1, 5, 0.05403, 0.22304, 0.0492, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 3, 0.04699, 0.19797, 0.0438, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 4, 0.05811, 0.17632, 0.034, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 5, 0.05695, 0.17388, 0.0346, 9900, 0, 0, 0, 0, 1, -360, 360],
    [3, 4, 0.06701, 0.17103, 0.0128, 9900, 0, 0, 0, 0, 1, -360, 360],
    [4, 5, 0.01335, 0.04211, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [4, 7, 0, 0.20912, 0, 9900, 0, 0, 0.978, 0, 1, -360, 360],
    # [4, 9, 0, 0.55618, 0, 9900, 0, 0, 0.969, 0, 1, -360, 360],  # replace with LFAC branch
    [5, 6, 0, 0.25202, 0, 9900, 0, 0, 0.932, 0, 1, -360, 360],
    [6, 11, 0.09498, 0.1989, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [6, 12, 0.12291, 0.25581, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [6, 13, 0.06615, 0.13027, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [7, 8, 0, 0.17615, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [7, 9, 0, 0.11001, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [9, 10, 0.03181, 0.0845, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [9, 14, 0.12711, 0.27038, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [10, 11, 0.08205, 0.19207, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [12, 13, 0.22092, 0.19988, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    # [13, 14, 0.17093, 0.34802, 0, 9900, 0, 0, 0, 0, 1, -360, 360],  # replace with DC branch
    [15, 2, 0.216, 0.01956, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 20 wf0 to Lianyungang 13km
    [16, 3, 0.7095, 0.0902, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 21 wf1 to Yancheng 42.7km
])
mpc.branch[18: 20, 2: 4] = mpc.branch[18: 20, 2: 4] / (
        Vbase ** 2 / (baseMVA * 1e6))

AC_branch_num = mpc.branch.shape[0]
# Create a mapping from (from_bus, to_bus) to branch index
bus_branch_mapping = {}
# mpc.branch[3] = mpc.branch[3] / 2
for idx, row in enumerate(mpc.branch):
    from_bus = row[0]
    to_bus = row[1]
    # Store the branch index for the (from_bus, to_bus) key
    bus_branch_mapping[(from_bus, to_bus)] = idx


# R 0.0114  X 0.9356*10-3
LFAC_bus = {17, 18, 19, 20}  # 20 is WF3
M3C_bus = {4, 9, 5}
LFAC_bus_num = len(LFAC_bus)
# 0              1          2(p.u)  3(p.u)    4             5           6            7           8            9
# LFAC_bus_from  LFAC_bus_to  R       X       M3C_idx_from  M3X_idx_to  AC_bus_from  AC_bus_to   WF_idx_from  WF_idx_to
mpc.LFAC_branch = np.array([
    [17, 18, 0, 0.18539, 1, 2, 4, 9, 0.0, 0.0],
    [19, 20, 0.798, 8.19, 3, 0.0, 5, 0.0, 4, 0.0],  # 70 km
])
LFAC_branch_num = mpc.LFAC_branch.shape[0]
mpc.LFAC_branch[1, [2, 3]] = mpc.LFAC_branch[1, [2, 3]] / (
        Vbase ** 2 / (baseMVA * 1e6))  # branch R, X are provided in p.u.

DC_bus = {21, 22, 23, 24, 25, 26, 27,28,29}  # 24--WF4
MMC_bus = {13, 14, 4}
DC_bus_num = len(DC_bus)
# 0            1          2(p.u)  3(p.u)  4             5           6            7           8            9
# DC_bus_from  DC_bus_to  R       X       MMC_idx_from  MMX_idx_to  AC_bus_from  AC_bus_to   WF_idx_from  WF_idx_to
mpc.DC_branch = np.array([
    [21, 22, 0.17093, 0, 1, 2, 13, 14, 0.0, 0.0],
    [23, 24, 218, 0, 3, 0.0, 4, 0.0, 3, 0.0],
    [24, 26, 218, 0, 0.0, 0.0, 0.0, 0.0, 4, 1],
    [25, 26, 218, 0, 0.0, 0.0, 0.0, 0.0, 1, 2],
    [25, 27, 218, 0, 0.0, 0.0, 0.0, 0.0, 2, 3],
    [24, 27, 218, 0, 0.0, 0.0, 0.0, 0.0, 3, 4],
    [25, 28, 218, 0, 0.0, 4, 0.0, 5, 1, 0.0],
    [26, 29, 218, 0, 0.0, 5, 0.0, 4, 2, 0.0],
])
mpc.DC_branch[1:, 2] = mpc.DC_branch[1:,2] / (
        Vbase ** 2 / (baseMVA * 1e6))  # branch R, X are provided in p.u.
DC_bus_branch_mapping = {}
DC_bus_MMC_mapping = {}
AC_bus_MMC_mapping = {}
DC_bus_AC_wind_mapping = {5: 1, 6: 2, 7: 3}
# mpc.branch[3] = mpc.branch[3] / 2
for idx, row in enumerate(mpc.DC_branch):
    from_bus = row[0] - AC_bus_number - LFAC_bus_num
    to_bus = row[1] - AC_bus_number - LFAC_bus_num
    MMC_from, MMC_to = row[4], row[5]
    if MMC_from != 0:
        DC_bus_MMC_mapping[from_bus] = MMC_from
        AC_bus_MMC_mapping[row[6]] = MMC_from
    if MMC_to !=0:
        DC_bus_MMC_mapping[to_bus] = MMC_to
        AC_bus_MMC_mapping[row[7]] = MMC_to
    # Store the branch index for the (from_bus, to_bus) key
    DC_bus_branch_mapping[(from_bus, to_bus)] = idx

DC_branch_num = mpc.DC_branch.shape[0]

mpc.gen = np.array([
    [1, 232.4, -16.9, 10, 0, 1.06, 100, 1, 332.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 40, 42.4, 50, -40, 1.045, 100, 1, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 23.4, 40, 0, 1.01, 100, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 12.2, 24, -6, 1.07, 100, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 17.4, 24, -6, 1.09, 100, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
mpc.gencost = np.array([
    [2, 0, 0, 3, 0.0430293, 20, 0],
    [2, 0, 0, 3, 0.25, 20, 0],
    [2, 0, 0, 3, 0.01, 40, 0],
    [2, 0, 0, 3, 0.01, 40, 0],
    [2, 0, 0, 3, 0.01, 40, 0]
])
P_gen_ub = mpc.gen[:, 8]
Q_gen_ub = mpc.gen[:, 3]
Q_gen_lb = mpc.gen[:, 4]
line_distance = [13, 42.7, 19.41, 13.20, 151.39, 145.96, 100.93]

BASE_V = 220e3
BR_R = 3 - 1  # minus 1 for the beginning index is 0 in python but 1 in MATLAB
BR_X = 4 - 1
PD = 3 - 1
QD = 4 - 1
# Vbase = mpc.bus[0, BASE_KV] * 1000
# Sbase = mpc.baseMVA * 1e6
# mpc.branch[:, [BR_R, BR_X]] = mpc.branch[:, [BR_R, BR_X]] / (Vbase ** 2 / Sbase) # branch R, X are provided in p.u.
# mpc.bus[:, [PD, QD]] = mpc.bus[:, [PD, QD]] / 1e3
bus = mpc.bus
branch = np.concatenate((mpc.branch[:, 0:4], mpc.LFAC_branch[:, 0:4], mpc.DC_branch[:, 0:4]), axis=0)
# branch = mpc.branch
Bus_Number = 16 + len(DC_bus) + len(LFAC_bus)
Branch_Number = branch.shape[0]
Gen_Number = mpc.gen.shape[0]

Z_ij = np.zeros(Branch_Number)
for i in range(Branch_Number):
    Z_ij[i] = np.sqrt(branch[i,2]**2 + branch[i,3]**2)
# adj_matrix = np.zeros((Bus_Number, Bus_Number))
# np.fill_diagonal(adj_matrix, 1)
def impedance_matrix_cal(branch_data, Bus_num):
    G_ij, B_ij = np.zeros((Bus_num, Bus_num)), np.zeros((Bus_num, Bus_num))
    for br in branch_data:
        bus_i = int(br[0]) - 1  # Convert to zero-indexed
        bus_j = int(br[1]) - 1  # Convert to zero-indexed
        # You can store impedances or admittances
        # adj_matrix[bus_i][bus_j] = 1  # Indicate connection
        # adj_matrix[bus_j][bus_i] = 1  # Assuming undirected connections
        R = br[BR_R]  # Branch resistance
        X = br[BR_X]  # Branch reactance

        Y = R / (R ** 2 + X ** 2)  # Conductance (positive value)
        B = -X / (R ** 2 + X ** 2)  # Susceptance (negative value)

        G_ij[bus_i][bus_j] = G_ij[bus_j][bus_i] = -Y  # Mutual conductance is negative
        B_ij[bus_i][bus_j] = B_ij[bus_j][bus_i] = -B

        G_ij[bus_i][bus_i] += Y  # Self-conductance remains positive
        G_ij[bus_j][bus_j] += Y
        B_ij[bus_i][bus_i] += B  # Self-susceptance
        B_ij[bus_j][bus_j] += B
    return G_ij, B_ij


G_ij, B_ij = impedance_matrix_cal(branch, Bus_Number)

# Initialize the reduced bus-branch incidence matrix Psi
Psi = np.zeros((Branch_Number, Bus_Number))

# Calculate the reduced bus-branch incidence matrix Psi
for idx, Branch in enumerate(mpc.branch):
    # Get the 'from' and 'to' bus indices, adjust for 0-based indexing
    fbus = int(Branch[0]) - 1
    tbus = int(Branch[1]) - 1

    # Set the corresponding entries in the Psi matrix
    Psi[idx, fbus] = 1
    Psi[idx, tbus] = -1

Psi_full = Psi.T
Psi = Psi_full[1:Bus_Number, :]
# Psi = csr_matrix(Psi)
M3C_capcity = [50, 50, 15]
MMC_capcity = [50, 50, 100, 100, 100]

ramp_limit = 0.6
Bus_V_min, Bus_V_max = 0.94, 1.06
Bus_V_min_DC, Bus_V_max_DC = 0.5, 1.5
# Pij_limit = np.zeros((Branch_Number, Branch_Number))
# branch_limit = [150, 60, 50, 50, 40, 30, 50, 20, 20, 20, 20, 40, 40, 60, 40, 20, 40, 40, 20, 20]
branch_limit = [120, 60, 50, 50, 40, 50, 50, 20, 20, 20, 40, 40, 60, 40, 20, 40, 40, 20, 5, 5]
branch_limit = [x / 5 / baseMVA for x in branch_limit]
# for i_br in range(Branch_Number):
#     node_head = int(branch[i_br, 0]) - 1
#     node_tail = int(branch[i_br, 1]) - 1
#     Pij_limit[node_head, node_tail], Pij_limit[node_tail, node_head] = branch_limit[i_br], branch_limit[i_br]

P_Wind_pu24 = pd.read_excel('Offshore_wind_farm.xlsx', header=None)
wind_pu = [P_Wind_pu24.iloc[:, i].values.reshape(365, 24) for i in range(P_Wind_pu24.shape[1])]
# wf_0, wf_1, wf_2, wf_3, wf_4 = wind_pu[0], wind_pu[1], wind_pu[2], wind_pu[4], wind_pu[3]

# Find the index of the column with the maximum standard deviation for each array
# contingency_indices = [
#     np.argmax(np.std(wf_0, axis=1)),
#     np.argmax(np.std(wf_1, axis=1)),
#     np.argmax(np.std(wf_2, axis=1)),
#     np.argmax(np.std(wf_3, axis=1)),
#     np.argmax(np.std(wf_4, axis=1))
# ]

P_PV_pu24 = [0.000246609751435038, 0.000248450946775893, 0.000248399498763841, 0.000249316722794134,
             0.000248670233384641, 0.00723408501397510, 0.0621928160401500, 0.234061219472568, 0.497097204199897,
             0.749499268962421, 0.912965571816002, 0.996509081013825, 0.999954564064567, 0.947315716395799,
             0.834299843330501, 0.659135371464226, 0.432249969429947, 0.196553782489059, 0.0524787324276500,
             0.00639183738678103, 0.000619545020385686, 0.000368541852373165, 0.000271747212638920,
             0.000268184305045309,
             0.000180599596881865, 0.000180691503655802, 0.000180361918810045, 0.000180383106898647,
             0.000180060022943712, 0.00349126984474899, 0.0249173845743947, 0.0821778885143541, 0.212785078897608,
             0.393107630032653, 0.526533209640061, 0.591911697966033, 0.596835187185100, 0.567933263641234,
             0.490691799777226, 0.353502023391330, 0.175926027565564, 0.0679888857102506, 0.0203255521278600,
             0.00308580543185550, 0.000429610578800545, 0.000304616368755321, 0.000260801018149626,
             0.000250357113469873,
             0.000248011128670850, 0.000247918638288448, 0.000247964858387304, 0.000248159574981835,
             0.000247702944493784, 0.000853786297844552, 0.00669775261828288, 0.0403112966776673, 0.227025973763246,
             0.518474532032379, 0.685410116100155, 0.762138626782030, 0.774046590379159, 0.733144512864345,
             0.641782959770999, 0.454371820721393, 0.184960897489237, 0.0453843223381480, 0.00627499644422406,
             0.000847355180093882, 0.000162706340638249, 0.000104559047411636, 9.15084687713078e-05,
             9.13432105894027e-05,
             0.000178112270181690, 0.000163221005393224, 0.000159108405579472, 0.000159447071331150,
             0.000160183411376603, 0.00504574881177423, 0.0514091425218062, 0.238441726755216, 0.522970769897627,
             0.758318647570983, 0.896614540050506, 0.952835823163475, 0.940910117634692, 0.890258090282033,
             0.795758039802389, 0.643851434137400, 0.418695635361550, 0.168560280638749, 0.0342357640114542,
             0.00302018763148175, 0.000508760143835456, 0.000285899210221312, 0.000196100797284490,
             0.000192067975119560]

city_load = pd.read_excel('City_load.xlsx', header=None)
anhui = np.array(city_load[0])
beijing = np.array(city_load[1])
fujian = np.array(city_load[2])
guangdong = np.array(city_load[3])
hebei = np.array(city_load[4])
henan = np.array(city_load[5])
hunan = np.array(city_load[6])
jiangsu = np.array(city_load[7])
jiangxi = np.array(city_load[8])
liaoning = np.array(city_load[9])
shandong = np.array(city_load[10])
shanghai = np.array(city_load[11])
shanxi = np.array(city_load[12])
zhejiang = np.array(city_load[13])

P_Eload_pu24 = np.empty([Bus_Number, N_time])
P_Eload_pu24[0] = np.tile(henan, int(N_time / 24))
P_Eload_pu24[1] = np.tile(shandong, int(N_time / 24))
P_Eload_pu24[2] = np.tile(jiangxi, int(N_time / 24))
P_Eload_pu24[3] = np.tile(liaoning, int(N_time / 24))
P_Eload_pu24[4] = np.tile(anhui, int(N_time / 24))
P_Eload_pu24[5] = np.tile(hunan, int(N_time / 24))
P_Eload_pu24[6] = np.tile(beijing, int(N_time / 24))
P_Eload_pu24[7] = np.tile(hebei, int(N_time / 24))
P_Eload_pu24[8] = np.tile(shanghai, int(N_time / 24))
P_Eload_pu24[9] = np.tile(guangdong, int(N_time / 24))
P_Eload_pu24[10] = np.tile(jiangsu, int(N_time / 24))
P_Eload_pu24[11] = np.tile(shanxi, int(N_time / 24))
P_Eload_pu24[12] = np.tile(fujian, int(N_time / 24))
P_Eload_pu24[13] = np.tile(zhejiang, int(N_time / 24))

# Pre-defined parameters
Bus_load_P_MW24 = np.zeros((Bus_Number, N_time))
Bus_load_Q_MW24 = np.zeros((Bus_Number, N_time))

for i in range(AC_bus_number):
    for t in range(N_time):
        Bus_load_P_MW24[i, t] = bus[i, 2] * 2 * P_Eload_pu24[i, t]
        Bus_load_Q_MW24[i, t] = bus[i, 3] * 1 * P_Eload_pu24[i, t]
    P_Wind_24 = 1 * P_Wind_pu24
    P_PV_24 = 1 * P_PV_pu24
total_load = np.sum(Bus_load_P_MW24)

# Parameters for reserve
SOC0 = 2.5
SOC_max = 5
ESS_P_max = 3  # Max charge/discharge power
Laimuta = 0.995
Yita = 0.95  # Efficiency of charge/discharge
cost_punish = 1000 * 2