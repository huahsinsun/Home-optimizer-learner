from pyomo.environ import *

from sys_data import *

# Define model
model = ConcreteModel()
# indices set
model.AC_bus = RangeSet(1, AC_bus_number)
model.DC_bus = RangeSet(1, DC_bus_num)
model.LFAC_bus = RangeSet(1, LFAC_bus_num)
model.gen = RangeSet(1, Gen_Number)
model.time = RangeSet(1, N_time)
model.MMC = RangeSet(1, MMC_Number)
model.WF_AC = RangeSet(1, 3)
model.WF_DC = RangeSet(1, 1)
model.M3C = RangeSet(1, M3C_Number)
model.AC_branch = RangeSet(1, AC_branch_num)
model.DC_branch = RangeSet(1, DC_branch_num)
model.LFAC_branch = RangeSet(1, LFAC_branch_num)


# Define a function that returns the bounds based on the row index
def generic_bounds_rule(lb_list, ub_list):
    def bounds_rule(model, i, j):
        # Use the provided lower and upper bounds lists
        lb = lb_list[i - 1]  # Access the lower bound for row i (adjust for 0-indexing)
        ub = ub_list[i - 1]  # Access the upper bound for row i (adjust for 0-indexing)
        return (lb, ub)

    return bounds_rule


model.V = Var(model.AC_bus, model.time, bounds=(Bus_V_min, Bus_V_max), initialize=1.0)
model.theta = Var(model.AC_bus, model.time, bounds=(-np.pi, np.pi), initialize=0.0)
model.P_gen = Var(model.gen, model.time,
                  bounds=generic_bounds_rule([0] * Gen_Number, P_gen_ub))  # Active power generation
model.Q_gen = Var(model.gen, model.time, bounds=generic_bounds_rule(Q_gen_lb, Q_gen_ub))
model.I2_branch = Var(model.AC_branch, model.time, bounds=(-1000, 1000))
# model.Q_branch = Var(model.AC_branch, model.time, bounds=(-1000, 1000))
model.V_DC = Var(model.DC_bus, model.time, bounds=(0, Bus_V_max_DC), initialize=0)
model.P_branch_DC = Var(model.DC_branch, model.time, bounds=(-1000, 1000))
model.P_MMC = Var(model.MMC, model.time, bounds=(-100, 100))
model.Q_MMC = Var(model.MMC, model.time, bounds=(-100, 100))
model.P_M3C = Var(model.M3C, model.time, bounds=(-100, 100))
model.Q_M3C = Var(model.M3C, RangeSet(1, 2), model.time, bounds=(-100, 100))
model.theta_LFAC = Var(model.LFAC_bus, model.time, bounds=(-np.pi, np.pi), initialize=0.0)
model.V_LFAC = Var(model.LFAC_bus, model.time, bounds=(Bus_V_min, Bus_V_max), initialize=1.0)
model.P_wind_AC = Var(model.WF_AC, model.time, RangeSet(1, 2), bounds=(0, 1000))
model.P_wind_DC = Var(model.WF_DC, model.time, bounds=(0, 1000))
model.P_lost = Var(model.AC_bus, model.time, bounds=(0, 100), initialize=0)

model.gen_ramp = ConstraintList()


def gen_ramp(model, i, t):
    if t > 1:
        model.gen_ramp.add(model.P_gen[i, t] - model.P_gen[i, t - 1] <= ramp_limit * P_gen_ub[i - 1])
        model.gen_ramp.add(model.P_gen[i, t] - model.P_gen[i, t - 1] >= -ramp_limit * P_gen_ub[i - 1])


model.gen_capacity = ConstraintList()


def gen_capacity(model, i, t):
    model.gen_capacity.add((model.P_gen[i, t]) ** 2 + (model.Q_gen[i, t]) ** 2 <= (P_gen_ub[i - 1] * 0.75) ** 2 + (
            Q_gen_ub[i - 1] * 0.75) ** 2)


model.WF_constraints_AC = ConstraintList()


def WF_cons_rule_AC(model, i, t):
    model.WF_constraints_AC.add(
        model.P_wind_AC[i, t, 1] + model.P_wind_AC[i, t, 2] <= wind_base[i - 1] * wind_pu[i - 1][0][t - 1])


model.WF_constraints_DC = ConstraintList()


def WF_cons_rule_DC(model, i, t):
    model.WF_constraints_DC.add(model.P_wind_DC[i, t] <= wind_base[i + 3 - 1] * wind_pu[i + 3 - 1][0][t - 1])


# model.P_branch_constraints = ConstraintList()
# model.Q_branch_constraints = ConstraintList()
# model.branch_constraints = ConstraintList()


# model.branch_capacity = ConstraintList()


def branch_constraints_rule(model, i, j, t):
    if (i, j) in bus_branch_mapping:
        br = bus_branch_mapping[i, j]
        R = branch[br][2]
        model.add_component(f'P_branch_{br}_at_{t}',
                            Constraint(expr=model.I2_branch[br + 1, t] == ((
                                        model.V[i, t] * cos(model.theta[i,t]) - model.V[j, t] * cos(
                                    model.theta[j, t])) ** 2 + (
                                                        model.V[i, t] * sin(model.theta[i, t]) - model.V[j, t] * sin(
                                                    model.theta[j, t])) ** 2) / Z_ij[br]))
        # model.add_component(f'Q_branch_{br}_at_{t}',
        #                     Constraint(expr=model.Q_branch[br + 1, t] == ((model.V[i, t] * model.V[
        #                         j, t]) / (R ** 2 + X ** 2)) * (R * sin(model.theta[i, t] + model.theta[j, t]) - X *
        #                                                        cos((model.theta[i, t] + model.theta[j, t]))) - (
        #                                             model.V[i, t] ** 2 /
        #                                             (R ** 2 + X ** 2)) * (
        #                                             R * sin(2 * model.theta[i, t]) - X * cos(
        #                                         2 * model.theta[i, t]))))
        model.add_component(f'branch_capacity_con_{br}_{t}', Constraint(
            expr=(model.I2_branch[br + 1, t]) * R <= branch_limit[br] ** 2))


model.LFAC_bus_P_balance = ConstraintList()
model.LFAC_bus_Q_balance = ConstraintList()


def LFAC_bus_power_balance(model, i, t):
    injected_P_LFAC = model.V_LFAC[i, t] * sum(model.V_LFAC[j, t] * (
            cos(model.theta_LFAC[i, t] - model.theta_LFAC[j, t]) * G_ij[
        i + AC_bus_number - 1, j + AC_bus_number - 1] +
            sin(model.theta_LFAC[i, t] - model.theta_LFAC[j, t]) * B_ij[
                i + AC_bus_number - 1, j + AC_bus_number - 1]) for j in model.LFAC_bus)
    injected_Q_LFAC = model.V_LFAC[i, t] * sum(model.V_LFAC[j, t] * (
            sin(model.theta_LFAC[i, t] - model.theta_LFAC[j, t]) * G_ij[
        i + AC_bus_number - 1, j + AC_bus_number - 1] -
            cos(model.theta_LFAC[i, t] - model.theta_LFAC[j, t]) * B_ij[
                i + AC_bus_number - 1, j + AC_bus_number - 1]
    ) for j in model.LFAC_bus)
    if i <= 3:
        model.LFAC_bus_P_balance.add(
            -model.P_M3C[i, t] / baseMVA == injected_P_LFAC)
        model.LFAC_bus_Q_balance.add(
            -model.Q_M3C[i, 1, t] / baseMVA == injected_Q_LFAC)
    elif i == 4:
        model.LFAC_bus_P_balance.add(model.P_wind_AC[3, t, 1] / baseMVA == injected_P_LFAC)
        model.LFAC_bus_Q_balance.add(0 == injected_Q_LFAC)


model.MMC_capacity_cons = ConstraintList()


def MMC_constraint(model, i, t):
    model.MMC_capacity_cons.add(model.P_MMC[i, t] ** 2 + model.Q_MMC[i, t] ** 2 <= MMC_capcity[i - 1] ** 2)


model.P_balance = ConstraintList()
model.Q_balance = ConstraintList()


def power_flow(model, i, t):
    # Check if there is a generator at this bus
    if i in {1, 2, 3, 6, 8}:  # Buses with generators
        gen_idx = {1: 1, 2: 2, 3: 3, 6: 4, 8: 5}[i]  # Mapping of bus to generator index
        gen_P_output = model.P_gen[gen_idx, t]  # Generator term
        gen_Q_output = model.Q_gen[gen_idx, t]
    else:
        gen_P_output, gen_Q_output = 0, 0  # No generation at this bus
    # Define the power balance constraint
    injected_P = model.V[i, t] * sum(model.V[j, t] * (
            cos(model.theta[i, t] - model.theta[j, t]) * G_ij[i - 1, j - 1] +
            sin(model.theta[i, t] - model.theta[j, t]) * B_ij[i - 1, j - 1]
    ) for j in model.AC_bus)
    injected_Q = model.V[i, t] * sum(model.V[j, t] * (
            sin(model.theta[i, t] - model.theta[j, t]) * G_ij[i - 1, j - 1] -
            cos(model.theta[i, t] - model.theta[j, t]) * B_ij[i - 1, j - 1]
    ) for j in model.AC_bus)
    if i in M3C_bus:
        M3C_idx = {4: 1, 9: 2, 5: 3}[i]
        I_M3C_LFAC = (model.P_M3C[M3C_idx, t] / baseMVA) / (
                model.V_LFAC[M3C_idx, t] * cos(model.theta_LFAC[M3C_idx, t]))
        # I_M3C_out = (injected_P / baseMVA) / (model.V[i, t] * cos(model.theta[i, t]))
        Loss_M3C = 0.05 * ((2 * I_M3C_LFAC) ** 2)
        M3C_P = model.P_M3C[M3C_idx, t] / baseMVA - Loss_M3C
        M3C_Q = model.Q_M3C[M3C_idx, 2, t] / baseMVA
    else:
        M3C_P, M3C_Q = 0, 0

    if i in AC_bus_MMC_mapping:
        MMC_idx = AC_bus_MMC_mapping[i]
        Loss_MMC = model.P_MMC[MMC_idx, t] * 0.05 + 1
        MMC_P = (model.P_MMC[MMC_idx, t] - Loss_MMC) / baseMVA
        MMC_Q = model.Q_MMC[MMC_idx, t] / baseMVA
    else:
        MMC_P, MMC_Q = 0, 0
    #
    if i in AC_WF_bus:
        WF_idx = {15: 1, 16: 2}[i]
        WF_P = model.P_wind_AC[WF_idx, t, 1]
    else:
        WF_P = 0
    # + M3C_P + MMC_P + WF_P
    model.P_balance.add(
        (gen_P_output + WF_P) / baseMVA + M3C_P + MMC_P + model.P_lost[i, t] / baseMVA - Bus_load_P_MW24[
            i - 1, t - 1] / baseMVA == injected_P)
    model.Q_balance.add(
        (gen_Q_output) / baseMVA + M3C_Q + MMC_Q - Bus_load_Q_MW24[i - 1, t - 1] / baseMVA == injected_Q)


# Objective: Minimize generation cost (quadratic)
def objective_function(model):
    gencost = sum(
        mpc.gencost[i - 1, 3] * model.P_gen[i, t] ** 2 + mpc.gencost[i - 1, 4] * model.P_gen[i, t] + mpc.gencost[
            i - 1, 5] for i in model.gen for t in model.time)
    load_loss = 100000 * sum(model.P_lost[i, t] for i in model.AC_bus for t in model.time)
    return gencost + load_loss


def model_initialization():
    model.obj = Objective(rule=objective_function, sense=minimize)

    # for t in model.time:
    #     balance_bus(model, t)

    for i in model.gen:
        for t in model.time:
            gen_ramp(model, i, t)
            gen_capacity(model, i, t)

    for i in model.WF_AC:
        for t in model.time:
            WF_cons_rule_AC(model, i, t)

    for i in model.WF_DC:
        for t in model.time:
            WF_cons_rule_DC(model, i, t)

    for i in model.AC_bus:
        for j in model.AC_bus:
            for t in model.time:
                branch_constraints_rule(model, i, j, t)

    for i in model.LFAC_bus:
        for t in model.time:
            LFAC_bus_power_balance(model, i, t)

    for i in model.MMC:
        for t in model.time:
            MMC_constraint(model, i, t)

    for i in model.M3C:
        for t in model.time:
            constraint_name = f'M3C_capacity_cons_{i}_{t}'
            model.add_component(constraint_name,
                                Constraint(
                                    expr=model.P_M3C[i, t] ** 2 + model.Q_M3C[i, 1, t] ** 2 <= M3C_capcity[i - 1] ** 2))

    for i in model.AC_bus:
        for t in model.time:
            power_flow(model, i, t)

    return model  # Return the model
