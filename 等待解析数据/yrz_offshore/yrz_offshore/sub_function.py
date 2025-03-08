import numpy
import torch
from pyomo.environ import *
from sys_data import *
import random
import concurrent.futures

# Connectivity definition
connections = {
    1: (4, 6),  # branch 1 connects bus 4 and bus 6
    2: (5, 6),  # branch 2 connects bus 5 and bus 6
    3: (5, 7),  # branch 3 connects bus 5 and bus 7
    4: (4, 7),  # branch 4 connects bus 7 and bus 4
    5: (5, 8),
    6: (6, 9),
}
Sub_SMOKE_test = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"dtype": dtype, "device": device}
random_scenario = False


# Define model


def sub_objective(x, discrete_idx, random_flag, full_output_flag, scenario_idx=0):
    wind_capacity = wind_base
    wind_capacity[-1] += x[-1] * 100
    wind_capacity[-2] += x[-2] * 100

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

    def WF_cons_rule_AC(model, i, t, sceario_number, full_output_flag):
        if not full_output_flag:
            model.WF_constraints_AC.add(
                model.P_wind_AC[i, t, 1] + model.P_wind_AC[i, t, 2] <= wind_capacity[i - 1] *
                wind_pu[i - 1][sceario_number][t - 1])
        else:
            model.WF_constraints_AC.add(
                model.P_wind_AC[i, t, 1] + model.P_wind_AC[i, t, 2] <= wind_capacity[i - 1] * 1)

    model.WF_constraints_DC = ConstraintList()

    def WF_cons_rule_DC(model, i, t):
        model.WF_constraints_DC.add(model.P_wind_DC[i, t] <= wind_capacity[i + 3 - 1] * wind_pu[i + 3 - 1][0][t - 1])

    def branch_constraints_rule(model, i, j, t):
        if (i, j) in bus_branch_mapping:
            br = bus_branch_mapping[i, j]
            R = branch[br][2]
            model.add_component(f'P_branch_{br}_at_{t}',
                                Constraint(expr=model.I2_branch[br + 1, t] == ((
                                                                                       model.V[i, t] * cos(
                                                                                   model.theta[i, t]) - model.V[
                                                                                           j, t] * cos(
                                                                                   model.theta[j, t])) ** 2 + (
                                                                                       model.V[i, t] * sin(
                                                                                   model.theta[i, t]) - model.V[
                                                                                           j, t] * sin(
                                                                                   model.theta[j, t])) ** 2) / Z_ij[
                                                    br] ** 2))
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
        load_loss = cost_punish * sum(model.P_lost[i, t] for i in model.AC_bus for t in model.time)
        return gencost + load_loss

    model.obj = Objective(rule=objective_function, sense=minimize)
    if random_flag:
        scenario_index = random.randint(1, 356) - 1
    else:
        scenario_index = scenario_idx

    # for t in model.time:
    #     balance_bus(model, t)

    for i in model.gen:
        for t in model.time:
            gen_ramp(model, i, t)
            gen_capacity(model, i, t)

    for i in model.WF_AC:
        for t in model.time:
            WF_cons_rule_AC(model, i, t, scenario_index, full_output_flag)

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

    # add the variable cons
    line_flag = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    connected_buses = {1, 2, 3, 4}

    discrete_number = len(discrete_idx)
    x = x.squeeze(0)
    capacity = numpy.zeros((max(x.shape) - discrete_number))
    for i in range(discrete_number):
        if x[i] == 1:
            capacity[i] = x[i + discrete_number] * 100
            # capacity[i + 3] = x[i + discrete_number] * 100
            if capacity[i] > 0:
                line_flag[i + 2] = 1
            # if capacity[i + 3] > 0:
            #     line_flag[i + 2 + 3] = 1

    DC_branch = mpc.DC_branch[line_flag == 1]
    DC_branch[:, [0, 1]] = DC_branch[:, [0, 1]] - AC_bus_number - LFAC_bus_num
    G_ij_DC, _ = impedance_matrix_cal(DC_branch, DC_bus_num)

    queue = [4]  # Start from bus 4
    # Perform BFS to find all connected buses
    while queue:
        current_bus = queue.pop(0)
        for br, (bus_a, bus_b) in connections.items():
            if capacity[br - 1] > 0:
                if current_bus == bus_a and bus_b not in connected_buses:
                    connected_buses.add(bus_b)
                    queue.append(bus_b)
                elif current_bus == bus_b and bus_a not in connected_buses:
                    connected_buses.add(bus_a)
                    queue.append(bus_a)

    def DC_branch_power_flow(model, from_bus, to_bus, t):
        if (from_bus, to_bus) in DC_bus_branch_mapping:
            DC_br_idx = DC_bus_branch_mapping[from_bus, to_bus]
            R = branch[DC_br_idx + AC_branch_num + LFAC_branch_num][2]
            constraint_name_p = f'Offshore_cable_capacity_cons_p_{DC_br_idx}_{t}'
            if hasattr(model, constraint_name_p):
                model.del_component(constraint_name_p)
            constraint_name_n = f'Offshore_cable_capacity_cons_n_{DC_br_idx}_{t}'
            if hasattr(model, constraint_name_n):
                model.del_component(constraint_name_n)
            constraint_name = f'DC_branch_power_flow_{DC_br_idx}_{t}'
            if hasattr(model, constraint_name):
                model.del_component(constraint_name)

            if DC_br_idx <= 1:
                model.add_component(
                    constraint_name,
                    Constraint(
                        expr=(model.P_branch_DC[DC_br_idx + 1, t]) == (
                                model.V_DC[from_bus, t] - model.V_DC[to_bus, t]) / R))
            elif capacity[DC_br_idx - 2] != 0:
                model.add_component(
                    constraint_name,
                    Constraint(
                        expr=(model.P_branch_DC[DC_br_idx + 1, t]) == (
                                model.V_DC[from_bus, t] - model.V_DC[to_bus, t]) / R))
                model.add_component(
                    constraint_name_p,
                    Constraint(
                        expr=(model.P_branch_DC[DC_br_idx + 1, t]) <= (capacity[DC_br_idx - 2] / baseMVA))
                )
                model.add_component(
                    constraint_name_n,
                    Constraint(
                        expr=(model.P_branch_DC[DC_br_idx + 1, t]) >= - (capacity[DC_br_idx - 2] / baseMVA))
                )

    for i in model.DC_bus:
        for j in model.DC_bus:
            for t in model.time:
                DC_branch_power_flow(model, i, j, t)

    def DC_bus_power_balance(model, i, t):

        constraint_name_V = f'DC_bus_voltage_constraint_{i}_{t}'
        if hasattr(model, constraint_name_V):
            model.del_component(constraint_name_V)
        constraint_name_P = f'DC_bus_power_balance_{i}_{t}'
        if hasattr(model, constraint_name_P):
            model.del_component(constraint_name_P)
        constraint_name_MMC_P = f'MMC_P_cons_{i}_{t}'
        if hasattr(model, constraint_name_MMC_P):
            model.del_component(constraint_name_MMC_P)
        constraint_name_MMC_Q = f'MMC_Q_cons_{i}_{t}'
        if hasattr(model, constraint_name_MMC_Q):
            model.del_component(constraint_name_MMC_Q)

        if i in connected_buses:
            model.add_component(
                constraint_name_V,
                Constraint(expr=model.V_DC[i, t] >= Bus_V_min_DC)
            )
            injected_P_DC = 0
            for j in connected_buses:
                injected_P_DC += model.V_DC[i, t] * model.V_DC[j, t] * G_ij_DC[i - 1, j - 1]

            if i in DC_bus_MMC_mapping:
                MMC_idx = DC_bus_MMC_mapping[i]
                MMC_P = -model.P_MMC[MMC_idx, t]
            else:
                MMC_P = 0

            if i == 4:
                P_wind = model.P_wind_DC[1, t]
            elif i in DC_bus_AC_wind_mapping:
                wf_ac_idx = DC_bus_AC_wind_mapping[i]
                P_wind = model.P_wind_AC[wf_ac_idx, t, 2]
            else:
                P_wind = 0

            model.add_component(
                constraint_name_P,
                Constraint(expr=(MMC_P + P_wind) / baseMVA == injected_P_DC)
            )
        else:
            if i in DC_bus_MMC_mapping:
                MMC_idx = DC_bus_MMC_mapping[i]
                model.add_component(
                    constraint_name_MMC_P,
                    Constraint(expr=model.P_MMC[MMC_idx, t] == 0)
                )
                model.add_component(
                    constraint_name_MMC_Q,
                    Constraint(expr=model.Q_MMC[MMC_idx, t] == 0)
                )

    # Check connectivity and add constraints for each bus
    for i in model.DC_bus:
        for t in model.time:
            DC_bus_power_balance(model, i, t)

    opt = SolverFactory('ipopt')
    opt.options['tol'] = 1e-6  # Set solver tolerance

    if Sub_SMOKE_test:
        results = opt.solve(model, tee=True)
        import sys
        # Define the output file
        output_file = 'model_output.txt'

        # Open the file in write mode
        with open(output_file, 'w') as f:
            # Redirect standard output to the file
            original_stdout = sys.stdout  # Save the original stdout
            sys.stdout = f  # Change stdout to the file

            # Display the model
            model.display()

            # Restore standard output to original
            sys.stdout = original_stdout
    else:
        results = opt.solve(model, tee=False)

    if results.solver.termination_condition == TerminationCondition.optimal:
        P_wind = sum(
            value(model.P_wind_AC[i, t, 1]) + value(model.P_wind_AC[i, t, 2]) + value(model.P_wind_DC[1, t]) for i in
            model.WF_AC for t in model.time)
        P_wind = torch.tensor(365 / (N_time / 24) * P_wind, **tkwargs).view(1, 1)
        cab_cost = 0  # in dollar_2024
        for i in range(capacity.size):
            cab_cost += HV_cable_cost(capacity[i], line_len[i]) * x[i]
        wind_cost = 0
        for i in range(1, 3):
            wind_cost += OWF_cost(wind_capacity[-i] - wind_base[-i])
        annual_investment = (cab_cost + wind_cost) * CRF_25
        gen_cost = 365 / (N_time / 24) * model.obj()

        annual_cost = (annual_investment + gen_cost).clone().detach().to(**tkwargs).view(1, 1)

        return -annual_cost, P_wind

    else:
        opt.options['tol'] = 1e-5  # Set a larger solver tolerance
        times = 1
        while results.solver.termination_condition != TerminationCondition.optimal:
            print(f'The {times} time resolve')
            results = opt.solve(model, tee=True if Sub_SMOKE_test else False)
            # import sys
            #
            # # Define the output file
            # output_file = 'model_output.txt'
            #
            # # Open the file in write mode
            # with open(output_file, 'w') as f:
            #     # Redirect standard output to the file
            #     original_stdout = sys.stdout  # Save the original stdout
            #     sys.stdout = f  # Change stdout to the file
            #
            #     # Display the model
            #     model.display()
            #
            #     # Restore standard output to original
            #     sys.stdout = original_stdout
            times += 1

        # if results.solver.termination_condition != TerminationCondition.optimal:
        #     raise ValueError("Optimization failed: Solver did not find an optimal solution. "
        #                      f"Termination condition: {results.solver.termination_condition}")
        # else:

        P_wind = sum(
            value(model.P_wind_AC[i, t, 1]) + value(model.P_wind_AC[i, t, 2]) + value(model.P_wind_DC[1, t]) for i in
            model.WF_AC for t in model.time)
        P_wind = torch.tensor(365 / (N_time / 24) * P_wind, **tkwargs).view(1, 1)
        cab_cost = 0  # in dollar_2024
        for i in range(capacity.size):
            cab_cost += HV_cable_cost(capacity[i], line_len[i]) * x[i]
        wind_cost = 0
        for i in range(1, 3):
            wind_cost += OWF_cost(wind_capacity[-i] - wind_base[-i])
        annual_investment = (cab_cost + wind_cost) * CRF_25
        gen_cost = 365 / (N_time / 24) * model.obj()

        annual_cost = (annual_investment + gen_cost).clone().detach().to(**tkwargs).view(1, 1)
        return -annual_cost, P_wind


def objective(x, discrete_idx, random_flag):
    # Use ProcessPoolExecutor for parallel CPU-bound tasks
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future_neg_annual_cost = executor.submit(sub_objective, x, discrete_idx, random_flag, 0)
    #     future_res_accombaility = executor.submit(sub_objective, x, discrete_idx, random_flag, 1)
    #
    #     # Get results from the futures
    #     neg_annual_cost, _ = future_neg_annual_cost.result()  # Block until result is available
    #     _, RES_accombaility = future_res_accombaility.result()  # Block until result is available
    neg_annual_cost, _ = sub_objective(x, discrete_idx, random_flag, 0)
    _, RES_accombaility = sub_objective(x, discrete_idx, random_flag, 1)

    return neg_annual_cost, RES_accombaility


if __name__ == "__main__":
    Sub_SMOKE_test = True
    # discrete_idx = [0, 1, 2, 3,4 ,5]
    # x = torch.tensor([[0.0000],
    #     [0.0000],
    #     [1.0000],
    #     [1.0000],
    #     [1.0000],
    #     [0.0000],
    #     [0.5381],
    #     [0.0769],
    #     [0.5359],
    #     [0.6160],
    #     [0.7378],
    #     [0.5058]], dtype=torch.float64)
    # obj = objective(x, discrete_idx)
    obj1, obj2 = objective(x=torch.tensor([0] * 14, **tkwargs), discrete_idx=[0, 1, 2, 3, 4, 5], random_flag=0)
    print(f'{obj1}, {obj2}')
