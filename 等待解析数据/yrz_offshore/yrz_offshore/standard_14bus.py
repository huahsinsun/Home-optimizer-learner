import numpy as np
from pyomo.environ import *

from sys_data import P_Eload_pu24


class Struct:
    pass


baseMVA = 100

mpc = Struct()

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
    [14, 1, 14.9, 5, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94]
])
# Pre-defined parameters
Bus_load_P_MW24 = np.zeros((14, 24))
Bus_load_Q_MW24 = np.zeros((14, 24))
for i in range(14):
    for t in range(24):
        Bus_load_P_MW24[i, t] = mpc.bus[i, 2] * 1 * P_Eload_pu24[i, t]
        Bus_load_Q_MW24[i, t] = mpc.bus[i, 3] * 1 * P_Eload_pu24[i, t]
mpc.branch = np.array([
    [1, 2, 0.01938, 0.05917, 0.0528, 9900, 0, 0, 0, 0, 1, -360, 360],
    [1, 5, 0.05403, 0.22304, 0.0492, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 3, 0.04699, 0.19797, 0.0438, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 4, 0.05811, 0.17632, 0.034, 9900, 0, 0, 0, 0, 1, -360, 360],
    [2, 5, 0.05695, 0.17388, 0.0346, 9900, 0, 0, 0, 0, 1, -360, 360],
    [3, 4, 0.06701, 0.17103, 0.0128, 9900, 0, 0, 0, 0, 1, -360, 360],
    [4, 5, 0.01335, 0.04211, 0, 9900, 0, 0, 0, 0, 1, -360, 360],
    [4, 7, 0, 0.20912, 0, 9900, 0, 0, 0.978, 0, 1, -360, 360],
    [4, 9, 0, 0.55618, 0, 9900, 0, 0, 0.969, 0, 1, -360, 360],
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
    [13, 14, 0.17093, 0.34802, 0, 9900, 0, 0, 0, 0, 1, -360, 360]
])
# Create a mapping from (from_bus, to_bus) to branch index
bus_branch_mapping = {}

for idx, row in enumerate(mpc.branch):
    from_bus = int(row[0])
    to_bus = int(row[1])
    # Store the branch index for the (from_bus, to_bus) key
    bus_branch_mapping[(from_bus, to_bus)] = idx
G_ij, B_ij = np.zeros((14, 14)), np.zeros((14, 14))
for br in mpc.branch:
    bus_i = int(br[0]) - 1  # Convert to zero-indexed
    bus_j = int(br[1]) - 1  # Convert to zero-indexed
    # You can store impedances or admittances
    # adj_matrix[bus_i][bus_j] = 1  # Indicate connection
    # adj_matrix[bus_j][bus_i] = 1  # Assuming undirected connections
    R = br[2]  # Branch resistance
    X = br[3]  # Branch reactance

    Y = R / (R ** 2 + X ** 2)  # Conductance (positive value)
    B = -X / (R ** 2 + X ** 2)  # Susceptance (negative value)

    G_ij[bus_i][bus_j] = G_ij[bus_j][bus_i] = -Y  # Mutual conductance is negative
    B_ij[bus_i][bus_j] = B_ij[bus_j][bus_i] = -B

    G_ij[bus_i][bus_i] += Y  # Self-conductance remains positive
    G_ij[bus_j][bus_j] += Y
    B_ij[bus_i][bus_i] += B  # Self-susceptance
    B_ij[bus_j][bus_j] += B
mpc.gen = np.array([
        [1, 232.4, -16.9, 10,   0, 1.06,  100, 1, 332.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2,  40,    42.4, 50, -40, 1.045, 100, 1, 140,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3,   0,    23.4, 40,   0, 1.01,  100, 1, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6,   0,    12.2, 24,  -6, 1.07,  100, 1, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8,   0,    17.4, 24,  -6, 1.09,  100, 1, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
P_gen_ub = mpc.gen[:, 8]
Q_gen_ub = mpc.gen[:, 3]
Q_gen_lb = mpc.gen[:, 4]
mpc.gencost = np.array([
    [2, 0, 0, 3, 0.0430293, 20, 0],
    [2, 0, 0, 3, 0.25, 20, 0],
    [2, 0, 0, 3, 0.01, 40, 0],
    [2, 0, 0, 3, 0.01, 40, 0],
    [2, 0, 0, 3, 0.01, 40, 0]
])
branch_limit = [150, 60, 50, 50, 40, 40, 50, 20, 20, 20, 20, 40, 40, 60, 40, 20, 40, 40, 20, 20]
branch_limit = [x / baseMVA for x in branch_limit]

# Define model
model = ConcreteModel()
# indices set
model.AC_bus = RangeSet(1, 14)
model.gen = RangeSet(1, 5)
model.time = RangeSet(1, 1)
model.branch = RangeSet(1, 20)


# Define a function that returns the bounds based on the row index
def generic_bounds_rule(lb_list, ub_list):
    def bounds_rule(model, i, j):
        # Use the provided lower and upper bounds lists
        lb = lb_list[i - 1]  # Access the lower bound for row i (adjust for 0-indexing)
        ub = ub_list[i - 1]  # Access the upper bound for row i (adjust for 0-indexing)
        return (lb, ub)

    return bounds_rule


model.V = Var(model.AC_bus, model.time, bounds=(0.94, 1.06), initialize=1.0)
model.theta = Var(model.AC_bus, model.time, bounds=(-np.pi, np.pi), initialize=0.0)
model.P_gen = Var(model.gen, model.time,
                  bounds=generic_bounds_rule([0] * 5, P_gen_ub))  # Active power generation
model.Q_gen = Var(model.gen, model.time, bounds=generic_bounds_rule(Q_gen_lb, Q_gen_ub))
model.P_branch = Var(model.branch, model.time, bounds=(-1000, 1000))
model.Q_branch = Var(model.branch, model.time, bounds=(-1000, 1000))
model.P_lost = Var(model.AC_bus, model.time, bounds=(0, 0), initialize=0)
model.balance_bus = ConstraintList()


# set bus one as reference bus
def balance_bus(model, t):
    # model.balance_bus.add(model.V[1, t] == 1)
    model.balance_bus.add(model.theta[1, t] == 0)


model.gen_ramp = ConstraintList()


def gen_ramp(model, i, t):
    if t > 1:
        model.gen_ramp.add(model.P_gen[i, t] - model.P_gen[i, t - 1] <= 0.6 * P_gen_ub[i - 1])
        model.gen_ramp.add(model.P_gen[i, t] - model.P_gen[i, t - 1] >= -0.6 * P_gen_ub[i - 1])


model.gen_capacity = ConstraintList()


def gen_capacity(model, i, t):
    model.gen_capacity.add((model.P_gen[i, t]) ** 2 + (model.Q_gen[i, t]) ** 2 <= (P_gen_ub[i - 1] * 0.75) ** 2 + (
            Q_gen_ub[i - 1] * 0.75) ** 2)


model.P_branch_constraints = ConstraintList()
model.Q_branch_constraints = ConstraintList()
model.branch_constraints = ConstraintList()


# model.branch_capacity = ConstraintList()


def branch_constraints_rule(model, i, j, t):
    if (i, j) in bus_branch_mapping:
        constraint_name = f'AC_branch_from_{i}_to_{j}_at_{t}'
        br = bus_branch_mapping[i, j]
        R, X = mpc.branch[br][2], mpc.branch[br][3]
        if constraint_name not in model.component_map(Constraint):
            model.add_component(f'P_branch_{br}_at_{t}',Constraint(expr=model.I2_branch[br + 1, t] == (model.V[i, t] * model.V[j, t] * R * cos(
                model.theta[i, t] + model.theta[j, t]) + model.V[i, t] * model.V[j, t] * X * sin(
                model.theta[i, t] + model.theta[j, t]) - (model.V[i, t] ** 2 * R * cos(2 * model.theta[i, t]) + model.V[
                i, t] ** 2 * X * sin(2 * model.theta[i, t]))) / (R ** 2 + X ** 2)))
            model.add_component(f'Q_branch_{br}_at_{t}',Constraint(expr=model.Q_branch[br + 1, t] == ((model.V[i, t] * model.V[
                j, t]) / (R ** 2 + X ** 2)) * (R * sin(model.theta[i, t] + model.theta[j, t]) - X * cos(
                (model.theta[i, t] + model.theta[j, t]))) - (model.V[i, t] ** 2 / (R ** 2 + X ** 2)) * (R * sin(
                2 * model.theta[i, t]) - X * cos(2 * model.theta[i, t]))))
            model.add_component(constraint_name, Constraint(
                expr=(model.I2_branch[br + 1, t]) ** 2 + (model.Q_branch[br + 1, t]) ** 2 <= branch_limit[br] ** 2))
    # if i != j:
    #     constraint_name = f'AC_branch_from_{i}_to_{j}_at_{t}'
    #     if (i, j) in bus_branch_mapping:
    #         idx = bus_branch_mapping[i, j]
    #         if constraint_name not in model.component_map(Constraint):
    #             model.add_component(constraint_name, Constraint(
    #                 expr=(model.P_branch[i, j, t]) ** 2 + (model.Q_branch[i, j, t]) ** 2 <= (1000) ** 2))
    #     elif (j, i) in bus_branch_mapping:
    #         idx = bus_branch_mapping[j, i]
    #         if constraint_name not in model.component_map(Constraint):
    #             model.add_component(constraint_name, Constraint(
    #                 expr=(model.P_branch[i, j, t]) ** 2 + (model.Q_branch[i, j, t]) ** 2 <= (1000) ** 2))
    #     else:
    #         if constraint_name not in model.component_map(Constraint):
    #             model.add_component(constraint_name, Constraint(
    #                 expr=(model.P_branch[i, j, t]) ** 2 + (model.Q_branch[i, j, t]) ** 2 <= 0))


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
    ) for j in range(1, 14 + 1))
    injected_Q = model.V[i, t] * sum(model.V[j, t] * (
            sin(model.theta[i, t] - model.theta[j, t]) * G_ij[i - 1, j - 1] -
            cos(model.theta[i, t] - model.theta[j, t]) * B_ij[i - 1, j - 1]
    ) for j in range(1, 14 + 1))

    model.P_balance.add(
        (gen_P_output) / baseMVA + model.P_lost[i, t] / baseMVA - Bus_load_P_MW24[
            i - 1, t - 1] / baseMVA == injected_P)
    model.Q_balance.add(
        (gen_Q_output) / baseMVA - Bus_load_Q_MW24[i - 1, t - 1] / baseMVA == injected_Q)


# Objective: Minimize generation cost (quadratic)
def objective_function(model):
    gencost = sum(
        mpc.gencost[i - 1, 3] * model.P_gen[i, t] ** 2 + mpc.gencost[i - 1, 4] * model.P_gen[i, t] + mpc.gencost[
            i - 1, 5] for i in model.gen for t in model.time)
    load_loss = 100000 * sum(model.P_lost[i, t] for i in model.AC_bus for t in model.time)
    return gencost + load_loss


model.obj = Objective(rule=objective_function, sense=minimize)

# for t in model.time:
#     balance_bus(model, t)

for i in model.gen:
    for t in model.time:
        gen_ramp(model, i, t)
        gen_capacity(model, i, t)

for i in model.AC_bus:
    for j in model.AC_bus:
        for t in model.time:
            branch_constraints_rule(model, i, j, t)

for i in model.AC_bus:
    for t in model.time:
        power_flow(model, i, t)

# model.test = Var(RangeSet(1,1),bounds=(0,100))
# model.add_component('test_cons',Constraint(expr= model.test[1] == cos(np.pi)))

opt = SolverFactory('ipopt')
opt.options['tol'] = 1e-6  # Set solver tolerance
results = opt.solve(model, tee=True)
import sys

# Define the output file
output_file = 'standard_14bus_output.txt'

# Open the file in write mode
with open(output_file, 'w') as f:
    # Redirect standard output to the file
    original_stdout = sys.stdout  # Save the original stdout
    sys.stdout = f  # Change stdout to the file

    # Display the model
    model.display()

    # Restore standard output to original
    sys.stdout = original_stdout
injected_Q = np.zeros([14, 24])
# if results.solver.termination_condition == TerminationCondition.optimal:
#     for i in model.AC_bus:
#         for t in model.time:
#             injected_Q[i - 1, t - 1] = sum(
#                 model.Q_branch[i, j, t].value
#                 for j in model.AC_bus
#             )
print()
