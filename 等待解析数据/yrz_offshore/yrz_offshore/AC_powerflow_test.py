from vars_and_cons import *


# Objective: Minimize generation cost (quadratic)
def objective_function(model):
    return sum(
        mpc.gencost[i - 1, 3] * model.P_gen[i, t] ** 2 + mpc.gencost[i - 1, 4] * model.P_gen[i, t] + mpc.gencost[
            i - 1, 5] for i in model.gen for t in model.time)


model.obj = Objective(rule=objective_function, sense=minimize)

model.balance_bus_constraint = Constraint(model.time, rule=balance_bus)

model.ref_phase_angle = Constraint(model.time, rule=ref_phase_angle)

for i in model.gen:
    for t in model.time:
        gen_ramp(model, i, t)
        gen_capacity(model, i, t)

for i in model.WF:
    for t in model.time:
        WF_cons_rule(model, i, t)

for i in model.AC_bus:
    for j in model.AC_bus:
        for t in model.time:
            branch_constraints_rule(model, i, j, t)

for i in model.LFAC_bus:
    for j in model.LFAC_bus:
        for t in model.time:
            branch_constraints_rule_LFAC(model, i, j, t)

for i in model.DC_bus:
    for j in model.DC_bus:
        for t in model.time:
            branch_constraints_rule_DC(model, i, j, t)

for i in model.LFAC_bus:
    for t in model.time:
        LFAC_bus_power_balance(model, i, t)

for i in model.DC_bus:
    for t in model.time:
        DC_bus_power_balance(model, i, t)

for i in model.MMC:
    for t in model.time:
        MMC_constraint(model, i, t)

for i in model.M3C:
    for t in model.time:
        M3C_constraint(model, i, t)

for i in model.AC_bus:
    for t in model.time:
        power_flow(model, i, t)

# Solve using IPOPT
opt = SolverFactory('ipopt')
opt.options['tol'] = 1e-6  # Set solver tolerance
results = opt.solve(model, tee=True)

if results.solver.termination_condition == TerminationCondition.optimal:
    # print("Optimal solution found.")
    objective_value = model.obj()
    # print(f"Objective value: {objective_value}")
else:
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

    print(f"Model output has been saved to {output_file}.")
