'''review for most praiticipents'''
from docplex.mp.model import Model
mdl = Model(name='n')
state = mdl.binary_var_list(keys=2)
cost = [20,30]
max = [100,80]
min = [50,30]
st = [1000,500]
power = mdl.continuous_var_list(keys=2,lb=0)

mdl.minimize(mdl.sum(state[i]*st[i]+power[i]*cost[i] for i in range(2)))

mdl.add_constraint(mdl.sum(power)>=150)
for i in range(2):
    mdl.add_constraint(power[i]>=min[i])
    mdl.add_constraint(power[i] <= max[i])
mdl.solve()
mdl.print_solution()
