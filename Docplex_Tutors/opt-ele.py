'''教程：发电调度（Unit Commitment）
问题描述
发电调度问题涉及在给定的时间段内确定哪些发电机应该运行以及各发电机的发电量，以最小化总发电成本，同时满足电力需求和各种约束（如发电机的启动/停机时间、功率输出限制等）。

假设：

有两个发电机，每个发电机有固定的启动成本和单位发电成本。
每个发电机有其最大和最小发电能力。
目标是满足电力需求，以最小化总成本。
参数：

电力需求：150 MW
发电机1：启动成本 1000, 单位发电成本 20, 最小发电 50 MW, 最大发电 100 MW
发电机2：启动成本 500, 单位发电成本 30, 最小发电 30 MW, 最大发电 80 MW'''
from docplex.mp.model import Model
# def unit_commitment_example():
#     mdl = Model(name='unit_commitment')
#     demand = 150
#     start_cost = [1000,50]
#     generation_cost = [20,30]
#     min_power=[50,30]
#     max_power=[100,80]
#     x = mdl.binary_var_list(2,name='x',lb=0) # # 发电机启动状态
#     p = mdl.continuous_var_list(2,name='p',lb=0)
#     # 目标函数：最小化总成本
#     mdl.minimize(mdl.sum(start_cost[i]*x[i]+generation_cost[i]*p[i] for i in range(2)))
#     #约束1 电力需要
#     mdl.add_constraint(mdl.sum(p[i] for i in range(2))>= demand)
#     #约束2
#     for i in range(2):
#         mdl.add_constraint(p[i]>=min_power[i]*x[i])
#         mdl.add_constraint(p[i] <= max_power[i] * x[i])
#         solution = mdl.solve()
#         # 输出结果
#     if solution:
#         print('Solution status:', solution.solve_status)
#         for i in range(2):
#             print(f'Generator {i + 1} status: {solution.get_value(x[i])}')
#             print(f'Generator {i + 1} power: {solution.get_value(p[i])}')
#         print(f'Total cost = {solution.objective_value}')
#     else:
#         print('No solution found')
#
# if __name__ == "__main__":
#     unit_commitment_example()

#shorter
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


'''
经济调度问题
问题描述：
在给定的负荷需求下，确定各发电机的功率输出，以最小化总发电成本，同时满足发电机的功率输出限制和其他约束。
假设：
有三个发电机，每个发电机的发电成本为二次函数。
电力需求：150 MW
发电机参数：
发电机1：a1=10, b1=0.05, c1=0.001, 最大发电 100 MW
发电机2：a2=20, b2=0.04, c2=0.002, 最大发电 80 MW
发电机3：a3=30, b3=0.03, c3=0.003, 最大发电 70 MW
约束条件：
总发电量必须满足电力需求：P1 + P2 + P3 = 150 MW
每个发电机的发电量必须在其最大发电功率范围内：
0 <= P1 <= 100 MW
0 <= P2 <= 80 MW
0 <= P3 <= 70 MW
目标：
最小化总发电成本：
总成本 = (a1 + b1 * P1 + c1 * P1^2) + (a2 + b2 * P2 + c2 * P2^2) + (a3 + b3 * P3 + c3 * P3^2)'''

from docplex.mp.model import Model
def Economic_Dispatch():
    mdl = Model(name='Economic Dispatch')
    demand = 150
    x = mdl.continuous_var_list(3,name='x',lb=0)
    a = [10,20,30]
    b = [0.05,0.04,0.03]
    c = [0.001,0.002,0.003]
    cost = [100,80,70]
    mdl.minimize(mdl.sum(a[i]*x[i]**2+b[i]*x[i]+c[i] for i in range(3)))
    mdl.add_constraint(mdl.sum(x[i] for i in range(3))>=150 )
    for i in range(3):
        mdl.add_constraint(x[i]<=cost[i])
    solution = mdl.solve()
    if solution:
        print('Solution status:', solution.solve_status)
        for i in range(3):
            print(f'Generator {i + 1} power: {solution.get_value(x[i])}')
        print(f'Total cost = {solution.objective_value}')
    else:
        print('No solution found')

Economic_Dispatch()

'''
问题描述：电话生产 一家电话公司生产和销售两种电话，即桌面电话和移动电话。
公司的目标是利润最大化，每种电话至少要生产 100 部。
假设桌面电话和移动电话的利润分别为50和40，目标是最大化总利润。
约束条件：
组装时间：假设桌面电话需要2小时，移动电话需要1小时，总组装时间不超过400小时。
喷漆时间：假设桌面电话需要1小时，移动电话需要3小时，总喷漆时间不超过490小时。
1.DeskProduction 应大于或等于 100。
2. CellProduction 应大于或等于 100。
DeskProduction 的组装时间加上 CellProduction 的组装时间不应超过 400 小时。
4. DeskProduction 的喷漆时间加上 CellProduction 的喷漆时间不应超过 490 小时。
'''
'''软约束'''

from docplex.mp.model import Model
def telephone_optimize():
    mdl = Model(name='tele_make')
    num = mdl.integer_var_list(2,lb=100)
    fee = [12,20]
    t_set = [0.2,0.4]
    q_set =[0.5,0.4]
    '''软约束'''
    over_time = mdl.continuous_var(name='overtime', ub=40)

    mdl.maximize(mdl.sum(fee[i]*num[i] for i in range(2))-2*over_time)


    for i in range(2):
        mdl.add_constraint(num[i]>=100)
    mdl.add_constraint(mdl.sum(t_set[i]*num[i] for i in range(2))<=400+over_time)
    mdl.add_constraint(mdl.sum(q_set[i] * num[i] for i in range(2)) <= 490)


    solution = mdl.solve()
    if solution:
        print('Solution status:', solution.solve_status)
        for i in range(2):
            print("num{}:{}".format(i + 1, solution.get_value(num[i])))
        print(f'Total cost = {solution.objective_value}')
    else:
        print('No solution found')
    # mdl.print_information() # 打印有关模型的信息
    mdl.print_solution()
telephone_optimize()




