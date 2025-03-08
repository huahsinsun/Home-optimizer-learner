from docplex.mp.model import Model

# 创建一个优化模型
mdl = Model(name='5k variables model')

# 创建5000个连续变量，范围在0到100之间
x = [mdl.continuous_var(lb=0, ub=100, name=f'x_{i}') for i in range(50000)]

# 添加一些约束条件作为示例
# 1. 所有变量的和不超过10000
mdl.add_constraint(mdl.sum(x) <= 10000, 'sum_constraint')

# 2. 相邻变量之间的关系约束(示例)
for i in range(10000):
    mdl.add_constraint(x[i] <= x[i+1], f'order_constraint_{i}')

# 设置目标函数 - 这里以最大化所有变量的和为例
objective = mdl.sum(x)
mdl.maximize(objective)

# 求解模型
solution = mdl.solve()

# 打印结果
if solution:
    print("找到最优解!")
    print(f"目标函数值: {solution.get_objective_value()}")
    # 打印前几个变量的值作为示例
    for i in range(5):
        print(f"x_{i} = {solution.get_value(x[i])}")
else:
    print("未找到可行解")
