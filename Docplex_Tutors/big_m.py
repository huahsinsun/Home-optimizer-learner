from docplex.mp.model import Model
# 创建模型
mdl = Model(name='learn')
# 大M值
M = 1e6
# 变量
b = mdl.binary_var(name='b')
x = mdl.continuous_var(name='x')
y = mdl.continuous_var(lb=0, name='y')
# 约束
mdl.add_constraint(y <= 10 + M * (1 - b))
mdl.add_constraint(x >= 5 * b)
# 目标函数（示例）
mdl.maximize(x + y)
# 求解
solution = mdl.solve()
# 检查并输出结果
if solution:
    print(f"x = {solution[x]}")
    print(f"y = {solution[y]}")
    print(f"b = {solution[b]}")
else:
    print("No solution found")