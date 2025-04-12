from docplex.mp.model import Model

# 创建模型实例
m = Model(name="ref")

# 1. 单一二元变量定义
# 使用binary_var()定义单一二元变量
single_var = m.binary_var(name="单一二元变量")

# 2. 二进制变量列表定义
# 使用binary_var_list()定义3个二进制变量组成的列表
var_list = m.binary_var_list(keys=3, name="二进制变量列表")

# 3. 二进制变量词典定义
# 定义项目列表和对应成本
items = ['A', 'B', 'C']
costs = {'A': 10, 'B': 15, 'C': 12}

# 使用binary_var_dict()为每个项目创建二进制变量（词典形式）
var_dict = m.binary_var_dict(items, name='x')  # 生成变量：x_A, x_B, x_C

# 添加约束示例
# 约束：项目B的变量值 <= 0.5
m.add_constraint(var_dict['B'] <= 0.5)

# 目标函数示例
# 计算总成本：sum(成本 * 变量)
total_cost = m.sum(costs[item] * var_dict[item] for item in items)


