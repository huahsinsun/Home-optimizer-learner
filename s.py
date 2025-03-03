 # 软约束：数学表达与实际建模
from docplex.mp.model import Model
# 1.违反变量法
mdl = Model(name="软约束示例")
# 定义决策变量
x = mdl.continuous_var(name="x",lb=0)
y = mdl.continuous_var(name="y",lb=0)
# 定义违反变量
v = mdl.continuous_var(name="v",lb=0)
# 添加软约束
mdl.add_constraint(x+y<=10+v)
penalty = 1000
mdl.maximize(x+y-penalty*v)
