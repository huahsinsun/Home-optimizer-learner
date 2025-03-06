'''
电力优化,IBM-Docplex-mp范式例子
时间2024/7/8
'''
# LP minx 3x+4y con:x+2y>=20 plus x,y >=0
from docplex.mp.model import Model

def linear_programming_example():
    mdl = Model(name='linear_programming')
    x = mdl.continuous_var(name='x',lb=0)
    y = mdl.continuous_var(name='y',lb=0)
    mdl.minimize(3*x+4*y)
    mdl.add_constraint(x+2*y>=20)
    solution = mdl.solve()
    if solution:
        print("Solution status:{}".format(solution.solve_status))
        print("x:{}".format(solution.get_value(x)))
        print("y:{}".format(solution.get_value(y)))
        print("Total cost = {}".format(solution.objective_value))
    else:
        print("no solution")
if __name__ == '__main__':
    linear_programming_example()

# 混合整数线性规划 (MILP)
# 求解线性目标函数在线性约束下，并且决策变量可以是连续的或离散（整数）的优化问题。
# min 5x+6y con x+y>=10 x be z ,y>=0
from docplex.mp.model import Model
def mix_integer_linear_programming_example():
    mdl = Model(name='MILP')
    x = mdl.integer_var(name='x')
    y = mdl.continuous_var(name='y',lb=0)
    mdl.minimize(5*x + 6*y)
    mdl.add_constraint(x+y>=10)
    solution = mdl.solve()
    if solution:
        print("Solution status:{}".format(solution.solve_status))
        print("x:{}".format(solution.get_value(x)))
        print("y:{}".format(solution.get_value(y)))
        print("Total cost = {}".format(solution.objective_value))
    else:
        print("no solution")
if __name__ == '__main__':
    mix_integer_linear_programming_example()
# 二次规划QP 求解二次目标函数在线性约束下的优化问题。
# minimize X2+Y2  CON x+y>=10, x,y>=0
from docplex.mp.model import Model
def quadratic_programming_example():
    mdl = Model(name='quadratic_programming')
    x = mdl.continuous_var(name='x',lb=0)
    y = mdl.continuous_var(name='y',lb=0)
    mdl.minimize(x**2+y**2)
    mdl.add_constraint(x+y>=10)
    solution = mdl.solve()
    if solution:
        print("Solution status:{}".format(solution.solve_status))
        print("x:{}".format(solution.get_value(x)))
        print("y:{}".format(solution.get_value(y)))
        print("Total cost = {}".format(solution.objective_value))
    else:
        print("no solution")
if __name__ == '__main__':
    quadratic_programming_example()

def qp():
    mdl = Model(name='qp')
    var = mdl.continuous_var_list(keys=2, lb=0)
    mdl.minimize(var[0] ** 2 + var[1] ** 2)
    mdl.add_constraint(mdl.sum(var) >= 10)
    mdl.solve()
    mdl.print_solution()
qp()





