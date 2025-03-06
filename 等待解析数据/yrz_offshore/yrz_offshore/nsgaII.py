import os
import torch
import numpy as np
import time
import pandas as pd
from datetime import datetime
# from sub_function import objective as obj
from platypus import NSGAII, Problem, Real
# from Sub_function import num_obj, Outcome_analysis
# from Op_sim import initilize_QP_model
# from main import exam_model, exam_times, NOISE_SE
# from main_mulit_obj import dim, integer_indices
from tqdm import trange
from algorithm_parameter import *
# from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

# tkwargs = {
#     "dtype": torch.double,
#     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# }
exam_model = 2
exam_times = 200
NOISE_SE = torch.tensor([1e-6, 1e6, 1e-6], **tkwargs)

pop_size = 12
Iteration = 150
ref_point = 7
num_obj =2
volume = np.empty([Iteration, ref_point])
t_nsaga = 0
# dim = 7
# P_model = initilize_QP_model()
bounds = torch.tensor([[0] * int(dim), [1] * int(dim)], **tkwargs)


# worst_obj23 = obj([1] * dim, P_model)
# worst_obj1 = obj([0] * dim, P_model)


class Cost(Problem):

    def __init__(self):
        super().__init__(int(dim), num_obj, )  # nvars nobjs
        self.types[:] = Real(0, 1)
        self.directions[:] = Problem.MAXIMIZE

    def evaluate(self, solution):
        x = solution.variables[:]
        x = torch.Tensor(x)
        x[0:6] = torch.round(x[0:6])
        solution.objectives[:] = base_function(x, integer_indices)


algorithm = NSGAII(Cost(), population_size=pop_size)

#
algorithm.run(1)
for solution in algorithm.result:
    decision_vars = torch.tensor(solution.variables, **tkwargs).resize(1, dim)
    obj_values = torch.tensor(solution.objectives, **tkwargs).resize(1, num_obj)

for iteration in trange(Iteration, desc='Optimizing', unit='Iteration'):
    # Run one iteration
    t0 = time.monotonic()
    algorithm.run(1)
    t1 = time.monotonic()
    t_nsaga += t1 - t0

    # Extract and analyze the results
    for solution in algorithm.result:
        decision_var = solution.variables
        obj_value = solution.objectives
        obj_values = torch.cat([obj_values, torch.tensor(obj_value, **tkwargs).resize(1, num_obj)])
        decision_vars = torch.cat([decision_vars, torch.tensor(decision_var, **tkwargs).resize(1, dim)])
    # print(volumn[iteration])

# normalized_obj = (obj_values - torch.mean(obj_values, dim=0)) / torch.std(obj_values, dim=0)
# for j in range(0, ref_point):
#     hvs_list = []
#     for i in range(int(normalized_obj.shape[0] / pop_size)):
#         # Extract the true objective values for the current iteration
#         train_obj_it = normalized_obj[0:(i + 1) * pop_size, :]
#         # Compute the hypervolume
#         bd = DominatedPartitioning(ref_point=torch.tensor([-j] * 3, **tkwargs), Y=train_obj_it)
#         volume1 = bd.compute_hypervolume().item()
#         # Append the computed hypervolume to the list
#         hvs_list.append(volume1)
#     volume[:, j] = hvs_list

# hvs_qparego, non_don_qpar_x, non_don_qpar_y, exam_value_qpar, post_mean_qpar = Outcome_analysis(
    # torch.tensor([-6, - 6, - 6], **tkwargs),
    # decision_vars,
    # obj_values,
    # exam_model,
    # exam_times,
    # P_model, NOISE_SE, bounds)
folder_name = "NSGAII"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

decision_vars_save = pd.DataFrame(decision_vars.numpy(),
                                  columns=[f'x_NSGAII_{i}' for i in range(decision_vars.shape[1])])
obj_values_save = pd.DataFrame(obj_values.numpy(), columns=[f'y_NSGAII_{i}' for i in range(obj_values.shape[1])])
# hvs = pd.DataFrame(volume, columns=['hypervolume0', 'hypervolume1', 'hypervolume2', 'hypervolume3', 'hypervolume4',
#                                     'hypervolume5', 'hypervolume6'])
# exam_value_qpar_save = pd.DataFrame(exam_value_qpar, columns=['exam_par'])

t_nsaga = [t_nsaga]
t_nsaga = pd.DataFrame(t_nsaga)
result = pd.concat(
    [decision_vars_save, obj_values_save, t_nsaga], axis=1)

current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
file_name = f'{current_time}_NSGAII.xlsx'

file_path = os.path.join(folder_name, file_name)
result.to_excel(file_path, index=False)
# decision_vars =
# volumn = np.empty(Iteration - pop_size)
# print(f"Iteration {iteration+1}: Decision Variables: {decision_vars}, Objectives: {obj_values}")
# feasible_solutions = [s for s in algorithm.result if s.feasible]

# # nondominated_solutions = nondominated(algorithm.result)
#
# # Get the number of objectives
# n_obj = len(algorithm.result[0].objectives)
#
# # Create an empty array to store the objective values
# obj_values = np.empty((len(algorithm.result), n_obj))
#
# # Loop over the solutions and get the objective values
# for i, solution in enumerate(algorithm.result):
#     obj_values[i,:] = solution.objectives

# Print the objective values
# print(obj_values)


# for i in range(Iteration-pop_size):
#     bd = DominatedPartitioning(ref_point=ref_point, Y=obj_values[i,:])
#     volumn[i] = bd.compute_hypervolume().item()
# print(volumn)
#
# for solution in algorithm.result:
#     decision_vars = solution.variables
#     obj_values = solution.objectives
#     print(f"Decision Variables: {decision_vars}, Objectives: {obj_values}")
