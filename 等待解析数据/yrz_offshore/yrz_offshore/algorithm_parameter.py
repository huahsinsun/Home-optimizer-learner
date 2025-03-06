import torch
import os
from sub_function import objective, tkwargs


SMOKE_TEST = False
load_option = False
random_flag = True
Noise_flag = True
Exam_flag = True if Noise_flag else False
if not Exam_flag:
    exam_time = 0
elif SMOKE_TEST:
    exam_time = 2
else:
    exam_time = 200
if not Noise_flag:
    NOISE_SE = torch.tensor([1e-6, 1e-6], **tkwargs)
else:
    NOISE_SE = torch.tensor([1e-6, 1e-6], **tkwargs)

# NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
from botorch.models.transforms.input import OneHotToNumeric
from botorch.test_functions.synthetic import Ackley
from botorch.test_functions.multi_objective import BraninCurrin
folder_name = "Outcome_noisy"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
dim = 14
N_TRIALS = 1
N_BATCH = 200 if not SMOKE_TEST else 2
Batch_size = 2
integer_indices = [0, 1, 2, 3, 4, 5]
discrete_len = len(integer_indices)
algorithm_option = [0, 0, 1] # cont_relax PR_montcarlo PR_analytic
# base_function = Ackley(dim=dim, negate=True).to(**tkwargs)
bounds = torch.zeros(2, dim, **tkwargs)
bounds[1] = 1
ref_rate = 3
ref_point_fixed = [-3 -3]
bounds[1] = 1
NUM_RESTARTS = 20 if not SMOKE_TEST else 2
RAW_SAMPLES = 1024 if not SMOKE_TEST else 32
MC_SAMPLES = 256 if not SMOKE_TEST else 16
sub_iteration = 80 if not SMOKE_TEST else 2
reopt_batch = 5 if not SMOKE_TEST else 2
def objective_fucntion(x, discrete_idx):
    neg_annual_cost, RES_accombaility = objective(x, discrete_idx, random_flag)
    return neg_annual_cost, RES_accombaility

base_function = objective_fucntion
