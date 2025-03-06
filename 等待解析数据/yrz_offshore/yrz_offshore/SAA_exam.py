import torch
from tqdm import trange
import algorithm_parameter as ap
from sub_function import sub_objective
from botorch.utils.sampling import draw_sobol_samples


def scheme_exam(scheme, scenario_num):
    cost_result = torch.zeros(scenario_num, **ap.tkwargs)
    if scenario_num == 364:  # full year simulation
        for i in range(scenario_num):
            cost_result[i], _ = sub_objective(x=scheme, discrete_idx=ap.integer_indices, random_flag=False,
                                              full_output_flag=0, scenario_idx=i)
            # _, RES_accombaility = sub_objective(x=scheme, discrete_idx=ap.integer_indices, random_flag=ap.random_flag,
            #                                     full_output_flag=1, scenario_idx=i)
    else:
        for i in range(scenario_num):
            cost_result[i], _ = sub_objective(x=scheme, discrete_idx=ap.integer_indices, random_flag=True,
                                              full_output_flag=0)
    return cost_result


def SAA_exam(Schemes, scenario_num):
    scheme_num = Schemes.shape[0]
    error = torch.zeros([scheme_num, scenario_num], **ap.tkwargs)
    for i in trange(scheme_num, desc='Examing', unit="Scheme"):
        error[i] = scheme_exam(Schemes[i], scenario_num)

    return error


if __name__ == "__main__":
    # raw_x = draw_sobol_samples(bounds=ap.bounds, n=2, q=1).squeeze(1)
    # train_x = torch.concat([torch.round(raw_x[:, 0:ap.discrete_len]), raw_x[:, ap.discrete_len:ap.dim]], dim=1)
    # SAA_exam(Schemes=train_x, scenario_num=2)
    files_dir = 'to_be_examed'
