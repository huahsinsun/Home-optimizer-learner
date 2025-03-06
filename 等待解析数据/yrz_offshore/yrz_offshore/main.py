import os
import pandas as pd
from sub_function import objective
from vars_and_cons import model_initialization
import torch
from botorch.models.transforms.factory import get_rounding_input_transform
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.logei import (
    qLogExpectedImprovement
)
import inspect
import numpy as np
from tqdm import trange
from datetime import datetime

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"dtype": dtype, "device": device}
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.models.transforms.input import OneHotToNumeric
from botorch.test_functions.synthetic import Ackley
folder_name = "Outcome_typical"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
SMOKE_TEST = False
# torch.manual_seed(0)
N_TRIALS = 1
N_Iteration = 100 if not SMOKE_TEST else 2
N_Batch = 3 if not SMOKE_TEST else 2
integer_indices = [0, 1, 2]#, 3, 4, 5]
dim = len(integer_indices) * 2
# base_function = Ackley(dim=dim, negate=True).to(**tkwargs)
algorithm_option = [1, 0, 1] # cont_relax PR_montcarlo PR_analytic

base_function = objective
bounds = torch.zeros(2, dim, **tkwargs)
bounds[1] = 1
initial_canditate = 20 if not SMOKE_TEST else 2
# construct a rounding function for initialization (equal probability for all discrete values)
init_exact_rounding_func = get_rounding_input_transform(
    one_hot_bounds=bounds, integer_indices=integer_indices, initialization=True
)
# construct a rounding function
exact_rounding_func = get_rounding_input_transform(
    one_hot_bounds=bounds, integer_indices=integer_indices, initialization=False
)


def eval_problem(X):
    X = exact_rounding_func(X)
    # model = model_initialization()
    # X_normalized = normalize(X, bounds) # Min-max normalize X w.r.t. the provided bounds.
    # unnormalize from unit cube to the problem space
    # raw_X = unnormalize(X_normalized, bounds)
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    train_obj_0, train_obj_1 = base_function(X, integer_indices)
    return train_obj_0 - 10 * train_obj_1


def generate_initial_data(n):
    r"""
    Generates the initial data for the experiments.
    Args:
        n: Number of training points..
    Returns:
        The train_X and train_Y. `n x d` and `n x 1`.
    """
    raw_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2)
    raw_x[..., integer_indices] = init_exact_rounding_func(raw_x[..., integer_indices])
    # bool_vars = raw_x[..., :len(integer_indices)]  # First n elements (boolean variables)
    # continuous_vars = raw_x[..., len(integer_indices):]  # Last n elements (continuous variables)
    # continuous_vars[bool_vars == 0] = 0
    # raw_x[..., len(integer_indices):] = continuous_vars

    train_obj = torch.zeros(n, **tkwargs)
    for i in trange(n, desc='Initializing', unit="Candidate"):
        train_obj[i] = eval_problem(raw_x[i])
    return raw_x, train_obj.unsqueeze(-1)


import torch
from botorch.models import FixedNoiseGP
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def get_kernel(dim: int) -> Kernel:
    """Helper function for kernel construction."""
    # ard kernel
    cont_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=dim,
        lengthscale_constraint=Interval(0.01, 20.0),
    )
    return ScaleKernel(cont_kernel)


NOISE_SE = 1e-6
train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)


def initialize_model(
        train_x, stdized_train_obj, state_dict=None, exact_rounding_func=None
):
    # define model
    model = FixedNoiseGP(
        train_x,
        stdized_train_obj,
        train_yvar.expand_as(stdized_train_obj),
        covar_module=get_kernel(dim=train_x.shape[1]),
    ).to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.optim import optimize_acqf

NUM_RESTARTS = 20 if not SMOKE_TEST else 2
RAW_SAMPLES = 1024 if not SMOKE_TEST else 32


def optimize_acqf_cont_relax_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=N_Batch,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        return_best_only=False,
    )
    # round the resulting candidates and take the best across restarts
    candidates = exact_rounding_func(candidates.detach())
    with torch.no_grad():
        af_vals = acq_func(candidates)
    best_idx = af_vals.argmax()
    new_x = candidates[best_idx]
    exact_obj = torch.zeros(N_Batch, 1, **tkwargs)
    for i in range(N_Batch):
        exact_obj[i] = torch.as_tensor(eval_problem(new_x[i]), **tkwargs)
        # exact_obj = torch.cat(exact_obj, dim=0)
    return new_x, exact_obj


def optimize_acqf_pr_and_get_observation(acq_func, analytic):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # construct PR
    if analytic:
        pr_acq_func = AnalyticProbabilisticReparameterization(
            acq_function=acq_func,
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            batch_limit=200,
        )
    else:
        pr_acq_func = MCProbabilisticReparameterization(
            acq_function=acq_func,
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            batch_limit=200,
            mc_samples=4 if SMOKE_TEST else 128,
        )
    candidates, _ = optimize_acqf(
        acq_function=pr_acq_func,
        bounds=bounds,
        q=N_Batch,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={
            "batch_limit": 5,
            "maxiter": 200,
            "rel_tol": float("-inf"),  # run for a full 200 steps
        },
        sequential=True,
        # use Adam for Monte Carlo PR
        gen_candidates=gen_candidates_torch if not analytic else gen_candidates_scipy,
    )
    # round the resulting candidates and take the best across restarts
    new_x = pr_acq_func.sample_candidates(X=candidates.detach())
    # observe new values
    exact_obj = torch.zeros(N_Batch, 1, **tkwargs)
    for i in range(N_Batch):
        exact_obj[i] = torch.as_tensor(eval_problem(new_x[i]), **tkwargs)
        # exact_obj = torch.cat(exact_obj, dim=0)
    return new_x, exact_obj


import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.transforms import standardize

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def tensor_to_dataframe_and_excel(tensor_or_array):
    # Ensure input is either a torch tensor or a numpy array
    if isinstance(tensor_or_array, torch.Tensor):
        # If input is a torch tensor, convert it to numpy
        data = tensor_or_array.cpu().numpy()  # Ensure tensor is moved to CPU if on GPU
    elif isinstance(tensor_or_array, np.ndarray):
        # If input is already a numpy array, use it as is
        data = tensor_or_array
    else:
        raise ValueError("Input must be a torch.Tensor or np.ndarray.")

    # Retrieve the variable name of the tensor/array in the current scope
    caller_locals = inspect.currentframe().f_back.f_locals
    tensor_name = next((name for name, value in caller_locals.items() if value is tensor_or_array), "tensor")

    # Create column names like 'tensorname_1', 'tensorname_2', ...
    n, m = data.shape
    column_names = [f"{tensor_name}_{i + 1}" for i in
                    range(m)]  # Create column names like 'tensorname_1', 'tensorname_2', ...

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)

    return df


verbose = True

(
    best_observed_all_pr,
    best_observed_all_pr_analytic,
    best_observed_all_cont_relax,
) = ([], [], [])
from botorch.acquisition.objective import GenericMCObjective
from typing import Optional

# average over multiple trials
if SMOKE_TEST:
    print("\n*** WARNING: SMOKE TEST IS ACTIVATED ***")
    print("Running a quick, limited set of trials and samples for testing purposes.\n")
for trial in range(1, N_TRIALS + 1):

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    (
        best_observed_pr,
        best_observed_pr_analytic,
        best_observed_cont_relax,
    ) = ([], [], [])

    # call helper functions to generate initial training data and initialize model
    train_x_pr, train_obj_pr = generate_initial_data(n=initial_canditate)
    best_observed_value_pr = train_obj_pr.max().item()
    stdized_train_obj_pr = standardize(train_obj_pr)  # z-score normalize
    mll_pr, model_pr = initialize_model(train_x_pr, stdized_train_obj_pr)

    train_x_pr_analytic, train_obj_pr_analytic, stdized_train_obj_pr_analytic = (
        train_x_pr,
        train_obj_pr,
        stdized_train_obj_pr,
    )
    best_observed_value_pr_analytic = best_observed_value_pr
    mll_pr_analytic, model_pr_analytic = initialize_model(
        train_x_pr_analytic,
        stdized_train_obj_pr_analytic,
    )

    train_x_cont_relax, train_obj_cont_relax, stdized_train_obj_cont_relax = (
        train_x_pr,
        train_obj_pr,
        stdized_train_obj_pr,
    )
    best_observed_value_cont_relax = best_observed_value_pr
    mll_cont_relax, model_cont_relax = initialize_model(
        train_x_cont_relax,
        stdized_train_obj_cont_relax,
    )

    best_observed_pr.append(best_observed_value_pr)
    best_observed_pr_analytic.append(best_observed_value_pr_analytic)
    best_observed_cont_relax.append(best_observed_value_cont_relax)

    # run N_Iteration rounds of BayesOpt after the initial random batch
    for iteration in trange(1, N_Iteration + 1, desc='Optimizing', unit="Iteration"):

        t0 = time.monotonic()

        # fit the models
        # fit_gpytorch_mll(mll_pr)
        fit_gpytorch_mll(mll_pr_analytic)
        fit_gpytorch_mll(mll_cont_relax)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([RAW_SAMPLES]))

        # for best_f, we use the best observed values
        # ei_pr = qLogExpectedImprovement(
        #     model=model_pr,
        #     best_f=stdized_train_obj_pr.max(),
        #     sampler=qmc_sampler,
        # )

        ei_pr_analytic = qLogExpectedImprovement(
            model=model_pr_analytic,
            best_f=stdized_train_obj_pr_analytic.max(),
            sampler=qmc_sampler,
        )

        ei_cont_relax = qLogExpectedImprovement(
            model=model_cont_relax,
            best_f=stdized_train_obj_cont_relax.max(),
            sampler=qmc_sampler,
        )

        # optimize and get new observation
        # new_x_pr, new_obj_pr = optimize_acqf_pr_and_get_observation(
        #     ei_pr, analytic=False
        # )
        new_x_pr_analytic, new_obj_pr_analytic = optimize_acqf_pr_and_get_observation(
            ei_pr_analytic, analytic=True
        )
        (
            new_x_cont_relax,
            new_obj_cont_relax,
        ) = optimize_acqf_cont_relax_and_get_observation(ei_cont_relax)

        # print(f"MC: {new_x_pr}")
        # print(f"analytic: {new_x_pr_analytic}")
        # update training points
        # train_x_pr = torch.cat([train_x_pr, new_x_pr])
        # train_obj_pr = torch.cat([train_obj_pr, new_obj_pr])
        # stdized_train_obj_pr = standardize(train_obj_pr)

        train_x_pr_analytic = torch.cat([train_x_pr_analytic, new_x_pr_analytic])
        train_obj_pr_analytic = torch.cat([train_obj_pr_analytic, new_obj_pr_analytic])
        stdized_train_obj_pr_analytic = standardize(train_obj_pr_analytic)

        train_x_cont_relax = torch.cat([train_x_cont_relax, new_x_cont_relax])
        train_obj_cont_relax = torch.cat([train_obj_cont_relax, new_obj_cont_relax])
        stdized_train_obj_cont_relax = standardize(train_obj_cont_relax)

        # update progress
        best_value_pr_analytic = train_obj_pr.max().item()
        # best_value_pr = train_obj_pr_analytic.max().item()
        best_value_cont_relax = train_obj_cont_relax.max().item()
        # best_observed_pr.append(best_value_pr)
        best_observed_pr_analytic.append(best_value_pr_analytic)
        best_observed_cont_relax.append(best_value_cont_relax)

        # reinitialize 2the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        # mll_pr, model_pr = initialize_model(
        #     train_x_pr,
        #     stdized_train_obj_pr,
        # )
        mll_pr_analytic, model_pr_analytic = initialize_model(
            train_x_pr_analytic,
            stdized_train_obj_pr_analytic,
        )
        mll_cont_relax, model_cont_relax = initialize_model(
            train_x_cont_relax,
            stdized_train_obj_cont_relax,
        )

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (Cont. Relax., PR (Analytic)) = "
                f"({best_value_cont_relax:>4.2f}, {best_value_pr_analytic:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        # if True:
            # train_x_pr_save = pd.DataFrame(train_x_pr.numpy(),
            #                                columns=[f'x_pr_{i}' for i in range(train_x_pr.shape[1])])
            # train_obj_pr_save = pd.DataFrame(train_obj_pr.numpy(),
            #                                  columns=[f'y_pr_{i}' for i in range(
            #                                      train_obj_pr.shape[1])])  # , 'line_par2', 'line_par3','rate_par'])

        if True:
            train_x_apr_save = pd.DataFrame(train_x_pr_analytic.numpy(),
                                            columns=[f'x_apr_{i}' for i in range(train_x_pr_analytic.shape[1])])
            train_obj_apr_save = pd.DataFrame(train_obj_pr_analytic.numpy(),
                                              columns=[f'y_apr_{i}' for i in range(
                                                  train_obj_pr_analytic.shape[1])])

        if True:
            train_x_cont_relax_save = pd.DataFrame(train_x_cont_relax.numpy(),
                                                   columns=[f'x_cont_{i}' for i in range(train_x_cont_relax.shape[1])])
            train_obj_cont_relax_save = pd.DataFrame(train_obj_cont_relax.numpy(),
                                                     columns=[f'y_cont_{i}' for i in range(
                                                         train_obj_cont_relax.shape[1])])

        # hp_data_save = pd.DataFrame(hp_data)
        defined_dataframes = []
        # List of dataframe names as strings
        dataframe_names = [
            'train_x_pr_save', 'train_x_apr_save', 'train_x_cont_relax_save',
            'train_obj_pr_save', 'train_obj_apr_save', 'train_obj_cont_relax_save',
        ]

        # Check each dataframe name and add it to the list if it is defined
        for df_name in dataframe_names:
            try:
                # Attempt to evaluate the dataframe name
                df = eval(df_name)
                # If successful, append the dataframe to the list
                defined_dataframes.append(df)
            except NameError:
                # If the dataframe is not defined, skip it
                pass
        # Concatenate the defined dataframes
        result = pd.concat(defined_dataframes, axis=1)

        current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        file_name = f'{current_time}.xlsx'

        file_path = os.path.join(folder_name, file_name)
        result.to_excel(file_path, index=False)

    best_observed_all_pr.append(best_observed_pr)
    best_observed_all_pr_analytic.append(best_observed_pr_analytic)
    best_observed_all_cont_relax.append(best_observed_cont_relax)




