import os
from sub_function import objective
from vars_and_cons import model_initialization
import torch
from botorch.models.transforms.factory import get_rounding_input_transform
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.logei import (
    qLogExpectedImprovement
)
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from tqdm import trange
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"dtype": dtype, "device": device}
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.models.transforms.input import OneHotToNumeric
from botorch.test_functions.synthetic import Ackley
SMOKE_TEST = False
torch.manual_seed(0)
N_TRIALS = 1
N_Iteration = 100 if not SMOKE_TEST else 2
N_Batch = 1 if not SMOKE_TEST else 1
dim = 8
integer_indices = [0, 1, 2, 3]
# base_function = Ackley(dim=dim, negate=True).to(**tkwargs)
base_function = objective
bounds = torch.zeros(2, dim, **tkwargs)
bounds[1] = 1
initial_canditate = 40 if not SMOKE_TEST else 2
initial_model = model_initialization()
# construct a rounding function for initialization (equal probability for all discrete values)
init_exact_rounding_func = get_rounding_input_transform(
    one_hot_bounds=bounds, integer_indices=integer_indices, initialization=True
)
# construct a rounding function
exact_rounding_func = get_rounding_input_transform(
    one_hot_bounds=bounds, integer_indices=integer_indices, initialization=False
)
def outcome_constraint(raw_x):
    X = raw_x.clone().view(-1)
    # Split X into the bool and continuous parts
    bool_vars = X[:len(integer_indices)].clone()        # First n elements (boolean variables)
    continuous_vars = X[len(integer_indices):]   # Last n elements (continuous variables)
    bool_vars[X[:len(integer_indices)] == 0] = 1
    bool_vars[X[:len(integer_indices)] == 1] = 0
    con = 0
    # Check the legality: if bool_vars[i] == 0, continuous_vars[i] must be 0
    for i in range(len(integer_indices)):
        con += bool_vars[i] * continuous_vars[i]

    return con  # Legal input if con == 0


def eval_problem(X):
    X = exact_rounding_func(X)
    # X_normalized = normalize(X, bounds) # Min-max normalize X w.r.t. the provided bounds.
    # unnormalize from unit cube to the problem space
    # raw_X = unnormalize(X_normalized, bounds)
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    fesibility = outcome_constraint(X)
    if fesibility != 0:
        return 0, fesibility
    else:
        return base_function(initial_model, X, integer_indices) * (fesibility == 0), fesibility


def generate_initial_data(n):
    r"""
    Generates the initial data for the experiments.
    Args:
        n: Number of training points..
    Returns:
        The train_X and train_Y. `n x d` and `n x 1`.
    """
    raw_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2)
    raw_x[...,integer_indices] = init_exact_rounding_func(raw_x[...,integer_indices])
    # bool_vars = raw_x[..., :len(integer_indices)]  # First n elements (boolean variables)
    # continuous_vars = raw_x[..., len(integer_indices):]  # Last n elements (continuous variables)
    # continuous_vars[bool_vars == 0] = 0
    # raw_x[..., len(integer_indices):] = continuous_vars

    train_obj, train_con = torch.zeros(n, **tkwargs), torch.zeros(n, **tkwargs)
    for i in trange(n, desc='Initializing', unit="Candidate"):
        train_obj[i], train_con[i] = eval_problem(raw_x[i])
    return raw_x, train_obj.unsqueeze(-1), train_con.unsqueeze(-1)


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
        train_x, stdized_train_obj, train_con, state_dict=None, exact_rounding_func=None
):
    # define model
    model_obj = FixedNoiseGP(
        train_x,
        stdized_train_obj,
        train_yvar.expand_as(stdized_train_obj),
        covar_module=get_kernel(dim=train_x.shape[1]),
    ).to(train_x)
    model_con = FixedNoiseGP(
        train_x,
        train_con,
        train_yvar.expand_as(train_con),
        input_transform=Normalize(d=train_x.shape[-1]),
    ).to(train_x)
    model = ModelListGP(model_obj, model_con)

    mll = SumMarginalLogLikelihood(model.likelihood, model)
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
    # observe new values
    exact_obj, new_con = eval_problem(new_x)
    return new_x, exact_obj, new_con.view(1,1)

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
        # use Adam for Monte Carlo PR
        gen_candidates=gen_candidates_torch if not analytic else gen_candidates_scipy,
    )
    # round the resulting candidates and take the best across restarts
    new_x = pr_acq_func.sample_candidates(X=candidates.detach())
    # observe new values
    exact_obj, new_con = eval_problem(new_x)
    return new_x, exact_obj, new_con.view(1,1)


def update_random_observations(best_random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(1, bounds.shape[1], **tkwargs)
    new_obj, new_con  = eval_problem(rand_x)
    next_random_best = new_obj.max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random


import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.transforms import standardize

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



verbose = True

(
    best_observed_all_pr,
    best_observed_all_pr_analytic,
    best_observed_all_cont_relax,
    best_random_all,
) = ([], [], [], [])
from botorch.acquisition.objective import GenericMCObjective
from typing import Optional

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable(Z):
    return Z[..., 1]


objective = GenericMCObjective(objective=obj_callable)
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
        best_random,
    ) = ([], [], [], [])

    # call helper functions to generate initial training data and initialize model
    train_x_pr, train_obj_pr, train_con_pr = generate_initial_data(n=initial_canditate)
    best_observed_value_pr = train_obj_pr.max().item()
    stdized_train_obj_pr = standardize(train_obj_pr)  # z-score normalize
    stdized_train_con_pr = standardize(train_con_pr)
    mll_pr, model_pr = initialize_model(train_x_pr, stdized_train_obj_pr, stdized_train_con_pr)

    train_x_pr_analytic, train_obj_pr_analytic, stdized_train_obj_pr_analytic, stdized_train_con_pr_analytic = (
        train_x_pr,
        train_obj_pr,
        stdized_train_obj_pr,
        stdized_train_con_pr
    )
    best_observed_value_pr_analytic = best_observed_value_pr
    mll_pr_analytic, model_pr_analytic = initialize_model(
        train_x_pr_analytic,
        stdized_train_obj_pr_analytic,
        stdized_train_con_pr_analytic
    )

    train_x_cont_relax, train_obj_cont_relax, stdized_train_obj_cont_relax, stdized_train_con_cont_relax = (
        train_x_pr,
        train_obj_pr,
        stdized_train_obj_pr,
        stdized_train_con_pr
    )
    best_observed_value_cont_relax = best_observed_value_pr
    mll_cont_relax, model_cont_relax = initialize_model(
        train_x_cont_relax,
        stdized_train_obj_cont_relax,
        stdized_train_con_cont_relax
    )

    best_observed_pr.append(best_observed_value_pr)
    best_observed_pr_analytic.append(best_observed_value_pr_analytic)
    best_observed_cont_relax.append(best_observed_value_cont_relax)
    best_random.append(best_observed_value_pr)

    # run N_Iteration rounds of BayesOpt after the initial random batch
    for iteration in trange(1, N_Iteration + 1, desc='Optimizing', unit="Iteration"):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_pr)
        fit_gpytorch_mll(mll_pr_analytic)
        fit_gpytorch_mll(mll_cont_relax)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([RAW_SAMPLES]))


        # for best_f, we use the best observed values
        ei_pr = qLogExpectedImprovement(
            model=model_pr,
            best_f=stdized_train_obj_pr.max(),
            sampler=qmc_sampler,
            objective = objective,
            constraints = [constraint_callable],
        )

        ei_pr_analytic = qLogExpectedImprovement(
            model=model_pr_analytic,
            best_f=stdized_train_obj_pr_analytic.max(),
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        ei_cont_relax = qLogExpectedImprovement(
            model=model_cont_relax,
            best_f=stdized_train_obj_cont_relax.max(),
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        # optimize and get new observation
        new_x_pr, new_obj_pr, new_con_pr = optimize_acqf_pr_and_get_observation(
            ei_pr, analytic=False
        )
        new_x_pr_analytic, new_obj_pr_analytic, new_con_pr_analytic = optimize_acqf_pr_and_get_observation(
            ei_pr_analytic, analytic=True
        )
        (
            new_x_cont_relax,
            new_obj_cont_relax,
            new_con_cont_relax
        ) = optimize_acqf_cont_relax_and_get_observation(ei_cont_relax)

        print(f"MC: {new_x_pr}")
        print(f"analytic: {new_x_pr_analytic}")
        # update training points
        train_x_pr = torch.cat([train_x_pr, new_x_pr])
        train_obj_pr = torch.cat([train_obj_pr, new_obj_pr])
        train_con_pr = torch.cat([train_con_pr, new_con_pr])
        stdized_train_obj_pr = standardize(train_obj_pr)
        stdized_con_obj_pr = standardize(train_con_pr)

        train_x_pr_analytic = torch.cat([train_x_pr_analytic, new_x_pr_analytic])
        train_obj_pr_analytic = torch.cat([train_obj_pr_analytic, new_obj_pr_analytic])
        train_con_pr_analytic = torch.cat([train_con_pr_analytic, new_con_pr_analytic])
        stdized_train_obj_pr_analytic = standardize(train_obj_pr_analytic)
        stdized_train_con_pr_analytic = standardize(train_con_pr_analytic)


        train_x_cont_relax = torch.cat([train_x_cont_relax, new_x_cont_relax])
        train_obj_cont_relax = torch.cat([train_obj_cont_relax, new_obj_cont_relax])
        train_con_cont_relax = torch.cat([train_con_cont_relax, new_con_cont_relax])
        stdized_train_obj_cont_relax = standardize(train_obj_cont_relax)
        stdized_train_con_cont_relax = standardize(train_con_cont_relax)


        # update progress
        best_random = update_random_observations(best_random)
        best_value_pr_analytic = train_obj_pr.max().item()
        best_value_pr = train_obj_pr_analytic.max().item()
        best_value_cont_relax = train_obj_cont_relax.max().item()
        best_observed_pr.append(best_value_pr)
        best_observed_pr_analytic.append(best_value_pr_analytic)
        best_observed_cont_relax.append(best_value_cont_relax)

        # reinitialize 2the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_pr, model_pr = initialize_model(
            train_x_pr,
            stdized_train_obj_pr,
            stdized_train_con_pr
        )
        mll_pr_analytic, model_pr_analytic = initialize_model(
            train_x_pr_analytic,
            stdized_train_obj_pr_analytic,
            stdized_train_con_pr_analytic
        )
        mll_cont_relax, model_cont_relax = initialize_model(
            train_x_cont_relax,
            stdized_train_obj_cont_relax,
            stdized_train_con_cont_relax
        )

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, Cont. Relax., PR (MC), PR (Analytic)) = "
                f"({max(best_random):>4.2f}, {best_value_cont_relax:>4.2f}, {best_value_pr:>4.2f}, {best_value_pr_analytic:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    best_observed_all_pr.append(best_observed_pr)
    best_observed_all_pr_analytic.append(best_observed_pr_analytic)
    best_observed_all_cont_relax.append(best_observed_cont_relax)
    best_random_all.append(best_random)

    import numpy as np
    from matplotlib import pyplot as plt


    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)


    iters = np.arange(N_Iteration + 1)
    y_cont_relax = np.asarray(best_observed_all_cont_relax)
    y_pr = np.asarray(best_observed_all_pr)
    y_pr_analytic = np.asarray(best_observed_all_pr_analytic)
    y_rnd = np.asarray(best_random_all)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="Random", linewidth=1.5)
    ax.errorbar(
        iters,
        y_cont_relax.mean(axis=0),
        yerr=ci(y_cont_relax),
        label="Cont. Relax.",
        linewidth=1.5,
    )
    ax.errorbar(iters, y_pr.mean(axis=0), yerr=ci(y_pr), label="PR (MC)", linewidth=1.5)
    ax.errorbar(iters, y_pr_analytic.mean(axis=0), yerr=ci(y_pr_analytic), label="PR (Analytic)", linewidth=1.5)
    ax.set(
        xlabel="number of observations (beyond initial points)",
        ylabel="best objective value",
    )
    ax.legend(loc="lower right")
