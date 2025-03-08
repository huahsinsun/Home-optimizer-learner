import torch
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.transforms import standardize
from vars_and_cons import model_initialization
from botorch.models.transforms.factory import get_rounding_input_transform
from algorithm_parameter import *
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from tqdm import trange
from botorch.models.transforms import Standardize
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
import numpy as np

initial_model = model_initialization()
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
    X_normalized = normalize(X, bounds)  # Min-max normalize X w.r.t. the provided bounds.
    # unnormalize from unit cube to the problem space
    raw_X = unnormalize(X_normalized, bounds)
    return base_function(X, integer_indices)


from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
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


def generate_initial_data(n):
    # generate training data
    raw_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    train_x = torch.concat([init_exact_rounding_func(raw_x[:,0:discrete_len*2]),raw_x[:,discrete_len*2:dim]],dim = 1)
    train_x[train_x == -0] = 0
    train_obj = torch.zeros([n, 2], **tkwargs)
    for i in trange(n, desc='Initializing', unit="Candidate"):
        train_obj[i, 0], train_obj[i, 1] = eval_problem(raw_x[i])
    # train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj


def initialize_model(
        train_x, train_obj, state_dict=None, exact_rounding_func=None
):
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i: i + 1]
        if not Noise_flag or i == 1:
            train_yvar = torch.full_like(train_y, 1e-6)
            models.append(
                FixedNoiseGP(
                    train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            )
        else:
            models.append(
                SingleTaskGP(
                    train_x, train_y, outcome_transform=Standardize(m=1)
                )
            )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.optim import optimize_acqf


def optimize_acqf_cont_relax_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=Batch_size,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        # return_best_only=False,
        sequential=True,
    )
    # round the resulting candidates and take the best across restarts
    candidates = exact_rounding_func(candidates.detach())
    # with torch.no_grad():
    #     af_vals = acq_func(candidates)
    # best_idx = af_vals.argmax()
    # new_x = candidates[best_idx]
    # observe new values
    exact_obj = torch.zeros(Batch_size, 2, **tkwargs)
    for i in range(Batch_size):
        exact_obj[i, :] = torch.as_tensor(eval_problem(candidates[i]), **tkwargs)
        # exact_obj = torch.cat(exact_obj, dim=0)

    return candidates, exact_obj


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
            mc_samples=4 if SMOKE_TEST else 256,
        )
    candidates, _ = optimize_acqf(
        acq_function=pr_acq_func,
        bounds=bounds,
        q=Batch_size,
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
    exact_obj = torch.zeros(Batch_size, 2, **tkwargs)
    for i in range(Batch_size):
        exact_obj[i, :] = torch.as_tensor(eval_problem(new_x[i]), **tkwargs)
        # exact_obj = torch.cat(exact_obj, dim=0)
    return new_x, exact_obj


def update_refpoint(stdized_obj):
    max, _ = torch.max(stdized_obj, 0)
    min, _ = torch.min(stdized_obj, 0)
    if ref_rate > 0:
        ref_point = min - ref_rate * max
    else:
        ref_point = min + ref_rate
    return ref_point


def hp_calculation(y_pr, y_apr, y_cont):
    y = torch.cat((y_pr.view(-1, 2), y_apr.view(-1, 2), y_cont.view(-1, 2)), dim=0)

    # mean_y = torch.mean(y, dim=0)
    # std_y = torch.std(y, dim=0)

    # def normalize_tensor(tensor, mean, std):
    #     # Normalize each group of 3 columns in the tensor
    #     for i in range(0, tensor.shape[1], 2):
    #         tensor[:, i:i + 2] = (tensor[:, i:i + 2] - mean) / std
    #     return tensor

    # Normalize each tensor
    # normalized_y_pr = normalize_tensor(y_pr, mean_y, std_y)
    # normalized_y_apr = normalize_tensor(y_apr, mean_y, std_y)
    # normalized_y_cont = normalize_tensor(y_cont, mean_y, std_y)
    nor_y = standardize(y)
    if ref_rate > 0:
        ref_point = torch.min(nor_y, dim=0).values - ref_rate * torch.max(nor_y, dim=0).values
    else:
        ref_point = torch.min(nor_y, dim=0).values + ref_rate
    if not SMOKE_TEST:
        hv_ambo, hv_par, hv_nehv = np.empty([(y_cont.shape[0] - 20) // 2, y_cont.shape[1] // 2]), np.empty(
            [(y_pr.shape[0] - 20) // 2, y_pr.shape[1] // 2]), np.empty(
            [(y_apr.shape[0] - 20) // 2, y_apr.shape[1] // 2])
    else:
        hv_ambo, hv_par, hv_nehv = np.empty([(y_cont.shape[0]) // 2, y_cont.shape[1] // 2]), np.empty(
            [(y_pr.shape[0]) // 2, y_pr.shape[1] // 2]), np.empty(
            [(y_apr.shape[0]) // 2, y_apr.shape[1] // 2])
    # hv_ambo, hv_par, hv_nehv = np.empty([(y_cont.shape[0]-20)//3,y_cont.shape[1]//3]), np.empty([(y_pr.shape[0]-20)//3,y_pr.shape[1]//3]), np.empty([(y_apr.shape[0]-20)//3,y_apr.shape[1]//3])

    for i in range(hv_par.shape[1]):
        for j in range(hv_par.shape[0]):
            bd = DominatedPartitioning(ref_point=ref_point, Y=y_pr[0:20 + j * 2, i * 2:i * 2 + 2])
            volume = bd.compute_hypervolume().item()
            # Append the computed hypervolume to the list
            hv_par[j, i] = volume
    hv_pr_mean = np.mean(hv_par, axis=1)

    for i in range(hv_nehv.shape[1]):
        for j in range(hv_nehv.shape[0]):
            bd = DominatedPartitioning(ref_point=ref_point, Y=y_apr[0:20 + j * 2, i * 2:i * 2 + 2])
            volume = bd.compute_hypervolume().item()
            # Append the computed hypervolume to the list
            hv_nehv[j, i] = volume
    hv_apr_mean = np.mean(hv_nehv, axis=1)

    for i in range(hv_ambo.shape[1]):
        for j in range(hv_ambo.shape[0]):
            bd = DominatedPartitioning(ref_point=ref_point, Y=y_cont[0:20 + j * 2, i * 2:i * 2 + 2])
            volume = bd.compute_hypervolume().item()
            # Append the computed hypervolume to the list
            hv_ambo[j, i] = volume
    hv_cont_mean = np.mean(hv_ambo, axis=1)

    return hv_pr_mean, hv_apr_mean, hv_cont_mean


def select_isolated_non_dominated_points(x, y):
    """
    Select the most isolated non-dominated points based on distance between their objective values.

    Parameters:
    - x: A 2D tensor of non-dominated decision variables (shape: N x D)
    - y: A 2D tensor of non-dominated objective values (shape: N x M)

    Returns:
    - isolated_points: Indices of the most isolated non-dominated points.
    """

    # 1. Check if y has been normalized (zero mean, unit variance)
    if not (torch.allclose(torch.min(y, dim=0).values, torch.zeros_like(torch.min(y, dim=0).values), atol=1e-5) and
            torch.allclose(torch.max(y, dim=0).values, torch.ones_like(torch.max(y, dim=0).values), atol=1e-5)):
        y_min, _ = torch.min(y, dim=0, keepdim=True)
        y_max, _ = torch.max(y, dim=0, keepdim=True)
        y = (y - y_min) / (y_max - y_min)  # Min-max normalization

    # 2. Calculate the pairwise Euclidean distances between the points in y
    N = y.shape[0]
    distance_matrix = torch.zeros((N, N), dtype=torch.float32)

    # Calculate the pairwise Euclidean distances
    for i in range(N):
        for j in range(i + 1, N):
            distance = torch.norm(y[i] - y[j])  # Euclidean distance
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric distance

    # 3. Calculate the isolation for each point: The isolation of a point is the sum of its distances to all other points
    isolation_scores = torch.sum(distance_matrix, dim=1)

    # 4. Select the points with the largest isolation scores (most isolated points)
    most_isolated_indices = torch.argsort(isolation_scores,
                                          descending=True)  # Sort in descending order (most isolated first)

    # Return the indices of the most isolated non-dominated points
    return x[most_isolated_indices[0], :]


def extract_isolated_points(x, y, isolated_x, dis_dim=len(integer_indices)):
    """
    Extracts the x and y values corresponding to the most isolated points based on the discrete variables.

    Parameters:
    - train_x_apr_df: A tensor of decision variables (shape: N x D)
    - train_obj_apr_df: A tensor of objective values (shape: N x M)
    - isolated_indices: Indices of the most isolated points based on their objective values
    - dis_dim: The number of discrete variables (default is 6)

    Returns:
    - filtered_x: Tensor of the decision variables of the isolated points with matching discrete parts
    - filtered_y: Tensor of the objective values of the isolated points with matching discrete parts
    """

    # Get the discrete part of the isolated points (first 'dis_dim' elements of x)
    isolated_discrete_part = isolated_x[:dis_dim]

    # Filter train_x_apr_df and train_obj_apr_df based on matching discrete parts
    filtered_x = []
    filtered_y = []

    # Filter the rows in train_x_apr_df where the first 'dis_dim' elements match
    matching_rows_mask = torch.all(x[:, :dis_dim] == isolated_discrete_part, dim=1)

    # Extract the corresponding x and y values from train_x_apr_df and train_obj_apr_df
    filtered_x.append(x[matching_rows_mask])
    filtered_y.append(y[matching_rows_mask])
    # Convert filtered lists back to tensors
    filtered_x = torch.cat(filtered_x, dim=0)
    filtered_y = torch.cat(filtered_y, dim=0)

    return filtered_x, filtered_y


def generate_initial_data_AMBO(discrete_part, bounds, n):
    # generate reference point
    train_obj = torch.empty(n, 2, **tkwargs)
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    train_x_complete = torch.empty(n, dim, **tkwargs)
    for i in trange(n, desc='Initializing', unit="Candidate"):
        sub_level_x = torch.cat((discrete_part.clone(), torch.tensor([0, 0], **tkwargs)))
        index = (discrete_part == 1).cpu().numpy().astype(bool).tolist() + [True, True] if discrete_part.is_cuda else (
                                                                                                                                  discrete_part == 1).numpy().astype(
            bool).tolist() + [True, True]
        sub_level_x[index] = train_x[i]
        train_x_complete[i] = torch.cat([discrete_part, sub_level_x], dim=0)
        train_obj[i] = torch.as_tensor(eval_problem(train_x_complete[i]), **tkwargs)

    return train_x, train_obj


def initialize_model_AMBO(train_x, train_obj, NOISE_SE, bounds):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds)

    models = []
    # normalized_ref_point = torch.zeros(train_obj.shape[-1], **tkwargs)
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i: i + 1]
        normalized_train_y = (train_y - torch.mean(train_y)) / torch.std(train_y)
        # train_y = normalized_train_y[..., i: i + 1]
        if not Noise_flag or i == 1:
            train_yvar = torch.full_like(normalized_train_y, 1e-6)
            models.append(
                FixedNoiseGP(
                    train_x, normalized_train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            )
        else:
            models.append(
                SingleTaskGP(
                    train_x, normalized_train_y, outcome_transform=Standardize(m=1)
                )
            )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_qnehvi_and_get_observation(discrete_part, ref_point, model, train_x, train_obj, sampler, bounds):
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=Batch_size,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    new_obj_true = torch.empty(Batch_size, 2, **tkwargs)
    for i in range(Batch_size):
        sub_level_x = torch.cat((discrete_part.clone(), torch.tensor([0, 0], **tkwargs)))
        index = (discrete_part == 1).cpu().numpy().astype(bool).tolist() + [True, True] if discrete_part.is_cuda else (
                                                                                                                              discrete_part == 1).numpy().astype(
            bool).tolist() + [True, True]
        sub_level_x[index] = train_x[i]
        x = torch.cat([discrete_part, sub_level_x], dim=0)
        new_obj_true[i] = torch.as_tensor(eval_problem(x), **tkwargs)

    new_obj = new_obj_true
    return new_x, new_obj, new_obj_true


def AMBO_optimization(discrete_x, continuous_x, obj, Iteration, index):
    dim_AMBO = torch.sum(discrete_x).numpy().astype(int) + 2 if not discrete_x.is_cuda else torch.sum(
        discrete_x).cpu().numpy().astype(int) + 2
    bound_AMBO = torch.zeros(2, dim_AMBO, **tkwargs)
    bound_AMBO[1] = 1
    continuous_part = continuous_x.clone()
    extract_idx = (discrete_x == 1).cpu().numpy().astype(bool).tolist() + [True, True] if discrete_x.is_cuda else (
                                                                                                                              discrete_x == 1).numpy().astype(
        bool).tolist() + [True, True]
    continuous_x = continuous_x[:, extract_idx]
    if continuous_x.shape[0] < 20:
        inital_x, initial_y = generate_initial_data_AMBO(discrete_x, bound_AMBO,
                                                         20 - continuous_x.shape[0] if not SMOKE_TEST else 2)

        continuous_x = torch.cat([continuous_x, inital_x])
        obj = torch.cat([obj, initial_y])

    for iter in trange(Iteration, desc=f'Sub-optimizing_{index}', unit="Iteration"):
        mll_qnehvi, model_qnehvi = initialize_model_AMBO(continuous_x, obj, NOISE_SE, bound_AMBO)
        ref_point = update_refpoint(standardize(obj))
        fit_gpytorch_mll(mll_qnehvi)
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        (
            new_x_qnehvi,
            new_obj_qnehvi,
            new_obj_true_qnehvi,
        ) = optimize_qnehvi_and_get_observation(discrete_x, ref_point, model_qnehvi, continuous_x, obj, qnehvi_sampler
                                                , bound_AMBO)
        continuous_x = torch.cat([continuous_x, new_x_qnehvi])
        obj = torch.cat([obj, new_obj_qnehvi])

    # Repeat the discrete_x across the rows to match the shape of continuous_x
    discrete_x_repeated = discrete_x.repeat(continuous_x.shape[0], 1)
    complete_x = torch.zeros([1, continuous_part.shape[1]], **tkwargs).repeat(continuous_x.shape[0], 1) # Start with a copy of discrete_x, so 1s stay in place
    for i in range(complete_x.shape[0]):
        complete_x[i][extract_idx] = continuous_x[i]
    # Concatenate along the feature dimension (axis=1)
    combined_x = torch.cat((discrete_x_repeated, complete_x), dim=1)

    return combined_x, obj
