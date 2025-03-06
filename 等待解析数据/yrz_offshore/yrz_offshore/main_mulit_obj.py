import pandas as pd
import torch
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import is_non_dominated
from datetime import datetime
from functions_for_main import *
from algorithm_parameter import *
import time
import warnings
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.transforms import standardize
from SAA_exam import SAA_exam
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt

# from botorch.test_functions.multi_objective import BraninCurrin

# ref_point = torch.tensor([-5, -5], **tkwargs)
# problem = BraninCurrin(negate=True).to(**tkwargs)
hvs_pr, hvs_analytic, hvs_cont_relax = [], [], []

verbose = True

(
    best_observed_all_pr,
    best_observed_all_pr_analytic,
    best_observed_all_cont_relax,
) = ([], [], [])

sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

if __name__ == "__main__":
    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        if not load_option:
            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            # call helper functions to generate initial training data and initialize model
            train_x_pr, train_obj_pr = generate_initial_data(n=2 if SMOKE_TEST else 20)
            # best_observed_value_pr = train_obj_pr.max().item()
            stdized_train_obj_pr = standardize(train_obj_pr)  # z-score normalize
            mll_pr, model_pr = initialize_model(train_x_pr, stdized_train_obj_pr)
            train_x_pr_analytic, train_obj_pr_analytic, stdized_train_obj_pr_analytic = (
                train_x_pr,
                train_obj_pr,
                stdized_train_obj_pr,
            )
            train_x_cont_relax, train_obj_cont_relax, stdized_train_obj_cont_relax = (
                train_x_pr,
                train_obj_pr,
                stdized_train_obj_pr,
            )

            if algorithm_option[2]:
                # best_observed_value_pr_analytic = best_observed_value_pr
                mll_pr_analytic, model_pr_analytic = initialize_model(
                    train_x_pr_analytic,
                    stdized_train_obj_pr_analytic,
                )

            if algorithm_option[0]:
                # best_observed_value_cont_relax = best_observed_value_pr
                mll_cont_relax, model_cont_relax = initialize_model(
                    train_x_cont_relax,
                    stdized_train_obj_cont_relax,
                )

            ref_point_pr = update_refpoint(stdized_train_obj_pr)
            ref_point_apr = ref_point_pr

            # compute hypervolume
            bd = DominatedPartitioning(ref_point=ref_point_pr, Y=stdized_train_obj_pr)
            volume = bd.compute_hypervolume().item()

            hvs_pr.append(volume)
            hvs_analytic.append(volume)
            hvs_cont_relax.append(volume)

            # run N_Iteration rounds of BayesOpt after the initial random batch
            for iteration in range(1, N_BATCH + 1):

                t0 = time.monotonic()

                # fit the models
                if algorithm_option[0]:
                    fit_gpytorch_mll(mll_cont_relax)
                    with torch.no_grad():
                        pred = model_pr.posterior(normalize(train_x_cont_relax, bounds)).mean
                    partitioning = FastNondominatedPartitioning(
                        ref_point=ref_point_fixed,
                        Y=pred,
                    )
                    ei_cont_relax = qLogExpectedHypervolumeImprovement(
                        model=model_cont_relax,
                        ref_point=ref_point_fixed,
                        partitioning=partitioning,
                        sampler=sampler,
                    )
                    (
                        new_x_cont_relax,
                        new_obj_cont_relax,
                    ) = optimize_acqf_cont_relax_and_get_observation(ei_cont_relax)
                    train_x_cont_relax = torch.cat([train_x_cont_relax, new_x_cont_relax])
                    train_obj_cont_relax = torch.cat([train_obj_cont_relax, new_obj_cont_relax.reshape(Batch_size, 2)])
                    stdized_train_obj_cont_relax = standardize(train_obj_cont_relax)
                    mll_cont_relax, model_cont_relax = initialize_model(
                        train_x_cont_relax,
                        stdized_train_obj_cont_relax,
                    )

                if algorithm_option[1]:
                    fit_gpytorch_mll(mll_pr)
                    # for best_f, we use the best observed values
                    with torch.no_grad():
                        pred = model_pr.posterior(normalize(train_x_pr, bounds)).mean
                    partitioning = FastNondominatedPartitioning(
                        ref_point=ref_point_pr,
                        Y=pred,
                    )
                    ei_pr = qLogExpectedHypervolumeImprovement(
                        model=model_pr,
                        ref_point=ref_point_pr,
                        partitioning=partitioning,
                        sampler=sampler,
                    )
                    # optimize and get new observation
                    new_x_pr, new_obj_pr = optimize_acqf_pr_and_get_observation(
                        ei_pr, analytic=False
                    )
                    train_x_pr = torch.cat([train_x_pr, new_x_pr])
                    train_obj_pr = torch.cat([train_obj_pr, new_obj_pr.reshape(Batch_size, 2)])
                    stdized_train_obj_pr = standardize(train_obj_pr)
                    ref_point_pr = update_refpoint(stdized_train_obj_pr)
                    mll_pr, model_pr = initialize_model(
                        train_x_pr,
                        stdized_train_obj_pr,
                    )

                if algorithm_option[2]:
                    fit_gpytorch_mll(mll_pr_analytic)
                    with torch.no_grad():
                        pred = model_pr.posterior(normalize(train_x_pr_analytic, bounds)).mean
                    partitioning = FastNondominatedPartitioning(
                        ref_point=ref_point_apr,
                        Y=pred,
                    )
                    ei_pr_analytic = qLogExpectedHypervolumeImprovement(
                        model=model_pr_analytic,
                        ref_point=ref_point_apr,
                        partitioning=partitioning,
                        sampler=sampler,
                    )
                    new_x_pr_analytic, new_obj_pr_analytic = optimize_acqf_pr_and_get_observation(
                        ei_pr_analytic, analytic=True
                    )
                    train_x_pr_analytic = torch.cat([train_x_pr_analytic, new_x_pr_analytic])
                    train_obj_pr_analytic = torch.cat(
                        [train_obj_pr_analytic, new_obj_pr_analytic.reshape(Batch_size, 2)])
                    stdized_train_obj_pr_analytic = standardize(train_obj_pr_analytic)
                    ref_point_apr = update_refpoint(stdized_train_obj_pr_analytic)
                    mll_pr_analytic, model_pr_analytic = initialize_model(
                        train_x_pr_analytic,
                        stdized_train_obj_pr_analytic,
                    )

                all_obj = torch.cat([train_obj_pr, train_obj_pr_analytic, train_obj_cont_relax])
                stdized_obj = standardize(all_obj)
                ref_point = update_refpoint(stdized_obj)

                # update progress
                for hvs_list, train_obj in zip(
                        (hvs_pr, hvs_analytic, hvs_cont_relax),
                        (
                                stdized_train_obj_pr,
                                stdized_train_obj_pr_analytic,
                                stdized_train_obj_cont_relax,
                        ),
                ):
                    # compute hypervolume
                    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
                    volume = bd.compute_hypervolume().item()
                    hvs_list.append(volume)

                t1 = time.monotonic()

                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: hypervolume_indicator (Cont. Relax., PR (MC), PR (Analytic)) = "
                        f"({hvs_cont_relax[iteration]:>4.2f}, {hvs_pr[iteration]:>4.2f}, {hvs_analytic[iteration]:>4.2f}), "
                        f"time = {t1 - t0:>4.2f}.",
                        end="\n",
                    )
                else:
                    print(".", end="")
            if reopt_batch == 0:
                if algorithm_option[1]:
                    # Convert tensors to CPU if they're on GPU before calling .numpy()
                    train_x_pr_cpu = train_x_pr.cpu() if train_x_pr.is_cuda else train_x_pr
                    train_x_pr_save = pd.DataFrame(train_x_pr_cpu.numpy(),
                                                   columns=[f'x_pr_{i}' for i in range(train_x_pr.shape[1])])

                    train_obj_pr_cpu = train_obj_pr.cpu() if train_obj_pr.is_cuda else train_obj_pr
                    train_obj_pr_save = pd.DataFrame(train_obj_pr_cpu.numpy(),
                                                     columns=[f'y_pr_{i}' for i in range(
                                                         train_obj_pr.shape[
                                                             1])])  # , 'line_par2', 'line_par3','rate_par'])

                if algorithm_option[2]:
                    # Convert tensors to CPU if they're on GPU before calling .numpy()
                    train_x_pr_analytic_cpu = train_x_pr_analytic.cpu() if train_x_pr_analytic.is_cuda else train_x_pr_analytic
                    train_x_apr_save = pd.DataFrame(train_x_pr_analytic_cpu.numpy(),
                                                    columns=[f'x_apr_{i}' for i in range(train_x_pr_analytic.shape[1])])

                    train_obj_pr_analytic_cpu = train_obj_pr_analytic.cpu() if train_obj_pr_analytic.is_cuda else train_obj_pr_analytic
                    train_obj_apr_save = pd.DataFrame(train_obj_pr_analytic_cpu.numpy(),
                                                      columns=[f'y_apr_{i}' for i in
                                                               range(train_obj_pr_analytic.shape[1])])

                if algorithm_option[0]:
                    # Convert tensors to CPU if they're on GPU before calling .numpy()
                    train_x_cont_relax_cpu = train_x_cont_relax.cpu() if train_x_cont_relax.is_cuda else train_x_cont_relax
                    train_x_cont_relax_save = pd.DataFrame(train_x_cont_relax_cpu.numpy(),
                                                           columns=[f'x_cont_{i}' for i in
                                                                    range(train_x_cont_relax.shape[1])])

                    train_obj_cont_relax_cpu = train_obj_cont_relax.cpu() if train_obj_cont_relax.is_cuda else train_obj_cont_relax
                    train_obj_cont_relax_save = pd.DataFrame(train_obj_cont_relax_cpu.numpy(),
                                                             columns=[f'y_cont_{i}' for i in
                                                                      range(train_obj_cont_relax.shape[1])])

                if sum(algorithm_option) != 3:
                    hp_pr, hp_apr, hp_cont = hp_calculation(y_pr=train_obj_pr_analytic, y_apr=train_obj_pr_analytic,
                                                            y_cont=train_obj_pr_analytic)
                else:
                    hp_pr, hp_apr, hp_cont = hp_calculation(y_pr=train_obj_pr, y_apr=train_obj_pr_analytic,
                                                            y_cont=train_obj_cont_relax)

                max_length = max(len(hp_pr), len(hp_apr), len(hp_cont))
                hp_data = {
                    'hp_pr': np.pad(hp_pr, (0, max_length - len(hp_pr)), constant_values=np.nan),
                    'hp_apr': np.pad(hp_apr, (0, max_length - len(hp_apr)), constant_values=np.nan),
                    'hp_cont': np.pad(hp_cont, (0, max_length - len(hp_cont)), constant_values=np.nan),
                }
                hp_data_save = pd.DataFrame(hp_data)
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
                defined_dataframes.append(hp_data_save)
                # Concatenate the defined dataframes
                result = pd.concat(defined_dataframes, axis=1)

                current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                file_name = f'{current_time}_{ref_rate}.xlsx'

                file_path = os.path.join(folder_name, file_name)
                result.to_excel(file_path, index=False)
        else:
            print("\nLoad existing optimization result\n")
            # Specify the folder where the file is saved
            folder_path = "./Outcome_noisy"

            # List all files in the folder
            files_in_folder = os.listdir(folder_path)

            # Filter for files with .xlsx extension
            excel_files = [f for f in files_in_folder if f.endswith('.xlsx')]

            file_path = os.path.join(folder_path, excel_files[0])

            # Read the Excel file
            result = pd.read_excel(file_path)

            # Assuming the sheet contains the columns as described in your code,
            # and that the names of the variables match the columns in the Excel file.
            # Extract the data for each of the variables from the columns of the dataframe.

            # Read the relevant parts based on the algorithm options
            if algorithm_option[1]:
                # Extract the training data for x_pr and y_pr
                train_x_pr_df = result[[col for col in result.columns if col.startswith('x_pr')]]
                train_obj_pr_df = result[[col for col in result.columns if col.startswith('y_pr')]]
                train_x_pr = torch.tensor(train_x_pr_df.values, **tkwargs)
                train_obj_pr = torch.tensor(train_obj_pr_df.values, **tkwargs)

            if algorithm_option[2]:
                # Extract the training data for x_apr and y_apr
                train_x_apr_df = result[[col for col in result.columns if col.startswith('x_apr')]]
                train_obj_apr_df = result[[col for col in result.columns if col.startswith('y_apr')]]
                train_x_pr_analytic = torch.tensor(train_x_apr_df.values, **tkwargs)
                train_obj_pr_analytic = torch.tensor(train_obj_apr_df.values, **tkwargs)

            if algorithm_option[0]:
                # Extract the training data for x_cont and y_cont
                train_x_cont_relax_df = result[[col for col in result.columns if col.startswith('x_cont')]]
                train_obj_cont_relax_df = result[[col for col in result.columns if col.startswith('y_cont')]]
                train_x_cont_relax = torch.tensor(train_x_cont_relax_df.values, **tkwargs)
                train_obj_cont_relax = torch.tensor(train_obj_cont_relax_df.values, **tkwargs)

            # For the hp_data
            # hp_pr_df = result[['hp_pr']].values.flatten()
            # hp_apr_df = result[['hp_apr']].values.flatten()
            # hp_cont_df = result[['hp_cont']].values.flatten()
            #
            # # Handle missing or padded values (replace NaNs if necessary)
            # hp_pr = torch.tensor(hp_pr_df, dtype=torch.float32)
            # hp_apr = torch.tensor(hp_apr_df, dtype=torch.float32)
            # hp_cont = torch.tensor(hp_cont_df, dtype=torch.float32)

            # Now you can use the tensors as needed in your optimization process.
        dis_dim = len(integer_indices)
        if algorithm_option[0]:
            pareto_mask = is_non_dominated(train_obj_cont_relax)
            # ndt_y_cont = train_obj_cont_relax[pareto_mask]
            ndt_x_cont = train_x_cont_relax[pareto_mask]
        if algorithm_option[1]:
            pareto_mask = is_non_dominated(train_obj_pr)
            # ndt_y_pr = train_obj_pr[pareto_mask]
            ndt_x_pr = train_x_pr[pareto_mask]
        if algorithm_option[2]:
            for i in range(reopt_batch):
                pareto_mask = is_non_dominated(train_obj_pr_analytic)
                ndt_y_apr = train_obj_pr_analytic[pareto_mask]
                ndt_x_apr = train_x_pr_analytic[pareto_mask]
                # dis_ndt_x_apr = ndt_x_apr[:,:dis_dim]
                isolated_ndt_x_apr = select_isolated_non_dominated_points(ndt_x_apr, ndt_y_apr)
                isolated_ndt_x_apr = isolated_ndt_x_apr.squeeze(-1)
                if verbose:
                    print(
                        f"Start Re-opt {i}: Discrete x = {isolated_ndt_x_apr[0:discrete_len]}",
                        end="\n"
                    )
                else:
                    print(".", end="")
                x, y = extract_isolated_points(train_x_pr_analytic, train_obj_pr_analytic, isolated_ndt_x_apr)
                x_apr_new, y_apr_new = AMBO_optimization(isolated_ndt_x_apr[0:discrete_len], x[:, discrete_len:], y,
                                                         Iteration=sub_iteration if not SMOKE_TEST else 2, index=i)
                train_x_pr_analytic = torch.cat([train_x_pr_analytic, x_apr_new])
                train_obj_pr_analytic = torch.cat([train_obj_pr_analytic, y_apr_new])
            if Exam_flag:
                exam_result_pr_analytic = torch.mean(SAA_exam(train_x_pr_analytic, exam_time),1).unsqueeze(-1)
                error_pr_analytic = (exam_result_pr_analytic - train_obj_pr_analytic[:,0].unsqueeze(-1))
                error_pr_analytic_cpu = error_pr_analytic.cpu() if error_pr_analytic.is_cuda else train_x_pr_analytic
                error_pr_analytic_save = pd.DataFrame(error_pr_analytic_cpu.numpy(),columns=[f'error_apr_{i}' for i in range(error_pr_analytic.shape[1])])
            # discrete_combinations = [tuple(row.numpy().astype(int)) for row in
            #                          dis_ndt_x_apr]  # Convert to int if they are 0 or 1
            #
            # # Count the frequency of each combination using Counter
            # counter = Counter(discrete_combinations)
            #
            # # Print the results
            # for combination, count in counter.items():
            #     print(f"Combination: {combination}, Count: {count}")
            if algorithm_option[1]:
                # Convert tensors to CPU if they're on GPU before calling .numpy()
                train_x_pr_cpu = train_x_pr.cpu() if train_x_pr.is_cuda else train_x_pr
                train_x_pr_save = pd.DataFrame(train_x_pr_cpu.numpy(),
                                               columns=[f'x_pr_{i}' for i in range(train_x_pr.shape[1])])

                train_obj_pr_cpu = train_obj_pr.cpu() if train_obj_pr.is_cuda else train_obj_pr
                train_obj_pr_save = pd.DataFrame(train_obj_pr_cpu.numpy(),
                                                 columns=[f'y_pr_{i}' for i in range(
                                                     train_obj_pr.shape[1])])  # , 'line_par2', 'line_par3','rate_par'])

            if algorithm_option[2]:
                # Convert tensors to CPU if they're on GPU before calling .numpy()
                train_x_pr_analytic_cpu = train_x_pr_analytic.cpu() if train_x_pr_analytic.is_cuda else train_x_pr_analytic
                train_x_apr_save = pd.DataFrame(train_x_pr_analytic_cpu.numpy(),
                                                columns=[f'x_apr_{i}' for i in range(train_x_pr_analytic.shape[1])])

                train_obj_pr_analytic_cpu = train_obj_pr_analytic.cpu() if train_obj_pr_analytic.is_cuda else train_obj_pr_analytic
                train_obj_apr_save = pd.DataFrame(train_obj_pr_analytic_cpu.numpy(),
                                                  columns=[f'y_apr_{i}' for i in range(train_obj_pr_analytic.shape[1])])

            if algorithm_option[0]:
                # Convert tensors to CPU if they're on GPU before calling .numpy()
                train_x_cont_relax_cpu = train_x_cont_relax.cpu() if train_x_cont_relax.is_cuda else train_x_cont_relax
                train_x_cont_relax_save = pd.DataFrame(train_x_cont_relax_cpu.numpy(),
                                                       columns=[f'x_cont_{i}' for i in
                                                                range(train_x_cont_relax.shape[1])])

                train_obj_cont_relax_cpu = train_obj_cont_relax.cpu() if train_obj_cont_relax.is_cuda else train_obj_cont_relax
                train_obj_cont_relax_save = pd.DataFrame(train_obj_cont_relax_cpu.numpy(),
                                                         columns=[f'y_cont_{i}' for i in
                                                                  range(train_obj_cont_relax.shape[1])])

            # if sum(algorithm_option) != 3:
            #     hp_pr, hp_apr, hp_cont = hp_calculation(y_pr=train_obj_pr_analytic, y_apr=train_obj_pr_analytic,
            #                                             y_cont=train_obj_pr_analytic)
            # else:
            #     hp_pr, hp_apr, hp_cont = hp_calculation(y_pr=train_obj_pr, y_apr=train_obj_pr_analytic,
            #                                             y_cont=train_obj_cont_relax)
            #
            # max_length = max(len(hp_pr), len(hp_apr), len(hp_cont))
            # hp_data = {
            #     'hp_pr': np.pad(hp_pr, (0, max_length - len(hp_pr)), constant_values=np.nan),
            #     'hp_apr': np.pad(hp_apr, (0, max_length - len(hp_apr)), constant_values=np.nan),
            #     'hp_cont': np.pad(hp_cont, (0, max_length - len(hp_cont)), constant_values=np.nan),
            # }
            # hp_data_save = pd.DataFrame(hp_data)
            defined_dataframes = []
            # List of dataframe names as strings
            dataframe_names = [
                'train_x_pr_save', 'train_x_apr_save','error_pr_analytic_save', 'train_x_cont_relax_save',
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
            # defined_dataframes.append(hp_data_save)
            # Concatenate the defined dataframes
            result = pd.concat(defined_dataframes, axis=1)

            current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            file_name = f'{current_time}_{ref_rate}.xlsx'

            file_path = os.path.join(folder_name, file_name)
            result.to_excel(file_path, index=False)

        #
        # from matplotlib.cm import ScalarMappable
        #
        # fig, axes = plt.subplots(1, 3, figsize=(23, 7), sharex=True, sharey=True)
        # algos = ["Cont. Relax.", "PR (MC)", "PR (Analytic)"]
        # cm = plt.get_cmap("viridis")
        #
        # batch_number = torch.cat(
        #     [
        #         torch.zeros(20),
        #         torch.arange(1, 54 + 1).repeat(1, 1).t().reshape(-1),
        #     ]
        # ).numpy()
        # for i, train_obj in enumerate(
        #         (
        #                 train_obj_cont_relax,
        #                 train_obj_pr,
        #                 train_obj_pr_analytic,
        #         )
        # ):
        #     sc = axes[i].scatter(
        #         train_obj[:, 0].cpu().numpy(),
        #         train_obj[:, 1].cpu().numpy(),
        #         c=batch_number,
        #         alpha=0.8,
        #     )
        #     axes[i].set_title(algos[i])
        #     axes[i].set_xlabel("Objective 1")
        # axes[0].set_ylabel("Objective 2")
        # norm = plt.Normalize(batch_number.min(), batch_number.max())
        # sm = ScalarMappable(norm=norm, cmap=cm)
        # sm.set_array([])
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        # cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar.ax.set_title("Iteration")
        # plt.show()
