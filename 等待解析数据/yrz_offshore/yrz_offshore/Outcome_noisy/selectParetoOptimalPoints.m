function [paretoPoints, nonDominantPoints, index] = selectParetoOptimalPoints(outcomes, problemType)
    % This function selects the Pareto optimal points from a set of outcomes
    % 'outcomes' is a matrix where each row represents a solution and each column an objective
    % 'problemType' is either 'min' or 'max' to specify the optimization problem type

    nSolutions = size(outcomes, 1);
    paretoPoints = true(nSolutions, 1);
    nonDominantPoints = false(nSolutions, 1);

    for i = 1:nSolutions
        for j = 1:nSolutions
            if i ~= j
                if strcmp(problemType, 'min')
                    % Minimization problem
                    if all(outcomes(j, :) <= outcomes(i, :)) && any(outcomes(j, :) < outcomes(i, :))
                        paretoPoints(i) = false;
                        break;
                    end
                elseif strcmp(problemType, 'max')
                    % Maximization problem
                    if all(outcomes(j, :) >= outcomes(i, :)) && any(outcomes(j, :) > outcomes(i, :))
                        paretoPoints(i) = false;
                        break;
                    end
                else
                    error('Invalid problem type. Specify ''min'' or ''max''.');
                end
            end
        end
    end

    nonDominantPoints = outcomes(~paretoPoints, :);
    index = paretoPoints;
    paretoPoints = outcomes(paretoPoints, :);
end
