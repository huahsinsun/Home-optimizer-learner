import torch


def fill_discrete_x_with_continuous(discrete_x, x_generated):
    # Get the number of discrete variables (1s in discrete_x)
    dim_AMBO = torch.sum(discrete_x).numpy().astype(int)

    # Ensure that the length of x_generated matches the number of 1s in discrete_x
    if len(x_generated) != dim_AMBO:
        raise ValueError("The number of continuous variables does not match the number of 1s in discrete_x.")

    # Create a new tensor that has the same shape as discrete_x
    sub_level_x = discrete_x.clone().float()  # Start with a copy of discrete_x, so 1s stay in place
    sub_level_x[discrete_x == 1] = x_generated # Replace 1s with x_generated values

    return sub_level_x


# Example
discrete_x = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float64)
x_generated = [0.5, 0.4]

# Call the function
sub_level_x = fill_discrete_x_with_continuous(discrete_x, x_generated)

print("Discrete x:", discrete_x)
print("Generated continuous variables:", x_generated)
print("Sub level input:", sub_level_x)
