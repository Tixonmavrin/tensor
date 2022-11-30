from tensor import Tensor

if __name__ == "__main__":
    # First example
    # You can see simple operation here
    print("First example")

    t = Tensor([1.0, 2.0, 3.0])
    print("Init tensor:", t)

    t_squared = t**2
    print("Tensor squared:", t_squared, "\n")

    # Second example
    # You can see grad usage here
    print("Second example")

    t = Tensor([1.0, 2.0, -1.0], requires_grad=True)
    print("Init tensor with grad:", t)

    t_squared = t**2
    print("Tensor squared:", t_squared)

    t_squared_sum = t_squared.sum()
    print("Tensor squared sum:", t_squared_sum)

    # Calculate gradients
    t_squared_sum.backward()

    t_grad = t.grad
    print("Tensor grad after backward:", t_grad)

    print("Success")
