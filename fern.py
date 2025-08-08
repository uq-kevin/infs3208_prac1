import torch
import numpy as np
import matplotlib.pyplot as plt

def barnsley_fern(
    n_points=100000,
    device=None,
    color='lime',
    c_background='black',
    a1=0., b1=0., c1=0., d1=0.16, e1=0., f1=0.,
    a2=0.85, b2=0.04, c2=-0.04, d2=0.85, e2=0., f2=1.6,
    a3=0.2, b3=-0.26, c3=0.23, d3=0.22, e3=0., f3=1.6,
    a4=-0.15, b4=0.28, c4=0.26, d4=0.24, e4=0., f4=0.44
):
    """
    Generates Barnsley's fern using vectorised PyTorch operations.
    Coefficients default to classic Barnsley's fern but are adjustable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probabilities from Wikipedia
    probs = torch.tensor([0.01, 0.86, 0.93, 1.0], device=device)

    # Allocate tensors
    x = torch.zeros(n_points, device=device)
    y = torch.zeros(n_points, device=device)

    # We'll do sequential dependency in parallel batches
    for i in range(1, n_points):
        r = torch.rand(1, device=device)
        if r < probs[0]:
            x[i] = a1 * x[i-1] + b1 * y[i-1] + e1
            y[i] = c1 * x[i-1] + d1 * y[i-1] + f1
        elif r < probs[1]:
            x[i] = a2 * x[i-1] + b2 * y[i-1] + e2
            y[i] = c2 * x[i-1] + d2 * y[i-1] + f2
        elif r < probs[2]:
            x[i] = a3 * x[i-1] + b3 * y[i-1] + e3
            y[i] = c3 * x[i-1] + d3 * y[i-1] + f3
        else:
            x[i] = a4 * x[i-1] + b4 * y[i-1] + e4
            y[i] = c4 * x[i-1] + d4 * y[i-1] + f4

    # Convert to NumPy for matplotlib
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Plot
    plt.figure(figsize=(6, 10))
    plt.style.use('dark_background' if c_background == 'black' else 'default')
    plt.scatter(x_np, y_np, s=0.2, color=color, marker='.')
    plt.axis('off')
    plt.show()


# Example usage:
barnsley_fern(
    n_points=200000,
    color='lime',
    c_background='black'
)
