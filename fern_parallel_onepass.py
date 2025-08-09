import torch
import matplotlib.pyplot as plt
import numpy as np

def barnsley_fern_one_pass(
    n_chains=1_000_000,
    burn_in=20, 
    collect_steps=5,
    device=None,
    seed=None,
    color='lime',
    background='black',
    point_size=0.2,
    figsize=(6,10),
    dpi=150,
    coeffs=None
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if coeffs is None:
        coeffs = dict(
            a1=0.,    b1=0.,     c1=0.,     d1=0.16,  e1=0.,    f1=0.,
            a2=0.85,  b2=0.04,   c2=-0.04,  d2=0.85,  e2=0.,    f2=1.6,
            a3=0.2,   b3=-0.26,  c3=0.23,   d3=0.22,  e3=0.,    f3=1.6,
            a4=-0.15, b4=0.28,   c4=0.26,   d4=0.24,  e4=0.,    f4=0.44
        ) # Original Barnsley's Fern matrix

        # coeffs = dict(
        #     a1=0.,    b1=0.,    c1=0.,     d1=0.25, e1=0.,     f1=-0.4,
        #     a2=0.95,  b2=0.005, c2=-0.005, d2=0.93, e2=-0.002, f2=0.5,
        #     a3=0.035, b3=-0.2,  c3=0.16,   d3=0.04, e3=-0.09,  f3=0.02,
        #     a4=-0.04, b4=0.2,   c4=0.16,   d4=0.04, e4=0.083,  f4=0.12
        # ) # Mutant varieties resembling the Cyclosorus or Thelypteridaceae 
        #   # Source: wikipedia

    a = torch.tensor([coeffs[f'a{i}'] for i in range(1,5)], device=device)
    b = torch.tensor([coeffs[f'b{i}'] for i in range(1,5)], device=device)
    c = torch.tensor([coeffs[f'c{i}'] for i in range(1,5)], device=device)
    d = torch.tensor([coeffs[f'd{i}'] for i in range(1,5)], device=device)
    e = torch.tensor([coeffs[f'e{i}'] for i in range(1,5)], device=device)
    f = torch.tensor([coeffs[f'f{i}'] for i in range(1,5)], device=device)

    probs = torch.tensor([0.01, 0.86, 0.93, 1.0], device=device)
    total_steps = burn_in + collect_steps

    # pre-sample transform choices for all chains and steps
    r = torch.rand((n_chains, total_steps), device=device)
    choices = torch.bucketize(r, probs)  # indices 0..3

    # parallel chains start at origin
    x = torch.zeros(n_chains, device=device)
    y = torch.zeros(n_chains, device=device)

    # collect storage
    total_collected = n_chains * collect_steps
    xs_collect = torch.empty(total_collected, device=device)
    ys_collect = torch.empty(total_collected, device=device)

    collect_idx = 0
    for t in range(total_steps):
        idx = choices[:, t]
        at = a[idx]; bt = b[idx]; ct = c[idx]; dt = d[idx]; et = e[idx]; ft = f[idx]

        x_new = at * x + bt * y + et
        y_new = ct * x + dt * y + ft

        x = x_new; y = y_new

        if t >= burn_in:
            end = collect_idx + n_chains
            xs_collect[collect_idx:end] = x
            ys_collect[collect_idx:end] = y
            collect_idx = end

    xs = xs_collect.cpu().numpy()
    ys = ys_collect.cpu().numpy()

    # plot
    plt.figure(figsize=figsize, dpi=dpi)
    if background == 'black':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.scatter(xs, ys, s=point_size, c=color, marker='.', alpha=1.0, linewidths=0)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    return xs, ys

thicker_leaves_coeffs = {'a1':0,'b1':0,'c1':0,'d1':0.16,'e1':0,'f1':0,
          'a2':0.9,'b2':0.03,'c2':-0.03,'d2':0.9,'e2':0,'f2':1.8,
          'a3':0.25,'b3':-0.3,'c3':0.28,'d3':0.26,'e3':0,'f3':1.7,
          'a4':-0.15,'b4':0.28,'c4':0.26,'d4':0.24,'e4':0,'f4':0.44}

barnsley_fern_one_pass(n_chains=1_000_000, coeffs=None, color='red')

