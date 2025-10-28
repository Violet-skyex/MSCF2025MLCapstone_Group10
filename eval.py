import numpy as np
import iisignature
import matplotlib.pyplot as plt

# plot the seed path and future simulations from ground truth process
def plot_true_series(hist_price, hist_logret, fut_prices, fut_logrets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    # --- Prices ---
    axes[0].plot(hist_price, color='black', label='realized')
    lines_p = axes[0].plot(len(hist_price) + np.arange(fut_prices.shape[-1]), fut_prices[:5].T, color='lightgrey')
    lines_p[0].set_label('future')
    for ln in lines_p[1:]: ln.set_label('_nolegend_')
    axes[0].axvline(len(hist_price), linestyle='--', label='today')
    axes[0].set_title("Stock Price Process")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("stock price")
    axes[0].legend()

    # --- Log returns ---
    axes[1].plot(hist_logret, color='black', label='realized')
    lines_r = axes[1].plot(len(hist_logret) + np.arange(fut_logrets.shape[-1]), fut_logrets[:5].T, color='lightgrey')
    lines_r[0].set_label('future')
    for ln in lines_r[1:]: ln.set_label('_nolegend_')
    axes[1].axvline(len(hist_logret), linestyle='--', label='today')  # use len(hist_logret) here
    axes[1].set_title("Log Return Process")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("log return")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# compare simulations and generations
def plot_series_compare(fut_logrets, fut_prices, logret_sims, price_sims):
    # hist_price, fut_logrets, fut_prices are macro variables

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: log returns ---
    axes[0].plot(np.arange(fut_logrets.shape[-1]), fut_logrets.T, color = 'grey')
    axes[0].plot(np.arange(logret_sims.shape[-1]), logret_sims.T, color = 'blue')
    axes[0].set_title("log returns")

    axes[1].plot(np.arange(fut_prices.shape[-1]), fut_prices.T, color = 'grey')
    axes[1].plot(np.arange(price_sims.shape[-1]), price_sims.T, color = 'blue')
    axes[1].set_title("prices")

    plt.tight_layout()
    plt.show()

def plot_dist_compare(fut_logrets, logret_gen):
    gt = np.ravel(fut_logrets)
    gen = np.ravel(logret_gen)

    all_data = np.concatenate([gt, gen])
    bins = np.histogram_bin_edges(all_data, bins="fd")

    def ecdf(x):
        x = np.sort(x)
        y = np.linspace(0, 1, len(x), endpoint=False)
        return x, y

    x1, y1 = ecdf(gt)
    x2, y2 = ecdf(gen)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # (1) histogram
    axes[0].hist(gt, bins=bins, density=True, alpha=0.5, label="ground truth")
    axes[0].hist(gen, bins=bins, density=True, alpha=0.5, label="generated")
    axes[0].set_xlabel("log return")
    axes[0].set_ylabel("density")
    axes[0].set_title("Histogram")
    axes[0].legend(frameon=False)

    # (2) ECDF
    axes[1].step(x1, y1, where="post", label="True ECDF")
    axes[1].step(x2, y2, where="post", label="Generated ECDF")
    axes[1].set_xlabel("log return")
    axes[1].set_ylabel("F(x)")
    axes[1].set_title("ECDFs")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.show()

def get_path_signatures(ts, paths, window_size, sig_level):
    """
    Calculates signatures for all sliding windows of a collection of paths.

    inputs:
    - ts: list or numpy array, (n_step)
    - paths: list or numpy array (n_sample, n_step)
    - window_size: int
    - sig_level: truncated level for signature calculation, int

    return:
    - all_signatures: numpy array, (num_windows, sig_dim)
    """
    all_signatures = []
    for path in paths:
        current_ts = ts[:len(path)] # allow path to differ in length
        path_2d = np.stack([current_ts, path], axis=1) # (n_step, 2)
        # TODO: Why are we stacking time metrics into the path??
        for i in range(len(path) - window_size + 1):
            window = path_2d[i : i + window_size] # (window_size, 2)
            sig = iisignature.sig(window, sig_level)
            all_signatures.append(sig)
    return np.array(all_signatures)

def mmd_signature_linear(ts, paths1, paths2, window_size, sig_level):
    """
    Calculates the MMD between two collections of paths using the linear signature kernel.
    This is the squared Euclidean distance between the mean signatures.

    inputs:
    - ts: (n_step)
    - paths1, paths2: (n_sample, n_step)
    
    return:
    - mmd_sq: double
    """
    sigs1 = get_path_signatures(ts, paths1, window_size, sig_level) # (num_windows, sig_dim)
    sigs2 = get_path_signatures(ts, paths2, window_size, sig_level)

    if sigs1.shape[0] == 0 or sigs2.shape[0] == 0:
        return 0.0

    mean_sig1 = np.mean(sigs1, axis=0)
    mean_sig2 = np.mean(sigs2, axis=0)

    mmd_sq = np.sum((mean_sig1 - mean_sig2)**2)
    return mmd_sq