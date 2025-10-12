import numpy as np

def prices_to_logreturns(prices, axis=1, check_positive=True):
    """
    Convert price paths to per-step log-returns: log(S_{t+1} / S_t).

    Parameters
    ----------
    prices : ndarray, shape (N, T) or (T,) or any array with time along `axis`
        Price series/paths; works for a single path or many paths.
    axis : int, default 1
        Axis corresponding to time (use 0 if your array is (T, N)).
    check_positive : bool, default True
        If True, raises ValueError if any price used in a ratio is <= 0.

    Returns
    -------
    logrets : ndarray
        Same shape as `prices` but with size along `axis` reduced by 1.
        Example: (N, T) -> (N, T-1).
    """
    p = np.asarray(prices)
    # Move time to last axis for simple slicing
    p = np.moveaxis(p, axis, -1)

    if p.shape[-1] < 2:
        raise ValueError("Need at least two time points to compute returns.")

    p_t   = p[..., :-1]
    p_t1  = p[...,  1:]

    if check_positive and (np.any(p_t <= 0) or np.any(p_t1 <= 0)):
        raise ValueError("Nonpositive prices encountered; log-returns undefined. "
                         "Set check_positive=False to proceed (will produce inf/NaN).")

    logrets = np.log(p_t1) - np.log(p_t)

    # Move time axis back to requested position
    return np.moveaxis(logrets, -1, axis)


def gbm(N, T, mu, sigma, S0=100.0, dt=1.0, seed=None):
    """
    Simulate Geometric Brownian Motion (GBM) price paths.

    dS_t = μ S_t dt + σ S_t dW_t
    ⇒ log-returns per step ~ Normal((μ - 0.5 σ^2) dt,  (σ^2 dt))

    Parameters
    ----------
    N : int
        Number of independent sample paths.
    T : int
        Number of time steps (periods) to simulate.
    mu : float or array-like of shape (T,)
        Drift per unit time. Scalar or time-varying (same for all paths).
    sigma : float or array-like of shape (T,)
        Volatility per unit time. Scalar or time-varying (same for all paths).
    S0 : float, default 100.0
        Initial price at time 0.
    dt : float, default 1.0
        Time step size.
    seed : int or None
        Random seed for reproducibility (np.random.default_rng).
    return_logrets : bool, default False
        If True, also returns the matrix of log-returns with shape (N, T).

    Returns
    -------
    prices : ndarray, shape (N, T+1)
        Simulated prices. Column 0 is S0.
    """
    rng = np.random.default_rng(seed)

    # Ensure mu and sigma are arrays shaped for broadcasting across N paths
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    if mu.ndim == 0:
        mu = np.full(T, float(mu))
    if sigma.ndim == 0:
        sigma = np.full(T, float(sigma))
    if mu.shape != (T,) or sigma.shape != (T,):
        raise ValueError("When not scalars, mu and sigma must have shape (T,)")

    Z = rng.standard_normal((N, T))

    # Per-step log-returns
    drift = (mu - 0.5 * sigma**2) * dt                  # shape (T,)
    diffu  = (sigma * np.sqrt(dt)) * Z                   # shape (N, T)
    logrets = drift[None, :] + diffu                     # broadcast to (N, T)

    # Prices: S_t = S0 * exp(cumsum(log-returns))
    cumlog = np.cumsum(logrets, axis=1)                  # (N, T)
    prices = S0 * np.exp(np.hstack([np.zeros((N,1)), cumlog]))  # (N, T+1)

    return prices


def cev(
    N, T, mu, sigma, beta, S0=100.0, dt=1.0,
    scheme="milstein",                  # "milstein" (default) or "euler"
    boundary="truncate",                # "truncate", "absorb", or "reflect"
    seed=None
):
    """
    Simulate Constant-Elasticity-of-Variance (CEV) prices ONLY.

        dS_t = μ_t S_t dt + σ_t S_t^{β_t} dW_t

    Parameters
    ----------
    N : int
        Number of paths.
    T : int
        Number of time steps.
    mu, sigma, beta : float or array-like of shape (T,)
        Time-step parameters (scalars broadcast to length T).
    S0 : float
        Initial price at t=0.
    dt : float
        Time step size.
    scheme : {"milstein","euler"}
        Discretization (Milstein recommended for state-dependent diffusion).
    boundary : {"truncate","absorb","reflect"}
        Handling when simulated step would produce nonpositive S.
    seed : int or None
        RNG seed.

    Returns
    -------
    prices : ndarray, shape (N, T+1)
        Price paths with column 0 equal to S0.
    """
    rng = np.random.default_rng(seed)

    def _as_T(x, name):
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full(T, float(x))
        if x.shape != (T,):
            raise ValueError(f"When not scalar, {name} must have shape (T,)")
        return x

    mu_t    = _as_T(mu,    "mu")
    sigma_t = _as_T(sigma, "sigma")
    beta_t  = _as_T(beta,  "beta")

    Z = rng.standard_normal((N, T))
    prices = np.empty((N, T + 1), dtype=float)
    S = np.full(N, float(S0))
    prices[:, 0] = S

    sqrt_dt = np.sqrt(dt)
    eps = 1e-12

    for t in range(T):
        mu_, sig_, b_ = mu_t[t], sigma_t[t], beta_t[t]
        S_pos = np.maximum(S, 0.0)

        if scheme.lower() == "milstein":
            # S_{t+1} = S + μ S dt + σ S^β sqrt(dt) Z + 0.5 σ^2 β S^{2β-1} dt (Z^2 - 1)
            diff = sig_ * (S_pos ** b_)
            S_pow = np.where(S_pos > 0, S_pos ** (2.0 * b_ - 1.0), 0.0)
            dS = (mu_ * S_pos) * dt \
                 + diff * sqrt_dt * Z[:, t] \
                 + 0.5 * (sig_ ** 2) * b_ * S_pow * dt * (Z[:, t] ** 2 - 1.0)
            S_new = S + dS
        elif scheme.lower() == "euler":
            # S_{t+1} = S + μ S dt + σ (S^+)^β sqrt(dt) Z
            diff = sig_ * (S_pos ** b_)
            S_new = S + (mu_ * S_pos) * dt + diff * sqrt_dt * Z[:, t]
        else:
            raise ValueError("scheme must be 'milstein' or 'euler'")

        if boundary == "truncate":
            S = np.maximum(S_new, eps)
        elif boundary == "absorb":
            S = np.where(S_new <= 0.0, 0.0, S_new)
        elif boundary == "reflect":
            S = np.where(S_new >= 0.0, S_new, -S_new)
        else:
            raise ValueError("boundary must be 'truncate', 'absorb', or 'reflect'")

        prices[:, t + 1] = S

    return prices

def merton(
    N, T, mu, sigma, lamb, mJ, sJ,
    S0=100.0, dt=1.0, adjust_drift=True, seed=None
):
    """
    Simulate prices under the Merton jump-diffusion model.

        dS_t / S_t = μ dt + σ dW_t + (J - 1) dN_t,
        with  ln J ~ Normal(mJ, sJ^2),  N_t ~ Poisson(λ t)

    Discrete-time exact step for log-price:
        log(S_{t+1}/S_t) = (μ - 0.5 σ^2 - λ κ) Δt + σ √Δt Z  +  sum_{i=1}^{K_t} Y_i
    where Y_i ~ Normal(mJ, sJ^2), K_t ~ Poisson(λ Δt), and κ = E[J-1] = exp(mJ + 0.5 sJ^2) - 1.
    Using Normal additivity, sum Y_i | K_t = k  ~ Normal(k mJ, k sJ^2).

    Parameters
    ----------
    N : int
        Number of paths.
    T : int
        Number of time steps.
    mu : float or array-like (T,)
        Drift per unit time.
    sigma : float or array-like (T,)
        Diffusion volatility per unit time (>=0).
    lamb : float or array-like (T,)
        Jump intensity λ per unit time (>=0).
    mJ : float or array-like (T,)
        Mean of jump log-size (ln J).
    sJ : float or array-like (T,)
        Std dev of jump log-size (ln J) (>=0).
    S0 : float, default 100.0
        Initial price.
    dt : float, default 1.0
        Time step size.
    adjust_drift : bool, default True
        If True, subtract λ κ so that E[exp(log-return)] matches the GBM part (common in
        risk-neutral pricing or to keep unconditional drift at μ).
    seed : int or None
        RNG seed.

    Returns
    -------
    prices : ndarray, shape (N, T+1)
        Simulated price paths; column 0 equals S0.
    """
    rng = np.random.default_rng(seed)

    def _as_T(x, name):
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full(T, float(x))
        if x.shape != (T,):
            raise ValueError(f"When not scalar, {name} must have shape (T,)")
        return x

    mu_t    = _as_T(mu,    "mu")
    sigma_t = _as_T(sigma, "sigma")
    lamb_t  = _as_T(lamb,  "lamb")
    mJ_t    = _as_T(mJ,    "mJ")
    sJ_t    = _as_T(sJ,    "sJ")

    # Diffusive shocks
    Z = rng.standard_normal((N, T))

    # Jump counts per step: K_t ~ Poisson(λ Δt)
    K = rng.poisson(lam=(lamb_t * dt)[None, :], size=(N, T))  # shape (N, T)

    # Sum of jump log-sizes given K: Normal(K*mJ, K*sJ^2)
    Zj = rng.standard_normal((N, T))
    sqrtK = np.sqrt(K)  # 0 when K=0
    jump_log = (K * mJ_t[None, :]) + (sqrtK * sJ_t[None, :] * Zj)  # (N, T)

    # Drift adjustment κ = E[J-1] = exp(mJ + 0.5 sJ^2) - 1
    kappa_t = np.exp(mJ_t + 0.5 * (sJ_t ** 2)) - 1.0
    drift_adj = (mu_t - 0.5 * sigma_t**2 - (lamb_t * kappa_t if adjust_drift else 0.0)) * dt

    # Per-step log-returns
    logrets = drift_adj[None, :] + (sigma_t[None, :] * np.sqrt(dt) * Z) + jump_log  # (N, T)

    # Prices: S_t = S0 * exp(cumsum(logrets))
    cumlog = np.cumsum(logrets, axis=1)  # (N, T)
    prices = S0 * np.exp(np.hstack([np.zeros((N, 1)), cumlog]))

    return prices

def variance_gamma(
    N, T, dt, S0 = 100.0,
    r = 0.0,            # risk-free rate
    q = 0.0,            # dividend yield
    theta = -0.1,       # VG asymmetry
    sigma = 0.2,        # VG diffusion scale (>0)
    nu = 0.2,           # VG variance of subordinator (>0)
    seed = None,
    martingale = True,   # apply omega so E[S_t]=S0*exp((r-q)t)
):
    """
    Simulate stock paths under the Variance–Gamma model via gamma subordination.

    Increments:
        G ~ Gamma(shape=dt/nu, scale=nu)   # mean=dt, var=dt*nu
        Z ~ N(0,1)
        dX = theta*G + sigma*sqrt(G)*Z

    Price:
        log S_{t+dt} = log S_t + (r-q + omega)*dt + dX,
        where omega = (1/nu) * log(1 - theta*nu - 0.5*sigma^2*nu) if martingale else 0.

    Returns
    -------
    S : (N, T+1) simulated prices
    X : (N, T+1) VG log-return process with X[:,0]=0
    """
    if sigma <= 0 or nu <= 0:
        raise ValueError("sigma and nu must be > 0")

    rng = np.random.default_rng(seed)
    shape = dt / nu
    scale = nu

    # martingale correction for risk-neutral measure
    omega = (1.0/nu) * np.log(1.0 - theta*nu - 0.5*sigma*sigma*nu) if martingale else 0.0
    drift = (r - q + omega) * dt

    S = np.empty((N, T+1), dtype=float)
    X = np.empty((N, T+1), dtype=float)
    S[:, 0] = S0
    X[:, 0] = 0.0

    # random numbers
    G = rng.gamma(shape=shape, scale=scale, size=(N, T))
    Z = rng.standard_normal((N, T))

    for t in range(T):
        dX = theta * G[:, t] + sigma * np.sqrt(G[:, t]) * Z[:, t]
        X[:, t+1] = X[:, t] + dX
        S[:, t+1] = S[:, t] * np.exp(drift + dX)

    return S


def heston(
    N, T, dt,
    mu=0.0,                 # drift of S (set r - q for risk-neutral)
    kappa=2.0,              # mean-reversion speed
    theta=0.04,             # long-run variance level (>=0)
    xi=0.5,                 # vol of vol (>=0)
    rho=-0.7,               # corr(dW^S, dW^v) in [-1,1]
    S0 = 100.0,      # initial price (>0)
    v0 = None,# initial variance (>=0); default = theta
    seed = None,
    clip_eps = 1e-12 # tiny floor to avoid log/exp issues
):
    """
    Simulate Heston (1993) paths:
        dS_t = mu*S_t dt + sqrt(v_t) * S_t dW^S_t
        dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW^v_t
    with corr(dW^S_t, dW^v_t) = rho.

    Scheme:
      - Variance: Full-Truncation Euler (FTE): use max(v_t,0) in drift & diffusion, then floor at 0.
      - Price: Log-Euler with same truncated variance, which prevents spurious drift and “straight lines”.

    Args
    ----
    N   : number of paths
    T   : number of time steps
    dt  : time step size (e.g., 1/252)
    mu, kappa, theta, xi, rho: floats or arrays of shape (T,) for time-varying params
    S0  : initial price
    v0  : initial variance (defaults to theta at t=0 if None)
    antithetic : if True, uses antithetic normals (N must be even)
    seed : PRNG seed
    clip_eps : numerical floor for S and v (post step)

    Returns
    -------
    S : (N, T+1) array of prices
    v : (N, T+1) array of variances
    """
    rng = np.random.default_rng(seed)

    # Broadcast parameters to shape (T,)
    def _as_tseries(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            arr = np.full(T, float(arr))
        elif arr.shape != (T,):
            raise ValueError(f"{name} must be scalar or shape (T,), got {arr.shape}")
        return arr

    mu_t    = _as_tseries(mu,    "mu")
    kappa_t = _as_tseries(kappa, "kappa")
    theta_t = _as_tseries(theta, "theta")
    xi_t    = _as_tseries(xi,    "xi")
    rho_t   = _as_tseries(rho,   "rho")

    if v0 is None:
        v0 = float(theta_t[0])
    v0 = max(float(v0), 0.0)

    # Allocate paths
    S = np.empty((N, T+1), dtype=float)
    v = np.empty((N, T+1), dtype=float)
    S[:, 0] = float(S0)
    v[:, 0] = v0

    sqrt_dt = np.sqrt(dt)

    # Draw correlated Brownian shocks
    # Z2 drives variance; Z1 is independent; combine to get corr(rho) for price
    Z1 = rng.standard_normal((N, T))
    Z2 = rng.standard_normal((N, T))

    for t in range(T):
        rho_t_clipped = np.clip(rho_t[t], -0.999999, 0.999999)  # robust
        sqrt_1m_rho2 = np.sqrt(1.0 - rho_t_clipped**2)

        # Build correlated increments
        dWv = Z2[:, t] * sqrt_dt
        dWs_indep = Z1[:, t] * sqrt_dt
        dWs = rho_t_clipped * dWv + sqrt_1m_rho2 * dWs_indep  # corr(dWs,dWv)=rho

        # Full-Truncation Euler for v
        v_pos = np.maximum(v[:, t], 0.0)                       # truncate
        v_next = (
            v[:, t]
            + kappa_t[t] * (theta_t[t] - v_pos) * dt
            + xi_t[t] * np.sqrt(v_pos) * dWv
        )
        v_next = np.maximum(v_next, 0.0)                       # keep non-negative

        # Log-Euler for S with the same truncated variance
        drift = (mu_t[t] - 0.5 * v_pos) * dt
        diff  = np.sqrt(v_pos) * dWs
        S_next = S[:, t] * np.exp(drift + diff)

        # Numerical hygiene
        S[:, t+1] = np.maximum(S_next, clip_eps)
        v[:, t+1] = np.maximum(v_next, 0.0)

    return S



def garch_ret(
    N, T,
    omega, alpha, beta,                 # GARCH(1,1): sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2
    mu=0.0, phi=0.0,                    # mean model: r_t = mu + phi*r_{t-1} + sigma_t * z_t  (phi=0 => constant mean)
    dist="normal", nu=8,                # innovations: "normal" or "t" (standardized)
    S0=100.0, v0=None, seed=None
):
    """
    Generate price paths whose (log-)returns follow a GARCH(1,1).

    Parameters
    ----------
    N : int
        Number of paths.
    T : int
        Number of periods.
    omega, alpha, beta : float
        GARCH parameters (require omega>0, alpha>=0, beta>=0, alpha+beta<1 for stationarity).
    mu : float, default 0.0
        Mean return intercept.
    phi : float, default 0.0
        AR(1) coefficient in mean. Set to 0 for constant mean.
    dist : {"normal","t"}, default "normal"
        Shock distribution. For "t", uses standardized Student-t with df=nu (unit variance).
    nu : int, default 8
        Degrees of freedom for t distribution (nu>2).
    S0 : float, default 100.0
        Initial price.
    v0 : float or None
        Initial variance. If None, uses unconditional variance omega/(1-alpha-beta).
    seed : int or None
        RNG seed.

    Returns
    -------
    prices : ndarray, shape (N, T+1)
        Price paths with column 0 equal to S0.
    """
    if not (omega > 0 and alpha >= 0 and beta >= 0):
        raise ValueError("Require omega>0, alpha>=0, beta>=0.")
    if alpha + beta >= 1:
        raise ValueError("Stationarity requires alpha + beta < 1.")
    if dist == "t" and nu <= 2:
        raise ValueError("For t distribution, require nu > 2 (finite variance).")

    rng = np.random.default_rng(seed)

    # Draw shocks z_t with unit variance
    if dist.lower() == "normal":
        Z = rng.standard_normal((N, T))
    elif dist.lower() == "t":
        # Standardize t_nu to unit variance: z = t / sqrt(nu/(nu-2))
        t_raw = rng.standard_t(df=nu, size=(N, T))
        Z = t_raw / np.sqrt(nu / (nu - 2))
    else:
        raise ValueError("dist must be 'normal' or 't'.")

    # Allocate arrays
    prices = np.empty((N, T + 1), dtype=float)
    r = np.empty((N, T), dtype=float)
    sigma2 = np.empty((N, T), dtype=float)

    # Initial conditions
    var_uncond = omega / (1.0 - alpha - beta)
    sigma2_0 = var_uncond if v0 is None else float(v0)
    r_prev = np.zeros(N)                 # r_{0}
    sigma2_prev = np.full(N, sigma2_0)   # sigma_{0}^2

    prices[:, 0] = S0
    S = np.full(N, float(S0))

    # Recursion
    for t in range(T):
        sigma_t = np.sqrt(sigma2_prev)
        r_t = mu + phi * r_prev + sigma_t * Z[:, t]  # log-return at step t
        r[:, t] = r_t

        # Update variance for next step
        sigma2_t = omega + alpha * (r_prev ** 2) + beta * sigma2_prev
        sigma2[:, t] = sigma2_t

        # Price update (log-price accumulation)
        S = S * np.exp(r_t)
        prices[:, t + 1] = S

        # Shift state
        r_prev = r_t
        sigma2_prev = sigma2_t

    return prices


