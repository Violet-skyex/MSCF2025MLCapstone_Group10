import numpy as np
import iisignature
#import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tqdm import trange
from typing import Union

def logrets_to_prices(logrets: np.ndarray, s0: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert log-returns r_t = log(S_t / S_{t-1}) into prices given starting price(s) s0.

    Parameters
    ----------
    logrets : np.ndarray
        Array of log-returns with shape (..., T) where the last axis is time.
        Works for 1D (T,) or batched, e.g. (n_paths, T).
    s0 : float or np.ndarray
        Starting price(s). Must be > 0.
        If array, its shape must broadcast to logrets.shape[:-1].

    Returns
    -------
    prices : np.ndarray
        Prices with shape (..., T+1). The first element along the last axis equals s0.
    """
    r = np.asarray(logrets, dtype=float)
    s0 = np.asarray(s0, dtype=float)

    if np.any(s0 <= 0):
        raise ValueError("s0 must be positive.")

    # Broadcast s0 to match leading dimensions of r
    lead_shape = r.shape[:-1]
    if s0.shape != lead_shape:
        s0 = np.broadcast_to(s0, lead_shape)

    # cumulative log-levels: log S_t = log(s0) + sum_{i=1..t} r_i
    # prepend a zero so we include the initial price at t=0
    zeros = np.zeros(lead_shape + (1,), dtype=float)
    log_levels = np.concatenate([zeros, np.cumsum(r, axis=-1)], axis=-1)

    return np.exp(np.log(s0)[..., None] + log_levels)


class BootstrapPathwise:
    """
    Non-parametric, signature-based pathwise bootstrap.

    Steps:
      1) fit(paths): build a library of (signature, future_segment) from historical 1D paths
      2) generate(seed_path, n_total_steps): repeatedly append a sampled neighbor's forward segment

    Parameters
    ----------
    lookback : int
        Length of lookback window used to compute the signature.
    sig_level : int
        Signature level for iisignature (2 channels: time & value).
    forward : int
        Length of the appended forward segment each step.
    dt : float
        Spacing used to create the time channel in the signature window.
    k : int, default 10
        Number of nearest neighbors in signature space.
    neighbor_weighting : {'uniform','softmax'}, default 'uniform'
        How to sample among the k neighbors.
    random_state : int or None
        Seed for reproducibility.
    use_tqdm : bool, default True
        Show a progress bar when building the library (if tqdm available).
    """

    def __init__(
        self,
        lookback: int,
        sig_level: int,
        forward: int,
        dt: float,
        k: int = 10,
        neighbor_weighting: str = "uniform",
        random_state: int | None = None,
    ):
        if lookback <= 0 or forward <= 0 or dt <= 0:
            raise ValueError("lookback, forward, and dt must be positive.")
        if neighbor_weighting not in {"uniform", "softmax"}:
            raise ValueError("neighbor_weighting must be 'uniform' or 'softmax'.")

        self.lookback = lookback
        self.sig_level = sig_level
        self.forward = forward
        self.dt = dt
        self.k = k
        self.neighbor_weighting = neighbor_weighting

        self.rng = np.random.default_rng(random_state)
        self.sig_dim = iisignature.siglength(2, sig_level)  # 2 channels: time & value

        # filled by fit(...)
        self.library_sigs: np.ndarray | None = None         # (num_windows, sig_dim)
        self.library_segments: np.ndarray | None = None     # (num_windows, forward)

    # ----------------------- public API ----------------------- #

    def fit_logrets(self, paths: np.ndarray):
        """
        Build the library from paths of 1D series of log-returns.
        """
        L, F, dt = self.lookback, self.forward, self.dt
        time_segment = np.arange(L) * dt

        sigs, segs = [], []
        for p in paths:
            if p.ndim != 1:
                raise ValueError("Each path must be 1D (n_steps,).")
            n = len(p)
            if n < L + F:
                continue
            for i in trange(L, n - F):
                past_segment = p[i - L:i]
                window = np.stack([time_segment, past_segment], axis=1)  # (L, 2)
                sig = iisignature.sig(window, self.sig_level)           # (sig_dim,)
                future_segment = p[i:i + F]                              # (F,)
                sigs.append(sig)
                segs.append(future_segment)

        if not sigs:
            raise ValueError("No windows extracted. Check lookback/forward vs. path lengths.")

        self.library_sigs = np.asarray(sigs)
        self.library_segments = np.asarray(segs)

        # quick sanity
        if self.library_sigs.shape[1] != self.sig_dim:
            raise RuntimeError("Signature dimension mismatch; check iisignature version/level.")

        return self

    def generate_logrets(
        self,
        seed_path: np.ndarray,
        n_total_steps: int,
        return_full_path: bool = False,
    ) -> np.ndarray:
        """
        Generate a continuation by repeatedly sampling forward segments from k-NN in signature space.

        Parameters
        ----------
        seed_path : array, shape (init_steps,)
            Initial history (same scale as training paths).
        n_total_steps : int
            Desired final length including the seed.
        return_full_path : bool, default False
            If True, returns the seed concatenated with the continuation.

        Returns
        -------
        np.ndarray
            If return_full_path=False: (n_total_steps - len(seed_path),)
            else: (n_total_steps,)
        """
        if self.library_sigs is None or self.library_segments is None:
            raise RuntimeError("Call fit(...) before generate(...).")

        L, F, dt = self.lookback, self.forward, self.dt
        if len(seed_path) < L:
            raise ValueError("seed_path must be at least `lookback` long.")
        if n_total_steps <= len(seed_path):
            raise ValueError("n_total_steps must exceed len(seed_path).")

        path = np.array(seed_path, dtype=float).tolist()
        k = min(self.k, len(self.library_sigs))  # guard when library is small

        while len(path) < n_total_steps:
            past = np.asarray(path[-L:], dtype=float)
            window = np.stack([np.arange(L) * dt, past], axis=1)
            qsig = iisignature.sig(window, self.sig_level)  # (sig_dim,)

            # brute-force kNN in signature space
            dists = np.linalg.norm(self.library_sigs - qsig, axis=1)  # (num_windows,)
            # grab top-k (fast partial sort); indices unordered within the top-k set
            topk_idx = np.argpartition(dists, kth=k - 1)[:k]

            if self.neighbor_weighting == "uniform":
                idx = self.rng.choice(topk_idx)
            else:  # softmax on negative distances (temperature = std of top-k)
                d = dists[topk_idx]
                # avoid zero std; add small epsilon
                temp = d.std() + 1e-12
                w = np.exp(-(d / temp))
                w /= w.sum()
                idx = self.rng.choice(topk_idx, p=w)

            seg = self.library_segments[idx]  # (F,)
            remain = n_total_steps - len(path)
            path.extend(seg[:remain].tolist())

        out = np.asarray(path, dtype=float)
        return out if return_full_path else out[len(seed_path):]


class HybridKRRBootstrap:
    """
    Generator that models log-returns as:
        r_t = (KRR_drift(window_{t-1})) * dt + residual_{NN(window_{t-1})}

    - Drift: kernel ridge regression (linear kernel on signatures)
    - Residual: bootstrap from k nearest neighbors in signature space

    Parameters
    ----------
    lookback : int
    sig_level : int
    dt : float
    lam : float
        Ridge regularization for KRR.
    k : int, default 10
        Number of NN for residual bootstrap.
    neighbor_weighting : {'uniform','softmax'}, default 'uniform'
        How to sample among the k neighbors.
    random_state : int | None
    """

    def __init__(
        self,
        lookback: int,
        sig_level: int,
        dt: float,
        lam: float,
        k: int = 10,
        neighbor_weighting: str = "uniform",
        random_state: int | None = None,
    ):
        if lookback <= 0 or dt <= 0 or lam <= 0:
            raise ValueError("lookback, dt, lam must be positive.")
        if neighbor_weighting not in {"uniform", "softmax"}:
            raise ValueError("neighbor_weighting must be 'uniform' or 'softmax'.")

        self.lookback = lookback
        self.sig_level = sig_level
        self.dt = dt
        self.lam = lam
        self.k = k
        self.neighbor_weighting = neighbor_weighting
        self.rng = np.random.default_rng(random_state)
        self.sig_dim = iisignature.siglength(2, sig_level)

        # learned / library attributes after fit(...)
        self.alpha: np.ndarray | None = None             # (N,)
        self.library_sigs: np.ndarray | None = None      # (N, sig_dim)
        self.library_residuals: np.ndarray | None = None # (N,)

    # --------- internal helpers --------- #

    @staticmethod
    def _train_krr(S: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        # alpha = (K + lam I)^{-1} y, with K = S S^T
        K = S @ S.T
        n = K.shape[0]
        return np.linalg.solve(K + lam * np.eye(n), y)

    @staticmethod
    def _krr_predict(alpha: np.ndarray, S_train: np.ndarray, sig_new: np.ndarray) -> float:
        kvec = S_train @ sig_new  # (N,)
        return float(kvec @ alpha)

    # --------- public API --------- #
    def fit_logrets(self, log_rets: np.ndarray):
        """
        Build signature library, train KRR drift, compute residuals.

        Parameters
        ----------
        log_rets : (n_paths, n_steps) array of log-returns
        """
        L, dt = self.lookback, self.dt
        times = np.arange(L) * dt

        X_sigs, y_drift = [], []
        # Build dataset for KRR: signatures and drift targets r_t / dt
        for log_r in log_rets:
            n = len(log_r)
            if n < L + 1:
                continue
            for i in trange(L, n):
                window = log_r[i - L:i]
                path2d = np.stack([times, window], axis=1)
                sig = iisignature.sig(path2d, self.sig_level)
                X_sigs.append(sig)
                y_drift.append(log_r[i] / dt)

        if not X_sigs:
            raise ValueError("No windows extracted; check lookback vs. sequence lengths.")

        X_sigs = np.vstack(X_sigs)               # (N, sig_dim)
        y_drift = np.asarray(y_drift, float)     # (N,)

        # Train KRR on signatures
        alpha = self._train_krr(X_sigs, y_drift, self.lam)

        # Compute residuals r_t - drift*dt for each window (aligned with X_sigs order)
        residuals = []
        idx = 0
        for log_r in log_rets:
            n = len(log_r)
            if n < L + 1:
                continue
            for i in trange(L, n):
                # signature is already in X_sigs[idx], but recomputing keeps the code clearer;
                # for speed, you could reuse X_sigs[idx] directly.
                window = log_r[i - L:i]
                path2d = np.stack([times, window], axis=1)
                sig = iisignature.sig(path2d, self.sig_level)

                drift = self._krr_predict(alpha, X_sigs, sig)
                resid = log_r[i] - drift * dt
                residuals.append(resid)
                idx += 1

        self.alpha = alpha
        self.library_sigs = X_sigs
        self.library_residuals = np.asarray(residuals, float)
        return self

    def generate_logrets(
        self,
        seed_path: np.ndarray,
        n_total_steps: int,
        return_full_path: bool = False,
    ) -> np.ndarray:
        """
        Generate log-returns using KRR drift + residual bootstrap.

        Parameters
        ----------
        seed_path : (init_steps,) array of log-returns (must be >= lookback)
        n_total_steps : final length (including seed)
        return_full_path : if True, return seed + continuation; else only continuation

        Returns
        -------
        np.ndarray
        """
        if self.alpha is None or self.library_sigs is None or self.library_residuals is None:
            raise RuntimeError("Call fit(...) before generate_logrets(...).")

        L, dt = self.lookback, self.dt
        if len(seed_path) < L:
            raise ValueError("seed_path must be at least lookback long.")
        if n_total_steps <= len(seed_path):
            raise ValueError("n_total_steps must exceed len(seed_path).")

        path = np.array(seed_path, float).tolist()
        k = min(self.k, len(self.library_sigs))
        times = np.arange(L) * dt

        while len(path) < n_total_steps:
            window = np.asarray(path[-L:], float)
            sig = iisignature.sig(np.stack([times, window], axis=1), self.sig_level)

            # drift via KRR
            drift = self._krr_predict(self.alpha, self.library_sigs, sig)

            # residual via kNN bootstrap
            dists = np.linalg.norm(self.library_sigs - sig, axis=1)
            topk = np.argpartition(dists, k - 1)[:k]
            if self.neighbor_weighting == "uniform":
                idx = self.rng.choice(topk)
            else:
                d = dists[topk]
                temp = d.std() + 1e-12
                w = np.exp(-(d / temp))
                w /= w.sum()
                idx = self.rng.choice(topk, p=w)

            resid = self.library_residuals[idx]
            next_val = drift * dt + resid
            path.append(next_val)

        out = np.asarray(path, float)
        return out if return_full_path else out[len(seed_path):]


class KRRSignature:
    """
    KRR-on-signatures model:
        inputs  : window of log-prices (length lookback+1) with time channel
        targets : mu_t (drift of dlog/dt), log(sigma_t)
        simulate: dlog = mu*dt + sigma*sqrt(dt)*Z

    Parameters
    ----------
    lookback : int               # window length (past steps)
    sig_level : int              # signature/logsignature level
    dt : float                   # step size
    lam_mu : float               # ridge for mu head
    lam_sig : float              # ridge for log-sigma head
    random_state : int | None
    """

    def __init__(
        self,
        lookback: int,
        sig_level: int,
        dt: float,
        lam_mu: float,
        lam_sig: float,
        random_state: int | None = None,
    ):
        if iisignature is None:
            raise ImportError("Install `iisignature` to use KRRSigMuSigmaGenerator.")
        if lookback <= 0 or dt <= 0 or lam_mu <= 0 or lam_sig <= 0:
            raise ValueError("lookback, dt, lam_mu, lam_sig must be positive.")

        self.lookback = lookback
        self.sig_level = sig_level
        self.dt = dt
        self.lam_mu = lam_mu
        self.lam_sig = lam_sig

        self.rng = np.random.default_rng(random_state)
        # learned params after fit
        self.alpha_mu: np.ndarray | None = None          # (N,1)
        self.alpha_sig: np.ndarray | None = None         # (N,1)
        self.S_train: np.ndarray | None = None           # (N, sig_dim)

    # ---------- helpers ---------- #
    @staticmethod
    def _krr_train(S: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        # y can be (N,) or (N,1); solve for alpha with linear kernel K = S S^T
        K = S @ S.T
        n = K.shape[0]
        y = y.reshape(n, 1)
        alpha = np.linalg.solve(K + lam * np.eye(n), y)  # (N,1)
        return alpha

    @staticmethod
    def _krr_pred(alpha: np.ndarray, S_train: np.ndarray, s_new: np.ndarray) -> float:
        # returns scalar prediction
        k = S_train @ s_new  # (N,)
        return float(k @ alpha[:, 0])

    # ---------- API ---------- #

    def fit_logrets(self, logrets: np.ndarray):
        """
        Fit KRR heads for drift and log-vol.
        log_paths : (n_paths, T) array of log-prices (NOT log-returns).
        """
        L, dt = self.lookback, self.dt
        S_list, y_mu, y_logsig = [], [], []
        times = np.arange(L)*dt

        for logr in logrets:
            n_steps = len(logr)
            if n_steps < L + 1:
                continue
            for i in range(L, n_steps):
                window = logr[i - L:i]                       # length L+1
                path2d = np.stack([times, window], axis=1)        # (L+1, 2)
                S_list.append(iisignature.sig(path2d, self.sig_level))
                
                y_mu.append(logr[i] / dt)
                sigma_est = np.std(window) / np.sqrt(dt)
                y_logsig.append(np.log(sigma_est + 1e-8))

        if not S_list:
            raise ValueError("No training windows found; increase data or reduce lookback.")

        S = np.vstack(S_list)                     # (N, sig_dim)
        ym = np.asarray(y_mu, dtype=float)        # (N,)
        ys = np.asarray(y_logsig, dtype=float)    # (N,)

        self.alpha_mu = self._krr_train(S, ym, self.lam_mu)       # (N,1)
        self.alpha_sig = self._krr_train(S, ys, self.lam_sig)     # (N,1)
        self.S_train = S
        return self

    def generate_logrets(
        self,
        seed_path: np.ndarray,
        n_total_steps: int,
        stochastic: bool = True,
        return_full_path: bool = False,
    ) -> np.ndarray:
        """
        Generate future *log-prices*.

        seed_path : (m,) with m >= lookback+1  (contains history ending at "today")
        n_total_steps   : desired final length including the seed
        stochastic      : if False, use Z=0 (mean path)
        return_full_path: if True, include the seed; else return continuation only
        """
        if self.alpha_mu is None or self.alpha_sig is None or self.S_train is None:
            raise RuntimeError("Call fit(...) before generate_log_prices(...).")

        L, dt = self.lookback, self.dt
        if len(seed_path) < L + 1:
            raise ValueError("seed_path must be at least lookback long.")
        if n_total_steps <= len(seed_path):
            raise ValueError("n_total_steps must exceed len(seed_path).")

        path = np.array(seed_path, float).tolist()
        times = np.arange(L) * dt

        while len(path) < n_total_steps:
            window = np.asarray(path[-L:], float)
            s = iisignature.sig(np.stack([times, window], axis=1), self.sig_level)

            mu = self._krr_pred(self.alpha_mu, self.S_train, s)                 # drift per unit time
            logsig = self._krr_pred(self.alpha_sig, self.S_train, s)            # log sigma
            sigma = np.exp(logsig)

            z = self.rng.standard_normal() if stochastic else 0.0
            dlog = mu * dt + sigma * np.sqrt(dt) * z
            path.append(dlog)

        out = np.asarray(path, float)
        return out if return_full_path else out[len(seed_path):]


class ARIMAGen:
    """
    ARIMA-based path generator.

    Fit on log-returns  r_t  with order=(p,0,q)
    After fit(...), call the corresponding generate_* method.

    Parameters
    ----------
    order : tuple[int,int,int]
        (p, d, q) as in ARIMA.
    trend : {'n','c','t','ct'} or None, default 'c'
        As in statsmodels (constant, time trend, both, none).
    enforce_stationarity : bool
    enforce_invertibility : bool
    random_state : int | None
    """

    def __init__(
        self,
        order: tuple[int, int, int],
        trend: str | None = "c",
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        random_state: int | None = None,
    ):
        if ARIMA is None:
            raise ImportError("Please install statsmodels: pip install statsmodels")

        self.order = order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.rng = np.random.default_rng(random_state)

        # Fitted models/results
        self._res_returns = None   # ARIMAResults for returns (d must be 0)


    def fit_logrets(self, log_returns: np.ndarray):
        """Fit ARMA(p,d,q) to a 1D array of log-returns (usually d=0)."""
        y = np.asarray(log_returns, float)
        model = ARIMA(
            y,
            order=self.order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self._res_returns = model.fit()
        return self

    def generate_logrets(self, n_steps: int,):
        """
        Generate future log-returns from the ARMA fit (fit_returns must be called).

        Returns
        -------
        If return_mean_and_se=False:
            rets : (n_steps,)
        Else:
            rets_mean : (n_steps,), rets_se : (n_steps,)
        Innovations (if requested) also returned as last item.
        """
        if self._res_returns is None:
            raise RuntimeError("Call fit_returns(...) first.")

        res = self._res_returns
        # Simulate from the state-space form (includes parameter uncertainty via fixed params)
        # statsmodels draws Gaussian shocks internally if `measurement_shocks`/`state_shocks` not passed.
        sim = res.simulate(nsimulations=n_steps, random_state=self.rng)
        return np.asarray(sim, float)


class GARCHGen:
    """
    Fit & generate log-returns using a constant-mean GARCH(p,q) with optional scaling.
    Scaling improves optimizer convergence (e.g., use 100 * returns).
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        random_state: int | None = None,
        fit_kwargs: dict | None = None,
        scale: float | str = "auto",   # NEW: 'auto' or a numeric factor (e.g., 100.0)
    ):
        if arch_model is None:
            raise ImportError("Please install `arch`: pip install arch")
        if p <= 0 or q <= 0:
            raise ValueError("p and q must be positive.")
        if dist not in {"normal", "t"}:
            raise ValueError("dist must be 'normal' or 't'.")

        self.p, self.q = p, q
        self.dist = dist
        self.rng = np.random.default_rng(random_state)
        self.fit_kwargs = fit_kwargs or {"disp": "off"}

        # learned state (in **scaled** units)
        self.mu = None
        self.omega = None
        self.alphas = None
        self.betas = None
        self.nu = None

        self._eps_lags = None
        self._sig_lags = None

        # scaling
        self.scale = scale
        self._scale = 1.0  # resolved numeric factor after fit

    # ---------------- Fit ---------------- #

    def _resolve_scale(self, y: np.ndarray) -> float:
        if isinstance(self.scale, (int, float)):
            return float(self.scale)
        # 'auto' heuristic: returns in decimals → scale by 100
        # If already large, keep as 1.0
        std = float(np.std(y))
        return 100.0 if std < 0.05 else 1.0

    def fit_logrets(self, log_returns: np.ndarray):
        y = np.asarray(log_returns, dtype=float).ravel()
        if y.size < max(self.p, self.q) + 5:
            raise ValueError("Series too short for the requested (p,q).")

        self._scale = self._resolve_scale(y)
        y_s = self._scale * y  # scale for estimation

        model = arch_model(
            y_s,
            mean="Constant",
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist="t" if self.dist == "t" else "normal",
        )
        res = model.fit(**self.fit_kwargs)

        # parameters in **scaled units**
        params = res.params
        self.mu = float(params.get("mu", params.get("const", 0.0)))
        self.omega = float(params["omega"])
        self.alphas = np.array([params[f"alpha[{i}]"] for i in range(1, self.p + 1)], float)
        self.betas  = np.array([params[f"beta[{j}]"]  for j in range(1, self.q + 1)], float)
        if self.dist == "t":
            self.nu = float(params["nu"])

        # lags from scaled fit
        eps_s = np.asarray(res.resid, dtype=float)
        sig_s = np.asarray(res.conditional_volatility, dtype=float)
        self._eps_lags = eps_s[-self.p:][::-1] if self.p > 0 else np.zeros(1)
        self._sig_lags = sig_s[-self.q:][::-1] if self.q > 0 else np.full(1, sig_s[-1])

        return self

    # ---------------- Generate ---------------- #

    def generate_logrets(
        self,
        n_steps: int,
        return_full_path: bool = False,
        seed_returns: np.ndarray | None = None,  # seed in ORIGINAL units
    ) -> np.ndarray:
        if any(v is None for v in (self.mu, self.omega, self.alphas, self.betas, self._sig_lags, self._eps_lags)):
            raise RuntimeError("Call fit(...) before generate(...).")

        p, q = self.p, self.q
        mu, omega = self.mu, self.omega
        alpha, beta = self.alphas, self.betas

        # initialize lags in **scaled** space
        if seed_returns is not None:
            seed = np.asarray(seed_returns, float).ravel()
            seed_s = self._scale * seed
            eps_hist_s = seed_s - mu  # constant-mean
            eps_lags = self._eps_lags.copy()
            sig_lags = self._sig_lags.copy()
            L = max(p, q)
            for r_s in eps_hist_s[-L:]:
                sigma2_s = omega
                if p:
                    sigma2_s += np.dot(alpha, (eps_lags[:p] ** 2))
                if q:
                    sigma2_s += np.dot(beta,  (sig_lags[:q] ** 2))
                sigma_s = np.sqrt(max(sigma2_s, 0.0))
                if p:
                    eps_lags = np.r_[r_s, eps_lags[:-1]]
                if q:
                    sig_lags = np.r_[sigma_s, sig_lags[:-1]]
        else:
            eps_lags = self._eps_lags.copy()
            sig_lags = self._sig_lags.copy()

        future_s = []
        for _ in range(n_steps):
            sigma2_s = omega
            if p:
                sigma2_s += np.dot(alpha, (eps_lags[:p] ** 2))
            if q:
                sigma2_s += np.dot(beta,  (sig_lags[:q] ** 2))
            sigma_s = np.sqrt(max(sigma2_s, 0.0))

            if self.dist == "normal":
                z = self.rng.standard_normal()
            else:
                assert self.nu is not None and self.nu > 2
                z = self.rng.standard_t(self.nu)
                z *= np.sqrt((self.nu - 2) / self.nu)  # standardize to Var=1

            eps_s = sigma_s * z
            r_next_s = mu + eps_s
            future_s.append(r_next_s)

            if p:
                eps_lags = np.r_[eps_s, eps_lags[:-1]]
            if q:
                sig_lags = np.r_[sigma_s, sig_lags[:-1]]

        future_s = np.asarray(future_s, float)
        future = future_s / self._scale  # UN-SCALE to original units

        if seed_returns is not None and return_full_path:
            return np.r_[seed_returns, future]
        return future


    def params_in_original_units(self) -> dict:
        """Convenience mapping of fitted parameters to original (unscaled) units."""
        if self.mu is None:
            raise RuntimeError("Fit the model first.")
        s = self._scale
        return {
            "mu": self.mu / s,
            "omega": self.omega / (s ** 2),
            "alphas": self.alphas.copy(),
            "betas": self.betas.copy(),
            "nu": self.nu,
            "scale": s,
        }