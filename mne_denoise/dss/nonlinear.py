"""Core nonlinear/iterative DSS algorithm and Estimator.

This module contains:
1. `iterative_dss`: The core mathematical implementation of Nonlinear DSS.
2. `IterativeDSS`: The Scikit-learn estimator compatible with MNE-Python objects or NumPy arrays.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._data import continuous_to_epochs, epochs_to_continuous, extract_data_from_mne
from .._logging import logger, verbose
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._whitening import whiten_from_data_covariance


def _resolve_callable(param, x, default=None):
    """Resolve parameters that can be float, callable, or None."""
    if param is None:
        return default
    if callable(param):
        return param
    # Wrap constant in callable
    return lambda *args: param


@verbose
def iterative_dss_one(
    X_whitened: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    *,
    w_init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float | Callable[[np.ndarray], float] | None = None,
    beta: float | Callable[[np.ndarray], float] | None = None,
    gamma: float | Callable[[np.ndarray, np.ndarray, int], float] | None = None,
    random_state: int | np.random.Generator | None = None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Fixed-point iteration for extracting a single DSS component.

    This implements Algorithm 1 from Särelä & Valpola (2005) [1]_ with optional
    Newton step acceleration (FastICA equivalence).

    The algorithm finds a spatial filter **w** that maximizes the objective:

    .. math:: J(w) = E[f(w^T X)^2]

    where f(·) is the nonlinear denoising function.

    **Algorithm**::

        Initialize: w = random unit vector

        Repeat until converged:
            1. source = w @ X                    # Project data → 1D source
            2. source_denoised = f(source)       # Apply nonlinearity
            3. source_denoised *= alpha          # (optional) signal normalization
            4. w_new = E[X · source_denoised]    # Gradient direction
            5. w_new += beta · w                 # (optional) Newton step
            6. w_new = normalize(w_new)          # Unit norm constraint
            7. w = w_old + gamma·(w_new - w_old) # (optional) relaxation
            8. Check convergence: |w · w_old| ≈ 1

    Parameters
    ----------
    X_whitened : ndarray, shape (n_components, n_times)
        Whitened data matrix. Must have identity covariance.
    denoiser : callable
        Nonlinear denoising function f(s) → s_denoised.
        Examples: TanhMaskDenoiser, GaussDenoiser, WienerMaskDenoiser.
    w_init : ndarray, shape (n_components,), optional
        Initial weight vector. If None, random initialization.
    max_iter : int
        Maximum iterations. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    alpha : float or callable, optional
        Signal normalization factor applied after denoising:
        ``source_denoised *= alpha``.
        Useful for denoisers with different output variance.
    beta : float or callable, optional
        Spectral shift parameter for Newton-like acceleration.
        For tanh denoiser: ``beta = -E[1 - tanh(s)²]`` (use ``beta_tanh``).
        For cubic denoiser: ``beta = -3`` (use ``beta_pow3``).
    gamma : float or callable, optional
        Learning rate / relaxation parameter. Controls step size:
        ``w = w_old + gamma · (w_new - w_old)``.
        Default None (gamma=1, full step).
    callback : callable | None
        Called synchronously after each completed fixed-point iteration with a
        progress event having ``method="iterative_dss"`` and
        ``stage="iteration"``. Callback return values are ignored and callback
        exceptions propagate unchanged. The event metric is the convergence
        change; it is ``None`` for a degenerate reinitialization iteration.
    verbose : bool | str | int | None
        MNE-style logging level. Iteration state is emitted at DEBUG.

    Returns
    -------
    w : ndarray, shape (n_components,)
        Optimal spatial filter (unit norm).
    source : ndarray, shape (n_times,)
        Extracted source time series.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged within max_iter.

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR, 6, 233-272.
    """
    callback = _validate_callback(callback)
    return _iterative_dss_one(
        X_whitened,
        denoiser,
        w_init=w_init,
        max_iter=max_iter,
        tol=tol,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_state=random_state,
        callback=callback,
        component=None,
    )


def _iterative_dss_one(
    X_whitened: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    *,
    w_init: np.ndarray | None,
    max_iter: int,
    tol: float,
    alpha: float | Callable[[np.ndarray], float] | None,
    beta: float | Callable[[np.ndarray], float] | None,
    gamma: float | Callable[[np.ndarray, np.ndarray, int], float] | None,
    random_state: int | np.random.Generator | None,
    callback: _ProgressCallback | None,
    component: int | None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Run one fixed-point DSS solve with validated progress state."""
    n_components, n_times = X_whitened.shape

    # Initialize RNG (handle both int and Generator)
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Initialize weight vector
    if w_init is not None:
        w = w_init.copy()
    else:
        w = rng.standard_normal(n_components)

    # Normalize
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w = np.ones(n_components) / np.sqrt(n_components)
    else:
        w = w / norm

    # Resolve parameters to callables
    alpha_func = _resolve_callable(alpha, None)
    beta_func = _resolve_callable(beta, None)
    gamma_func = _resolve_callable(gamma, None)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1
        w_old = w.copy()

        # Step 2: Extract source
        source = w @ X_whitened  # (n_times,)

        # Step 3: Apply denoiser
        source_denoised = denoiser(source)

        # Apply alpha (signal normalization)
        if alpha_func is not None:
            source_denoised = alpha_func(source) * source_denoised

        # Calculate beta step if applicable
        step_beta = 0.0
        if beta_func is not None:
            step_beta = beta_func(source)

        # Step 4: Update weights
        # w_new = E[X * f(s)] + beta * w
        # Standard DSS: w_new = E[X * f(s)]
        # FastICA Newton: beta = -E[f'(s)]
        gradient_part = X_whitened @ source_denoised / n_times
        w_new = gradient_part + step_beta * w

        # Step 5: Normalize
        norm = np.linalg.norm(w_new)
        if norm < 1e-12:
            # Denoiser killed the signal, reinitialize
            logger.debug(
                "IterativeDSS single-component iteration %d/%d: "
                "degenerate update; reinitializing.",
                iteration + 1,
                max_iter,
            )
            w = rng.standard_normal(n_components)
            w = w / np.linalg.norm(w)
            _emit_progress(
                callback,
                method="iterative_dss",
                stage="iteration",
                current=iteration + 1,
                total=max_iter,
                component=component,
                metric=None,
            )
            continue

        w_normalized = w_new / norm

        # Apply gamma (learning rate / relaxation)
        if gamma_func is not None:
            step_gamma = gamma_func(w_normalized, w_old, iteration)
            # w = w_old + gamma * (w_new - w_old)
            w = w_old + step_gamma * (w_normalized - w_old)
            # Re-normalize after relaxation
            w = w / np.linalg.norm(w)
        else:
            w = w_normalized

        # Check convergence (using abs because sign can flip)
        correlation = np.abs(np.dot(w, w_old))
        change = 1 - correlation
        logger.debug(
            "IterativeDSS single-component iteration %d/%d: change %.3e.",
            iteration + 1,
            max_iter,
            change,
        )
        _emit_progress(
            callback,
            method="iterative_dss",
            stage="iteration",
            current=iteration + 1,
            total=max_iter,
            component=component,
            metric=float(change),
        )
        if change < tol:
            converged = True
            logger.debug(
                "IterativeDSS single-component converged at iteration %d.",
                iteration + 1,
            )
            break
    else:
        logger.debug(
            "IterativeDSS single-component reached max_iter=%d without convergence.",
            max_iter,
        )

    # Final source extraction
    source = w @ X_whitened

    return w, source, n_iter, converged


@verbose
def iterative_dss(
    data: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    n_components: int,
    *,
    method: str = "deflation",
    rank: int | None = None,
    reg: float = 1e-9,
    max_iter: int = 100,
    tol: float = 1e-6,
    w_init: np.ndarray | None = None,
    verbose: bool | str | int | None = None,
    alpha: float | Callable | None = None,
    beta: float | Callable | None = None,
    gamma: float | Callable | None = None,
    random_state: int | np.random.Generator | None = None,
    callback=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract multiple DSS components using iterative (nonlinear) algorithm.

    This implements the Iterative DSS algorithm from Särelä & Valpola (2005) [1]_.
    Unlike linear DSS which uses a closed-form eigendecomposition, iterative DSS
    uses fixed-point iteration with a nonlinear denoising function.

    **Algorithm Overview**::

        1. Center data: X = X - mean(X)
        2. Whiten data: X_white = Whitener @ X  (identity covariance)
        3. Extract components using deflation or symmetric method:

           Deflation (sequential):
               For each component i = 1..n_components:
                   w_i = iterative_dss_one(X_deflated)
                   Orthogonalize w_i against w_1..w_{i-1}
                   Deflate: X_deflated -= w_i @ w_i.T @ X_deflated

           Symmetric (parallel):
               Initialize W = [w_1, ..., w_n] randomly
               Repeat until converged:
                   Update all w_i simultaneously
                   W = symmetric_orthogonalize(W)

        4. Convert filters to sensor space: filters = W @ Whitener

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Input multichannel data.
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function f(s) → s_denoised.
        Examples: TanhMaskDenoiser, GaussDenoiser, WienerMaskDenoiser.
    n_components : int
        Number of components to extract.
    method : {'deflation', 'symmetric'}
        Component extraction method:

        - ``'deflation'``: Extract one-by-one, orthogonalizing after each.
          More stable, but order-dependent.
        - ``'symmetric'``: Update all simultaneously, then orthogonalize.
          Order-independent, may be less stable.

        Default ``'deflation'``.
    rank : int, optional
        Rank for whitening. If None, auto-determined from eigenvalue threshold.
    reg : float
        Regularization for whitening eigenvalue cutoff. Default 1e-9.
    max_iter : int
        Maximum iterations per component. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    w_init : ndarray, shape (n_components, n_whitened), optional
        Initial weight matrix. If None, random initialization.
    verbose : bool | str | int | None
        MNE-style logging level. Convergence details are emitted at DEBUG and
        the final numerical result is summarized at INFO.
    alpha : float or callable, optional
        Signal normalization factor (see ``iterative_dss_one``).
    beta : float or callable, optional
        Newton step parameter (see ``iterative_dss_one``).
    gamma : float or callable, optional
        Learning rate / relaxation (see ``iterative_dss_one``).
    callback : callable | None
        Called synchronously after each completed fixed-point iteration with a
        progress event having ``method="iterative_dss"`` and
        ``stage="iteration"``. The event metric is the convergence change.
        For deflationary extraction, ``component`` is the 1-based component
        being extracted; it is ``None`` for symmetric extraction. Callback
        return values are ignored and callback exceptions propagate unchanged.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters in sensor space.
        Apply as: ``sources = filters @ data``.
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    patterns : ndarray, shape (n_channels, n_components)
        Spatial patterns for visualization / reconstruction.
        Note: These are returned in original sensor units (not normalized),
        satisfying the identity :math:`X_{recon} = patterns @ sources`.
    convergence_info : ndarray, shape (n_components, 2)
        ``[n_iterations, converged]`` for each component.


    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import iterative_dss
    >>> from mne_denoise.dss.denoisers import TanhMaskDenoiser
    >>> # Basic usage with numpy array
    >>> data = np.random.randn(10, 1000)
    >>> denoiser = TanhMaskDenoiser()
    >>> filters, sources, patterns, _ = iterative_dss(
    ...     data, denoiser, n_components=2, method="symmetric"
    ... )

    See Also
    --------
    iterative_dss_one : Single component extraction.
    IterativeDSS : Sklearn-style estimator wrapper.

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR, 6, 233-272.
    """
    callback = _validate_callback(callback)
    data_2d, _, _, _, _, _ = extract_data_from_mne(
        data,
        concatenate_epochs=True,
    )

    if data_2d.ndim != 2:
        raise ValueError(f"Data must be 2D or 3D, got {data_2d.ndim}D")

    n_channels, n_samples = data_2d.shape

    # Center data
    data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)

    # Whiten data
    X_whitened, whitener, dewhitener = whiten_from_data_covariance(
        data_centered, rank=rank, reg=reg
    )
    n_whitened = X_whitened.shape[0]

    # Limit components to whitened dimension
    n_components = min(n_components, n_whitened)

    if method == "deflation":
        filters_whitened, sources, convergence_info = _iterative_dss_deflation(
            X_whitened,
            denoiser,
            n_components,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            verbose=verbose,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            random_state=random_state,
            callback=callback,
        )
    elif method == "symmetric":
        filters_whitened, sources, convergence_info = _iterative_dss_symmetric(
            X_whitened,
            denoiser,
            n_components,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            verbose=verbose,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            random_state=random_state,
            callback=callback,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'deflation' or 'symmetric'")

    # Convert filters from whitened to sensor space
    # filters_whitened: (n_components, n_whitened)
    # whitener: (n_whitened, n_channels)
    # sensor_filter = whitened_filter @ whitener
    filters = filters_whitened @ whitener  # (n_components, n_channels)

    # patterns = C @ filters.T
    C = data_centered @ data_centered.T / n_samples
    patterns = C @ filters.T

    n_converged = int(np.sum(convergence_info[:, 1] > 0.5))
    logger.info(
        "Iterative DSS: method=%s, denoiser=%s, components=%d, "
        "converged=%d/%d, iterations=%s.",
        method,
        type(denoiser).__name__,
        n_components,
        n_converged,
        n_components,
        convergence_info[:, 0].astype(int).tolist(),
    )

    return filters, sources, patterns, convergence_info


def _iterative_dss_deflation(
    X_whitened: np.ndarray,
    denoiser: Callable,
    n_components: int,
    *,
    max_iter: int,
    tol: float,
    w_init: np.ndarray | None,
    verbose: bool | str | int | None,
    alpha: float | Callable | None = None,
    beta: float | Callable | None = None,
    gamma: float | Callable | None = None,
    random_state: int | np.random.Generator | None = None,
    callback: _ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components one-by-one using deflation.

    **Algorithm**::

        For i = 1..n_components:
            1. w_i = iterative_dss_one(X_deflated)     # Extract one component
            2. Orthogonalize: w_i -= W_prev.T @ (W_prev @ w_i)
            3. Normalize: w_i = w_i / ||w_i||
            4. s_i = w_i @ X_whitened                  # Extract source
            5. Deflate: X_deflated -= w_i @ w_i.T @ X_deflated

    Parameters
    ----------
    X_whitened : ndarray, shape (n_whitened, n_times)
        Whitened data with identity covariance.
    denoiser : callable
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    max_iter, tol : int, float
        Convergence parameters passed to ``iterative_dss_one``.
    w_init : ndarray, optional
        Initial weight matrix.
    verbose : bool | str | int | None
        Print progress.
    alpha, beta, gamma : optional
        Passed to ``iterative_dss_one``.

    Returns
    -------
    W : ndarray, shape (n_components, n_whitened)
        Weight matrix (spatial filters in whitened space).
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    convergence_info : ndarray, shape (n_components, 2)
        [n_iter, converged] per component.
    """
    n_whitened, n_times = X_whitened.shape

    # Initialize RNG (handle both int and Generator)
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Storage
    W = np.zeros((n_components, n_whitened))
    sources = np.zeros((n_components, n_times))
    convergence_info = np.zeros((n_components, 2))

    X_deflated = X_whitened.copy()

    for i in range(n_components):
        # Get initial weight
        if w_init is not None and i < w_init.shape[0]:
            w_i = w_init[i]
        else:
            w_i = None

        # Run single-component iteration
        w, source, n_iter, converged = _iterative_dss_one(
            X_deflated,
            denoiser,
            w_init=w_i,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            random_state=rng,
            callback=callback,
            component=i + 1,
        )

        status = "converged" if converged else "max_iter"
        logger.debug(
            "IterativeDSS component %d/%d: %d iteration(s) (%s).",
            i + 1,
            n_components,
            n_iter,
            status,
        )

        # Orthogonalize against previous components (vectorized)
        if i > 0:
            W_prev = W[:i]  # (i, n_whitened)
            # Vectorized: w - W_prev.T @ (W_prev @ w)
            w = w - W_prev.T @ (W_prev @ w)
            norm = np.linalg.norm(w)
            if norm < 1e-12:
                logger.debug(
                    "IterativeDSS component %d/%d was degenerate after "
                    "orthogonalization; using random initialization.",
                    i + 1,
                    n_components,
                )
                w = rng.standard_normal(n_whitened)
                w = w - W_prev.T @ (W_prev @ w)
                norm = np.linalg.norm(w)
            w = w / norm

        W[i] = w
        sources[i] = w @ X_whitened
        convergence_info[i] = [n_iter, float(converged)]

        # Deflate: remove component from data
        outer = np.outer(w, w)
        X_deflated = X_deflated - outer @ X_deflated

    return W, sources, convergence_info


def _iterative_dss_symmetric(
    X_whitened: np.ndarray,
    denoiser: Callable,
    n_components: int,
    *,
    max_iter: int,
    tol: float,
    w_init: np.ndarray | None,
    verbose: bool | str | int | None,
    alpha: float | Callable | None = None,
    beta: float | Callable | None = None,
    gamma: float | Callable | None = None,
    random_state: int | np.random.Generator | None = None,
    callback: _ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components simultaneously with symmetric orthogonalization.

    **Algorithm**::

        Initialize: W = random (n_components, n_whitened)
        W = symmetric_orthogonalize(W)

        Repeat until converged:
            S = W @ X
            S_denoised = f(S)
            W_new = E[S_denoised @ X.T] + beta * W
            W = symmetric_orthogonalize(W_new)
            Check convergence

    Parameters
    ----------
    X_whitened : ndarray, shape (n_whitened, n_times)
        Whitened data with identity covariance.
    denoiser : callable
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    max_iter, tol : int, float
        Convergence parameters.
    w_init : ndarray, optional
        Initial weight matrix.
    verbose : bool | str | int | None
        Print progress.
    alpha, beta, gamma : optional
        Iteration parameters.

    Returns
    -------
    W : ndarray, shape (n_components, n_whitened)
        Weight matrix (spatial filters in whitened space).
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    convergence_info : ndarray, shape (n_components, 2)
        [n_iter, converged] (same for all components in symmetric).
    """
    n_whitened, n_times = X_whitened.shape

    # Initialize weight matrix
    if w_init is not None:
        W = w_init[:n_components, :n_whitened].copy()
    else:
        rng = np.random.default_rng(random_state)
        W = rng.standard_normal((n_components, n_whitened))

    # Symmetric orthogonalization (decorrelation)
    W = _symmetric_orthogonalize(W)

    # Resolve parameters to callables
    alpha_func = _resolve_callable(alpha, None)
    beta_func = _resolve_callable(beta, None)
    # Gamma not typically used in vectorized symmetric step, but could be added

    convergence_info = np.zeros((n_components, 2))

    for iteration in range(max_iter):
        W_old = W.copy()

        # 1. Project to sources
        # W: (n_comp, n_white), X: (n_white, n_times) -> S: (n_comp, n_times)
        S = W @ X_whitened

        # 2. Apply denoiser (vectorized)
        S_denoised = denoiser(S)

        # Apply alpha
        if alpha_func is not None:
            # Broadcast alpha across columns if it returns (n_comp,) or scalar
            a = alpha_func(S)
            if np.ndim(a) == 1:
                a = a[:, np.newaxis]
            S_denoised = a * S_denoised

        # 3. Update weights
        # Gradient part: E[S_denoised @ X.T] -> (n_comp, n_white)
        gradient = S_denoised @ X_whitened.T / n_times

        # Beta part
        step_beta = 0.0
        if beta_func is not None:
            b = beta_func(S)
            if np.ndim(b) == 1:
                b = b[:, np.newaxis]
            step_beta = b

        W = gradient + step_beta * W

        # 4. Symmetric orthogonalization
        W = _symmetric_orthogonalize(W)

        # 5. Check convergence (max change across components)
        # Dot product of rows: diag(W @ W_old.T)
        correlations = np.abs(np.sum(W * W_old, axis=1))
        max_change = np.max(1 - correlations)

        logger.debug(
            "IterativeDSS symmetric iteration %d/%d: max change %.3e.",
            iteration + 1,
            max_iter,
            max_change,
        )
        _emit_progress(
            callback,
            method="iterative_dss",
            stage="iteration",
            current=iteration + 1,
            total=max_iter,
            component=None,
            metric=float(max_change),
        )
        if max_change < tol:
            logger.debug(
                "IterativeDSS symmetric converged at iteration %d.",
                iteration + 1,
            )
            convergence_info[:, 0] = iteration + 1
            convergence_info[:, 1] = 1.0
            break
    else:
        logger.debug(
            "IterativeDSS symmetric reached max_iter=%d without convergence.",
            max_iter,
        )
        convergence_info[:, 0] = max_iter
        convergence_info[:, 1] = 0.0

    # Extract final sources
    sources = W @ X_whitened

    return W, sources, convergence_info


def _symmetric_orthogonalize(W: np.ndarray) -> np.ndarray:
    """Symmetric orthogonalization using (W * W.T)^{-1/2} * W."""
    # EVD of W @ W.T
    gram = W @ W.T
    D, E = np.linalg.eigh(gram)

    # Handle numerical issues
    D = np.maximum(D, 1e-12)

    # W_orth = E @ diag(1/sqrt(D)) @ E.T @ W
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    W_orth = E @ D_inv_sqrt @ E.T @ W

    return W_orth


class IterativeDSS:
    """Iterative (Nonlinear) Denoising Source Separation Transformer.

    Implements Iterative DSS as a scikit-learn compatible transformer that
    fits natively on MNE-Python objects (Raw, Epochs) or numpy arrays.

    Unlike linear DSS which uses closed-form eigendecomposition, Iterative DSS
    uses fixed-point iteration with a nonlinear denoising function, making it
    equivalent to FastICA when using ICA contrast functions (tanh, gauss, cube).

    Parameters
    ----------
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function f(s) → s_denoised. Must be an instance of
        `mne_denoise.dss.NonlinearDenoiser` (e.g. `TanhMaskDenoiser`,
        `WienerMaskDenoiser`) or a callable.
    n_components : int
        Number of components to extract.
    method : {'deflation', 'symmetric'}
        Component extraction method:

        - ``'deflation'``: Extract one-by-one, orthogonalizing after each.
        - ``'symmetric'``: Update all simultaneously, then orthogonalize.

        Default ``'deflation'``.
    rank : int, optional
        Rank for whitening. If None, auto-determined from eigenvalue threshold.
    reg : float
        Regularization for whitening. Default 1e-9.
    max_iter : int
        Maximum iterations per component. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    verbose : bool | str | int | None
        MNE-style logging level.
    alpha : float or callable, optional
        Signal normalization factor applied after denoising.
    beta : float or callable, optional
        Spectral shift (Newton step) for faster convergence.
        Use ``beta_tanh`` for TanhMaskDenoiser, ``beta_pow3`` for cubic.
    gamma : float or callable, optional
        Learning rate / relaxation parameter.
    random_state : int, RandomState, Generator, optional
        Seed for random initialization.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        The spatial filters (un-mixing matrix). Apply as: ``sources = filters_ @ data``.
    patterns_ : ndarray, shape (n_channels, n_components)
        The spatial patterns (mixing matrix). Reconstruct as: ``data = patterns_ @ sources``.
    sources_ : ndarray, shape (n_components, n_times)
        Extracted sources from fit data.
    convergence_info_ : ndarray, shape (n_components, 2)
        [n_iterations, converged] for each component.

    Examples
    --------
    >>> from mne_denoise.dss import IterativeDSS
    >>> from mne_denoise.dss.denoisers import TanhMaskDenoiser, beta_tanh
    >>>
    >>> # With numpy array
    >>> dss = IterativeDSS(denoiser=denoiser, n_components=3)
    >>> dss.fit(data)
    >>> sources = dss.transform(data)
    >>>
    >>> # With MNE Raw object
    >>> dss.fit(raw)
    >>> sources = dss.transform(raw)

    See Also
    --------
    DSS : Linear DSS transformer.
    iterative_dss : Functional API.
    """

    def __init__(
        self,
        denoiser: Callable[[np.ndarray], np.ndarray],
        n_components: int,
        *,
        method: str = "deflation",
        rank: int | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool | str | int | None = None,
        alpha: float | Callable | None = None,
        beta: float | Callable | None = None,
        gamma: float | Callable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.n_components = n_components
        self.method = method
        self.rank = rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.sources_: np.ndarray | None = None
        self.convergence_info_: np.ndarray | None = None
        self._mne_info = None

    @verbose
    def fit(
        self,
        X,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> IterativeDSS:
        """Compute Iterative DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            The data to fit. Accepts:

            - ``mne.io.Raw``: Continuous data
            - ``mne.Epochs``: Epoched data
            - ``ndarray``: Shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        callback : callable | None
            Called synchronously after each completed fixed-point iteration
            with a progress event whose ``method`` is ``"iterative_dss"`` and
            whose ``stage`` is ``"iteration"``. Callback return values are
            ignored and callback exceptions propagate unchanged. The event
            metric is the convergence change; ``component`` is the 1-based
            component for deflation and ``None`` for symmetric extraction.

        Returns
        -------
        self : IterativeDSS
            The fitted transformer.
        """
        callback = _validate_callback(callback)
        # Validate and extract data using shared helper
        data, _, mne_type, mne_info, picks, ch_names = extract_data_from_mne(
            X,
            concatenate_epochs=True,
        )
        self._mne_ch_names_ = ch_names

        # Store MNE info for later use if available
        if (
            mne_type in ("raw", "epochs")
            and mne_info is not None
            and hasattr(mne_info, "info")
        ):
            self._mne_info = mne_info.info

        if self.normalize_input:
            self.channel_norms_ = np.std(data, axis=1)
            self.channel_norms_ = np.where(
                self.channel_norms_ > 0, self.channel_norms_, 1.0
            )
            data = data / self.channel_norms_[:, np.newaxis]

        filters, sources, patterns, conv_info = iterative_dss(
            data,
            self.denoiser,
            self.n_components,
            method=self.method,
            rank=self.rank,
            reg=self.reg,
            max_iter=self.max_iter,
            tol=self.tol,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            random_state=self.random_state,
            callback=callback,
        )

        self.filters_ = filters
        self.patterns_ = patterns
        self.sources_ = sources
        self.convergence_info_ = conv_info

        return self

    @verbose
    def transform(
        self,
        X,
        *,
        verbose: bool | str | int | None = None,
    ) -> np.ndarray:
        """Apply fitted filters to extract sources.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to transform. Same formats as ``fit()``.

        Returns
        -------
        sources : ndarray, shape (n_components, n_times) or (n_components, n_times, n_epochs)
            Extracted source time series.
        """
        if self.filters_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")

        # Validate and extract data
        data, _, mne_type, _, picks, _ = extract_data_from_mne(
            X, ch_names=getattr(self, "_mne_ch_names_", None)
        )

        original_shape = data.shape

        data_2d = epochs_to_continuous(data) if data.ndim == 3 else data

        if self.normalize_input:
            if self.channel_norms_ is None:
                raise RuntimeError(
                    "IterativeDSS not fitted with normalize_input=True. Call fit() first."
                )
            data_2d = data_2d / self.channel_norms_[:, np.newaxis]

        # Center
        data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)

        # Apply filters
        sources = self.filters_ @ data_centered

        # Reshape to original 3D if needed
        # (n_components, n_epochs * n_times) -> (n_epochs, n_components, n_times)
        source_shape = original_shape
        if data.ndim == 3:
            source_shape = (original_shape[0], sources.shape[0], original_shape[2])
        sources = continuous_to_epochs(sources, source_shape)

        return sources

    @verbose
    def inverse_transform(
        self,
        sources: np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> np.ndarray:
        """Reconstruct data from sources.

        Parameters
        ----------
        sources : ndarray, shape (n_sources, n_times)
            Source time series. Can use fewer sources than fitted.

        Returns
        -------
        reconstructed : ndarray, shape (n_channels, n_times)
            Reconstructed data: ``patterns_[:, :n_sources] @ sources``.
        """
        if self.patterns_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")

        n_comp_sources = sources.shape[1] if sources.ndim == 3 else sources.shape[0]
        patterns = self.patterns_[:, :n_comp_sources]

        if sources.ndim == 3:
            # Assume MNE format (n_epochs, n_comp, n_times)
            rec = np.tensordot(sources, patterns, axes=(1, 1)).transpose(0, 2, 1)
            if self.normalize_input:
                if self.channel_norms_ is None:
                    raise RuntimeError(
                        "IterativeDSS not fitted with normalize_input=True. Call fit() first."
                    )
                rec *= self.channel_norms_[np.newaxis, :, np.newaxis]
        else:
            rec = patterns @ sources
            if self.normalize_input:
                if self.channel_norms_ is None:
                    raise RuntimeError(
                        "IterativeDSS not fitted with normalize_input=True. Call fit() first."
                    )
                rec *= self.channel_norms_[:, np.newaxis]

        return rec

    def get_normalized_patterns(self) -> np.ndarray:
        """Get L2-normalized spatial patterns for visualization.

        Returns
        -------
        patterns_norm : ndarray, shape (n_channels, n_components)
            L2-normalized spatial patterns.
        """
        if self.patterns_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")

        norms = np.linalg.norm(self.patterns_, axis=0)
        # Use relative threshold for physical units
        max_norm = np.max(norms)
        threshold = 1e-15 * max_norm if max_norm > 0 else 1e-30
        norms = np.where(norms > threshold, norms, 1.0)
        return self.patterns_ / norms

    @verbose
    def fit_transform(
        self,
        X,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to fit and transform.
        callback : callable | None
            Called synchronously after each completed fixed-point iteration
            during fitting. Callback return values are ignored and callback
            exceptions propagate unchanged. Deflation events include the
            1-based component; symmetric events use ``component=None``.

        Returns
        -------
        sources : ndarray
            Extracted sources.
        """
        return self.fit(X, callback=callback).transform(X)
