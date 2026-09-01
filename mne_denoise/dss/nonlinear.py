"""Iterative DSS algorithms."""

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
    """Extract one DSS component by fixed-point iteration.

    Parameters
    ----------
    X_whitened : ndarray, shape (n_features, n_times)
        Whitened data matrix.
    denoiser : callable
        Nonlinear source transformation.
    w_init : ndarray, shape (n_features,), or None, default=None
        Initial unit-vector estimate. ``None`` uses random initialization.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Sign-insensitive convergence tolerance.
    alpha : float, callable, or None, default=None
        Optional source-normalization factor.
    beta : float, callable, or None, default=None
        Optional Newton/fixed-point correction.
    gamma : float, callable, or None, default=None
        Optional relaxation factor.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for initialization.
    callback : callable or None, default=None
        Synchronous progress callback receiving ``ProgressEvent`` objects.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    w : ndarray, shape (n_features,)
        Unit-norm spatial filter.
    source : ndarray, shape (n_times,)
        Extracted source.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the tolerance was reached.

    Notes
    -----
    Convergence compares the absolute dot product of successive unit filters, so a
    sign flip is treated as no change.
    This follows the iterative DSS formulation :footcite:p:`sarela2005_dss`.

    References
    ----------
    .. footbibliography::
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
    """Extract multiple components with iterative DSS.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Channel-first data; epochs are flattened for the solve.
    denoiser : callable
        Nonlinear source transformation.
    n_components : int
        Number of components to extract, limited by the whitening rank.
    method : {"deflation", "symmetric"}, default="deflation"
        Sequential deflation or simultaneous symmetric extraction.
    rank : int or None, default=None
        Whitening rank.
    reg : float, default=1e-9
        Relative whitening threshold.
    max_iter : int, default=100
        Maximum iterations per component or symmetric solve.
    tol : float, default=1e-6
        Convergence tolerance.
    w_init : ndarray or None, default=None
        Initial filter vector or matrix.
    verbose : bool, str, int, or None, default=None
        Logging level.
    alpha, beta, gamma : float, callable, or None
        Optional fixed-point parameters passed to :func:`iterative_dss_one`.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for initialization.
    callback : callable or None, default=None
        Synchronous progress callback for fixed-point iterations.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        Sensor-space spatial filters.
    sources : ndarray, shape (n_components, n_times)
        Extracted sources.
    patterns : ndarray, shape (n_channels, n_components)
        Sensor-space mixing patterns.
    convergence_info : ndarray, shape (n_components, 2)
        Iteration count and convergence flag for each component.

    Notes
    -----
    This follows the iterative DSS formulation :footcite:p:`sarela2005_dss`.

    References
    ----------
    .. footbibliography::
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
    """Extract components sequentially with deflation."""
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
    """Extract components jointly with symmetric orthogonalization."""
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
    """Iterative DSS transformer.

    The estimator fits fixed-point nonlinear DSS filters on NumPy arrays or MNE
    ``Raw``/``Epochs`` inputs.

    Parameters
    ----------
    denoiser : callable
        Nonlinear source transformation.
    n_components : int
        Number of components to extract.
    method : {"deflation", "symmetric"}, default="deflation"
        Component extraction strategy.
    rank : int or None, default=None
        Whitening rank.
    reg : float, default=1e-9
        Relative whitening threshold.
    normalize_input : bool, default=True
        Normalize channels during fitting and undo the scaling on reconstruction.
    max_iter : int, default=100
        Maximum fixed-point iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    verbose : bool, str, int, or None, default=None
        Logging level.
    alpha, beta, gamma : float, callable, or None
        Optional fixed-point parameters.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for initialization.

    Attributes
    ----------
    filters_ : ndarray
        Fitted sensor-space filters.
    patterns_ : ndarray
        Fitted sensor-space patterns.
    sources_ : ndarray
        Sources from the fit data.
    convergence_info_ : ndarray, shape (n_components, 2)
        Iteration count and convergence flag per component.

    See Also
    --------
    DSS
        Linear covariance-bias DSS.
    iterative_dss
        Functional iterative DSS interface.

    Notes
    -----
    ``transform`` returns source arrays. ``inverse_transform`` reconstructs arrays
    using the fitted patterns; MNE metadata is used for channel extraction during
    fitting and transformation. This follows the iterative DSS formulation
    :footcite:p:`sarela2005_dss`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import IterativeDSS
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = IterativeDSS(
    ...     lambda source: source**3, n_components=2, rank=4, random_state=0
    ... )
    >>> sources = model.fit_transform(data)
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
        """Fit the iterative DSS filters.

        Parameters
        ----------
        X : mne.io.BaseRaw, mne.BaseEpochs, or ndarray
            Training data in channel-first NumPy layout or an MNE container.
        callback : callable or None, default=None
            Synchronous fixed-point progress callback.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        IterativeDSS
            The fitted estimator.
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
        """Apply fitted filters and return source data.

        Parameters
        ----------
        X : mne.io.BaseRaw, mne.BaseEpochs, or ndarray
            Data compatible with the fitted channel layout.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ndarray
            Sources with continuous or epoch-preserving layout.
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
        """Reconstruct sensor data from fitted sources.

        Parameters
        ----------
        sources : ndarray, shape (n_components, n_times) or (n_epochs, n_components, n_times)
            Source data.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ndarray
            Reconstructed sensor-space data.
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
        """Return L2-normalized fitted spatial patterns.

        Returns
        -------
        ndarray, shape (n_channels, n_components)
            Normalized patterns.
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
        """Fit the estimator and return the extracted sources.

        Parameters
        ----------
        X : ndarray
            Channel-first data.
        callback : callable or None, default=None
            Synchronous progress callback for fitting.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ndarray
            Extracted sources.
        """
        return self.fit(X, callback=callback).transform(X)
