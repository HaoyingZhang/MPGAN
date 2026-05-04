import numpy as np
from numba import jit
import opendp.prelude as dp

dp.enable_features("contrib")

# ── Geometric (Discrete Laplace) Mechanism ───────────────────────────────────

def geometric_obfuscate(input_data: int, scale: float) -> int:
    """
    Obfuscate an integer value using the Geometric (Discrete Laplace) mechanism via OpenDP.

    Parameters
    ----------
    input_data : int
        The user's true integer value to be obfuscated.
    scale : float
        The noise scale (sensitivity / epsilon). The bigger, more noise added.

    Returns
    -------
    int
        The obfuscated integer value.
    """
    dp.enable_features("contrib")
    input_space = dp.atom_domain(T=dp.i32), dp.absolute_distance(T=dp.i32)
    geometric = dp.m.make_geometric(*input_space, scale=scale)
    return geometric(np.int32(input_data))


class GeometricMechanism:
    def __init__(self, sensitivity: int, epsilon: float):
        """
        Initialize the Geometric (Discrete Laplace) mechanism.

        Parameters
        ----------
        sensitivity : int
            The L1 sensitivity of the integer-valued query (typically 1 for LDP).
        epsilon : float
            The privacy budget. Must be positive.

        Raises
        ------
        ValueError
            If sensitivity <= 0 or epsilon <= 0.
        """
        if sensitivity <= 0:
            raise ValueError("sensitivity must be a positive integer.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")

        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon
        self._alpha = np.exp(-1.0 / self.scale)  # exp(-epsilon / sensitivity)

    def obfuscate(self, input_data: int) -> int:
        """
        Obfuscate the input data using the Geometric mechanism.

        Parameters
        ----------
        input_data : int
            The user's true integer value.

        Returns
        -------
        int
            The sanitized integer value after adding discrete Laplace noise.
        """
        return geometric_obfuscate(input_data, self.scale)

    # def estimate(self, noisy_reports: list) -> float:
    #     """
    #     Estimate the true mean from noisy reports collected via the Geometric mechanism.

    #     The discrete Laplace noise has zero mean, so the sample mean of the noisy
    #     reports is an unbiased estimator of the true value.

    #     Parameters
    #     ----------
    #     noisy_reports : list of int
    #         Noisy integer values collected from users.

    #     Returns
    #     -------
    #     float
    #         Unbiased estimate of the true integer value.

    #     Raises
    #     ------
    #     ValueError
    #         If noisy_reports is empty.
    #     """
    #     if len(noisy_reports) == 0:
    #         raise ValueError("Noisy reports cannot be empty.")
    #     return float(np.mean(noisy_reports))

    # def attack(self, obfuscated_value: int) -> int:
    #     """
    #     Perform a privacy attack on a value obfuscated by the Geometric mechanism.

    #     The optimal attack is to report the noisy value as the inferred true value,
    #     since the noise is symmetric and zero-mean.

    #     Parameters
    #     ----------
    #     obfuscated_value : int
    #         The obfuscated value produced by the mechanism.

    #     Returns
    #     -------
    #     int
    #         The inferred true value (equal to the obfuscated value).
    #     """
    #     return obfuscated_value

    def get_variance(self) -> float:
        """
        Compute the variance of the Geometric mechanism.

        For the discrete Laplace distribution with scale b = sensitivity / epsilon:
            Var = 2 * alpha / (1 - alpha)^2,  where alpha = exp(-1 / b).

        Returns
        -------
        float
            The variance of the added noise.
        """
        return 2.0 * self._alpha / (1.0 - self._alpha) ** 2

    # def get_asr(self) -> float:
    #     """
    #     Compute the Adversarial Success Rate (ASR) for the Geometric mechanism.

    #     ASR = P(noise = 0) = (1 - alpha) / (1 + alpha), where alpha = exp(-1 / scale).
    #     This is the probability that the noisy output equals the true input.

    #     Returns
    #     -------
    #     float
    #         The probability that an attacker correctly infers the original input value.
    #     """
    #     return (1.0 - self._alpha) / (1.0 + self._alpha)