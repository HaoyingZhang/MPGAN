import numpy as np

class GaussianMechanism:
    def __init__(self, sensitivity: float, epsilon: float):
        """
        Gaussian mechanism for continuous-valued queries.

        Parameters
        ----------
        sensitivity : float
            The L2 sensitivity of the query.
        epsilon : float
            The privacy budget. Must be positive.
        """
        if sensitivity <= 0:
            raise ValueError("sensitivity must be a positive value.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")

        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon

    def obfuscate(self, input_data: float) -> float:
        return float(input_data) + np.random.normal(0.0, self.scale)

