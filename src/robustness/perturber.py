from src.robustness.ldp_protocols.geometric import GeometricMechanism
from src.robustness.ldp_protocols.gaussian import GaussianMechanism
from src.robustness.ldp_protocols.laplace import LaplaceMechanism
import numpy as np

def perturb_mpi(mpi_original, epsilon_global):
    max_index = len(mpi_original) - 1
    epsilon_individual = epsilon_global/len(mpi_original)
    geo = GeometricMechanism(sensitivity=max_index, epsilon=epsilon_individual)

    mpi_perturbed = [geo.obfuscate(mpi) for mpi in mpi_original]
    mpi_perturbed_clipped = np.clip(mpi_perturbed, 0, max_index).astype(np.int32)
    return mpi_perturbed_clipped


def perturb_mpd(mpd_original, epsilon_global, m=100, mechanism=LaplaceMechanism):
    max_distance = 2*np.sqrt(m)
    epsilon_individual = epsilon_global / len(mpd_original)
    gauss = mechanism(sensitivity=max_distance, epsilon=epsilon_individual)
    mpd_perturbed = np.array([gauss.obfuscate(v) for v in mpd_original])
    return np.clip(mpd_perturbed, 0.0, max_distance).astype(np.float32)