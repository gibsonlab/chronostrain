from typing import List
import torch

from chronostrain import cfg, logger
from chronostrain.model import Population, NoiselessErrorModel, PhredErrorModel, GenerativeModel


def initialize_seed(seed: int):
    torch.manual_seed(seed)


def create_model(population: Population,
                 window_size: int,
                 time_points: List[float],
                 disable_quality: bool) -> GenerativeModel:
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param window_size: Fragment read length to use.
    @param time_points: List of time points for which samples are taken from.
    @param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    @return A Generative model object.
    """
    mu = torch.zeros(population.num_strains(), device=cfg.torch_cfg.device)

    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
        error_model = PhredErrorModel(read_len=window_size)

    model = GenerativeModel(
        bacteria_pop=population,
        read_length=window_size,
        times=time_points,
        mu=mu,
        tau_1_dof=cfg.model_cfg.sics_dof_1,
        tau_1_scale=cfg.model_cfg.sics_scale_1,
        tau_dof=cfg.model_cfg.sics_dof,
        tau_scale=cfg.model_cfg.sics_scale,
        read_error_model=error_model
    )

    return model
