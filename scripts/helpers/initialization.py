from typing import List
import torch

from chronostrain import cfg, logger
from chronostrain.model import Population, NoiselessErrorModel, PhredErrorModel, GenerativeModel, FragmentSpace


def initialize_seed(seed: int):
    torch.manual_seed(seed)


def create_model(population: Population,
                 fragments: FragmentSpace,
                 time_points: List[float],
                 disable_quality: bool) -> GenerativeModel:
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param fragments: The collection of fragments coming from the population.
    @param time_points: List of time points for which samples are taken from.
    @param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    @return A Generative model object.
    """
    mu = torch.zeros(population.num_strains(), device=cfg.torch_cfg.device)

    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
        error_model = PhredErrorModel()

    model = GenerativeModel(
        bacteria_pop=population,
        times=time_points,
        mu=mu,
        tau_1_dof=cfg.model_cfg.sics_dof_1,
        tau_1_scale=cfg.model_cfg.sics_scale_1,
        tau_dof=cfg.model_cfg.sics_dof,
        tau_scale=cfg.model_cfg.sics_scale,
        read_error_model=error_model,
        fragments=fragments,
        mean_frag_length=cfg.model_cfg.mean_read_length
    )

    return model
