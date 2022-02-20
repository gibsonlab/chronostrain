from typing import List
import torch

from chronostrain import cfg, logger
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, NoiselessErrorModel, PEPhredErrorModel, FragmentSpace
from chronostrain.model.generative import GenerativeModel


def initialize_seed(seed: int):
    torch.manual_seed(seed)


def create_model(population: Population,
                 fragments: FragmentSpace,
                 time_points: List[float],
                 disable_quality: bool,
                 db: StrainDatabase) -> GenerativeModel:
    """
    Simple wrapper for creating a generative model.
    :param population: The bacteria population.
    :param fragments: The collection of fragments coming from the population.
    :param time_points: List of time points for which samples are taken from.
    :param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    :param db: A Strain database instance.
    :return A Generative model object.
    """
    mu = torch.zeros(population.num_strains(), device=cfg.torch_cfg.device)

    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
        error_model = PEPhredErrorModel(
            insertion_error_ll_1=cfg.model_cfg.get_float("INSERTION_LL_1"),
            deletion_error_ll_1=cfg.model_cfg.get_float("DELETION_LL_1"),
            insertion_error_ll_2=cfg.model_cfg.get_float("INSERTION_LL_2"),
            deletion_error_ll_2=cfg.model_cfg.get_float("DELETION_LL_2")
        )

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
        frag_negbin_n=cfg.model_cfg.frag_len_negbin_n,
        frag_negbin_p=cfg.model_cfg.frag_len_negbin_p,
        min_overlap_ratio=cfg.model_cfg.min_overlap_ratio,
        db=db
    )

    return model
