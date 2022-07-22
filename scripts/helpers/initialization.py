from typing import List
import numpy as np
import torch

from chronostrain import cfg, logger
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, NoiselessErrorModel, PEPhredErrorModel, FragmentSpace, PhredErrorModel
from chronostrain.model.generative import GenerativeModel


def initialize_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_model(population: Population,
                 mean: torch.Tensor,
                 fragments: FragmentSpace,
                 time_points: List[float],
                 disable_quality: bool,
                 db: StrainDatabase,
                 pair_ended: bool = True) -> GenerativeModel:
    """
    Simple wrapper for creating a generative model.
    :param mean: The prior mean for the underlying gaussian process.
    :param population: The bacteria population.
    :param fragments: The collection of fragments coming from the population.
    :param time_points: List of time points for which samples are taken from.
    :param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    :param db: A Strain database instance.
    :param pair_ended: Indicates whether the read model is paired-end or single-ended.
    :return A Generative model object.
    """
    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    elif not pair_ended:
        error_model = PhredErrorModel(
            insertion_error_ll=cfg.model_cfg.get_float("INSERTION_LL"),
            deletion_error_ll=cfg.model_cfg.get_float("DELETION_LL")
        )
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
        mu=mean,
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
