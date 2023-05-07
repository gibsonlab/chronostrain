from logging import Logger
from typing import List, Set

import jax.numpy as np
from chronostrain.database import StrainDatabase
from chronostrain.model import *
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import ReadType


def create_model(population: Population,
                 read_types: Set[ReadType],
                 mean: np.ndarray,
                 fragments: FragmentSpace,
                 time_points: List[float],
                 disable_quality: bool,
                 db: StrainDatabase,
                 logger: Logger) -> GenerativeModel:
    """
    Simple wrapper for creating a generative model.
    :param mean: The prior mean for the underlying gaussian process.
    :param population: The bacteria population.
    :param fragments: The collection of fragments coming from the population.
    :param time_points: List of time points for which samples are taken from.
    :param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    :param db: A Strain database instance.
    :return A Generative model object.
    """
    if disable_quality:
        logger.info("Using NoiselessErrorModel (Are you trying to debug?).")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
        """
        Possible todo item: allow for mixing & matching of different experiments across timepoints?
        """
        if read_types == {ReadType.SINGLE_END}:
            is_paired_end = False
        elif read_types == {ReadType.PAIRED_END_1}:
            is_paired_end = True
            logger.warning("Read set specified paired-ended reads, but only found forward reads. Is this correct?")
        elif read_types == {ReadType.PAIRED_END_2}:
            is_paired_end = True
            logger.warning("Read set specified paired-ended reads, but only found reverse reads. Is this correct?")
        elif read_types == {ReadType.PAIRED_END_1, ReadType.PAIRED_END_2}:
            is_paired_end = True
        else:
            raise RuntimeError("Only single-ended datasets or paired-end datasets are supported. "
                               "Mixtures or other combinations are not supported.")

        if is_paired_end:
            logger.debug("Using paired-end error model.")
            error_model = PEPhredErrorModel(
                insertion_error_ll_1=cfg.model_cfg.get_float("INSERTION_LL_1"),
                deletion_error_ll_1=cfg.model_cfg.get_float("DELETION_LL_1"),
                insertion_error_ll_2=cfg.model_cfg.get_float("INSERTION_LL_2"),
                deletion_error_ll_2=cfg.model_cfg.get_float("DELETION_LL_2")
            )
        else:
            logger.debug("Using single-end error model.")
            error_model = PhredErrorModel(
                insertion_error_ll=cfg.model_cfg.get_float("INSERTION_LL"),
                deletion_error_ll=cfg.model_cfg.get_float("DELETION_LL")
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
