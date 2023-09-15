from logging import Logger

from chronostrain.model import *
from chronostrain.config import cfg


def create_error_model(
        observed_reads: TimeSeriesReads,
        disable_quality: bool,
        logger: Logger
) -> AbstractErrorModel:
    """
    Simple wrapper for creating a gaussian posterior model.
    """
    read_types = set()
    for reads_t in observed_reads.time_slices:
        for src in reads_t.sources:
            if isinstance(src, SampleReadSourceSingle):
                read_types.add(src.read_type)
            elif isinstance(src, SampleReadSourcePaired):
                read_types.add(ReadType.PAIRED_END_1)
                read_types.add(ReadType.PAIRED_END_2)

    if disable_quality:
        logger.info("Using NoiselessErrorModel (Are you trying to debug?).")
        return NoiselessErrorModel(mismatch_likelihood=0.)

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
        return PEPhredErrorModel(
            insertion_error_ll_1=cfg.model_cfg.get_float("INSERTION_LL_1"),
            deletion_error_ll_1=cfg.model_cfg.get_float("DELETION_LL_1"),
            insertion_error_ll_2=cfg.model_cfg.get_float("INSERTION_LL_2"),
            deletion_error_ll_2=cfg.model_cfg.get_float("DELETION_LL_2")
        )
    else:
        logger.debug("Using single-end error model.")
        return PhredErrorModel(
            insertion_error_ll=cfg.model_cfg.get_float("INSERTION_LL"),
            deletion_error_ll=cfg.model_cfg.get_float("DELETION_LL")
        )


def create_gaussian_prior(
        population: Population,
        observed_reads: TimeSeriesReads,
) -> AbundanceGaussianPrior:
    return AbundanceGaussianPrior(
        times=[reads_t.time_point for reads_t in observed_reads],
        tau_1_dof=cfg.model_cfg.sics_dof_1,
        tau_1_scale=cfg.model_cfg.sics_scale_1,
        tau_dof=cfg.model_cfg.sics_dof,
        tau_scale=cfg.model_cfg.sics_scale,
        population=population
    )
