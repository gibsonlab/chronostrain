from abc import abstractmethod
from typing import Optional, Iterator, Tuple, List

import torch

from chronostrain.database import StrainDatabase
from chronostrain.model import Population, GenerativeModel, FragmentSpace, Strain, PhredErrorModel
from chronostrain.model.io import TimeSeriesReads
from .base import StrainVariant
from ..inference import BBVISolver
from ..subroutines.alignments import CachedReadMultipleAlignments

from chronostrain.config import create_logger, cfg
logger = create_logger(__name__)


class AbstractVariantBBVISolver(object):
    def __init__(self,
                 db: StrainDatabase,
                 reads: TimeSeriesReads,
                 time_points: List[float],
                 bbvi_iters: int,
                 bbvi_lr: int,
                 bbvi_num_samples: int,
                 seed_with_database: bool = False):
        """
        :param db: The database of markers and strains.
        :param reads: The input time-series reads.
        :param time_points: A list of floats representing timepoints.
        :param seed_with_database: Indicates whether or not to include the database reference strains (for test cases,
        where reads were actually sampled from the database strains.
        :param bbvi_num_samples:
        :param bbvi_lr:
        :param bbvi_iters:
        """
        self.db = db
        self.reads = reads
        self.time_points = time_points
        self.bbvi_lr = bbvi_lr
        self.bbvi_iters = bbvi_iters
        self.bbvi_num_samples = bbvi_num_samples
        self.seed_with_database = seed_with_database

        self.multi_alignments = list(CachedReadMultipleAlignments(reads, db).get_alignments())

    @abstractmethod
    def propose_variants(self) -> Iterator[StrainVariant]:
        pass

    def construct_variants(self) -> Tuple[Population, BBVISolver]:
        """
        :return: A strain population instance, which may or may not include StrainVariant instances (depending on whether
        seed_with_database was set, and on whether any improvement is made in the log-likelihood).
        """

        # Determine initialization based on seed_with_database param.
        if self.seed_with_database:
            # set best_model, best_result from initial seeded run.
            reference_pop, cumulative_fragments = self.database_reference()
            best_strains = reference_pop.strains
            best_model, best_result, best_data_ll_estimate = self.run_bbvi(reference_pop, cumulative_fragments)
            best_num_variants: int = 0
        else:
            best_strains: List[Strain] = []
            # noinspection PyTypeChecker
            best_model: GenerativeModel = None

            # noinspection PyTypeChecker
            best_result: BBVISolver = None

            best_data_ll_estimate: float = float('-inf')
            best_num_variants: int = 0
            cumulative_fragments = FragmentSpace()

        for strain_variant in self.propose_variants():
            population = Population(best_strains + [strain_variant])

            # Add all fragments implied by this new strain variant.
            for marker_variant in strain_variant.variant_markers:
                for time_slice in self.reads:
                    for read in time_slice:
                        for subseq, insertions, deletions in marker_variant.subseq_from_read(read):
                            cumulative_fragments.add_seq(
                                subseq,
                                metadata=f"Subseq_{marker_variant.id}"
                            )

            # obtain the solution and likelihood.
            model, solver, data_ll_estimate = self.run_bbvi(population, cumulative_fragments)

            if data_ll_estimate <= best_data_ll_estimate:
                # Found local max.
                logger.debug(
                    "Data LL decrease ({:.3f} --> {:.3f}). Terminating search at {} strains ({} non-base variants).".format(
                        best_data_ll_estimate,
                        data_ll_estimate,
                        best_model.bacteria_pop.num_strains(),
                        best_num_variants
                    )
                )
                return best_model.bacteria_pop, best_result
            else:
                # Keep searching.
                best_data_ll_estimate = data_ll_estimate
                best_model = model
                best_result = solver
                best_num_variants += 1

    def database_reference(self) -> Tuple[Population, FragmentSpace]:
        fragments = FragmentSpace()
        reference_strains = self.db.all_strains()
        reference_pop = Population(reference_strains)

        # Create fragments using reference markers.
        logger.debug("Using fragment construction from alignments.")
        for multi_align in self.multi_alignments:
            if not reference_pop.contains_marker(multi_align.marker):
                continue

            for reverse in [False, True]:
                for read in multi_align.reads(reverse=reverse):
                    subseq, insertions, deletions = multi_align.get_aligned_reference_region(
                        read, reverse=reverse
                    )

                    fragments.add_seq(
                        subseq,
                        metadata=f"ClustalO({read.id}->{multi_align.marker.id})"
                    )
        return reference_pop, fragments

    def run_bbvi(self,
                 population: Population,
                 fragments: FragmentSpace
                 ) -> Tuple[GenerativeModel, BBVISolver, float]:
        model = GenerativeModel(
            bacteria_pop=population,
            times=self.time_points,
            mu=torch.zeros(population.num_strains(), device=cfg.torch_cfg.device),
            tau_1_dof=cfg.model_cfg.sics_dof_1,
            tau_1_scale=cfg.model_cfg.sics_scale_1,
            tau_dof=cfg.model_cfg.sics_dof,
            tau_scale=cfg.model_cfg.sics_scale,
            read_error_model=PhredErrorModel(
                insertion_error_ll=cfg.model_cfg.insertion_error_log10,
                deletion_error_ll=cfg.model_cfg.deletion_error_log10
            ),
            fragments=fragments,
            mean_frag_length=cfg.model_cfg.mean_read_length
        )

        solver = BBVISolver(model=model, data=self.reads, correlation_type="strain", db=self.db)
        solver.solve(
            optim_class=torch.optim.Adam,
            optim_args={'lr': self.bbvi_lr, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
            iters=self.bbvi_iters,
            num_samples=self.bbvi_num_samples,
            print_debug_every=100
        )

        x_latent_mean = solver.gaussian_posterior.mean()
        prior_ll = model.log_likelihood_x(x_latent_mean)
        data_ll = model.data_likelihood(x_latent_mean, solver.data_likelihoods.matrices)
        posterior_ll_est = solver.gaussian_posterior.log_likelihood(x_latent_mean)
        data_ll_estimate = (data_ll + prior_ll - posterior_ll_est).item()
        return model, solver, data_ll_estimate
