from abc import abstractmethod, ABC
from multiprocessing import Pool
from typing import Iterator, Tuple, List

import numpy as np
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model import Population, GenerativeModel, FragmentSpace, PEPhredErrorModel
from chronostrain.model.io import TimeSeriesReads
from .base import StrainVariant
from ..inference import BBVISolverV2
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
                 bbvi_num_samples: int):
        """
        :param db: The database of markers and strains.
        :param reads: The input time-series reads.
        :param time_points: A list of floats representing timepoints.
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

        self.multi_alignments = list(CachedReadMultipleAlignments(reads, db).get_alignments())

    @abstractmethod
    def propose_variants(self, used_variants: List[StrainVariant]) -> Iterator[StrainVariant]:
        pass

    def construct_variants(self) -> GenerativeModel:
        """
        :return: A strain population instance, which may or may not include StrainVariant instances (depending on
        whether seed_with_database was set, and on whether any improvement is made in the log-likelihood).
        """

        # Determine initialization based on seed_with_database param.
        best_strains: List[StrainVariant] = []
        # noinspection PyTypeChecker
        best_model: GenerativeModel = None

        best_data_ll_estimate: float = float('-inf')
        best_num_variants: int = 0

        for strain_variant in self.propose_variants([]):
            cur_variants = best_strains + [strain_variant]
            logger.debug("Included strains: {}".format(
                cur_variants
            ))

            # obtain the solution and likelihood.
            model, data_ll_estimate = self.run_bbvi(cur_variants)

            if best_model is not None and data_ll_estimate < best_data_ll_estimate:
                # Found local max.
                logger.debug(
                    "Data LL decrease ({:.3f} --> {:.3f}). "
                    "Terminating search at {} strains ({} non-base variants).".format(
                        best_data_ll_estimate,
                        data_ll_estimate,
                        best_model.bacteria_pop.num_strains(),
                        best_num_variants
                    )
                )
                return best_model
            else:
                # Keep searching.
                logger.debug("New data ll estimate: {:.3f}".format(data_ll_estimate))
                best_data_ll_estimate = data_ll_estimate
                best_model = model
                best_strains = best_strains + [strain_variant]
                best_num_variants += 1
        if best_model is None:
            raise RuntimeError("Unable to decide on variant-only population.")
        else:
            return best_model

    def database_reference(self) -> Tuple[Population, FragmentSpace]:
        fragments = FragmentSpace()
        reference_strains = self.db.all_strains()
        reference_pop = Population(reference_strains)

        # Create fragments using reference markers.
        logger.debug("Using fragment construction from alignments.")
        for multi_align in self.multi_alignments:
            for marker in multi_align.markers():
                if not reference_pop.contains_marker(marker):
                    continue

                for reverse in [False, True]:
                    for read in multi_align.reads(revcomp=reverse):
                        subseq, insertions, deletions, start_clip, end_clip = multi_align.get_aligned_reference_region(
                            marker, read, revcomp=reverse
                        )

                        fragments.add_seq(
                            subseq,
                            metadata=f"MultiAlign({read.id}->{marker.id})"
                        )
        return reference_pop, fragments

    def construct_fragments(self, variants: List[StrainVariant]):
        fragments = FragmentSpace()
        for strain_idx, strain_variant in enumerate(variants):
            # Add all fragments implied by this new strain variant.
            for marker_variant in strain_variant.variant_markers:
                for t_idx, time_slice in enumerate(self.reads):
                    for read in time_slice:
                        for subseq, _, _, _, _, _ in marker_variant.subseq_from_read(read):
                            fragments.add_seq(subseq)
        return fragments

    def create_model(self, variants: List[StrainVariant]):
        population = Population(variants)
        fragments = self.construct_fragments(variants)
        return GenerativeModel(
            bacteria_pop=population,
            times=self.time_points,
            mu=torch.zeros(population.num_strains(), device=cfg.torch_cfg.device),
            tau_1_dof=cfg.model_cfg.sics_dof_1,
            tau_1_scale=cfg.model_cfg.sics_scale_1,
            tau_dof=cfg.model_cfg.sics_dof,
            tau_scale=cfg.model_cfg.sics_scale,
            read_error_model=PEPhredErrorModel(
                insertion_error_ll=cfg.model_cfg.insertion_error_log10,
                deletion_error_ll=cfg.model_cfg.deletion_error_log10
            ),
            fragments=fragments,
            mean_frag_length=cfg.model_cfg.mean_read_length
        )

    def run_bbvi(self,
                 variants: List[StrainVariant],
                 num_cores: int = 1
                 ) -> float:
        model = self.create_model(variants)

        if model.num_strains() > 1:
            solver = BBVISolverV2(model=model,
                                data=self.reads,
                                correlation_type="strain",
                                db=self.db,
                                num_cores=num_cores)
            solver.solve(
                optim_class=torch.optim.Adam,
                optim_args={'lr': self.bbvi_lr, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
                iters=self.bbvi_iters,
                num_samples=self.bbvi_num_samples,
                print_debug_every=500
            )

            x_latent_mean = solver.gaussian_posterior.mean()
            prior_ll = model.log_likelihood_x(x_latent_mean)
            data_conditional_ll = solver.data_likelihoods.conditional_likelihood(x_latent_mean)
            posterior_ll_est = solver.gaussian_posterior.log_likelihood(x_latent_mean)
            logger.debug("Log likelihoods: Data={}, Prior={}, Posterior={}".format(
                data_conditional_ll,
                prior_ll.item(),
                posterior_ll_est.item()
            ))

            # Bayes Rule: Pr(Data|Variants) = Pr(Data|X,Variants) * Pr(X|Variants) / Pr(X|Data,Variants)
            data_ll = (data_conditional_ll + prior_ll - posterior_ll_est).item()
        else:
            # Special case when there is only one strain in the population. (Nothing to do)
            solver = BBVISolverV2(model=model, data=self.reads, correlation_type="strain", db=self.db)
            data_ll = solver.data_likelihoods.conditional_likelihood(
                torch.ones((model.num_times(), 1), device=cfg.torch_cfg.device)
            ).item()

        logger.info("Data LL = {}, Variants = {}".format(data_ll, model.bacteria_pop.strains))
        return data_ll


class ExhaustiveVariantBBVISolver(AbstractVariantBBVISolver, ABC):
    def __init__(self,
                 db: StrainDatabase,
                 reads: TimeSeriesReads,
                 time_points: List[float],
                 bbvi_iters: int,
                 bbvi_lr: int,
                 bbvi_num_samples: int,
                 num_cores: int = 1):
        """
        :param db: The database of markers and strains.
        :param reads: The input time-series reads.
        :param time_points: A list of floats representing timepoints.
        :param bbvi_num_samples:
        :param bbvi_lr:
        :param bbvi_iters:
        """
        super().__init__(db, reads, time_points, bbvi_iters, bbvi_lr, bbvi_num_samples)
        self.num_cores = num_cores

    def construct_variants(self) -> GenerativeModel:
        """
        :return: A strain population instance.
        """
        chosen_variants = []
        best_data_ll_est = -float("inf")
        done = False

        """Perform a greedy search."""
        while not done:
            try:
                next_variant, data_ll_est = self.next_variant_to_include(chosen_variants)
            except NoVariantFoundException:
                break

            logger.info("Found next best variant: {}".format(next_variant.id))

            if data_ll_est < best_data_ll_est:
                break

            chosen_variants.append(next_variant)
            best_data_ll_est = data_ll_est

        return self.create_model(chosen_variants)

    def next_variant_to_include(self,
                                included_variants: List[StrainVariant],
                                ) -> Tuple[StrainVariant, float]:
        best_variant = None
        best_data_ll = -float("inf")

        if self.num_cores == 1:
            for variant in self.propose_variants(included_variants):
                test_variants: List[StrainVariant] = included_variants + [variant]
                test_variants.sort(key=lambda v: v.id)

                data_ll = self.run_bbvi(test_variants, num_cores=1)
                if data_ll > best_data_ll or best_variant is None:
                    best_data_ll = data_ll
                    best_variant = variant
            if best_variant is None:
                raise NoVariantFoundException()

            return best_variant, best_data_ll
        else:
            arguments = []
            variants = []

            for variant in self.propose_variants(included_variants):
                print("Trying variant {}".format(variant.id))
                test_variants: List[StrainVariant] = included_variants + [variant]
                test_variants.sort(key=lambda v: v.id)

                arguments.append((test_variants,))
                variants.append(variant)

            if len(arguments) == 0:
                raise NoVariantFoundException()

            thread_pool = Pool(self.num_cores)
            print("Starting thread.")
            results = thread_pool.starmap(self.run_bbvi, arguments)
            print("done.")
            i = np.argmax(results)

            best_variant = variants[i]
            best_data_ll = results[i]
            return best_variant, best_data_ll


class NoVariantFoundException(Exception):
    pass
