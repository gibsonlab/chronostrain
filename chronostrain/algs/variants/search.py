from typing import Dict, Optional, Tuple

import torch
import numpy as np

from chronostrain.config import cfg
from chronostrain.model import Population, GenerativeModel, Marker, MarkerMetadata, Strain
from chronostrain.model.io import TimeSeriesReads

from .. import BBVISolver
from ..subroutines import CachedReadAlignments
from .pileup import MarkerPileups, NoSuchMarkerException

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class VariantSearchAlgorithm(object):
    def __init__(self,
                 base_model: GenerativeModel,
                 reads: TimeSeriesReads,
                 optim_iters: int = 1000,
                 optim_mc_samples: int = 200,
                 optim_kwargs: Optional[Dict] = None):
        self.db = cfg.database_cfg.get_database()
        self.base_model = base_model
        self.reads = reads
        self.optim_iters = optim_iters
        self.optim_mc_samples = optim_mc_samples
        if optim_kwargs is None:
            self.optim_kwargs = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.}
        else:
            self.optim_kwargs = optim_kwargs

        self.pileups = MarkerPileups(self.db)
        self._prepare_pileups()

    def _prepare_pileups(self):
        alignments = CachedReadAlignments(
            cfg.database_cfg.get_database().multifasta_file,
            self.reads
        )

        for t_idx in range(len(self.reads)):
            for alignment_result in alignments.get_alignments(t_idx):
                self.pileups.add_aligned_evidence(alignment_result)

    def evaluate(self, population: Population):
        model = GenerativeModel(
            bacteria_pop=population,
            read_length=self.base_model.read_length,
            times=self.base_model.times,
            mu=torch.zeros(population.num_strains(), device=cfg.torch_cfg.device),
            tau_1_dof=self.base_model.tau_1_dof,
            tau_1_scale=self.base_model.tau_1_scale,
            tau_dof=self.base_model.tau_dof,
            tau_scale=self.base_model.tau_scale,
            read_error_model=self.base_model.error_model
        )

        solver = BBVISolver(model=model, data=self.reads, correlation_type="strain")
        solver.solve(
            optim_class=torch.optim.Adam,
            optim_args=self.optim_kwargs,
            iters=self.optim_iters,
            num_samples=self.optim_mc_samples,
            print_debug_every=100
        )

        x = solver.gaussian_posterior.mean()
        prior_ll = model.log_likelihood_x(x)
        data_ll = model.data_likelihood(x, solver.data_likelihoods.matrices)
        posterior_ll_est = solver.gaussian_posterior.log_likelihood(x)
        data_ll_estimate = data_ll + prior_ll - posterior_ll_est

        population.clear_fragment_space()
        return data_ll_estimate.item(), solver

    def perform_search(self) -> Tuple[Population, float, BBVISolver]:
        current_pop = self.base_model.bacteria_pop
        likelihood, solution = self.evaluate(current_pop)

        while True:
            # Retrieve the marker to try and resolve.
            try:
                unresolved_marker = self.pileups.marker_with_largest_pileup()
            except NoSuchMarkerException:
                break

            # Try the best available variant.
            variant_seq, variant_selector = next(self.pileups.proposal_variants(unresolved_marker))

            # Create necessary marker/strain instances.
            locs = np.where(variant_selector)[0]
            new_id = "{}%({})".format(
                unresolved_marker.id,
                "|".join(["{}:{}".format(i, variant_seq[i]) for i in locs])
            )

            new_marker = Marker(
                id=new_id,
                name=unresolved_marker.name,
                seq=variant_seq,
                metadata=MarkerMetadata(
                    parent_accession="N/A",
                    file_path=None
                ) if unresolved_marker.metadata is not None else None
            )
            new_strains = [
                construct_derived_strain(tgt_strain, unresolved_marker, new_marker)
                for tgt_strain in self.db.get_strains_with_marker(unresolved_marker)
            ]

            # Create the new Population instance.
            new_pop = Population(
                strains=current_pop.strains + new_strains,
                extra_strain=False
            )

            # Use this population instance to evaluate.
            new_likelihood, new_solution = self.evaluate(new_pop)
            if new_likelihood > likelihood:
                current_pop = new_pop
                likelihood = new_likelihood
                solution = new_solution

                self.pileups.accept_variant(unresolved_marker, new_marker.seq)
                for new_strain in new_strains:
                    self.db.backend.add_strain(new_strain)
                logger.debug("Variant improvement found. # Strains = {}".format(new_pop.num_strains))
            else:
                logger.debug("Variant improvement not found. Terminating search.")
                break
        return current_pop, likelihood, solution


def construct_derived_strain(base_strain: Strain, base_marker: Marker, variant_marker: Marker):
    new_id = "{}#{}".format(
        base_strain.id,
        variant_marker.id
    )
    new_markers = []

    for marker in base_strain.markers:
        if marker.id == base_marker.id:
            new_markers.append(variant_marker)
        else:
            new_markers.append(marker)
    return Strain(
        genome_length=base_strain.genome_length,
        id=new_id,
        markers=new_markers,
        metadata=None
    )
