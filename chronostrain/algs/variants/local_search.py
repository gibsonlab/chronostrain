from dataclasses import dataclass
from typing import Iterator, List

import torch

from .. import BBVISolver
from ..subroutines import CachedReadAlignments
from ... import cfg
from ...model import GenerativeModel
from ...model.io import TimeSeriesReads
from ...util.sam_handler import SamHandler


class VariantSearchSpace(object):
    def __init__(self, reads: TimeSeriesReads):
        self.visited_map = {}
        self._preprocess(reads)

    def _preprocess(self, reads):
        alignments = CachedReadAlignments(
            cfg.database_cfg.get_database().multifasta_file,
            reads
        )

        for t_idx in range(len(reads)):
            for sam_handler in alignments.get_alignments(t_idx):
                self._preprocess_alignment(sam_handler)

    def _preprocess_alignment(self, sam_output: SamHandler):
        """
        Keep a pileup of aligned variants.
        :param sam_output:
        :return:
        """
        for samline in sam_output.mapped_lines():
            read_quality = samline.read_quality
            raise NotImplementedError("TODO!")


    def neighbors(self, model: GenerativeModel) -> Iterator[GenerativeModel]:
        """
        Look at the list of variants, propose all of them one by one.
        :param model:
        :return:
        """
        # Idea: look at the alignment to each strain, tally up all sites with variants.

    def mark_visited(self, model: GenerativeModel):
        self.visited_map[model.bacteria_pop] = True

    def already_visited(self, model: GenerativeModel) -> bool:
        return self.visited_map.get(model.bacteria_pop, default=False)


@dataclass
class SearchNode:
    model: GenerativeModel
    metric: float


class VariantSearchAlgorithm(object):
    def __init__(self,
                 search_space: VariantSearchSpace,
                 iters: int = 2000,
                 learning_rate: float = 1e-3,
                 num_bbvi_samples: int = 100):
        self.search_space = search_space
        self.learning_rate = learning_rate
        self.iters = iters
        self.num_bbvi_samples = num_bbvi_samples

    def likelihood_metric(self, model: GenerativeModel, reads) -> float:
        # TODO: initializing this class automatically starts read likelihood calculation,
        #  but there must be a way to modify previous calculations. (models differ by known bases,
        #  so only a few fragments change.)
        solver = BBVISolver(model=model, data=reads, correlation_type="strain")

        solver.solve(
            optim_class=torch.optim.Adam,
            optim_args={'lr': self.learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
            iters=self.iters,
            num_samples=self.num_bbvi_samples,
            print_debug_every=100
        )

        x = solver.gaussian_posterior.mean()
        prior_ll = model.log_likelihood_x(x)
        data_ll = model.data_likelihood(x, solver.data_likelihoods.matrices)
        posterior_ll_est = solver.gaussian_posterior.log_likelihood(x)
        data_ll_estimate = data_ll + prior_ll - posterior_ll_est
        return data_ll_estimate

    def do_search(self, initial: GenerativeModel, reads: TimeSeriesReads) -> GenerativeModel:
        """
        A depth-first search with pruning based on the implemented metric.
        :param initial:
        :param reads:
        :return:
        """
        stack: List[SearchNode] = [SearchNode(initial, self.likelihood_metric(initial, reads))]
        self.search_space.mark_visited(initial)
        best_node = stack[0]

        while len(stack) > 0:
            v = stack.pop()
            for new_model in self.search_space.neighbors(v.model):
                # If visited, don't do anything.
                if self.search_space.already_visited(new_model):
                    continue

                # New node instance (with evaluated metric).
                new_node = SearchNode(new_model, self.likelihood_metric(new_model, reads))

                # Remember the best-fitting model so far.
                if new_node.metric > best_node.metric:
                    best_node = new_node

                # If metric decreases, prune search.
                prune = new_node.metric < v.metric
                if not prune:
                    stack.append(new_node)

                # Mark as visited.
                self.search_space.mark_visited(new_model)
        return best_node.model
