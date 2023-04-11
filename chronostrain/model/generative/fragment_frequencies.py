from abc import abstractmethod
from pathlib import Path
from typing import Union, List, Tuple, Iterator

import torch
import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
from scipy.special import logsumexp

from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, Fragment, Population, Marker, Strain
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap
from chronostrain.config import cfg
from chronostrain.util.math.matrices import RowSectionedSparseMatrix, SparseMatrix

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class FragmentFrequencyComputer(object):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase):
        self.frag_length_rv = frag_length_rv
        self.db: StrainDatabase = db

    def get_frequencies(self,
                        fragments: FragmentSpace,
                        population: Population
                        ) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        logger.debug("Loading fragment frequencies of {} fragments on {} strains.".format(
            len(fragments),
            population.num_strains()
        ))
        cache = ComputationCache(
            CacheTag(
                markers=self.db.multifasta_file,
                strains=population.strains,
                fragments=fragments
            )
        )

        bwa_output_path = cache.cache_dir / self.relative_matrix_path().with_name('bwa_fastmap.output')

        # ====== Run the cached computation.
        matrix = cache.call(
            relative_filepath=self.relative_matrix_path(),
            fn=lambda: self.compute_frequencies(fragments, population, bwa_output_path),
            save=lambda path, obj: self.save_matrix(obj, path),
            load=lambda path: self.load_matrix(path)
        )

        # Validate the matrix.
        if isinstance(matrix, RowSectionedSparseMatrix):
            for frag_idx, frag_locs in enumerate(matrix.locs_per_row):
                if len(frag_locs) == 0:
                    logger.warning(f"Fragment IDX={frag_idx} contains no hits across strains. ELBO might return -inf.")

        return matrix

    @abstractmethod
    def relative_matrix_path(self) -> Path:
        pass

    @abstractmethod
    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         all_frag_hits: Iterator[Tuple[Fragment, List[Tuple[Marker, Strain, int]]]]
                         ) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        pass

    @abstractmethod
    def save_matrix(self, matrix: Union[RowSectionedSparseMatrix, torch.Tensor], out_path: Path):
        pass

    @abstractmethod
    def load_matrix(self, matrix_path: Path) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        pass

    def compute_frequencies(self,
                            fragments: FragmentSpace,
                            population: Population,
                            bwa_fastmap_output_path: Path
                            ) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        self.search(fragments, bwa_fastmap_output_path, max_num_hits=10 * self.db.num_strains())
        return self.construct_matrix(
            fragments,
            population,
            self.parse(fragments, bwa_fastmap_output_path)
        )

    def search(self, fragments: FragmentSpace, output_path: Path, max_num_hits: int):
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file, bwa_cmd='bwa')

        output_path.parent.mkdir(exist_ok=True, parents=True)
        fragments_path = fragments.to_fasta(output_path.parent)
        bwa_fastmap(
            output_path=output_path,
            reference_path=self.db.multifasta_file,
            query_path=fragments_path,
            max_interval_size=max_num_hits,
            min_smem_len=min(len(f) for f in fragments)
        )

    def parse(self,
              fragments: FragmentSpace,
              fastmap_output_path: Path) -> Iterator[Tuple[Fragment, List[Tuple[Marker, Strain, int]]]]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
        total_entries = 0
        with open(fastmap_output_path, 'rt') as f:
            for line in f:
                if line.startswith('SQ'):
                    total_entries += 1

        from tqdm import tqdm
        pbar = tqdm(total=total_entries)

        with open(fastmap_output_path, 'rt') as f:
            for fragment_line in f:
                fragment_line = fragment_line.strip()
                if fragment_line == '':
                    continue

                frag_line_tokens = fragment_line.split('\t')
                if frag_line_tokens[0] != 'SQ':
                    raise ValueError(f"Expected header line to start with `SQ`. Got: {fragment_line}")

                # Parse fragment
                fragment = fragments.from_fasta_record_id(frag_line_tokens[1])
                fragment_mapping_locations: List[Tuple[Marker, Strain, int]] = []

                # Parse matches
                exact_match_found = False
                total_matches = 0
                for match_line in f:
                    match_line = match_line.strip()

                    # Section end
                    if match_line == '//':
                        break

                    if exact_match_found:
                        continue

                    match_tokens = match_line.split('\t')
                    if match_tokens[0] != 'EM':
                        raise ValueError(f"Expected header line to start with `EM`. Got: {match_line}")
                    total_matches += len(match_tokens)

                    # Only accept a match if it spans the whole fragment (we are looking for exact matches)
                    frag_start = int(match_tokens[1])
                    frag_end = int(match_tokens[2])
                    if frag_end - frag_start == len(fragment):
                        # Parse the strain/marker hits and tally them up.
                        exact_match_found = True
                        for marker_hit_token in match_tokens[4:]:
                            if marker_hit_token == "*":
                                raise ValueError(
                                    f"Output of bwa fastmap didn't report output for {frag_line_tokens[1]} "
                                    "(usually occurs because there were too many hits). "
                                    "Try raising the value of the -w option."
                                )

                            marker_desc, pos = marker_hit_token.split(':')
                            gene_name, gene_id = marker_desc.split('|')

                            marker = self.db.get_marker(gene_id)
                            for strain in self.db.get_strains_with_marker(marker):
                                pos = int(pos)
                                if pos < 0:
                                    # Skip `-` strands (since fragments are canonically defined using the forward strand)
                                    continue

                                fragment_mapping_locations.append(
                                    (marker, strain, pos)
                                )
                if not exact_match_found:
                    logger.warning(
                        f"No exact matches found for fragment ID={fragment.id}."
                        f"Validate the output of bwa fastmap!"
                    )
                else:
                    yield fragment, fragment_mapping_locations
                pbar.update(1)


class SparseFragmentFrequencyComputer(FragmentFrequencyComputer):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase, fragments: FragmentSpace, min_overlap_ratio: float):
        super().__init__(frag_length_rv, db)
        self.fragments = fragments
        self.strains = self.db.all_strains()
        self.min_overlap_ratio: float = min_overlap_ratio

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.npz'

    # def construct_matrix_fast(self,
    #                           fragments: FragmentSpace,
    #                           population: Population,
    #                           all_frag_hits: np.ndarray,
    #                           ) -> RowSectionedSparseMatrix:
    #     prior_lls = {
    #         self.frag_length_rv.logpmf()
    #         for fl in set(frag_lens)
    #     }
    #
    #
    #     # Use acceleration/multithreading-enabled libraries.
    #     strain_indices = []
    #     frag_indices = []
    #     matrix_values = []
    #
    #     return RowSectionedSparseMatrix(
    #         indices=torch.tensor([frag_indices, strain_indices], device=cfg.torch_cfg.device, dtype=torch.long),
    #         values=torch.tensor(matrix_values, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype),
    #         dims=(len(fragments), population.num_strains()),
    #         force_coalesce=False
    #     )

    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         all_frag_hits: Iterator[Tuple[Fragment, List[Tuple[Marker, Strain, int]]]]
                         ) -> RowSectionedSparseMatrix:
        """
        :param fragments:
        :param population:
        :param all_frag_hits: Represents the mapping <Fragment> -> [ <hit_1_marker, hit_1_strain, hit_1_pos>, ... ]
        :return:
        """
        strain_indices = []
        frag_indices = []
        matrix_values = []

        all_strain_marker_lengths = {}
        strain_index_mapping = {
            strain.id: idx
            for idx, strain in enumerate(self.strains)
        }

        for fragment, frag_hits in all_frag_hits:
            hits_df = pd.DataFrame([
                {'strain_idx': strain_index_mapping[hit_strain.id], 'hit_marker_len': len(hit_marker), 'hit_pos': hit_pos}
                for hit_marker, hit_strain, hit_pos in frag_hits
            ])
            for strain_idx, section in hits_df.groupby('strain_idx'):
                if strain_idx not in all_strain_marker_lengths:
                    all_strain_marker_lengths[strain_idx] = np.array([len(m) for m in self.strains[strain_idx].markers])

                try:
                    strain_indices.append(strain_idx)
                except KeyError:
                    continue
                frag_indices.append(fragment.index)
                matrix_values.append(
                    self.frag_log_ll_numpy(
                        len(fragment), all_strain_marker_lengths[strain_idx], section['hit_marker_len'].to_numpy(), section['hit_pos'].to_numpy()
                    )
                )
        return RowSectionedSparseMatrix(
            indices=torch.tensor([frag_indices, strain_indices], device=cfg.torch_cfg.device, dtype=torch.long),
            values=torch.tensor(matrix_values, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype),
            dims=(len(fragments), population.num_strains()),
            force_coalesce=False
        )

    def save_matrix(self, matrix: RowSectionedSparseMatrix, out_path: Path):
        matrix.save(out_path)

    def load_matrix(self, matrix_path: Path) -> RowSectionedSparseMatrix:
        return RowSectionedSparseMatrix.from_sparse_matrix(SparseMatrix.load(
            matrix_path,
            device=cfg.torch_cfg.device,
            dtype=cfg.torch_cfg.default_dtype
        ))

    # def frag_log_ll(self, frag: Fragment, strain: Strain, hits: List[Tuple[Marker, int]]) -> float:
    #     window_lens = torch.arange(
    #         len(frag),
    #         1 + max(int(self.frag_length_rv.mean() + 2 * self.frag_length_rv.std()), len(frag)),
    #         dtype=cfg.torch_cfg.default_dtype
    #     )  # 1-dimensional, [|f|, |f|+1, ..., µ+2σ], length = W
    #
    #     n_total_windows = strain.metadata.total_len - window_lens + 1
    #
    #     def is_edge_positioned(marker: Marker, pos: int) -> bool:
    #         return (pos == 1) or (pos == len(marker) - len(frag) + 1)
    #
    #     # the number of windows which ``look like'' the read.
    #     n_matching_windows = torch.sum(
    #         torch.unsqueeze(torch.tensor([is_edge_positioned(m, p) for m, p in hits], dtype=torch.bool), 1)  # (N_HITS) x 1
    #         | torch.unsqueeze(window_lens == len(frag), 0),  # (1 x W)
    #         dim=0
    #     )  # length W
    #
    #     return float(torch.logsumexp(
    #         torch.tensor(self.frag_length_rv.logpmf(window_lens))  # window length prior
    #         + torch.log(n_matching_windows) - torch.log(n_total_windows),  # proportion of hitting windows versus entire strain.
    #         dim=0,
    #         keepdim=False
    #     ).item())

    def frag_log_ll_numpy(self, frag_len: int, strain_marker_lengths: np.ndarray, hit_marker_lens: np.ndarray, hit_pos: np.ndarray) -> float:
        window_lens = np.arange(
            frag_len,
            1 + max(int(self.frag_length_rv.mean() + 2 * self.frag_length_rv.std()), frag_len)
        )  # length-W

        n_windows = np.sum(strain_marker_lengths) + len(strain_marker_lengths) * (
                (2 * (1 - self.min_overlap_ratio) - 1) * window_lens + 1
        )  # length-W

        cond1 = window_lens == frag_len
        cond2 = (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)
        n_matching_windows = np.sum(cond1[None, :] | cond2[:, None], axis=0)
        _mask = n_matching_windows > 0
        result: np.ndarray = logsumexp(
            self.frag_length_rv.logpmf(window_lens[_mask]) + np.log(n_matching_windows[_mask]) - np.log(n_windows[_mask]),
            keepdims=False
        )
        return float(result)
    #
    #
    # def frag_log_ll(self, frag: Fragment, strain: Strain, hits: List[Tuple[Marker, int]]) -> float:
    #     marker_lengths = torch.tensor([len(marker) for marker in strain.markers], dtype=cfg.torch_cfg.default_dtype)
    #
    #     window_lens = torch.arange(
    #         len(frag),
    #         1 + max(int(self.frag_length_rv.mean() + 2 * self.frag_length_rv.std()), len(frag)),
    #         dtype=cfg.torch_cfg.default_dtype
    #     )
    #
    #     n_windows = torch.sum(
    #         torch.unsqueeze(marker_lengths, 1)  # (M x 1)
    #         + torch.unsqueeze((2 * (1 - self.min_overlap_ratio) * window_lens) - window_lens + 1, 0),  # (1 x W)
    #         dim=0
    #     )  # length W
    #
    #     def is_edge_positioned(marker: Marker, pos: int) -> bool:
    #         return (pos == 1) or (pos == len(marker) - len(frag) + 1)
    #
    #     n_matching_windows = torch.sum(
    #         torch.unsqueeze(torch.tensor([is_edge_positioned(m, p) for m, p in hits], dtype=torch.bool), 1)  # H x 1
    #         | torch.unsqueeze(window_lens == len(frag), 0),  # (1 x W)
    #         dim=0
    #     )  # length W
    #
    #     return float(torch.logsumexp(
    #         torch.tensor(self.frag_length_rv.logpmf(window_lens))
    #         + torch.log(n_matching_windows)
    #         - torch.log(n_windows),
    #         dim=0,
    #         keepdim=False
    #     ).item())
