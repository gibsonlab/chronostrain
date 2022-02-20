from abc import abstractmethod
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import torch
from Bio import SeqIO
import scipy.special
from scipy.stats import rv_discrete

from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, Fragment, Population, Marker, Strain
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap

from chronostrain.config import create_logger, cfg
from chronostrain.util.sparse import RowSectionedSparseMatrix, SparseMatrix

logger = create_logger(__name__)


class FragmentFrequencyComputer(object):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase, min_overlap_ratio: float):
        self.frag_length_rv = frag_length_rv
        self.db: StrainDatabase = db
        self.min_overlap_ratio = min_overlap_ratio

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
                         counts: Dict[Fragment, List[Tuple[Marker, Strain, int]]]
                         ) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        pass

    @abstractmethod
    def save_matrix(self, matrix: Union[RowSectionedSparseMatrix, torch.Tensor], out_path: Path):
        pass

    @abstractmethod
    def load_matrix(self, matrix_path: Path) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        pass

    def frag_log_ll(self, frag: Fragment, marker: Marker, frag_position: int) -> float:
        """
        Compute the partial likelihood
            P(M=m|S=s) * SUM_{k >= |f|} P(F=f|M=m,K=k) * P(K = k)
        where
            P(F=f|M=m,K=k) = SUM_{i in <positions_with_0.5_coverage>}
                (1/# sliding windows of length k)
                * 1_{window at position i is f}
        by marginalizing over all possible positions and all possible "parent lengths" k (insert measurement lengths,
        e.g. the chunk of the insert that makes it into the read),
        and across all positions i.
        """
        def length_normalizer(k: int, min_overlap_ratio: float) -> float:
            """
            Computes the number of sliding windows of length `k`, with padding on either end of the marker,
            such that the window always overlaps for half of its length with the marker.
            """
            return len(marker) + (2 * (1 - min_overlap_ratio) * k) - k + 1

        if (frag_position == 1) or (frag_position == len(marker) - len(frag) + 1):
            # Edge case: at the start or end of marker.
            return np.log(len(marker)) + scipy.special.logsumexp([
                self.frag_length_rv.logpmf(k) - np.log(length_normalizer(k, self.min_overlap_ratio))
                for k in range(
                    len(frag),
                    int(np.max([self.frag_length_rv.mean() + 2 * self.frag_length_rv.std(), len(frag)]))
                )
            ])
        else:
            n_sliding_windows = length_normalizer(len(frag), self.min_overlap_ratio)
            ans = np.log(len(marker)) + self.frag_length_rv.logpmf(len(frag)) - np.log(n_sliding_windows)
            return ans

    def compute_frequencies(self,
                            fragments: FragmentSpace,
                            population: Population,
                            bwa_fastmap_output_path: Path
                            ) -> Union[RowSectionedSparseMatrix, torch.Tensor]:
        self.search(fragments, bwa_fastmap_output_path, max_num_hits=10 * population.num_strains())
        return self.construct_matrix(
            fragments,
            population,
            self.parse(fragments, bwa_fastmap_output_path)
        )

    def search(self, fragments: FragmentSpace, output_path: Path, max_num_hits: int):
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        fragments_path = self.fragments_to_fasta(fragments, output_path.parent)
        bwa_fastmap(
            output_path=output_path,
            reference_path=self.db.multifasta_file,
            query_path=fragments_path,
            min_smem_len=fragments.min_frag_len,
            max_interval_size=max_num_hits
        )

    @staticmethod
    def fragments_to_fasta(fragments: FragmentSpace, data_dir: Path) -> Path:
        out_path = data_dir / "all_fragments.fasta"

        logger.debug(f"Writing {len(fragments)} records to fasta.")
        with open(out_path, 'w') as f:
            for fragment in fragments:
                SeqIO.write([fragment.to_seqrecord()], f, 'fasta')
        return out_path

    def parse(self,
              fragments: FragmentSpace,
              fastmap_output_path: Path) -> Dict[Fragment, List[Tuple[Marker, Strain, int]]]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
        hits_dict: Dict[Fragment, List[Tuple[Marker, Strain, int]]] = {}
        with open(fastmap_output_path) as f:
            for fragment_line in f:
                fragment_line = fragment_line.strip()
                if fragment_line == '':
                    continue

                frag_line_tokens = fragment_line.split('\t')
                if frag_line_tokens[0] != 'SQ':
                    raise ValueError(f"Expected header line to start with `SQ`. Got: {fragment_line}")

                # Parse fragment
                frag_idx = int(frag_line_tokens[1][len('FRAGMENT_'):])
                fragment = fragments.get_fragment_by_index(frag_idx)

                # Parse matches
                exact_match_found = False
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

                    # Only accept a match if it spans the whole fragment (we are looking for exact matches)
                    frag_start = int(match_tokens[1])
                    frag_end = int(match_tokens[2])

                    if frag_end - frag_start == len(fragment):
                        # Parse the strain/marker hits and tally them up.
                        exact_match_found = True
                        mapping_locations: List[Tuple[Marker, Strain, int]] = []
                        for marker_hit_token in match_tokens[4:]:
                            if marker_hit_token == "*":
                                raise ValueError(
                                    f"Output of bwa fastmap didn't report output for {frag_line_tokens[0]} "
                                    "(usually occurs because there were too many hits). "
                                    "Try raising the value of the -w option."
                                )

                            marker_desc, pos = marker_hit_token.split(':')
                            strain_id, gene_name, gene_id = marker_desc.split('|')

                            marker = self.db.get_marker(gene_id)
                            strain = self.db.get_strain(strain_id)
                            pos = int(pos)
                            if pos < 0:
                                # Skip `-` strands (since fragments are canonically defined using the forward strand)
                                continue

                            mapping_locations.append(
                                (marker, strain, pos)
                            )
                        hits_dict[fragment] = mapping_locations
                if not exact_match_found:
                    logger.warning(
                        f"No exact matches found for fragment {fragment.index} [{fragment.nucleotide_content()}]."
                        f"Validate the output of bwa fastmap!"
                    )

        return hits_dict


class SparseFragmentFrequencyComputer(FragmentFrequencyComputer):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase, min_overlap_ratio: float):
        super().__init__(frag_length_rv, db, min_overlap_ratio)

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.npz'

    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         all_frag_hits: Dict[Fragment, List[Tuple[Marker, Strain, int]]]) -> RowSectionedSparseMatrix:
        """
        :param fragments:
        :param population:
        :param all_frag_hits: Represents the mapping <Fragment> -> [ <hit_1_marker, hit_1_strain, hit_1_pos>, ... ]
        :return:
        """
        strain_indices = []
        frag_indices = []
        matrix_values = []

        for fragment, frag_hits in all_frag_hits.items():
            strains = np.array([
                population.strain_index(hit_strain)
                for _, hit_strain, _ in frag_hits
            ], dtype=int)

            frag_lls = np.array([
                self.frag_log_ll(fragment, hit_marker, hit_pos)
                for hit_marker, _, hit_pos in frag_hits
            ], dtype=float)

            for strain_idx in np.unique(strains):
                strain_indices.append(strain_idx)
                frag_indices.append(fragment.index)
                matrix_values.append(scipy.special.logsumexp(
                    frag_lls[strains == strain_idx]
                ))
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
