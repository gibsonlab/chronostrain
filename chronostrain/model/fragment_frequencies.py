from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from Bio import SeqIO
from scipy.stats import rv_discrete

from chronostrain.model import FragmentSpace, Fragment, Population
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap

from chronostrain.config import create_logger, cfg
from chronostrain.util.sparse import RowSectionedSparseMatrix, SparseMatrix

logger = create_logger(__name__)


class FragmentFrequencyComputer(object):
    def __init__(self, all_markers_path: Path, length_rv: rv_discrete):
        self.all_markers_path = all_markers_path
        self.length_rv = length_rv

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
                markers=self.all_markers_path,
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
                         counts: Dict[Fragment, Counter]
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
        self.search(fragments, bwa_fastmap_output_path, max_num_hits=10 * population.num_strains())
        return self.construct_matrix(
            fragments,
            population,
            self.parse(fragments, population, bwa_fastmap_output_path)
        )

    def search(self, fragments: FragmentSpace, output_path: Path, max_num_hits: int):
        logger.debug("Creating index for exact matches.")
        bwa_index(self.all_markers_path)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        fragments_path = self.fragments_to_fasta(fragments, output_path.parent)
        bwa_fastmap(
            output_path=output_path,
            reference_path=self.all_markers_path,
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

    @staticmethod
    def parse(fragments: FragmentSpace,
              population: Population,
              fastmap_output_path: Path) -> Dict[Fragment, Counter]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
        strain_indices = {
            strain.id: strain_idx
            for strain_idx, strain in enumerate(population.strains)
        }

        hits_dict: Dict[Fragment, Counter] = {}
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
                        strain_counts = Counter()
                        for marker_hit_token in match_tokens[4:]:
                            if marker_hit_token == "*":
                                raise ValueError(
                                    f"Output of bwa fastmap didn't report output for {frag_line_tokens[0]} "
                                    "(usually occurs because there were too many hits). "
                                    "Try raising the value of the -w option."
                                )

                            marker_desc, pos = marker_hit_token.split(':')
                            strain_id, gene_name, gene_id = marker_desc.split('|')
                            strain_idx = strain_indices[strain_id]
                            strain_counts[strain_idx] += 1
                        hits_dict[fragment] = strain_counts
                if not exact_match_found:
                    logger.warning(
                        f"No exact matches found for fragment {fragment.index} [{fragment.nucleotide_content()}]."
                        f"Validate the output of bwa fastmap!"
                    )

        return hits_dict


class SparseFragmentFrequencyComputer(FragmentFrequencyComputer):
    def __init__(self, all_markers_path: Path, length_rv: rv_discrete):
        super().__init__(all_markers_path, length_rv)

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.npz'

    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         counts: Dict[Fragment, Counter]) -> RowSectionedSparseMatrix:
        strain_indices = []
        frag_indices = []
        matrix_values = []

        for fragment, frag_counts in counts.items():
            for strain_idx, strain_frag_hits in frag_counts.items():
                strain_indices.append(strain_idx)
                frag_indices.append(fragment.index)

                length_ll = self.length_rv.logpmf(len(fragment))
                frag_log_ll = (
                        length_ll
                        + np.log(strain_frag_hits)
                        - np.log(population.strains[strain_idx].num_marker_frags(len(fragment)))
                )

                matrix_values.append(frag_log_ll)

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
