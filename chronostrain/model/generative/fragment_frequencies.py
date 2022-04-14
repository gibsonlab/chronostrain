from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Tuple, Iterator, Dict

import torch
from Bio import SeqIO
from scipy.stats import rv_discrete

from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, Fragment, Population, Marker, Strain
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap

from chronostrain.config import create_logger, cfg
from chronostrain.util.sparse import RowSectionedSparseMatrix, SparseMatrix

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
        self.search(fragments, bwa_fastmap_output_path, max_num_hits=10 * population.num_strains())
        return self.construct_matrix(
            fragments,
            population,
            self.parse(fragments, bwa_fastmap_output_path)
        )

    def search(self, fragments: FragmentSpace, output_path: Path, max_num_hits: int):
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file, bwa_cmd='bwa')

        output_path.parent.mkdir(exist_ok=True, parents=True)
        fragments_path = self.fragments_to_fasta(fragments, output_path.parent)
        bwa_fastmap(
            output_path=output_path,
            reference_path=self.db.multifasta_file,
            query_path=fragments_path,
            max_interval_size=max_num_hits,
            min_smem_len=min(len(f) for f in fragments)
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
              fastmap_output_path: Path) -> Iterator[Tuple[Fragment, List[Tuple[Marker, Strain, int]]]]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
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
                fragment_mapping_locations: List[Tuple[Marker, Strain, int]] = []

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
                        for marker_hit_token in match_tokens[4:]:
                            if marker_hit_token == "*":
                                raise ValueError(
                                    f"Output of bwa fastmap didn't report output for {frag_line_tokens[1]} "
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

                            fragment_mapping_locations.append(
                                (marker, strain, pos)
                            )
                if not exact_match_found:
                    logger.warning(
                        f"No exact matches found for fragment {fragment.index} [{fragment.nucleotide_content()}]."
                        f"Validate the output of bwa fastmap!"
                    )
                else:
                    yield fragment, fragment_mapping_locations


class SparseFragmentFrequencyComputer(FragmentFrequencyComputer):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase, min_overlap_ratio: float):
        super().__init__(frag_length_rv, db)
        self.min_overlap_ratio: float = min_overlap_ratio

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.npz'

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

        for fragment, frag_hits in all_frag_hits:
            hits_per_strain: Dict[Strain, List[Tuple[Marker, int]]] = defaultdict(list)
            for hit_marker, hit_strain, hit_pos in frag_hits:
                hits_per_strain[hit_strain].append((hit_marker, hit_pos))

            for strain, hits in hits_per_strain.items():
                strain_indices.append(population.strain_index(strain))
                frag_indices.append(fragment.index)
                matrix_values.append(self.frag_log_ll(fragment, strain, hits))
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

    def frag_log_ll(self, frag: Fragment, strain: Strain, hits: List[Tuple[Marker, int]]) -> float:
        marker_lengths = torch.tensor([len(marker) for marker in strain.markers], dtype=cfg.torch_cfg.default_dtype)

        window_lens = torch.arange(
            len(frag),
            1 + max(int(self.frag_length_rv.mean() + 2 * self.frag_length_rv.std()), len(frag)),
            dtype=cfg.torch_cfg.default_dtype
        )

        n_windows = torch.sum(
            torch.unsqueeze(marker_lengths, 1)  # (M x 1)
            + torch.unsqueeze((2 * (1 - self.min_overlap_ratio) * window_lens) - window_lens + 1, 0),  # (1 x W)
            dim=0
        )  # length W

        def is_edge_positioned(marker: Marker, pos: int) -> bool:
            return (pos == 1) or (pos == len(marker) - len(frag) + 1)

        n_matching_windows = torch.sum(
            torch.unsqueeze(torch.tensor([is_edge_positioned(m, p) for m, p in hits], dtype=torch.bool), 1)  # H x 1
            | torch.unsqueeze(window_lens == len(frag), 0),  # (1 x W)
            dim=0
        )  # length W

        return float(torch.logsumexp(
            torch.tensor(self.frag_length_rv.logpmf(window_lens))
            + torch.log(n_matching_windows)
            - torch.log(n_windows),
            dim=0,
            keepdim=False
        ).item())
