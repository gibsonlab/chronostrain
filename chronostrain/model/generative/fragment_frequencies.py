from pathlib import Path
from typing import List, Tuple, Iterator

import jax.numpy as jnp
import numpy as cnp
import jax.experimental.sparse as jsparse
import pandas as pd
import multiprocessing
import scipy.special
from scipy.stats import rv_discrete

from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, Fragment, Population, Marker, Strain
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap
from chronostrain.config import cfg

from chronostrain.logging import create_logger
from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix

logger = create_logger(__name__)


class FragmentFrequencyComputer(object):
    def __init__(self, frag_length_rv: rv_discrete, db: StrainDatabase, fragments: FragmentSpace, min_overlap_ratio: float):
        self.frag_length_rv = frag_length_rv
        self.db: StrainDatabase = db
        self.fragments = fragments
        self.strains = self.db.all_strains()
        self.min_overlap_ratio: float = min_overlap_ratio

    def get_frequencies(self,
                        fragments: FragmentSpace,
                        population: Population
                        ) -> jsparse.BCOO:
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
            save=save_sparse_matrix,
            load=load_sparse_matrix
        )

        # Validate the matrix.
        if not isinstance(matrix, jsparse.BCOO):
            raise ValueError("Expected fragment frequencies to be jsparse.BCOO, but got {}".format(type(matrix)))

        return matrix

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.npz'

    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         all_frag_hits: Iterator[Tuple[Fragment, List[Tuple[Marker, Strain, int]]]]
                         ) -> jsparse.BCOO:
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

        logger.debug("Using {} cores for fragment frequency task.".format(cfg.model_cfg.num_cores))
        p = multiprocessing.Pool(cfg.model_cfg.num_cores)
        def _arg_gen():
            for fragment, frag_hits in all_frag_hits:
                hits_df = pd.DataFrame([
                    {'strain_idx': strain_index_mapping[hit_strain.id], 'hit_marker_len': len(hit_marker), 'hit_pos': hit_pos}
                    for hit_marker, hit_strain, hit_pos in frag_hits
                ])
                for _sidx, section in hits_df.groupby('strain_idx'):
                    # noinspection PyTypeChecker
                    _sidx = int(_sidx)
                    if _sidx not in all_strain_marker_lengths:
                        all_strain_marker_lengths[_sidx] = cnp.array([len(m) for m in self.strains[_sidx].markers])
                    yield [
                        _sidx,
                        fragment.index,
                        self.frag_length_rv,
                        self.min_overlap_ratio,
                        len(fragment),
                        all_strain_marker_lengths[_sidx],
                        cnp.array(section['hit_marker_len']),
                        cnp.array(section['hit_pos'])
                    ]

        for strain_idx, fragment_idx, ll in p.imap_unordered(_ll_wrapper, _arg_gen()):
            strain_indices.append(strain_idx)
            frag_indices.append(fragment_idx)
            matrix_values.append(ll)

        indices = jnp.array(list(zip(frag_indices, strain_indices)))
        matrix_values = jnp.array(matrix_values)
        return jsparse.BCOO(
            (matrix_values, indices),
            shape=(len(fragments), population.num_strains())
        )

    def compute_frequencies(self,
                            fragments: FragmentSpace,
                            population: Population,
                            bwa_fastmap_output_path: Path
                            ) -> jsparse.BCOO:
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
                        f"No exact matches found for fragment ID={fragment.index}."
                        f"Validate the output of bwa fastmap!"
                    )
                else:
                    yield fragment, fragment_mapping_locations
                pbar.update(1)


def frag_log_ll_numpy(strain_idx:int,
                      frag_idx: int,
                      frag_len_rv: rv_discrete,
                      min_overlap: float,
                      frag_len: int,
                      strain_marker_lengths: cnp.ndarray,
                      hit_marker_lens: cnp.ndarray,
                      hit_pos: cnp.ndarray) -> Tuple[float, float, float]:
    window_lens = cnp.arange(
        frag_len,
        1 + max(int(frag_len_rv.mean() + 2 * frag_len_rv.std()), frag_len)
    )  # length-W

    n_windows = cnp.sum(strain_marker_lengths) + len(strain_marker_lengths) * (
            (2 * (1 - min_overlap) - 1) * window_lens + 1
    )  # length-W

    cond1 = window_lens == frag_len
    cond2 = (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)
    n_matching_windows = cnp.sum(cond1[None, :] | cond2[:, None], axis=0)
    _mask = n_matching_windows > 0
    # noinspection PyTypeChecker
    result: cnp.ndarray = scipy.special.logsumexp(
        a=frag_len_rv.logpmf(window_lens[_mask]) + cnp.log(n_matching_windows[_mask]) - cnp.log(n_windows[_mask]),
        keepdims=False
    )
    return strain_idx, frag_idx, float(result)


def _ll_wrapper(args):
    return frag_log_ll_numpy(*args)
