from pathlib import Path
from typing import List, Tuple, Iterator, Callable, Dict

import jax.numpy as jnp
import numpy as cnp
import scipy
import jax.experimental.sparse as jsparse
import pandas as pd

from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, Fragment, Population
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import bwa_index, bwa_fastmap

from chronostrain.logging import create_logger
from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix

logger = create_logger(__name__)


def convert_tabs_to_newlines(target_path: Path):
    import subprocess
    src_path = target_path.parent / '{}.original'.format(target_path.name)
    target_path.rename(src_path)

    with open(src_path, 'r') as infile, open(target_path, 'w') as outfile:
        process = subprocess.Popen(
            ['tr', '\"\t\"', '\"\n\"'],
            stdin=infile,
            stdout=outfile
        )
        process.communicate()
    src_path.unlink()


class FragmentFrequencyComputer(object):
    def __init__(self,
                 frag_nbinom_n: float,
                 frag_nbinom_p: float,
                 db: StrainDatabase,
                 fragments: FragmentSpace,
                 min_overlap_ratio: float,
                 dtype='bfloat16'):
        self.frag_nbinom_n = frag_nbinom_n
        self.frag_nbinom_p = frag_nbinom_p
        self.db: StrainDatabase = db
        self.fragments = fragments
        self.strains = self.db.all_strains()
        self.min_overlap_ratio: float = min_overlap_ratio
        self.dtype = dtype

    def get_frequencies(self,
                        fragments: FragmentSpace,
                        population: Population,
                        cache: ComputationCache
                        ) -> jsparse.BCOO:
        logger.debug("Loading fragment frequencies of {} fragments on {} strains.".format(
            len(fragments),
            population.num_strains()
        ))

        bwa_output_path = cache.cache_dir / self.relative_matrix_path().with_name('bwa_fastmap.output')

        # ====== Run the cached computation.
        matrix = cache.call(
            relative_filepath=self.relative_matrix_path(),
            fn=lambda: self.compute_frequencies(fragments, population, bwa_output_path),
            save=save_sparse_matrix,
            load=load_sparse_matrix
        )

        logger.debug("Cleaning up. (Deleting fastmap output to save disk space)")
        bwa_output_path.unlink()

        # Validate the matrix.
        if not isinstance(matrix, jsparse.BCOO):
            raise ValueError("Expected fragment frequencies to be jsparse.BCOO, but got {}".format(type(matrix)))

        logger.debug("Finished loading fragment frequencies.")
        return matrix

    def relative_matrix_path(self) -> Path:
        return Path('fragment_frequencies') / 'sparse_frag_freqs.{}.npz'.format(self.dtype)

    def construct_matrix(self,
                         fragments: FragmentSpace,
                         population: Population,
                         all_frag_hits: Iterator[Tuple[Fragment, List[Tuple[int, int, int]]]]
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

        frag_len_rv = scipy.stats.nbinom(self.frag_nbinom_n, self.frag_nbinom_p)
        frag_len_mean = frag_len_rv.mean()
        frag_len_std = frag_len_rv.std()

        for fragment, frag_hits in all_frag_hits:
            hits_df = pd.DataFrame([
                {
                    'strain_idx': hit_strain_idx,
                    'hit_marker_len': hit_marker_len,
                    'hit_pos': hit_pos
                }
                for hit_marker_len, hit_strain_idx, hit_pos in frag_hits
            ])

            for _sidx, section in hits_df.groupby('strain_idx'):
                # noinspection PyTypeChecker
                _sidx = int(_sidx)
                strain_indices.append(_sidx)
                frag_indices.append(fragment.index)
                matrix_values.append(
                    frag_log_ll_numpy(
                        frag_len_mean,
                        frag_len_std,
                        len(fragment),
                        frag_len_rv.logpmf,
                        cnp.array(section['hit_marker_len']),
                        cnp.array(section['hit_pos'])
                    )
                )

        indices = jnp.array(list(zip(frag_indices, strain_indices)))
        matrix_values = jnp.array(matrix_values, dtype=self.dtype)
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
        if not fragments_path.exists():
            raise FileNotFoundError("Expected fragments file, but it does not exist. "
                                    "This is an unexpected bug. Verify the cache to continue.")
        bwa_fastmap(
            output_path=output_path,
            reference_path=self.db.multifasta_file,
            query_path=fragments_path,
            max_interval_size=max_num_hits,
            min_smem_len=min(len(f) for f in fragments)
        )
        logger.debug("Converting tabs to newlines for parsing.")
        convert_tabs_to_newlines(output_path)

    def parse(self,
              fragments: FragmentSpace,
              fastmap_output_path: Path) -> Iterator[Tuple[Fragment, List[Tuple[int, int, int]]]]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
        total_entries = 0
        with open(fastmap_output_path, 'rt') as f:
            for line in f:
                if line.startswith('SQ'):
                    total_entries += 1

        # pandas-specific!
        # TODO: make search queries optimized by ID directly into the backend implementation.
        from chronostrain.database.backend import PandasAssistedBackend
        assert isinstance(self.db.backend, PandasAssistedBackend)
        marker_id_dataframe = self.db.backend.strain_df.merge(
            self.db.backend.marker_df,
            on='MarkerIdx',
            how='right'
        ).set_index('MarkerId')
        # ==========================
        strain_id_to_idx: Dict[str, int] = {
            s.id: i
            for i, s in enumerate(self.db.all_strains())
        }
        # ==========================

        from tqdm import tqdm
        with open(fastmap_output_path, 'rt') as f:
            for _ in tqdm(range(total_entries)):
                fragment_mapping_locations: List[Tuple[int, int, int]] = []
                sq_line = f.readline()
                if sq_line is None or (not sq_line.startswith("SQ")):
                    break

                frag_id = f.readline().strip()
                fragment = fragments.from_fasta_record_id(frag_id)
                frag_len = int(f.readline().strip())

                em_line = f.readline()
                if em_line is None or (not em_line.startswith("EM")):
                    raise RuntimeError("Expected `EM` tag, but found {}".format(em_line))

                frag_start = int(f.readline().strip())
                frag_end = int(f.readline().strip())
                num_hits = int(f.readline().strip())

                for hit_idx in range(num_hits):
                    marker_hit_token = f.readline().strip()
                    marker_desc, pos = marker_hit_token.split(':')
                    pos = int(pos)
                    gene_name, marker_id = marker_desc.split('|')

                    if frag_end - frag_start != frag_len:
                        continue  # don't do anything if it's not exact hits.

                    if pos < 0:
                        continue  # Skip `-` strands (since fragments are canonically defined using the forward strand)

                    res = marker_id_dataframe.loc[marker_id]
                    if isinstance(res, pd.DataFrame):
                        for _, row in res.iterrows():
                            strain_id = row['StrainId']
                            marker_idx = row['MarkerIdx']
                            strain_idx = strain_id_to_idx[strain_id]
                            marker = self.db.backend.markers[marker_idx]
                            hit_marker_len = len(marker)
                            fragment_mapping_locations.append((hit_marker_len, strain_idx, pos))
                    else:
                        strain_id = res['StrainId']
                        marker_idx = res['MarkerIdx']
                        strain_idx = strain_id_to_idx[strain_id]
                        marker = self.db.backend.markers[marker_idx]
                        hit_marker_len = len(marker)
                        fragment_mapping_locations.append((hit_marker_len, strain_idx, pos))

                if len(fragment_mapping_locations) > 0:
                    yield fragment, fragment_mapping_locations
                else:
                    logger.warning(
                        f"No exact matches found for fragment ID={fragment.index}."
                        f"Validate the output of bwa fastmap!"
                    )

                break_line = f.readline().strip()
                if break_line is None or break_line != "//":
                    raise RuntimeError("Expected break token `//`, but  found {}".format(break_line))



def frag_log_ll_numpy(frag_len_mean: float,
                      frag_len_std: float,
                      frag_len: int,
                      logpmf: Callable,
                      hit_marker_lens: cnp.ndarray,
                      hit_pos: cnp.ndarray) -> Tuple[float, float, float]:
    window_lens = cnp.arange(
        frag_len,
        1 + max(int(frag_len_mean + 2 * frag_len_std), frag_len)
    )  # length-W

    cond1 = window_lens == frag_len
    cond2 = (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)

    # n_matching_windows = cnp.sum(cond1[None, :] | cond2[:, None], axis=0)
    n_matching_windows = cnp.where(cond1, len(cond2), cond2.sum())

    _mask = n_matching_windows > 0
    # noinspection PyTypeChecker
    result: cnp.ndarray = scipy.special.logsumexp(
        # a=frag_len_rv.logpmf(window_lens[_mask]) + cnp.log(n_matching_windows[_mask]),
        a=logpmf(window_lens[_mask]) + cnp.log(n_matching_windows[_mask]),
        keepdims=False
    )
    return result.item()


def _ll_wrapper(args):
    return frag_log_ll_numpy(*args)
