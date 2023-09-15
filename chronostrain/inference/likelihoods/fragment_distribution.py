from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Union

import jax.numpy as jnp
import numpy as cnp
import scipy
from jax.experimental.sparse import BCOO
import pandas as pd
import shutil

from scipy.stats import rv_discrete, nbinom as negative_binomial

from chronostrain.database import StrainDatabase
from chronostrain.model import Strain, Fragment, FragmentSpace, FragmentPairSpace, \
    FragmentFrequencySparse, TimeSeriesReads
from chronostrain.util.external import bwa_index, bwa_fastmap
from .cache import ReadStrainCollectionCache

from chronostrain.logging import create_logger
logger = create_logger(__name__)


def convert_tabs_to_newlines(src_path: Path, target_path: Path, append: bool):
    import subprocess

    if append:
        write_mode = 'a'
    else:
        write_mode = 'w'
    with open(src_path, 'r') as infile, open(target_path, write_mode) as outfile:
        process = subprocess.Popen(
            ['tr', '\"\t\"', '\"\n\"'],
            stdin=infile,
            stdout=outfile
        )
        process.communicate()


class FragmentFrequencyComputer(object):
    def __init__(self,
                 frag_nbinom_n: float,
                 frag_nbinom_p: float,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 strains: List[Strain],
                 fragments: FragmentSpace,
                 fragment_pairs: FragmentPairSpace,
                 dtype='bfloat16'):
        self.frag_nbinom_n = frag_nbinom_n
        self.frag_nbinom_p = frag_nbinom_p
        self.db: StrainDatabase = db
        self.fragments = fragments
        self.fragment_pairs = fragment_pairs
        self.strains = strains
        self.cache = ReadStrainCollectionCache(reads, db)
        self.dtype = dtype
        self.exact_match_df: Union[None, pd.DataFrame] = None

    def get_frequencies(
            self,
    ) -> Tuple[FragmentFrequencySparse, FragmentFrequencySparse]:
        """
        Computes a list of fragment frequencies; one per specified ordering instance (usually one per timepoint).
        """
        logger.debug("Loading fragment frequencies of {} fragments on {} strains.".format(
            len(self.fragments),
            len(self.strains)
        ))

        # ====== Run the cached computation.
        freqs = self.cache.call(
            relative_filepath=Path('fragment_frequencies') / 'frag_freqs.{}.npz'.format(self.dtype),
            fn=self.compute_frequencies,
            save=lambda p, x: x.save(p),
            load=FragmentFrequencySparse.load
        )

        pair_freqs = self.cache.call(
            relative_filepath=Path('fragment_frequencies') / 'frag_pair_freqs.{}.npz'.format(self.dtype),
            fn=lambda: self.compute_paired_frequencies(freqs),
            save=lambda p, x: x.save(p),
            load=FragmentFrequencySparse.load
        )

        logger.debug("Finished loading fragment frequencies.")
        return freqs, pair_freqs

    def get_exact_matches(self) -> pd.DataFrame:
        if self.exact_match_df is None:
            fastmap_tmp_dir = self.cache.cache_dir / 'bwa_fastmap'
            self.exact_match_df = self.cache.call(
                relative_filepath='fragment_frequencies/matches_all.feather',
                fn=lambda: self.compute_exact_matches(
                    self.fragments, fastmap_tmp_dir,
                    max_num_hits=10 * self.db.num_strains()
                ),
                save=lambda p, df: df.to_feather(p),
                load=lambda p: pd.read_feather(p)
            )
        return self.exact_match_df

    def compute_frequencies(self) -> FragmentFrequencySparse:
        strain_indices = []
        frag_indices = []
        matrix_values = []

        frag_len_rv = negative_binomial(self.frag_nbinom_n, self.frag_nbinom_p)

        for (frag_idx, s_idx), section in self.get_exact_matches().groupby(['FragIdx', 'StrainIdx']):
            # noinspection PyTypeChecker
            strain_indices.append(s_idx)
            frag_indices.append(frag_idx)
            matrix_values.append(
                frag_log_ll_numpy(
                    len(self.fragments[frag_idx]),
                    frag_len_rv.logpmf,
                    cnp.array(section['HitMarkerLen']),
                    cnp.array(section['HitPos'])
                )
            )

        indices = jnp.array(list(zip(frag_indices, strain_indices)))
        matrix_values = jnp.array(matrix_values, dtype=self.dtype)
        return FragmentFrequencySparse(
            BCOO((matrix_values, indices), shape=(len(self.fragments), len(self.strains)))
        )

    def compute_paired_frequencies(self, single_freqs: FragmentFrequencySparse) -> FragmentFrequencySparse:
        strain_indices = []
        frag_pair_indices = []
        matrix_values = []
        for f_idx1, f_idx2, pair_idx in self.fragment_pairs:
            s_idxs1, values1 = single_freqs.slice_by_fragment(f_idx1)
            s_idxs2, values2 = single_freqs.slice_by_fragment(f_idx2)
            s_intersect, locs1, locs2 = jnp.intersect1d(s_idxs1, s_idxs2, return_indices=True)
            val_intersect_1 = values1[locs1]
            val_intersect_2 = values2[locs2]

            # s_intersect, val_intersect_1, val_intersect_2 all have matching shapes.
            strain_indices.append(s_intersect)
            frag_pair_indices.append(jnp.array([pair_idx] * len(s_intersect)))
            matrix_values.append(val_intersect_1 * val_intersect_2)  # evaluate the product.
        return FragmentFrequencySparse(
            BCOO(
                (
                    jnp.concatenate(matrix_values),  # data; this should be a length N array
                    jnp.stack(  # indices; this should be an Nx2 array
                        [
                            jnp.concatenate(frag_pair_indices),
                            jnp.concatenate(strain_indices)
                        ],
                        axis=1  # this ensures concatenation across 2nd dimension
                    )
                ),
                shape=(len(self.fragment_pairs), len(self.strains))
            )
        )

    def compute_exact_matches(self, fragments: FragmentSpace, tmp_dir: Path, max_num_hits: int) -> pd.DataFrame:
        bwa_fastmap_output = self._invoke_bwa_fastmap(fragments, tmp_dir, max_num_hits)
        df_entries = []
        for fragment, frag_mapping_locs in self._parse(fragments, bwa_fastmap_output):
            for hit_marker_len, strain_idx, hit_pos in frag_mapping_locs:
                df_entries.append(
                    (fragment.index, strain_idx, hit_marker_len, hit_pos)
                )
        df = pd.DataFrame(
            df_entries,
            columns=['FragIdx', 'StrainIdx', 'HitMarkerLen', 'HitPos'],
        )
        shutil.rmtree(tmp_dir)
        return df

    def _invoke_bwa_fastmap(
            self,
            fragments: FragmentSpace,
            tmp_dir: Path,
            max_num_hits: int
    ) -> Path:
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file, bwa_cmd='bwa', check_suffix='bwt')

        tmp_dir.mkdir(exist_ok=True, parents=True)
        output_path = tmp_dir / 'bwa_fastmap.out'
        for frag_len, frag_fasta in fragments.fragment_files_by_length(output_path.parent):
            fastmap_output_path = output_path.with_stem(
                f'{output_path.stem}.{frag_len}'
            )
            bwa_fastmap(
                output_path=fastmap_output_path,
                reference_path=self.db.multifasta_file,
                query_path=frag_fasta,
                max_interval_size=max_num_hits,
                min_smem_len=frag_len
            )

            convert_tabs_to_newlines(fastmap_output_path, output_path, append=True)
            fastmap_output_path.unlink()
            frag_fasta.unlink()
        return output_path

    def _parse(
            self,
            fragments: FragmentSpace,
            fastmap_output_path: Path
    ) -> Iterator[Tuple[Fragment, List[Tuple[int, int, int]]]]:
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
            self.db.backend.marker_df.reset_index(),
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
                while em_line.startswith("EM"):
                    frag_start = int(f.readline().strip())
                    frag_end = int(f.readline().strip())
                    num_hits = int(f.readline().strip())

                    # Parse all hits on this "EM" record.
                    for hit_idx in range(num_hits):
                        marker_hit_token = f.readline().strip()
                        if frag_end - frag_start != frag_len:
                            continue  # don't do anything if it's not exact hits.

                        tokens = marker_hit_token.split(':')
                        pos = tokens[-1]
                        marker_desc = ":".join(tokens[:-1])
                        pos = int(pos)
                        gene_name, marker_id = marker_desc.split('|')

                        res = marker_id_dataframe.loc[marker_id]
                        if isinstance(res, pd.DataFrame):
                            for _, row in res.iterrows():
                                strain_id = row['StrainId']
                                marker_idx = row['MarkerIdx']
                                strain_idx = strain_id_to_idx[strain_id]
                                marker = self.db.backend.markers[marker_idx]
                                hit_marker_len = len(marker)
                        else:
                            strain_id = res['StrainId']
                            marker_idx = res['MarkerIdx']
                            strain_idx = strain_id_to_idx[strain_id]
                            marker = self.db.backend.markers[marker_idx]
                            hit_marker_len = len(marker)

                        if pos < 0:
                            """
                            Note: BWA fastmap output convention is to write +/-[position], where position is the first
                            base on the FORWARD strand. The sign tells us whether or not to apply revcomp afterwards.
                            """
                            pos = -pos  # minus means negative strand; just flip the sign.

                        fragment_mapping_locations.append((hit_marker_len, strain_idx, pos))
                    # Attempt to read next section. Either another "EM" or a break "//".
                    em_line = f.readline()

                if len(fragment_mapping_locations) > 0:
                    yield fragment, fragment_mapping_locations
                else:
                    logger.warning(
                        f"No exact matches found for fragment ID={fragment.index}."
                        f"Validate the output of bwa fastmap!"
                    )


def frag_log_ll_numpy(frag_len: int,
                      frag_len_rv: rv_discrete,
                      hit_marker_lens: cnp.ndarray,  # an array of marker lengths that the fragment hits.
                      hit_pos: cnp.ndarray  # an array of marker positions that the fragment hit at.
                      ) -> float:
    window_lens = cnp.arange(
        frag_len,
        1 + max(int(frag_len_rv.mean() + 2 * frag_len_rv.std()), frag_len)
    )  # length-W

    n_hits = len(hit_pos)
    cond1 = window_lens == frag_len  # window is exactly frag_len
    cond2 = (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)  # edge map

    # ==== this is old code
    # n_matching_windows = cnp.sum(cond1[None, :] | cond2[:, None], axis=0)

    # ==== slight speedup...
    # per window length w, get the # of sliding windows (of length w) that yields fragment f.
    n_matching_windows = cnp.where(cond1, n_hits, cond2.sum())

    _mask = n_matching_windows > 0
    # noinspection PyTypeChecker
    result: cnp.ndarray = scipy.special.logsumexp(
        # a=frag_len_rv.logpmf(window_lens[_mask]) + cnp.log(n_matching_windows[_mask]),
        a=frag_len_rv.logpmf(window_lens[_mask]) + cnp.log(n_matching_windows[_mask]),
        keepdims=False
    )  # This calculation marginalizes across possible window lengths.
    return result.item()
