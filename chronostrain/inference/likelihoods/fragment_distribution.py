from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Union

import jax.numpy as jnp
import numpy as cnp
import scipy
import scipy.special
import scipy.sparse
from jax.experimental.sparse import BCOO
import pandas as pd

from scipy.stats import rv_discrete, nbinom as negative_binomial
from tqdm import tqdm

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
            fastmap_out_dir = self.cache.cache_dir / 'bwa_fastmap'

            def _save(p, df):
                p.parent.mkdir(exist_ok=True, parents=True)
                df.to_feather(p)

            self.exact_match_df = self.cache.call(
                relative_filepath='fragment_frequencies/matches_all.feather',
                fn=lambda: self.compute_exact_matches(
                    self.fragments, fastmap_out_dir,
                    max_num_hits=10 * self.db.num_strains()
                ),
                save=_save,
                load=lambda p: pd.read_feather(p)
            )
        return self.exact_match_df

    def compute_frequencies(self) -> FragmentFrequencySparse:
        strain_indices = []
        frag_indices = []
        matrix_values = []

        frag_len_rv = negative_binomial(self.frag_nbinom_n, self.frag_nbinom_p)
        min_frag_len = self.fragments.min_len
        max_frag_len = max(int(frag_len_rv.mean() + 2 * frag_len_rv.std()), min_frag_len)

        window_lens = cnp.arange(min_frag_len, 1 + max_frag_len)
        window_len_logpmf = frag_len_rv.logpmf(window_lens)

        matches_df = self.get_exact_matches()
        groupings = matches_df.groupby(['FragIdx', 'StrainIdx'])
        count = groupings.ngroups
        pbar = tqdm(total=count, unit='matrix-entry')
        for (frag_idx, s_idx), section in groupings:
            # noinspection PyTypeChecker
            strain_indices.append(s_idx)
            frag_indices.append(frag_idx)
            # a = frag_log_ll_numpy(
            #     len(self.fragments[frag_idx]),
            #     frag_len_rv,
            #     section['HitMarkerLen'].to_numpy(),
            #     section['HitPos'].to_numpy()
            # )
            # b = frag_log_ll_numpy_new(
            #     frag_len=len(self.fragments[frag_idx]),
            #     window_lens=window_lens,
            #     window_len_log_pmf=window_len_logpmf,
            #     hit_marker_lens=section['HitMarkerLen'].to_numpy(),
            #     hit_pos=section['HitPos'].to_numpy()
            # )
            # assert a == b
            matrix_values.append(
                frag_log_ll_numpy_new(
                    frag_len=len(self.fragments[frag_idx]),
                    window_lens=window_lens,
                    window_len_log_pmf=window_len_logpmf,
                    hit_marker_lens=section['HitMarkerLen'].to_numpy(),
                    hit_pos=section['HitPos'].to_numpy()
                )
            )
            pbar.update(1)
        # exit(1)

        indices = jnp.array(list(zip(frag_indices, strain_indices)))
        matrix_values = jnp.array(matrix_values, dtype=self.dtype)
        return FragmentFrequencySparse(
            BCOO((matrix_values, indices), shape=(len(self.fragments), len(self.strains)))
        )

    def compute_paired_frequencies(self, single_freqs: FragmentFrequencySparse) -> FragmentFrequencySparse:
        indices = cnp.array(single_freqs.matrix.indices)
        log_counts = cnp.array(single_freqs.matrix.data)

        """
        Main strategy here is to first convert the sparse matrix into scipy's CSR array,
        then take two slices: X[frag1], X[frag2] and then compute the log-space product of expected counts, X[frag1] + X[frag2].
        But... need to be careful here! 
        
        The sparse arrays (for us) are stored such that "empty" locations have value -inf, but Scipy's CSR assumes that 
        empty locs are Zeroes. Thus, doing a simple sum X[frag1] + X[frag2] may accidentally cancel things out.
        (e.g. log(x) + log(y) = 0 means that x*y = 1.0, but scipy's sparse logic EMPTIES out this entry, 
        so log(x) + log(y) incorrectly becomes -inf.)
        
        The solution here is to force ALL entries to be positive, so no cancellations occur.
        """
        # ================= pick an offset for numerical stability.
        n_positives = cnp.sum(log_counts > 0)
        n_negatives = cnp.sum(log_counts < 0)
        eps = 1.0
        if n_positives > 0 and n_negatives > 0:
            # possible canceellations; do some preprocessing.
            offset = cnp.min(log_counts) - eps  # add an extra epsilon factor to avoid canceelation on the minimal element.
            log_counts = log_counts - offset  # now all entries are positive.
        else:
            offset = 0.0

        # ================ do sparse operations using CPU sparse arrays.
        n_zeros = cnp.sum(log_counts == 0)
        if n_zeros > 0:
            raise RuntimeError(
                "Sparse matrix's data array has zero values. Since the interpretation is that empty entries are -inf, "
                "this contradicts the model. This is an unusual error; try running the program again with a "
                "different/randomized offset."
            )

        cpu_sparse = scipy.sparse.csr_array(
            (log_counts, (indices[:, 0], indices[:, 1])),
            shape=(len(self.fragments), len(self.strains))
        )
        del indices
        del log_counts

        # =============== Extract the slice indexes
        f1 = cnp.zeros(len(self.fragment_pairs), dtype=int)
        f2 = cnp.zeros(len(self.fragment_pairs), dtype=int)
        for f_idx1, f_idx2, pair_idx in self.fragment_pairs:
            f1[pair_idx] = f_idx1
            f2[pair_idx] = f_idx2

        # =============== Take the slices.
        freq1 = cpu_sparse[f1, :]
        freq2 = cpu_sparse[f2, :]
        logger.debug("freq1 nnz = {}, freq2 nnz = {}".format(freq1.nnz, freq2.nnz))

        # =============== Compute the operations.
        _mutual_nnz = (freq1 != 0) * (freq2 != 0)
        _sum = (freq1 + freq2) * _mutual_nnz  # the offsets have combined; it is now 2 times the original offset.
        del _mutual_nnz

        _sum = _sum.tocoo()
        jax_coo_data = jnp.array(_sum.data + 2 * offset)
        jax_coo_indices = jnp.stack(
            [_sum.row, _sum.col],
            axis=1
        )
        logger.debug("Result nnz = {}".format(_sum.nnz))
        return FragmentFrequencySparse(
            BCOO(
                (jax_coo_data, jax_coo_indices),
                shape=(len(self.fragment_pairs), len(self.strains))
            )
        )

    def compute_exact_matches(self, fragments: FragmentSpace, target_dir: Path, max_num_hits: int) -> pd.DataFrame:
        output_path = target_dir / 'bwa_fastmap.out'
        target_dir.mkdir(exist_ok=True, parents=True)
        bwa_fastmap_output = self._invoke_bwa_fastmap(fragments, output_path, max_num_hits)

        return pd.DataFrame(
            [
                (frag_idx, strain_idx, hit_marker_len, hit_pos)
                for frag_idx, strain_idx, hit_marker_len, hit_pos in self._parse(fragments, bwa_fastmap_output)
            ],
            columns=['FragIdx', 'StrainIdx', 'HitMarkerLen', 'HitPos'],
        )

    def _invoke_bwa_fastmap(
            self,
            fragments: FragmentSpace,
            output_path: Path,
            max_num_hits: int
    ) -> Path:
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file, bwa_cmd='bwa', check_suffix='bwt')

        output_path.unlink(missing_ok=True)
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
    ) -> Iterator[Tuple[Fragment, int, int, int]]:
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
            for _ in tqdm(range(total_entries), unit='frag'):
                sq_line = f.readline()
                if sq_line is None or (not sq_line.startswith("SQ")):
                    break

                frag_id = f.readline().strip()
                frag_idx = self.fragments.frag_index_from_record_id(frag_id)
                frag_len = int(f.readline().strip())
                n_hits_for_fragment = 0

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

                        try:
                            tokens = marker_hit_token.split(':')
                            pos = tokens[-1]
                            marker_desc = ":".join(tokens[:-1])
                            pos = int(pos)
                            gene_name, marker_id = marker_desc.split('|')
                        except Exception as e:
                            logger.error(f"Encountered a hit line that couldn't be parsed: `{marker_hit_token}`")
                            raise e

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

                        yield frag_idx, strain_idx, hit_marker_len, pos
                        n_hits_for_fragment += 1

                    # Attempt to read next section. Either another "EM" or a break "//".
                    em_line = f.readline()

                if n_hits_for_fragment == 0:
                    frag = self.fragments.get_fragment_by_index(frag_idx)
                    logger.warning(
                        f"No exact matches found for fragment (IDX={frag_idx}, SEQ={frag.seq.nucleotides()})."
                        f"Validate the output of bwa fastmap!"
                    )


def frag_log_ll_numpy_new(
        frag_len: int,
        window_lens: cnp.ndarray,
        window_len_log_pmf: cnp.ndarray,
        hit_marker_lens: cnp.ndarray,
        hit_pos: cnp.ndarray,
) -> float:
    n_hits = hit_pos.shape[0]  # int
    n_edge_hits = cnp.sum(
        (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)  # framgent maps to edge
    )

    n_matching_windows = cnp.where(window_lens == frag_len, n_hits, n_edge_hits)
    n_matching_windows = cnp.where(window_lens >= frag_len, n_matching_windows, 0)
    _mask = n_matching_windows > 0  # we are about to take logs...

    # noinspection PyUnresolvedReferences
    return scipy.special.logsumexp(
        a=window_len_log_pmf[_mask] + cnp.log(n_matching_windows[_mask]),
        keepdims=False,
        return_sign=False
    ).item()

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
