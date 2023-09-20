from pathlib import Path
from typing import List, Tuple, Iterator, Union

import jax.numpy as jnp
import numpy as cnp
import scipy
import scipy.special
import scipy.sparse
from jax.experimental.sparse import BCOO
import pandas as pd
from numba import prange, njit

from scipy.stats import nbinom as negative_binomial
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
                 dtype='bfloat16',
                 n_threads: int = 1):
        self.frag_nbinom_n = frag_nbinom_n
        self.frag_nbinom_p = frag_nbinom_p
        self.db: StrainDatabase = db
        self.fragments = fragments
        self.fragment_pairs = fragment_pairs
        self.strains = strains
        self.cache = ReadStrainCollectionCache(reads, db)
        self.dtype = dtype
        self.n_threads = n_threads
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
                    fastmap_out_dir,
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

        cnp.seterr(divide='ignore')
        all_marker_lens = matches_df['HitMarkerLen'].to_numpy()
        all_pos = matches_df['HitPos'].to_numpy()
        for (frag_idx, s_idx), section in groupings:
            # noinspection PyTypeChecker
            strain_indices.append(s_idx)
            frag_indices.append(frag_idx)
            matrix_values.append(
                frag_log_ll_numpy(
                    frag_len=len(self.fragments[frag_idx]),
                    window_lens=window_lens,
                    window_lens_log_pmf=window_len_logpmf,
                    hit_marker_lens=all_marker_lens[section.index],
                    hit_pos=all_pos[section.index]
                )
            )
            pbar.update(1)
        cnp.seterr(divide='warn')

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

    def compute_exact_matches(self, target_dir: Path, max_num_hits: int) -> pd.DataFrame:
        output_path = target_dir / 'bwa_fastmap.out'
        target_dir.mkdir(exist_ok=True, parents=True)
        bwa_fastmap_output = self._invoke_bwa_fastmap(output_path, max_num_hits)

        return pd.DataFrame(
            [
                (frag_idx, strain_idx, hit_marker_len, hit_pos)
                for frag_idx, strain_idx, hit_marker_len, hit_pos in self._parse(bwa_fastmap_output)
            ],
            columns=['FragIdx', 'StrainIdx', 'HitMarkerLen', 'HitPos'],
        ).astype({
            'FragIdx': 'int', 'StrainIdx': 'int', 'HitMarkerLen': 'int', 'HitPos': 'int'
        })

    def _invoke_bwa_fastmap(
            self,
            output_path: Path,
            max_num_hits: int
    ) -> Path:
        logger.debug("Creating index for exact matches.")
        bwa_index(self.db.multifasta_file, bwa_cmd='bwa', check_suffix='bwt')

        output_path.unlink(missing_ok=True)
        for frag_len, frag_fasta in self.fragments.fragment_files_by_length(output_path.parent):
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
            fastmap_output_path: Path
    ) -> Iterator[Tuple[int, int, int, int]]:
        """
        :return: A dictionary mapping (fragment) -> List of (strain, num_hits) pairs
        """
        total_entries = 0
        with open(fastmap_output_path, 'rt') as f:
            for line in f:
                if line.startswith('SQ'):
                    total_entries += 1

        # === Construct dataframe.
        marker_lookup = {
            marker.id: (strain_idx, len(marker))
            for strain_idx, strain in enumerate(self.strains)
            for marker in strain.markers
        }

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

                        if marker_id not in marker_lookup:
                            raise ValueError(f"Couldn't find marker with ID {marker_id} in table.")
                        strain_idx, hit_marker_len = marker_lookup[marker_id]

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


@njit
def frag_log_ll_numpy(
        frag_len: int,
        window_lens: cnp.ndarray,
        window_lens_log_pmf: cnp.ndarray,
        hit_marker_lens: cnp.ndarray,
        hit_pos: cnp.ndarray,
) -> float:
    assert len(window_lens.shape) == 1
    assert len(window_lens_log_pmf.shape) == 1
    assert window_lens.shape[0] == window_lens_log_pmf.shape[0]

    n_hits = hit_pos.shape[0]  # int
    n_edge_hits = cnp.sum(
        (hit_pos == 1) | (hit_pos == hit_marker_lens - frag_len + 1)  # fragment maps to edge
    )

    """
    The following np.where() chain is equivalent to:
    
    n_matching_windows = cnp.zeros(window_lens.shape)
    for i in prange(len(window_lens)):
        w = window_lens[i]
        if w == frag_len:
            n_matching_windows[i] = n_hits
        elif w > frag_len:
            n_matching_windows[i] = n_edge_hits
    """
    n_matching_windows = cnp.where(window_lens == frag_len, n_hits, n_edge_hits)
    n_matching_windows = cnp.where(window_lens >= frag_len, n_matching_windows, 0)

    return numba_logsumexp_1d(
        window_lens_log_pmf + cnp.log(n_matching_windows)
    )


@njit
def numba_logsumexp_1d(x) -> float:
    offset = cnp.max(x)
    return offset + cnp.log(
        cnp.sum(
            cnp.exp(x - offset)
        )
    )
