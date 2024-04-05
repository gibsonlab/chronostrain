from typing import *
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special

from chronostrain.config import cfg
from chronostrain.database import StrainDatabase
from chronostrain.model import Strain, TimeSeriesReads
from chronostrain.inference import GaussianWithGumbelsPosterior

from matplotlib.axes import Axes


def parse_strains(db: StrainDatabase, strain_txt: Path):
    with open(strain_txt, 'rt') as f:
        return [
            db.get_strain(l.strip())
            for l in f
        ]


def total_marker_len(strain: Strain) -> int:
    return sum(len(m) for m in strain.markers)


def posterior_with_bf_threshold(
        infant_id: str,
        posterior: GaussianWithGumbelsPosterior,
        inference_strains: List[Strain],
        output_strains: List[Strain],
        adhoc_clustering: Dict[str, Strain],
        posterior_threshold: float
) -> Tuple[Dict[str, float], np.ndarray]:
    # Raw random samples.
    n_samples = 5000
    rand = posterior.random_sample(n_samples)
    g_samples = np.array(posterior.reparametrized_gaussians(rand['std_gaussians'], posterior.get_parameters()))  # T x N x S
    z_samples = np.array(posterior.reparametrized_zeros(rand['std_gumbels'], posterior.get_parameters()))
    # print(posterior.get_parameters())# N x S
    
    n_times = g_samples.shape[0]
    n_inference_strains = g_samples.shape[-1]
    assert n_inference_strains == len(inference_strains)

    # Calculate bayes factors.
    posterior_inclusion_p = scipy.special.expit(-posterior.get_parameters()['gumbel_diff'])
    # print(posterior_inclusion_p)
    # posterior_inclusion_bf = (posterior_inclusion_p / (1 - posterior_inclusion_p)) * ((1 - prior_p) / prior_p)

    # Calculate abundance estimates using BF thresholds.
    indicators = np.full(n_inference_strains, fill_value=False, dtype=bool)
    indicators[posterior_inclusion_p > posterior_threshold] = True
    print("{} of {} inference strains passed Posterior p(Z_s|Data) > {}".format(np.sum(indicators), n_inference_strains, posterior_threshold))
    
    log_indicators = np.empty(n_inference_strains, dtype=float)
    log_indicators[indicators] = 0.0
    log_indicators[~indicators] = -np.inf
    pred_abundances_raw = scipy.special.softmax(g_samples + np.expand_dims(log_indicators, axis=[0, 1]), axis=-1)
    
    # Unwind the adhoc grouping.
    pred_abundances = np.zeros(shape=(n_times, n_samples, len(output_strains)), dtype=float)
    adhoc_indices = {s.id: i for i, s in enumerate(inference_strains)}
    output_indices = {s.id for s in output_strains}
    for s_idx, s in enumerate(output_strains):
        adhoc_rep = adhoc_clustering[s.id]
        adhoc_idx = adhoc_indices[adhoc_rep.id]
        adhoc_clust_ids = set(s_ for s_, clust in adhoc_clustering.items() if clust.id == adhoc_rep.id)
        adhoc_sz = len(adhoc_clust_ids.intersection(output_indices))
        # if adhoc_sz > 1:
        #     print(f"{s.id} [{s.metadata.genus} {s.metadata.species}, {s.name}] --> adhoc sz = {adhoc_sz} (Adhoc Cluster {adhoc_rep.id} [{adhoc_rep.metadata.genus} {adhoc_rep.metadata.species}, {adhoc_rep.name}])")
        pred_abundances[:, :, s_idx] = pred_abundances_raw[:, :, adhoc_idx] / adhoc_sz
    return {
        s.id: posterior_inclusion_p[
            adhoc_indices[adhoc_clustering[s.id].id]
        ]
        for i, s in enumerate(output_strains)
    }, pred_abundances


def parse_adhoc_clusters(db: StrainDatabase, txt_file: Path) -> Dict[str, Strain]:
    clust = {}
    with open(txt_file, "rt") as f:
        for line in f:
            tokens = line.strip().split(":")
            rep = tokens[0]
            members = tokens[1].split(",")
            for member in members:
                clust[member] = db.get_strain(rep)
    return clust


class Taxon:
    def __init__(self, genus: str, species: str):
        self.genus = genus
        self.species = species


class InferenceNotFinishedError(BaseException):
    pass


class ChronostrainResult(object):
    def __init__(self, name: str, db: StrainDatabase, posterior_class, out_dir: Path, input_path: Path, posterior_threshold: float = 0.95):
        self.name = name
        if not (out_dir.parent / 'inference.DONE').exists():
            raise InferenceNotFinishedError()

        self.adhoc_clusters: Dict[str, Strain] = parse_adhoc_clusters(db, out_dir / "adhoc_cluster.txt")
        self.inference_strains: List[Strain] = parse_strains(db, out_dir / 'strains.txt')
        self.display_strains = self.inference_strains
        # self.display_strains: List[Strain] = list(db.get_strain(x) for x in self.adhoc_clusters.keys())

        self.filt_input_path = input_path
        self.reads = TimeSeriesReads.load_from_file(input_path)

        self.num_filtered_reads: np.ndarray = np.array([len(reads_t) for reads_t in self.reads], dtype=int)
        self.read_depths: np.ndarray = np.array([reads_t.read_depth for reads_t in self.reads], dtype=int)
        self.time_points = np.array([reads_t.time_point for reads_t in self.reads], dtype=float)

        posterior = posterior_class(
            len(self.inference_strains),
            len(self.time_points),
            cfg.engine_cfg.dtype
        )
        posterior.load(Path(out_dir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)))
        self.posterior_p, self.posterior_samples = posterior_with_bf_threshold(
            self.name,
            posterior, self.inference_strains, self.display_strains, self.adhoc_clusters, posterior_threshold
        )
        self.posterior_threshold = posterior_threshold
        self.timeseries_df = self._timeseries_dataframe()
        self.strain_df = self._strain_dataframe()

    def marker_lens(self) -> np.ndarray:
        return np.array([total_marker_len(strain) for strain in self.display_strains], dtype=int)

    def genome_lens(self) -> np.ndarray:
        return np.array([strain.metadata.total_len for strain in self.display_strains], dtype=int)

    def filt_ra(self) -> np.ndarray:
        return self.posterior_samples

    def overall_ra(self) -> np.ndarray:
        abundance_samples = self.filt_ra()
        marker_ratio = np.reciprocal(np.sum(
            np.expand_dims(self.marker_lens() / self.genome_lens(), axis=[0, 1]) * abundance_samples,
            axis=-1
        ))  # (T x N)
        read_ratio = self.num_filtered_reads / self.read_depths  # length T
        weights = marker_ratio * np.expand_dims(read_ratio, axis=1)  # (T x N)
        return abundance_samples * np.expand_dims(weights, axis=2)

    def _timeseries_dataframe(self) -> pd.DataFrame:
        df_entries = []

        filt_ras = self.filt_ra()
        overall_ras = self.overall_ra()

        for t_idx, t in enumerate(self.time_points):
            for strain_idx, strain in enumerate(self.display_strains):
                filt_relabunds = filt_ras[t_idx, :, strain_idx]
                overall_relabunds = overall_ras[t_idx, :, strain_idx]

                df_entries.append({
                    'StrainIdx': strain_idx,
                    'T': t,
                    'FilterRelAbundLower': np.quantile(filt_relabunds, 0.025),
                    'FilterRelAbundMedian': np.quantile(filt_relabunds, 0.5),
                    'FilterRelAbundUpper': np.quantile(filt_relabunds, 0.975),
                    'FilterRelAbundVar': np.var(filt_relabunds, ddof=1),
                    'OverallRelAbundLower': np.quantile(overall_relabunds, 0.025),
                    'OverallRelAbundMedian': np.quantile(overall_relabunds, 0.5),
                    'OverallRelAbundUpper': np.quantile(overall_relabunds, 0.975),
                    'OverallRelAbundVar': np.var(overall_relabunds, ddof=1),
                    'LatentMean': np.mean(self.posterior_samples[t_idx, :, strain_idx]),
                    'LatentVar': np.var(self.posterior_samples[t_idx, :, strain_idx], ddof=1)

                })

        return pd.DataFrame(df_entries)

    def _strain_dataframe(self) -> pd.DataFrame:
        df_entries = []

        for strain_idx, strain in enumerate(self.display_strains):
            df_entries.append({
                'StrainIdx': strain_idx,
                'StrainId': strain.id,
                'Genus': strain.metadata.genus,
                'Species': strain.metadata.species,
                'StrainName': strain.name,
                'PosteriorProb': self.posterior_p[strain.id],
            })

        return pd.DataFrame(df_entries).astype(dtype={
            'StrainIdx': 'int64',
            'StrainId': 'object',
            'Genus': 'object',
            'Species': 'object',
            'StrainName': 'object',
            'PosteriorProb': 'float',
        })

    def annot_df_with_lower_bound(self, abund_lb: float, target_taxon: Union[Taxon, None] = None) -> pd.DataFrame:
        df = self.timeseries_df.merge(self.strain_df, on='StrainIdx')
        # df = df.loc[df['PosteriorProb'] > posterior_lb, :]
        # df = df.loc[df['OverallRelAbundMedian'] > abund_lb]
        df = df.loc[df['FilterRelAbundMedian'] > abund_lb]
        # df = df.loc[df['FilterRelAbundMedian'] > (1/len(self.display_strains))]

        if target_taxon is not None:
            df = df.loc[
                (df['Genus'] == target_taxon.genus)
                & (df['Species'] == target_taxon.species)
                ]
        return df


class ChronostrainRenderer:
    def __init__(
            self,
            abund_lb: float,
            target_taxon: Taxon,
            color_palette: Dict[str, np.ndarray]
    ):
        self.abund_lb = abund_lb
        self.target_taxon = target_taxon
        self.color_palette = color_palette

    def get_color(self, strain_id: str) -> np.ndarray:
        return self.color_palette[strain_id]

    def plot_overall_relabund(
            self,
            res: ChronostrainResult,
            ax: Axes,
            yscale: str = 'log'
    ) -> Tuple[pd.DataFrame, float, float]:
        df = res.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon)

        ymin = 1.0  # max possible value
        ymax = 0.0  # min possible value
        for s_idx in pd.unique(df['StrainIdx']):
            section = res.timeseries_df.loc[res.timeseries_df['StrainIdx'] == s_idx].sort_values('T')
            color = self.get_color(res.display_strains[s_idx].id)
            ax.plot(res.time_points, section['OverallRelAbundMedian'], marker='.', linewidth=2, color=color)
            ax.fill_between(res.time_points, section['OverallRelAbundLower'], section['OverallRelAbundUpper'],
                            color=color, alpha=0.3)
            ymin = min(ymin, np.min(section['OverallRelAbundMedian']))
            ymax = max(ymax, np.max(section['OverallRelAbundMedian']))
        ax.set_yscale(yscale)
        ax.set_xticks(res.time_points)
        return df, ymin, ymax

    def plot_clade_presence(
            self,
            ax: Axes,
            res: ChronostrainResult,
            strain_y: Optional[Dict[str, int]] = None,
            scatter_kwargs: Optional[Dict[str, Any]] = None,
            show_ylabels: bool = True
    ):
        df = res.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon)
        if strain_y is None:
            _ids = sorted(pd.unique(df['StrainId']))
            strain_y = {_id: _i for _i, _id in enumerate(_ids)}
        df = df.assign(Y=df['StrainId'].map(strain_y))

        for strain_idx, group in df.groupby("StrainIdx"):
            strain_id = group['StrainId'].head(1).item()
            color = self.get_color(strain_id)

            kwargs = {'edgecolors': color, 'facecolors': color, 'zorder': 2}
            if scatter_kwargs is not None:
                for k, v in scatter_kwargs.items():
                    kwargs[k] = v
            ax.scatter(group['T'], group['Y'], **kwargs)

        # Other configurations
        y_min = df['Y'].min() - 1
        y_max = df['Y'].max() + 1

        ax.set_yticks(sorted(strain_y.values()))
        ax.set_xticks(res.time_points)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=y_min, top=y_max)

        if show_ylabels:
            labels = []
            for y, _df in df.sort_values('Y').groupby('Y'):
                labels.append(_df.head(1)['StrainName'].item())
            ax.set_yticklabels(labels=labels)
        else:
            ax.set_yticklabels(labels=["" for _ in range(len(strain_y))])
