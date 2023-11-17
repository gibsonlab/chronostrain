from typing import *
from pathlib import Path

import numpy as np
import scipy.special
import pandas as pd
from matplotlib.pyplot import Axes
import matplotlib.transforms as transforms

from chronostrain.inference import GaussianWithGumbelsPosterior
from chronostrain.config import cfg
from chronostrain.database import StrainDatabase
from chronostrain.model import *

from Bio.Nexus.Trees import Tree
from Bio.Phylo.Newick import Clade
from .tree import pruned_subtree, phylo_draw_custom


def parse_strains(db: StrainDatabase, strain_txt: Path):
    with open(strain_txt, 'rt') as f:
        return [
            db.get_strain(l.strip())
            for l in f
        ]


def total_marker_len(strain: Strain) -> int:
    return sum(len(m) for m in strain.markers)


def posterior_with_bf_threshold(
        posterior: GaussianWithGumbelsPosterior,
        inference_strains: List[Strain],
        output_strains: List[Strain],
        adhoc_clustering: Dict[str, Strain],
        bf_threshold: float, 
        prior_p: float = 0.001
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
    posterior_inclusion_bf = (posterior_inclusion_p / (1 - posterior_inclusion_p)) * ((1 - prior_p) / prior_p)

    # Calculate abundance estimates using BF thresholds.
    indicators = np.full(n_inference_strains, fill_value=False, dtype=bool)
    indicators[posterior_inclusion_bf > bf_threshold] = True
    print("{} of {} inference strains passed BF Threshold > {}".format(np.sum(indicators), n_inference_strains, bf_threshold))
    
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


class ChronostrainResult(object):
    def __init__(self, name: str, db: StrainDatabase, posterior_class, out_dir: Path, input_path: Path, target_bayes_factor: float = 100.0):
        self.name = name
        self.adhoc_clusters: Dict[str, Strain] = parse_adhoc_clusters(db, out_dir / "adhoc_cluster.txt")
        self.inference_strains: List[Strain] = parse_strains(db, out_dir / 'strains.txt')
        self.display_strains: List[Strain] = list(db.get_strain(x) for x in self.adhoc_clusters.keys())

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
            posterior, self.inference_strains, self.display_strains, self.adhoc_clusters, target_bayes_factor
        )
        self.target_bayes_factor = target_bayes_factor
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
            strain_palette: Dict[str, np.ndarray],
            stool_result: ChronostrainResult,
            urine_result: ChronostrainResult,
            plate_results: List[Tuple[float, ChronostrainResult]],
            abx_df: pd.DataFrame,
            abx_palette: Dict[str, np.ndarray],
            abx_label: Dict[str, str],
            uti_df: pd.DataFrame,
            sample_df: pd.DataFrame
    ):
        self.abund_lb = abund_lb
        self.target_taxon = target_taxon
        self.strain_palette = strain_palette
        self.stool_result = stool_result
        self.urine_result = urine_result
        self.plate_results = plate_results
        self.abx_df = abx_df
        self.abx_label = abx_label
        self.abx_palette = abx_palette
        self.uti_df = uti_df
        self.sample_df = sample_df

        t_mins = [np.min(self.stool_result.time_points)]
        if self.urine_result is not None:
            t_mins.append(np.min(self.urine_result.time_points))
        self.min_t = np.min(t_mins)

    def get_merged_df(self):
        dfs = [
            self.stool_result.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon).assign(Src='stool')
        ]
        if self.urine_result is not None:
            dfs.append(
                self.urine_result.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon).assign(Src='urine')
            )
        for t, plate_result in self.plate_results:
            dfs.append(
                plate_result.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon).assign(
                    Src='plate',
                    T=t
                )
            )

        return pd.concat(dfs, ignore_index=True)
        
    def get_color(self, strain_id: str) -> np.ndarray:
        return self.strain_palette[strain_id]
    
    def plot_overall_relabund(
            self,
            ax: Axes,
            mode: str = 'stool',
            yscale: str = 'log'
    ) -> Tuple[float, float]:
        if mode == 'stool':
            res = self.stool_result
        elif mode == 'urine':
            res = self.urine_result
        else:
            raise ValueError(f"Unknown plotting mode `{mode}`.")
        df = res.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon)
        if df.shape[0] == 0:
            return [0.1, 1.0]

        overall_relabund = res.overall_ra()
        
        ymin = 1.0  # max possible value
        ymax = 0.0  # min possible value
        for s_idx in pd.unique(df['StrainIdx']):
            # section = res.timeseries_df.loc[res.timeseries_df['StrainIdx'] == s_idx].sort_values('T')
            # color = self.get_color(res.display_strains[s_idx].id)
            # ax.plot(res.time_points, section['OverallRelAbundMedian'], marker='.', linewidth=2, color=color)
            # ax.fill_between(res.time_points, section['OverallRelAbundLower'], section['OverallRelAbundUpper'], color=color, alpha=0.3)
            # ymin = min(ymin, np.min(section['OverallRelAbundMedian']))
            # ymax = max(ymax, np.max(section['OverallRelAbundMedian']))
            section = overall_relabund[:, :, s_idx]
            color = self.get_color(res.display_strains[s_idx].id)
            upper = np.quantile(section, axis=-1, q=0.975)
            lower = np.quantile(section, axis=-1, q=0.025)
            median = np.quantile(section, axis=-1, q=0.5)
            ax.plot(res.time_points, median, marker='.', linewidth=2, color=color)
            ax.fill_between(res.time_points, lower, upper, color=color, alpha=0.3)
            ymin = min(ymin, np.min(median))
            ymax = max(ymax, np.max(median))
        ax.set_yscale(yscale)
        ax.set_xticks(res.time_points)
        return ymin, ymax
    
    def plot_filt_relabund(
            self,
            ax: Axes,
            yscale: str = 'log'
    ) -> Tuple[float, float]:
        res = self.stool_result
        df = res.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon)
        
        ymin = 1.0  # max possible value
        ymax = 0.0  # min possible value
        for s_idx in pd.unique(df['StrainIdx']):
            section = res.timeseries_df.loc[res.timeseries_df['StrainIdx'] == s_idx].sort_values('T')
            color = self.get_color(res.display_strains[s_idx].id)
            ax.plot(res.time_points, section['FilterRelAbundLower'], marker='.', linewidth=2, color=color)
            ax.fill_between(res.time_points, section['OverallRelAbundLower'], section['FilterRelAbundUpper'], color=color, alpha=0.3)
            ymin = min(ymin, np.min(section['FilterRelAbundMedian']))
            ymax = max(ymax, np.max(section['FilterRelAbundMedian']))
        ax.set_yscale(yscale)
        ax.set_xticks(res.time_points)
        return ymin, ymax

    def get_plotted_strain_names(self, res: ChronostrainResult) -> Dict[str, str]:
        df = res.annot_df_with_lower_bound(self.abund_lb, target_taxon=self.target_taxon)
        return {
            row['StrainId']: row['StrainName']
            for _, row in df.iterrows()
        }

    def plot_clade_presence(
            self,
            ax: Axes,
            strain_y: Optional[Dict[str, int]] = None,
            show_ylabels: bool = True
    ):
        df = self.get_merged_df()
        if df.shape[0] == 0:
            return
        if strain_y is None:
            _ids = sorted(pd.unique(df['StrainId']))
            strain_y = {_id: _i for _i, _id in enumerate(_ids)}
        df = df.assign(Y=df['StrainId'].map(strain_y))

        marker_sz = 1.0
        for (src, strain_idx), group in df.groupby(["Src", "StrainIdx"]):
            strain_id = group['StrainId'].head(1).item()
            color = self.get_color(strain_id)

            # Pick style
            if src == 'stool':
                ax.scatter(group['T'], group['Y'], edgecolor=color, facecolors=[1, 1, 1, 0], marker='o', linewidths=2 * marker_sz, s=200 * marker_sz, zorder=2)
            elif src == 'plate':
                ax.scatter(group['T'], group['Y'], facecolors=color, marker='x', linewidths=1.5 * marker_sz, s=100 * marker_sz, zorder=2)
            elif src == 'urine':
                ax.scatter(group['T'], group['Y'], edgecolor=color, facecolors=color, marker='.', linewidths=2 * marker_sz, s=200 * marker_sz, zorder=2)

        time_points = sorted(df['T'])
        ax.set_yticks(sorted(strain_y.values()))
        ax.set_xticks(time_points)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # ======= Other settings.
        y_min = df['Y'].min() - 1
        y_max = df['Y'].max() + 1
        if np.isnan(y_min) or np.isinf(y_min):
            y_min = -1
        if np.isnan(y_max) or np.isinf(y_max):
            y_max = 1
        ax.set_ylim(bottom=y_min, top=y_max)
        
        if show_ylabels:
            labels = []
            for y, _df in df.sort_values('Y').groupby('Y'):
                labels.append(_df.head(1)['StrainName'].item())
            ax.set_yticklabels(labels=labels)
        else:
            ax.set_yticklabels(labels=["" for _ in range(len(strain_y))])

    def plot_tree(
            self,
            ax: Axes,
            tree: Tree,
    ) -> Tuple[Dict, Dict]:
        strain_id_to_names = {
            row['StrainId']: row['StrainName']
            for _, row in self.get_merged_df().iterrows()
        }
        ax.axis('off')

        strain_leaves = set(strain_id_to_names.keys())
        if len(strain_leaves) == 0:
            return {}, {}

        subtree = pruned_subtree(tree, strain_leaves)
        if isinstance(subtree, Clade) and subtree.is_terminal():
            return {subtree: 0}, {subtree: 0}
        
        def color_fn(s):
            if s.is_terminal() and len(s.name) != 0:
                return self.get_color(s.name)
            else:
                return "black"
        def label_fn(s):
            if s.is_terminal() and len(s.name) != 0:
                return "{}".format(strain_id_to_names[s.name])
            else:
                return ""
            
        subtree.name = ''
        x_posns, y_posns = phylo_draw_custom(
            subtree, 
            label_func=label_fn,
            axes=ax,
            do_show=False,
            show_confidence=False,
            label_colors=color_fn,
            branch_labels=lambda c: '{:.03f}'.format(c.branch_length) if (c.branch_length is not None and c.branch_length > 0.003) else ''
        )
        return x_posns, y_posns
    
    def plot_abx(self, ax: Axes, draw_labels: bool = True):
        for _, row in self.abx_df.iterrows():
            d = row['experiment_day_ended']
            if d < self.min_t:
                continue
                
            abx_class = row['abx_class']
            abx_label = self.abx_label.get(abx_class, '?')
                
            color = self.abx_palette.get(abx_class, 'gray')
            ax.axvline(x=d, color=color, linestyle='--', zorder=1)
            if draw_labels:
                ax.text(
                    x=d, 
                    y=1.01, 
                    s=abx_label, 
                    color=color,
                    transform=transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    horizontalalignment='center', verticalalignment='bottom'
                )
                
    def plot_infections(self, ax: Axes, color='black'):
        for _, row in self.uti_df.iterrows():
            d = row['UTIDay']
            ax.axvline(x=d, color=color, alpha=1.0, zorder=1)

    def set_xtick_dates(self, ax: Axes, show_t: bool = False):
        """
        read from the UMB experiment dataframe and change x-axis labels to the sample dates.
        """
        xticks = []
        xlabels = []
        prev_t = -10000
        for _, row in self.sample_df.groupby('T').head(1).sort_values('T').iterrows():
            date = row['date']
            samplename = row['SampleName']
            t = row['T']
            if show_t:
                lbl_t = f'{date.date()} ({samplename}): {t}'
            else:
                lbl_t = f'{date.date()}'
            if t - prev_t < 7:  # Within a week
                xticks.append(t)
                xlabels.append("")
                # xlabels[-1] = lbl_t
            else:
                xticks.append(t)
                xlabels.append(lbl_t)
            prev_t = t
        t_min = self.sample_df['T'].min()
        t_max = self.sample_df['T'].max()
        dt = (t_max - t_min) * 0.02
        ax.set_xlim(
            left=t_min - dt,
            right=t_max + dt
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=90)
        