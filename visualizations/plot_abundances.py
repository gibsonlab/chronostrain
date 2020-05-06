import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_abundances_comparison(
            inferred_abnd_path: str,
            real_abnd_path: str,
            title: str,
            plots_out_path: str):

    real_df = (pd.read_csv(real_abnd_path)
                      .assign(Truth="Real")
                      .melt(id_vars=['T', "Truth"],
                            var_name="Strain",
                            value_name="Abundance")
                      .rename(columns={"T": "Time"}))

    inferred_df = (pd.read_csv(inferred_abnd_path)
                          .assign(Truth="Inferred")
                          .melt(id_vars=['T', "Truth"],
                                var_name="Strain",
                                value_name="Abundance")
                          .rename(columns={"T": "Time"}))

    result_df = pd.concat([real_df, inferred_df])

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=result_df, style="Truth", markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')

def plot_abundances(
            abnd_path: str,
            title: str,
            plots_out_path: str):

    inferred_df = (pd.read_csv(abnd_path)
                          .melt(id_vars=['T', "Truth"],
                                var_name="Strain",
                                value_name="Abundance")
                          .rename(columns={"T": "Time"}))

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=inferred_df, markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')