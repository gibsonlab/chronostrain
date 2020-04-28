import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_abundances_comparison(
        inferred_abnd_path: str,
        real_abnd_path: str,
        title: str,
        output_dir: str,
        output_filename: str):

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

    save_location = os.path.join(output_dir, output_filename)
    plt.savefig(save_location, bbox_inches='tight')
    return save_location


def plot_abundances(
        inferred_abnd_path: str,
        title: str,
        output_dir: str,
        output_filename: str):

    inferred_df = (pd.read_csv(inferred_abnd_path)
                   .melt(id_vars=['T', "Truth"],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"}))

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=inferred_df, markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    save_location = os.path.join(output_dir, output_filename)
    plt.savefig(save_location, bbox_inches='tight')
    return save_location
