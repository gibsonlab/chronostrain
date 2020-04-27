import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import os


def plot_abundances(directory: str, abundance_file: str):

    dataframe = pd.read_csv(os.path.join(directory, abundance_file))

    melted_df = pd.melt(dataframe, ['T'], var_name="Strain", value_name="Abundance")
    melted_df = melted_df.rename(columns={"T": "Time"})

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=melted_df)
    plt.show()


def plot_abundances_comparsion(inferred_abnd_dir: str, inferred_abnd_file: str,
                               real_abnd_dir: str, real_abnd_file: str,
                               title: str, output_dir: str, output_file: str):

    # inferred_abnd_dir = "output/test_2/"
    # inferred_abnd_file = "inferred_abundances_2.csv"
    # real_abnd_dir = "simulated_data/test_2/"
    # real_abnd_file = "sim_abundances.csv"

    real_df_melted = (pd.read_csv(os.path.join(real_abnd_dir, real_abnd_file))
                        .assign(Truth="Real")
                        .melt(id_vars=['T', "Truth"],
                              var_name="Strain",
                              value_name="Abundance")
                        .rename(columns={"T": "Time"}))

    inferred_df_melted = (pd.read_csv(os.path.join(inferred_abnd_dir, inferred_abnd_file))
                          .assign(Truth="Inferred")
                          .melt(id_vars=['T', "Truth"],
                                var_name="Strain",
                                value_name="Abundance")
                          .rename(columns={"T": "Time"}))

    result_df = pd.concat([real_df_melted, inferred_df_melted])

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=result_df, style="Truth", markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(output_dir + output_file)


# %%
if __name__ == "__main__":
    directory = "simulated_data/test_1/"
    abundance_file = "sim_abundances.csv"