import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import os



def plot_abundances_comparsion(inferred_abnd_dir: str, inferred_abnd_file: str,
                               reads_dir: str, abnd_file: str,
                               title: str, output_dir: str, output_file: str):

    real_df_melted = (pd.read_csv(os.path.join(reads_dir, abnd_file))
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
    plt.title(title)

    save_location = os.path.join(output_dir, output_file)
    plt.savefig(save_location, bbox_inches='tight')
    plt.show()


def plot_abundances(directory: str, abundance_file: str):
    pass


# %%
# if __name__ == "__main__":
#     directory = "simulated_data/test_1/"
#     abundance_file = "sim_abundances.csv"