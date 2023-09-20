from chronostrain.config import cfg
from chronostrain.util.external import bowtie2_build


def main():
    db = cfg.database_cfg.get_database()

    marker_reference_path = db.multifasta_file
    bowtie2_build(
        refs_in=[marker_reference_path],
        index_basepath=marker_reference_path.parent,
        index_basename=marker_reference_path.stem,
        offrate=1,  # default is 5; but we want to optimize for the -a option.
        ftabchars=13,
        quiet=True,
        threads=cfg.model_cfg.num_cores,
    )


if __name__ == "__main__":
    main()
