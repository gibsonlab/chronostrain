"""
    Convert a chronostrain database to metaphlan database, by appending markers one at a time and passing it
    through the tutorial found at https://github.com/biobakery/MetaPhlAn/wiki/MetaPhlAn-3.0#customizing-the-database.
"""
import os
import argparse
import pickle
import bz2
import tarfile
import shutil
import hashlib
from typing import Tuple

from chronostrain.database import AbstractStrainDatabase
from chronostrain.model import Strain
from chronostrain.util.external import bowtie2
from chronostrain import cfg, logger


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Chronostrain database to MetaPhlAn database.")
    parser.add_argument('-i', '--input_path', dest='metaphlan_input_path',
                        required=True, type=str,
                        help='<Required> The pickle file containing the default MetaPhlAn database '
                             'Example: metaphlan_databases/mpa_v30_CHOCOPhlAn_201901.pkl')
    parser.add_argument('-o', '--out_dir', dest='metaphlan_out_dir',
                        required=True, type=str,
                        help='<Required> The output directory to store all of the files.')

    parser.add_argument('-b', '--basename', dest='basename',
                        required=False, type=str, default='mpa_chronostrain',
                        help='<Optional> The basename of the resulting MetaPhlAn database.')
    return parser.parse_args()


def search_taxonomy(input_metaphlan_db: dict, strain: Strain) -> Tuple[str, str]:
    """
    Get the 7-level gtdb taxonomy with clade names of the given strain, as well as the NCBI 7-level taxID.
    Currently, (as a stopgap), this is implemented by looking up the name on metaphlan database.
    :param input_metaphlan_db: The dictionary corresponding to the pre-defined metaphlan database.
    :param strain: The Strain instance.
    :return: See the description.
    """
    for tax_clade in input_metaphlan_db['taxonomy']:
        if strain.metadata.species in tax_clade:
            ncbi_taxid, _ = input_metaphlan_db['taxonomy'][tax_clade]
            return tax_clade, ncbi_taxid


def get_strain_clade_taxon(input_metaphlan_db: dict, strain: Strain) -> Tuple[str, str]:
    for marker_name in input_metaphlan_db['markers']:
        entry = input_metaphlan_db['markers'][marker_name]
        species_name = "{}_{}".format(strain.metadata.genus, strain.metadata.species)
        if species_name in entry['taxon']:
            return entry['clade'], entry['taxon']


def convert_to_metaphlan_db(chronostrain_db, metaphlan_in_path, metaphlan_out_dir, basename):
    input_metaphlan_db: dict = pickle.load(bz2.open(metaphlan_in_path, 'r'))

    new_metaphlan_db = {
        'markers': dict(),
        'taxonomy': dict(),
        'merged_taxon': dict()
    }

    for s_idx, strain in enumerate(chronostrain_db.all_strains()):
        logger.info("Strain {} of {} -- {} ({} {})".format(
            s_idx + 1,
            chronostrain_db.num_strains(),
            strain.id,
            strain.metadata.genus,
            strain.metadata.species
        ))

        strain_clade, strain_taxon = get_strain_clade_taxon(input_metaphlan_db, strain)
        strain_gtdb_levels, strain_ncbi_levels = search_taxonomy(input_metaphlan_db, strain)

        new_metaphlan_db['taxonomy'][strain_gtdb_levels] = (strain_ncbi_levels, strain.genome_length)

        for marker in strain.markers:
            marker_id = "{}_{}".format(strain.metadata.ncbi_accession, marker.name)
            # Add the information of the new marker as the other markers
            new_metaphlan_db['markers'][marker_id] = {
                'clade': strain_clade,
                'ext': [],  # This appears to be optional.
                'len': len(marker),
                'taxon': strain_taxon
            }

    # Save the new mpa_pkl file
    pkl_path = os.path.join(metaphlan_out_dir, "{}.pkl".format(basename))
    with bz2.BZ2File(pkl_path, 'w') as outfile:
        pickle.dump(new_metaphlan_db, outfile, pickle.HIGHEST_PROTOCOL)
        logger.info("Output new database to {}.".format(pkl_path))

    # Bzip2 the fasta file.
    fasta_bz2_path = os.path.join(metaphlan_out_dir, "{}.fna.bz2".format(basename))
    with open(chronostrain_db.get_multifasta_file(), 'rb') as f_in:
        with bz2.open(fasta_bz2_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Tarball these two files.
    tar_filename = "{}.tar".format(basename)
    tar_path = os.path.join(metaphlan_out_dir, tar_filename)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkl_path)
        tar.add(fasta_bz2_path)

    # Generate MD5 hash.
    md5_path = os.path.join(metaphlan_out_dir, "{}.md5".format(basename))
    md5 = hashlib.md5(open(tar_path, 'rb').read()).hexdigest()
    with open(md5_path, "w") as md5file:
        print("{}  {}".format(md5, tar_filename), file=md5file)


def build_bowtie(chronostrain_db: AbstractStrainDatabase, index_basename: str):
    marker_multifasta_path = chronostrain_db.get_multifasta_file()
    # TODO: Refactor code so that this multifasta file and the strain-marker pair
    #  iteration order in `convert_to_metaphlan_db()` are identical.
    # Note: bowtie_build 'refs_in' argument is optionally a comma-separated list.

    bowtie2.bowtie2_build(
        refs_in=marker_multifasta_path,
        output_index_base=index_basename
    )


def main():
    args = parse_args()

    chronostrain_db = cfg.database_cfg.get_database()

    logger.info("Building bowtie2 index.")
    build_bowtie(
        chronostrain_db=chronostrain_db,
        index_basename=os.path.join(args.metaphlan_out_dir, args.basename)
    )

    logger.info("Converting metaphlan pickle files.")
    pkl_out_path = os.path.join(args.metaphlan_out_dir, "{}.pkl".format(args.basename))
    convert_to_metaphlan_db(
        chronostrain_db=chronostrain_db,
        metaphlan_in_path=args.metaphlan_input_path,
        metaphlan_out_dir=args.metaphlan_out_dir,
        basename=args.basename
    )


if __name__ == "__main__":
    main()
