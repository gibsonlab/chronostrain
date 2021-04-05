"""
    Convert a chronostrain database to metaphlan database, by appending markers one at a time and passing it
    through the tutorial found at https://github.com/biobakery/MetaPhlAn/wiki/MetaPhlAn-3.0#customizing-the-database.
"""
import argparse
import bz2
import hashlib
import os
import pickle
import shutil
import tarfile
from typing import Tuple

from chronostrain import cfg, logger
from chronostrain.model import Strain
from chronostrain.util.external import bowtie2


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
    species_token = "s__{}_{}".format(strain.metadata.genus, strain.metadata.species)
    for tax_clade in input_metaphlan_db['taxonomy']:
        if species_token in tax_clade:
            ncbi_taxid, _ = input_metaphlan_db['taxonomy'][tax_clade]
            tax_clade_tokens = tax_clade.split("|")
            if tax_clade_tokens[-1].startswith("t__"):
                tax_clade_tokens[-1] = "t__{}".format(strain.metadata.assembly_id)
            else:
                tax_clade_tokens.append("t__{}".format(strain.metadata.assembly_id))
            tax_clade_new = "|".join(tax_clade_tokens)
            return tax_clade_new, ncbi_taxid


def get_strain_clade_taxon(input_metaphlan_db: dict, strain: Strain) -> Tuple[str, str]:
    species_name = "{}_{}".format(strain.metadata.genus, strain.metadata.species)
    for marker_name in input_metaphlan_db['markers']:
        entry = input_metaphlan_db['markers'][marker_name]
        if species_name in entry['taxon']:
            return entry['clade'], entry['taxon']


def fill_higher_levels(master_strain, strain_gtdb, strain_ncbi_levels, metaphlan_db, marker_fasta_path):
    print(strain_gtdb)
    gtdb_tokens = strain_gtdb.split("|")
    ncbi_tokens = strain_ncbi_levels.split("|")
    master_marker = master_strain.markers[0]

    for i in range(1, len(ncbi_tokens)):
        gtdb = "|".join(gtdb_tokens[:i])
        ncbi = "|".join(ncbi_tokens[:i])
        clade = gtdb_tokens[i - 1]

        if gtdb not in metaphlan_db['taxonomy']:
            logger.info("Adding parent taxon {}".format(gtdb))
            metaphlan_db['taxonomy'][gtdb] = (ncbi, master_strain.genome_length)

            marker_id = "{}-16S_V4".format(master_strain.metadata.ncbi_accession)
            metaphlan_db['markers'][marker_id] = {
                'clade': clade,
                'ext': [master_strain.metadata.assembly_id],
                'len': len(master_marker),
                'taxon': gtdb
            }

            with open(marker_fasta_path, "a") as fasta_file:
                print(">{}".format(marker_id), file=fasta_file)
                print(master_marker.seq, file=fasta_file)


def convert_to_metaphlan_db(chronostrain_db, metaphlan_in_path, metaphlan_out_dir, basename):
    input_metaphlan_db: dict = pickle.load(bz2.open(metaphlan_in_path, 'r'))

    new_metaphlan_db = {
        'markers': dict(),
        'taxonomy': dict(),
        'merged_taxon': dict()
    }

    marker_fasta_path = os.path.join(metaphlan_out_dir, "{}.fasta".format(basename))

    with open(marker_fasta_path, "w") as _:
        pass

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
        new_metaphlan_db['merged_taxon'] = input_metaphlan_db['merged_taxon']

        ecoli_master_strain = chronostrain_db.get_strain("NC_000913.3")

        if strain.metadata.ncbi_accession != "NC_000913.3":
            new_metaphlan_db['taxonomy'][strain_gtdb_levels] = (strain_ncbi_levels, strain.genome_length)

            for marker in strain.markers:
                marker_id = "{}-{}".format(strain.metadata.ncbi_accession, marker.name)
                # Add the information of the new marker as the other markers

                new_metaphlan_db['markers'][marker_id] = {
                    'clade': strain_clade,
                    'ext': [strain.metadata.assembly_id],
                    'len': len(marker),
                    'taxon': strain_taxon
                }
                with open(marker_fasta_path, "a") as fasta_file:
                    print(">{}".format(marker_id), file=fasta_file)
                    print(marker.seq, file=fasta_file)
            fill_higher_levels(ecoli_master_strain, strain_gtdb_levels, strain_ncbi_levels, new_metaphlan_db, marker_fasta_path)

    # Build the bowtie2 database.
    bowtie2.bowtie2_build(
        refs_in=marker_fasta_path,
        output_index_base=basename
    )
    logger.info("Ran bowtie2-build on {}.".format(marker_fasta_path))

    # Save the new mpa_pkl file
    pkl_path = os.path.join(metaphlan_out_dir, "{}.pkl".format(basename))
    with bz2.BZ2File(pkl_path, 'w') as outfile:
        pickle.dump(new_metaphlan_db, outfile, pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote pickle file {}.".format(pkl_path))

    # Bzip2 the fasta file.
    fasta_bz2_path = os.path.join(metaphlan_out_dir, "{}.fna.bz2".format(basename))
    with open(marker_fasta_path, 'rb') as f_in:
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


def main():
    args = parse_args()

    chronostrain_db = cfg.database_cfg.get_database()

    for strain in chronostrain_db.all_strains():
        accession_to_gca = {
            'CP001071.1': 'GCA_000020225.1',
            'CR626927.1': 'GCA_000025985.1',
            "U00096": "GCA_000005845.2",
            "NZ_CP012938.1": "GCF_001314995.1",
            "CP000139.1": "GCA_000012825.1",
            "NZ_CP013243.1": "GCF_001889325.1",
            "CP068242.1": "GCA_016743835.1",
            "CP026285.1": "GCA_002902905.1",
            "NC_009615.1": "GCF_000012845.1",
            "CP044436.1": "GCA_016772335.1",
            "CP027002.1": "GCA_009831375.1",
            "NC_000913.3": "GCA_000005845.2"
        }
        strain.metadata.assembly_id = accession_to_gca[strain.metadata.ncbi_accession]

    logger.info("Converting metaphlan pickle files.")
    convert_to_metaphlan_db(
        chronostrain_db=chronostrain_db,
        metaphlan_in_path=args.metaphlan_input_path,
        metaphlan_out_dir=args.metaphlan_out_dir,
        basename=args.basename
    )


if __name__ == "__main__":
    main()
