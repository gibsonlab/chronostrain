import os
import csv
import urllib.request as urllib
from util.logger import logger


_base_dir = "data"
_filename = "{accession}.fasta"
_ncbi_api_url = "https://www.ncbi.nlm.nih.gov/search/api/sequence/{accession}/?report=fasta"
_refs_file_csv = "ncbi_refs.csv"


def get_ncbi_url(accession):
    return _ncbi_api_url.format(accession=accession)


def get_filename(accession):
    return os.path.join(_base_dir, _filename.format(accession=accession))


def fetch_filename(accession):
    """
    Return the FASTA filename of the accession. Try to download the file if not found.
    :param accession: NCBI accession number
    :return:
    """
    filename = get_filename(accession)
    if os.path.exists(filename):
        logger.info("[{}] file found: {}".format(accession, filename))
    else:
        logger.info("[{}] file \"{}\" not found. Downloading... ".format(accession, filename))
        filedata = urllib.urlopen(get_ncbi_url(accession))
        with open(filename, 'w') as f:
            f.write(str(filedata.read()).replace("\r", ""))
    return filename


def fetch_sequences(refs_file_csv):
    """
    Read CSV file, and download FASTA from accessions if doesn't exist.
    :return: a dictionary mapping accessions to strain-accession-filename wrappers.
    """
    strains_map = {}

    csv_filename = os.path.join(_base_dir, refs_file_csv)
    line_count = 0
    with open(csv_filename, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            strain_name = row[0]
            accession = row[1]
            strains_map[accession] = {
                "strain": strain_name,
                "accession": accession,
                "file": fetch_filename(accession)
            }
            line_count += 1

    logger.info("Found {} records.".format(len(strains_map.keys())))
    return strains_map


if __name__ == "__main__":
    fetch_sequences(_refs_file_csv)
