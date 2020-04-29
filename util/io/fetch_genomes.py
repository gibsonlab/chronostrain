import os
import csv
import urllib.request as urllib
from util.io.logger import logger
from util.io.filesystem import convert_size, get_filesize_bytes

_base_dir = "data"
_filename = "{accession}.fasta"
_ncbi_api_url = "https://www.ncbi.nlm.nih.gov/search/api/sequence/{accession}/?report=fasta"


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
            logger.info("[{ac}] download completed. ({sz})".format(
                ac=accession, sz=convert_size(get_filesize_bytes(filename))
            ))
    return filename


def fetch_sequences(refs_file_csv: str):
    """
    Read CSV file, and download FASTA from accessions if doesn't exist.
    :return: a dictionary mapping accessions to strain-accession-filename wrappers.
    """
    strains_map = {}

    line_count = 0
    with open(refs_file_csv, "r") as f:
        csv_reader = csv.reader(f)
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

