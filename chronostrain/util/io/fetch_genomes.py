import os
import urllib.request as urllib
from chronostrain.util.io.logger import logger
from chronostrain.util.io.filesystem import convert_size, get_filesize_bytes

_base_dir = "data"
_fasta_filename = "{accession}.fasta"
_genbank_filename = "{accession}.gb"
_ncbi_fasta_api_url = "https://www.ncbi.nlm.nih.gov/search/api/sequence/{accession}/?report=fasta"
_ncbi_genbank_api_url = "https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?tool=portal&save=file&log$=seqview&db=nuccore&report=genbank&id={accession}&conwithfeat=on&basic_feat=on&hide-sequence=on&hide-cdd=on"


def get_ncbi_fasta_url(accession):
    return _ncbi_fasta_api_url.format(accession=accession)
    
def get_ncbi_genbank_url(accession):
    return _ncbi_genbank_api_url.format(accession=accession)


def get_fasta_filename(accession):
    return os.path.join(_base_dir, _fasta_filename.format(accession=accession))


def get_genbank_filename(accession):
    return os.path.join(_base_dir, _genbank_filename.format(accession=accession))


def fetch_filenames(accession):
    """
    Return the FASTA and GenBank filename of the accession. Try to download the file if not found.
    :param accession: NCBI accession number
    :return:
    """
    for filename, url in [
        (get_fasta_filename(accession), get_ncbi_fasta_url(accession)),
        (get_genbank_filename(accession), get_ncbi_genbank_url(accession))
    ]:
        if os.path.exists(filename):
            logger.info("[{}] file found: {}".format(accession, filename))
        else:
            logger.info("[{}] file \"{}\" not found. Downloading... ".format(accession, filename))
            filedata = urllib.urlopen(url)
            with open(filename, 'w') as f:
                f.write(str(filedata.read()).replace('\\r','').replace('\\n','\n'))
                logger.info("[{ac}] download completed. ({sz})".format(
                    ac=accession, sz=convert_size(get_filesize_bytes(filename))
                ))
    return (get_fasta_filename(accession), get_genbank_filename(accession))



        
