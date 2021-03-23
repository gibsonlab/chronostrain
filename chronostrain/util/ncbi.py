"""
    A python wrapper implementation of relevant NCBI API calls.
"""
import os
import urllib.request
import urllib.error

from chronostrain.util.logger import logger
from chronostrain.util.filesystem import convert_size, get_filesize_bytes


_fasta_filename = "{accession}.fasta"
_genbank_filename = "{accession}.gb"
_ncbi_fasta_api_url = "https://www.ncbi.nlm.nih.gov/search/api/sequence/{accession}/?report=fasta"
_ncbi_genbank_api_url = "https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?tool=portal&save=file&log$=seqview&db=nuccore&report=genbank&id={accession}&conwithfeat=on&basic_feat=on&hide-sequence=on&hide-cdd=on"


class NCBIAPIException(BaseException):
    pass


def _fasta_get_url(accession: str) -> str:
    """
    Returns the HTTP GET url corresponding to NCBI's FASTA sequence API.
    :param accession: the NCBI accession tag.
    :return: the URL corresponding to the target accession.
    """
    return _ncbi_fasta_api_url.format(accession=accession)


def _genbank_get_url(accession: str) -> str:
    """
    Returns the HTTP GET url corresponding to NCBI's GenBank API.
    :param accession: the NCBI accession tag.
    :return: the URL corresponding to the target accession.
    """
    return _ncbi_genbank_api_url.format(accession=accession)


def fasta_filename(accession: str, base_dir: str) -> str:
    """
    The default filename to output fasta files to.
    """
    return os.path.join(base_dir, _fasta_filename.format(accession=accession))


def genbank_filename(accession: str, base_dir: str) -> str:
    """
    The default filename to output genbank files to.
    """
    return os.path.join(base_dir, _genbank_filename.format(accession=accession))


def fetch_fasta(accession: str, base_dir: str) -> str:
    filename = fasta_filename(accession, base_dir)
    url = _fasta_get_url(accession)
    _fetch_from_api(accession, filename, url)
    return filename


def fetch_genbank(accession: str, base_dir: str) -> str:
    filename = genbank_filename(accession, base_dir)
    url = _genbank_get_url(accession)
    _fetch_from_api(accession, filename, url)
    return filename


def _fetch_from_api(accession: str, filename: str, url: str):
    """
    Check if file exists. If not, try to download the files.
    :param filename: The target file to check.
    :param url: The url to access.
    """
    if os.path.exists(filename):
        logger.debug("[{}] file found: {}".format(accession, filename))
    else:
        try:
            logger.debug("HTTP GET {}".format(url))
            conn = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            raise NCBIAPIException("NCBI API Error {code} [{url}]".format(
                code=e.code,
                url=url
            ))
        except urllib.error.URLError as e:
            raise NCBIAPIException("URLError: {}".format(e.reason))
        else:
            with open(filename, 'w') as f:
                content = str(conn.read().decode("utf-8")).replace('\\r', '').replace('\\n', '\n')
                if content.startswith("# ERROR"):
                    raise NCBIAPIException("Error encountered from {url}. Message=`{msg}`".format(
                        url=url,
                        msg=content.split("\n")[0]
                    ))

                f.write(content)
                logger.debug("[{ac}] download completed. ({sz})".format(
                    ac=accession, sz=convert_size(get_filesize_bytes(filename))
                ))
