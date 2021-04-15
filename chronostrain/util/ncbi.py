"""
    A python wrapper implementation of relevant NCBI API calls.
"""
from pathlib import Path
import urllib.request
import urllib.error

from . import logger
from chronostrain.util.filesystem import convert_size


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


def fasta_filename(accession: str, base_dir: Path) -> Path:
    """
    The default filename to output fasta files to.
    """
    return base_dir / _fasta_filename.format(accession=accession)


def genbank_filename(accession: str, base_dir: Path) -> Path:
    """
    The default filename to output genbank files to.
    """
    return base_dir / _genbank_filename.format(accession=accession)


def fetch_fasta(accession: str, base_dir: Path, force_download: bool = False) -> Path:
    file_path = fasta_filename(accession, base_dir)
    url = _fasta_get_url(accession)
    _fetch_from_api(accession, file_path, url, force_download=force_download)
    return file_path


def fetch_genbank(accession: str, base_dir: Path, force_download: bool = False) -> Path:
    file_path = genbank_filename(accession, base_dir)
    url = _genbank_get_url(accession)
    _fetch_from_api(accession, file_path, url, force_download=force_download)
    return file_path


def _fetch_from_api(accession: str, file_path: Path, url: str, force_download: bool):
    """
    Check if file exists. If not, try to download the files.
    :param filename: The target file to check.
    :param url: The url to access.
    """
    if not force_download and file_path.exists() and file_path.stat().st_size > 0:
        logger.debug("[{}] file found: {}".format(accession, file_path))
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
            with open(file_path, 'w') as f:
                content = str(conn.read().decode("utf-8")).replace('\\r', '').replace('\\n', '\n')
                if content.startswith("# ERROR"):
                    raise NCBIAPIException("Error encountered from {url}. Message=`{msg}`".format(
                        url=url,
                        msg=content.split("\n")[0]
                    ))

                f.write(content)
                logger.debug("[{ac}] download completed. ({sz})".format(
                    ac=accession, sz=convert_size(file_path.stat().st_size)
                ))
