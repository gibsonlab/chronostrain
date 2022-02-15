from typing import Union, List
from pathlib import Path
from Bio import Entrez

from chronostrain.util.filesystem import convert_size
from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


_fasta_filename = "{accession}.fasta"
_genbank_filename = "{accession}.gb"


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


def fetch_fasta(accession, base_dir: Path, force_download: bool = False) -> Path:
    file_path = fasta_filename(accession, base_dir)
    fetch_entrez(entrez_db="nucleotide", accession=accession, rettype="fasta", retmode="text",
                 file_path=file_path, force_download=force_download)
    return file_path


def fetch_genbank(accession: str, base_dir: Path, force_download: bool = False) -> Path:
    file_path = genbank_filename(accession, base_dir)
    fetch_entrez(entrez_db="nucleotide", accession=accession, rettype="gb", retmode="text",
                 file_path=file_path, force_download=force_download)
    return file_path


def fetch_entrez(entrez_db: str,
                 accession: Union[str, List[str]],
                 rettype: str,
                 file_path: Path,
                 retmode: str = "text",
                 force_download: bool = False):
    logger.info(f"Using email `{cfg.entrez_cfg.email}` for Entrez.")
    Entrez.email = cfg.entrez_cfg.email

    if not force_download and file_path.exists() and file_path.stat().st_size > 0:
        logger.debug("[{}] File found: {}".format(
            accession[0] if isinstance(accession, list) else accession,
            file_path
        ))
    else:
        logger.debug("[{}] Downloading entrez file ({})...".format(
            accession[0] if isinstance(accession, list) else accession,
            str(file_path.name)
        ))
        net_handle = Entrez.efetch(
            db=entrez_db, id=accession, rettype=rettype, retmode=retmode
        )
        with open(file_path, "w") as f:
            f.write(net_handle.read())
        net_handle.close()

        logger.debug("[{ac}] download completed. ({sz})".format(
            ac=accession[0] if isinstance(accession, list) else accession,
            sz=convert_size(file_path.stat().st_size)
        ))
