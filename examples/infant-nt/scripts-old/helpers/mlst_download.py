import argparse
from typing import Tuple, Iterator, List, Dict, Set, Optional

from io import StringIO
from pathlib import Path
from xml.etree import ElementTree
import urllib.error
import urllib.request
import csv

from Bio import SeqIO

from chronostrain import create_logger
from chronostrain.config import cfg
from chronostrain.database import StrainDatabase, PickleParser
from chronostrain.database.backend import PandasAssistedBackend
from chronostrain.util.filesystem import convert_size
from chronostrain.model import Strain, Marker
from chronostrain.util.sequences import Sequence, AllocatedSequence

logger = create_logger('script.mlst_download')


def download_from_url(url: str) -> str:
    try:
        logger.debug(f"Fetching URL resource {url}")
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        raise RuntimeError("Failed to retrieve from resource {} due to HTTP error {}.".format(
            url, e.code
        ))
    except urllib.error.URLError as e:
        raise RuntimeError("Failed to retrieve from resource {} due to error. Reason: {}.".format(
            url, e.reason
        ))

    r_raw = response.read()
    logger.debug("Got a response of size {}.".format(
        convert_size(len(r_raw))
    ))
    return r_raw.decode('utf-8')


class TaxaIdentifier:
    def __init__(self, genus: str, species: str, typeid: str, profile_url: str):
        self.genus = genus
        self.species = species
        self.typeid = typeid
        self.profile_url = profile_url

    def __repr__(self):
        return "g_{}__s_{}__t_{}".format(self.genus, self.species, self.typeid)

    def __hash__(self):
        return hash(self.__repr__())


class LocusContainer:
    def __init__(self, locus_name: str, url: str):
        self.locus_name = locus_name
        self.url = url
        self._fasta_contents: Dict[str, Sequence] = {}
        self._fasta_loaded = False

    def __str__(self):
        return f"{self.locus_name}[{self.url}]"

    def variant(self, seq_id: str) -> Sequence:
        # Lazy initialization
        if not self._fasta_loaded:
            self._load_fasta_contents()

        return self._fasta_contents[seq_id]

    def _load_fasta_contents(self):
        fasta_txt = download_from_url(self.url)
        fasta_io = StringIO(fasta_txt)
        records = SeqIO.parse(fasta_io, "fasta")

        for rec in records:
            if not rec.id.startswith(f'{self.locus_name}_'):
                raise ValueError(f"Found fasta record ID {rec.id}; expected prefix `{self.locus_name}_`")
            rec_id = rec.id[len(self.locus_name)+1:]
            self._fasta_contents[rec_id] = AllocatedSequence(str(rec.seq))

        fasta_io.close()
        self._fasta_loaded = True


def parse_taxa_id(xml_taxa_text: str, profile_url: str) -> TaxaIdentifier:
    tokens = xml_taxa_text.strip().split()
    if len(tokens) == 1:
        return TaxaIdentifier(tokens[0], "", "", profile_url)
    else:
        genus = tokens[0]
        sub_tokens = tokens[1].split('#')
        if len(sub_tokens) == 1:
            return TaxaIdentifier(genus, sub_tokens[0], "", profile_url)
        elif len(sub_tokens) == 2:
            return TaxaIdentifier(genus, sub_tokens[0], sub_tokens[1], profile_url)
        else:
            raise ValueError("Expected second token to have at most two parts (species, typeid). Text: {}".format(
                tokens[1]
            ))
    # else:
    #     raise ValueError(
    #         "Expected at most two tokens in XML species node text. Text: {}".format(xml_taxa_text)
    #     )


def parse_mlst_sequences(xml_path: Path) -> Iterator[Tuple[TaxaIdentifier, List[LocusContainer]]]:
    """
    Parse the XML into URL elements.
    Adapted from parseXML() implementation of FastMLST (https://github.com/EnzoAndree/FastMLST).
    :param xml_path: The filesystem path to the metadata XML file.
    :return:
    """
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    for parent in root.iter('species'):
        profile_url = next(next(parent.iter('profiles')).iter('url')).text.strip()
        t_id = parse_taxa_id(parent.text.strip(), profile_url)
        loci = [
            LocusContainer(locus_node.text.strip(), next(locus_node.iter('url')).text.strip())
            for locus_node in parent.iter('locus')
        ]
        yield t_id, loci


def parse_mlst_types(taxa: TaxaIdentifier, loci: List[LocusContainer], prefix: str = "") -> Iterator[Strain]:
    profile_text = download_from_url(taxa.profile_url)
    loci: Dict[str, LocusContainer] = {locus.locus_name: locus for locus in loci}

    # We expect this to be a TSV.
    profile_reader = csv.reader(profile_text.split('\n'), delimiter='\t')
    header = next(profile_reader)

    # columns = set(header)
    # has_clonal_complex_label = 'clonal_complex' in columns
    # has_species_label = 'species' in columns
    loci_columns = {
        c: i
        for i, c in enumerate(header)
        if c not in {'ST', 'clonal_complex', 'species'}
    }

    for row in profile_reader:
        if len(row) == 0:
            continue

        s_id = 'ST{}'.format(row[0])
        markers = [
            Marker(
                id='{}-{}'.format(locus_name, row[locus_col]),
                name=locus_name,
                seq=loci[locus_name].variant(row[locus_col]),
                canonical=True,
                metadata=None
            )
            for locus_name, locus_col in loci_columns.items()
        ]
        yield Strain(
            id=f'{taxa.genus}_{taxa.species}_T{taxa.typeid}_{s_id}',
            name=s_id,
            markers=markers,
            metadata=None,
        )


def generate_mlst_db(mlst_xml_path: Path, target_genera: Optional[Set[str]], db_data_dir: Path, db_name: str) -> StrainDatabase:
    db_backend = PandasAssistedBackend()
    for taxon, loci in parse_mlst_sequences(mlst_xml_path):
        # Filter target genus
        if target_genera is None or taxon.genus in target_genera:
            logger.debug(f'Schema type id: {taxon.typeid}')
            logger.debug("Found loci: {}".format(
                ",".join(locus.locus_name for locus in loci)
            ))
            db_backend.add_strains(parse_mlst_types(taxon, loci))
    return StrainDatabase(
        db_backend,
        data_dir=db_data_dir,
        name=db_name
    )


def parse_target_genera(in_path: Path) -> Set[str]:
    with open(in_path, "r") as f:
        return {x.strip() for x in f if len(x.strip()) > 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--db_name', type=str, required=True)
    parser.add_argument('-o', '--output_db_name', type=str, required=True)
    parser.add_argument('-t', '--target_genera_path', type=str, required=False, dest='target_genera_path',
                        help='A text file listing out all Genera to include, one per line.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_genera_path is not None:
        logger.debug("Using target genera specified in {}.".format(args.target_genera_path))
        target_genera = parse_target_genera(Path(args.target_genera_path))
        logger.debug("Found {} genera.".format(len(target_genera)))
        logger.debug(str(target_genera))
    else:
        target_genera = None

    out_dir = StrainDatabase.database_named_dir(cfg.database_cfg.data_dir, args.db_name)
    out_dir.mkdir(exist_ok=True, parents=True)
    mlst_xml_path = out_dir / 'mlst-dbases.xml'

    # Fetch Database XML resource
    xml_db_text = download_from_url("https://pubmlst.org/static/data/dbases.xml")
    with open(mlst_xml_path, "w") as f:
        f.write(xml_db_text)

    db = generate_mlst_db(mlst_xml_path, target_genera, cfg.database_cfg.data_dir, args.output_db_name)
    serializer = PickleParser(db.name, cfg.database_cfg.data_dir)
    serializer.save_to_disk(db)
    logger.debug("Saved database [{}] to {}".format(db.name, serializer.disk_path()))


if __name__ == "__main__":
    main()
