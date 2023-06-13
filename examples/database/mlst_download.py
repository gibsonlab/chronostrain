from typing import Tuple, Iterator, List, Set, Optional
import argparse
import math

from io import StringIO
from pathlib import Path
from xml.etree import ElementTree
import urllib.error
import urllib.request

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def convert_size(size_bytes: int) -> str:
    """
    Converts bytes to the nearest useful meaningful unit (B, KB, MB, GB, etc.)
    Code credit to https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python/14822210
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def download_from_url(url: str) -> str:
    try:
        print(f"Fetching URL resource {url}")
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
    print("Got a response of size {}.".format(convert_size(len(r_raw))))
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

    def __str__(self):
        return f"{self.locus_name}[{self.url}]"

    def variants(self) -> Iterator[Tuple[str, str]]:
        fasta_txt = download_from_url(self.url)
        fasta_io = StringIO(fasta_txt)
        records = SeqIO.parse(fasta_io, "fasta")
        for rec in records:
            if not rec.id.startswith(f'{self.locus_name}_'):
                raise ValueError(f"Found fasta record ID {rec.id}; expected prefix `{self.locus_name}_`")
            yield rec.id, str(rec.seq)
        fasta_io.close()


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


def generate_marker_index(mlst_xml_path: Path, target_taxa: Optional[Set[str]], index_path: Path):
    f = open(index_path, 'wt')
    for taxon, loci in parse_mlst_sequences(mlst_xml_path):
        # Filter target genus
        if target_taxa is None or f'{taxon.genus} {taxon.species}'.lower() in target_taxa:
            print(f'Schema type id: {taxon.typeid}')
            for locus in loci:
                print(f"Handling locus {locus.locus_name}")

                first_variant_id, first_variant_seq = next(iter(locus.variants()))
                fasta_path = index_path.parent / f"{taxon.genus}_{taxon.species}_{taxon.typeid}_{locus.locus_name}.fasta"
                SeqIO.write(
                    SeqRecord(seq=Seq(first_variant_seq), id=locus.locus_name, description=first_variant_id),
                    fasta_path, "fasta"
                )
                print(
                    f"{locus.locus_name}\t{fasta_path}\tMLST:{taxon.genus}_{taxon.species}_{taxon.typeid}",
                    file=f
                )
    f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-index', dest='output_index',
                        type=str, required=True,
                        help='The path to the marker index TSV file. If one already exists, will append to the index.')
    parser.add_argument(
        '-t', '--target-taxa', dest='target_taxa',
        type=str, required=False,
        help='A comma-separated list of all taxa to include.'
             '(We expect MLST scheme names to follow the naming convention <genus> <species><modifier>, '
             'e.g. `Eshcerichia coli#1` or `Enterrococcus faecalis`)'
    )
    parser.add_argument(
        '-w', '--work-dir', dest='work_dir',
        type=str, required=False,
        help='A directory to store intermediate files (e.g. the XML Schema) into. '
             'If not specified, will use the directory containing output-index.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index_path = Path(args.output_index)
    print("Downloading marker seeds from MLST schema.")

    if args.target_taxa is not None:
        target_taxa = {x.strip().lower() for x in args.target_taxa.split(',')}
        print("Targeting {} taxa using MLST scheme.".format(len(target_taxa)))
    else:
        target_taxa = None

    if args.work_dir is not None:
        work_dir = Path(args.work_dir)
    else:
        work_dir = index_path.parent

    work_dir.mkdir(exist_ok=True, parents=True)
    index_path.parent.mkdir(exist_ok=True, parents=True)

    # Fetch Database XML resource
    mlst_xml_path = work_dir / 'mlst-dbases.xml'
    xml_db_text = download_from_url("https://pubmlst.org/static/data/dbases.xml")
    with open(mlst_xml_path, "w") as f:
        f.write(xml_db_text)

    # Save FASTA records and write to index
    generate_marker_index(mlst_xml_path, target_taxa, index_path)


if __name__ == "__main__":
    main()
