import os
import re
import json
import urllib.request as urllib
from util.io.logger import logger
from util.io.filesystem import convert_size, get_filesize_bytes
from dataclasses import dataclass

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


def fetch_sequences(refs_file_json: str):
    """
    Read JSON file, and download FASTA from accessions if doesn't exist.
    :return: a dictionary mapping accessions to strain-accession-filename-subsequences
             wrappers.
    """
    strains_map = {}

    with open(refs_file_json, "rb") as f:
        strain_infos = json.load(f, object_hook = lambda info: StrainInfo(info))
        for info in strain_infos:
            info.validate()
            fasta_filename, genbank_filename = fetch_filenames(info['accession'])
            strains_map[info['accession']] = {
                "strain": info['name'],
                "accession": info['accession'],
                "file": fasta_filename,
                "subsequences": parse_genbank_tags(genbank_filename, info['tags'])
            }

    logger.info("Found {} records.".format(len(strains_map.keys())))
    return strains_map
    

def parse_subsequence_tag(genbank_tag: str, sequence_name, max_index):
    '''
    Parses a genome location tag, e.g. "complement(312..432)" to create SubsequenceMetadata
    and pads indices by 100bp
    '''
    indices = re.findall(r'[0-9]+', genbank_tag)
    if not len(indices) == 2:
        logger.warning('Encountered match to malformed tag: ' + genbank_tag)
        return None
    
    return SubsequenceMetadata(
        name=sequence_name, start_index=max(0, int(indices[0])-100), 
        end_index=min(int(indices[1])+100, max_index), complement='complement' in genbank_tag
    )
    

def parse_genbank_tags(filename, tags_to_names):
    '''
    :param filename: Name of local genbank file
    :param tags_to_names: Dictionary of locus tags to the name of their subsequence
    :return: Dictionary mapping input locus tags to SubsequenceMetadata
    '''
    try:
        genbank_file = open(filename, 'rb')
    except IOError:
        logger.warning("Unable to read file: " + filename)
        return {}
		
    parent_genome_length = float('inf')
    chunk_designation = None
    potential_index_tag = ''
    locus_tags = set(tags_to_names.keys())
    tag_to_index = {}
    
    for line in genbank_file:
        # Split on >1 space
        split_line = re.split(r'\s{2,}', line.decode('utf-8').strip())
        
        if len(split_line) == 2:
            potential_index_tag = ''
            chunk_designation = split_line[0]
            if chunk_designation == 'source':
                # Tag data begins with the full genome declaration as follows
                # source    start_index..end_index
                parent_genome_length = int(re.findall(r'[0-9]+', split_line[1])[1])
            elif chunk_designation == 'gene':
                potential_index_tag = split_line[1]
            
        if chunk_designation == 'gene':
            if 'locus_tag' in split_line[-1]:
                # Tags are declared by: /locus_tag=""
                tag = split_line[-1].split('"')[1]
                if tag in locus_tags:
                    tag_index = parse_subsequence_tag(potential_index_tag, tags_to_names[tag], parent_genome_length)
                    if tag_index is not None:
                        tag_to_index[tag] = tag_index
                    
    genbank_file.close()
    return tag_to_index
    

class StrainInfo(dict):
    def __getattr__(self, key):
        return self[key]
 
    def __setattr__(self, key, value):
        if key not in ['name', 'accession', 'tags']:
            logger.error('Invalid attribute name in refs JSON')
            raise KeyError
        self[lower(key)] = value

    def validate(self):
        if 'name' not in self.keys() or 'accession' not in self.keys() or 'tags' not in self.keys():
            logger.error('Missing name, accession, or tags for entry in refs JSON: ' + str(self))
            class StrainInfoIncompleteError(Exception):
                pass
            raise StrainInfoIncompleteError


@dataclass
class SubsequenceMetadata:
    name: str
    start_index: int
    end_index: int
    complement: bool # Indicates whether the intended sequence is the complement of the given sequence
    
    def get_subsequence(self, genome: str):
        complement_translation = {'A': 'T', 'T':'A', 'G':'C', 'C':'G'}
        complement_func = (lambda base: complement_translation[base]) if self.complement else lambda base: base
        subsequence = ''
        for base in genome[self.start_index:self.end_index]:
            subsequence += complement_func(base)
        return subsequence
        

