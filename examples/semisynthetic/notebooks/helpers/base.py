from typing import Dict
from pathlib import Path
from collections import defaultdict


def replicate_dir(mut_ratio: str, replicate: int) -> Path:
    return Path('/mnt/e/semisynthetic_data') / f'mutratio_{mut_ratio}' / f'replicate_{replicate}'


def trial_dir(mut_ratio: str, replicate: int, read_depth: int, trial: int) -> Path:
    return replicate_dir(mut_ratio, replicate) / f'reads_{read_depth}' / f'trial_{trial}'


def parse_runtime(p: Path) -> int:
    with open(p, 'rt') as f:
        return int(f.readline().strip())


def parse_phylogroups() -> Dict[str, str]:
    """To each strain, add a phylogroup annotation."""
    phylogroup_path = Path("/mnt/e/chronostrain/phylogeny/ClermonTyping/umb_phylogroups_complete.txt")
    _dict = defaultdict(lambda: '?')
    with open(phylogroup_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            accession = Path(tokens[0]).with_suffix('').with_suffix('').name
            phylogroup = tokens[4]
            _dict[accession] = phylogroup
    return _dict
