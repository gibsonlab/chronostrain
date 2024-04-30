from typing import *
from pathlib import Path
import gzip

import numpy as np
import pandas as pd
from tqdm import tqdm
from chronostrain.util.external import call_command


def combine_umb_reads_for_kraken(read_paths: List[Path], target_path: Path, gzip_compress: bool):
    """
    Concatenate paired-end files with an "x", and also add reads from the unpaired files.
    """
    paired_1: Path = None
    paired_2: Path = None
    unpaired: List[Path] = []
    for p in read_paths:
        if 'paired_1' in p.name:
            paired_1 = p
        elif 'paired_2' in p.name:
            paired_2 = p
        else:
            unpaired.append(p)

    raw_out = target_path.parent / 'combined.fastq'
    out_f = open(raw_out, 'wt')

    # Sort the fastq files.
    tmpdir = target_path.parent / 'TMP'
    tmpdir.mkdir(exist_ok=True)
    paired_1_tmp = target_path.parent / 'paired_1.fastq'
    paired_2_tmp = target_path.parent / 'paired_2.fastq'
    paired_1_sorted = target_path.parent / 'paired_1.sorted.fastq'
    paired_2_sorted = target_path.parent / 'paired_2.sorted.fastq'
    if paired_1.suffix == '.gz':
        call_command('pigz', ['-cd', paired_1], stdout=paired_1_tmp)
    else:
        paired_1_tmp.symlink_to(paired_1)

    if paired_2.suffix == '.gz':
        call_command('pigz', ['-cd', paired_2], stdout=paired_2_tmp)
    else:
        paired_2_tmp.symlink_to(paired_2)
    call_command('fastq-sort', ['-n', f'--temporary-directory={tmpdir}', paired_1_tmp], stdout=paired_1_sorted)
    call_command('fastq-sort', ['-n', f'--temporary-directory={tmpdir}', paired_2_tmp], stdout=paired_2_sorted)
    paired_1_tmp.unlink()  # clean up
    paired_2_tmp.unlink()  # clean up
    
    with open(paired_1_sorted, 'rt') as f1, open(paired_2_sorted, 'rt') as f2:
        for l1, l2 in zip(f1, f2):
            l1 = l1.rstrip()
            l2 = l2.rstrip()
            assert l1.startswith("@")
            assert l2.startswith("@")
            assert l1.endswith("/1")
            assert l2.endswith("/2")
            read_id = l1[1:-2]
            read_id2 = l2[1:-2]
            assert read_id == read_id2

            seq1 = f1.readline().rstrip()
            seq2 = f2.readline().rstrip()
            f1.readline()
            f2.readline()
            qual1 = f1.readline().rstrip()
            qual2 = f2.readline().rstrip()

            print(f"@{read_id}", file=out_f)
            print(f"{seq1}x{seq2}", file=out_f)
            print("+", file=out_f)
            print(f"{qual1}x{qual2}", file=out_f)
    
    paired_1_sorted.unlink()
    paired_2_sorted.unlink()
    tmpdir.rmdir()
    for unpaired_path in unpaired:
        with gzip.open(unpaired_path, 'rt') as f:
            for line in f:
                out_f.write(line)
    out_f.close()
    if gzip_compress:
        call_command('pigz', ['-6', '-c', raw_out], stdout=target_path)
        raw_out.unlink()
    else:
        raw_out.rename(target_path)


def parse_path_csv(reads_csv: Path, time_points: List[float]) -> Dict[str, List[Path]]:
    timepoint_to_full_reads = {t: [] for t in time_points}
    input_df = pd.read_csv(
        reads_csv,
        sep=',',
        header=None,
        names=['T', 'SampleName', 'ReadDepth', 'ReadPath', 'ReadType', 'QualityFormat']
    ).astype(
        {
            'T': 'float32',
            'SampleName': 'string',
            'ReadDepth': 'int64',
            'ReadPath': 'string',
            'ReadType': 'string',
            'QualityFormat': 'string'
        }
    )
    for _, row in input_df.iterrows():
        t = row['T']
        path_str = row['ReadPath']
        # ====== Development-specific fix: some file paths are broken. Fix them.
        from_prefix = "/data/cctm/youn/umb"
        to_prefix = "/mnt/e/umb_data"
        if path_str.startswith(from_prefix):
            suffix = path_str[len(from_prefix):]
            path_str = f"{to_prefix}{suffix}"
        # ====== END fix
        
        if t not in timepoint_to_full_reads:
            raise KeyError(f"Couldn't find timepoint {t} in filtered reads file.")
        timepoint_to_full_reads[t].append(Path(path_str))
    
    return timepoint_to_full_reads


def quantify_ecoli(dataset_name: str, prefilt_csv_path: Path, time_points: List[float], kraken_db: Path) -> np.ndarray:
    cwd = Path().absolute()
    workdir = cwd / "_kraken" / dataset_name
    
    breadcrumb = workdir / "quantify_ecoli.DONE"
    if not breadcrumb.exists():
        print("[*] Kraken+Bracken results not found. Running species-level quantification.")
        print("[**] Target dir = {}".format(workdir))
        workdir.mkdir(exist_ok=True, parents=True)
        original_reads = parse_path_csv(prefilt_csv_path, time_points)
        for t_idx, t in enumerate(tqdm(time_points)):
            t_workdir = workdir / f'timepoint_{t_idx}'
            t_workdir.mkdir(exist_ok=True)

            bracken_breadcrumb = t_workdir / 'bracken.DONE'
            if bracken_breadcrumb.exists():
                continue
                
            combined_read_path = t_workdir / 'reads.fastq.gz'
            print(f"Combining reads for t_idx={t_idx}...")
            combine_umb_reads_for_kraken(original_reads[t], combined_read_path, gzip_compress=True)

            # Run Kraken to classify and report bins/bin abundances.
            kraken_report = t_workdir / 'reads.kreport'
            kraken_out = t_workdir / 'output.kraken'
            print("Running Kraken.")
            # kraken2 --db ${KRAKEN_DB} --threads ${THREADS} --report ${SAMPLE}.kreport ${SAMPLE}.fq > ${SAMPLE}.kraken
            call_command(
                command='kraken2',
                args=[
                    '--db', kraken_db,  # TODO replace this with configuration.
                    '--threads', 24,
                    '--report', kraken_report,
                    '--gzip-compressed',
                    combined_read_path
                ],
                stdout=kraken_out
            )

            # Run bracken to re-estimate abundances at species level.
            # bracken -d ${KRAKEN_DB} -i ${SAMPLE}.kreport -o ${SAMPLE}.bracken -r ${READ_LEN} -l ${LEVEL} -t ${THRESHOLD}
            braken_output = t_workdir / 'output.bracken'
            bracken_report = t_workdir / 'report.bracken'
            print("Running Bracken.")
            call_command(
                command='bracken',
                args=[
                    '-d', kraken_db,
                    '-i', kraken_report,
                    '-o', braken_output,
                    '-w', bracken_report,
                    '-r', 150,
                    '-l', 'S'
                ]
            )
            bracken_breadcrumb.touch()
            combined_read_path.unlink()
        breadcrumb.touch()
    
    # parse bracken results.
    # from the suggestion in: https://github.com/jenniferlu717/Bracken/issues/177
    ecoli_concentrations = np.zeros(len(time_points), dtype=float)
    for t_idx, t in enumerate(time_points):
        out_path_absolute = workdir / f'timepoint_{t_idx}' / 'output.bracken'
        _df = pd.read_csv(out_path_absolute, sep='\t')
        ecoli_r = _df.loc[_df['name'] == 'Escherichia coli', 'new_est_reads'].head(1).item()
        total_r = _df['new_est_reads'].sum()
        ecoli_concentrations[t_idx] = ecoli_r / total_r
    return ecoli_concentrations