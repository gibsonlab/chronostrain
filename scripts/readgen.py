import sys
import os
import subprocess
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

from chronostrain import cfg

'''
Abundance profile should begin with a header row containing accessions and
each row after should contain the time point followed by relative abundances
'''


def parse_abundance_profile(abundance_profile):
    profile_rows = abundance_profile.readlines()
    time_abundance_map = {}
    accessions = [accession.strip().strip('"') for accession in profile_rows[0].strip().split(',')[1:]]

    for row in profile_rows[1:]:
        entries = row.strip().split(',')
        time_abundance_map[entries[0]] = {}
        for i in range(len(accessions)):
            time_abundance_map[entries[0]][accessions[i]] = float(entries[i + 1])

    return time_abundance_map


'''
Spawns a parallel subprocess call to ART Illumina, outputting to output_path. 
Note all parameters must be Strings and file names relative to output_path or
absolute
'''


def invoke_art(reference_path, reads, output_path, output_prefix, profile_first, profile_second, read_length, seed):
    try:
        p = subprocess.run(['art_illumina', '--qprof1', profile_first, '--qprof2', profile_second, \
                            '-sam', '-i', reference_path, '-l', read_length, '-c', reads, '-p', '-m', '200', '-s', \
                            '10', '-o', output_prefix, '-rs', str(seed)], cwd=output_path, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE)
        # print(p.stderr.decode("utf-8"))
        return os.path.join(output_path, output_prefix), None
    except Exception as e:
        return None, e


'''
Creates a thread pool and zipped invoke_art parameters. Manages them in parallel, suppresses stdout
'''


def generate_reads(sample_size,
                   read_length,
                   chronostrain_db,
                   output_path,
                   trial,
                   profile_first,
                   profile_second,
                   abundances,
                   seed):
    thread_pool = Pool(cpu_count())
    configs = []
    for strain in chronostrain_db.all_strains():
        ref_file = strain.metadata.file_path
        configs.append(
            (ref_file,
             str(int(abundances[strain.metadata.ncbi_accession] * sample_size)),
             output_path,
             trial + '-' + strain.metadata.ncbi_accession,
             profile_first,
             profile_second,
             str(read_length), seed)
        )
    resulting_files = []
    for output, error in thread_pool.starmap(invoke_art, configs):
        if error is not None:
            print(error)
        else:
            resulting_files.append(output)
    return resulting_files


def concat_results(result_files, output_path, time_point):
    output_filepath = os.path.join(output_path, time_point + '-all_reads.fq')
    with open(output_filepath, 'w') as output:
        for f_in in result_files:
            for f in [f_in + str(i) + '.fq' for i in range(1, 3)]:
                with open(f, 'r') as input:
                    for line in input:
                        output.write(line)
                output.write('\n')
    return output_filepath


def create_chronostrain_input_map(output_path, read_files):
    with open(os.path.join(output_path, 'input_files.csv'), 'w') as in_files:
        for time_point, file in read_files:
            in_files.write('"' + time_point + '","' + file + '"\n')


def parse_args():
    if len(sys.argv) != 9:
        print("Unexpected number of arguments")
        sys.exit()
    try:
        open(sys.argv[4], 'r')
        open(sys.argv[5], 'r')
    except IOError:
        print("Quality profiles do not exist")
        sys.exit()

    # map_file = sys.argv[6]
    # try:
    #     map_file = open(map_file, 'r')
    # except IOError:
    #     print("Community profile does not exist")
    #     sys.exit()

    abundance_profile = sys.argv[6]
    try:
        abundance_profile = open(abundance_profile, 'r')
    except IOError:
        print("Abundance profile does not exist")
        sys.exit()

    return (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5],
            abundance_profile, sys.argv[7], int(sys.argv[8]))


if __name__ == "__main__":
    sample_size, read_length, trial, first_profile, second_profile, abundance_profile, output_path, seed = parse_args()

    final_read_files = []
    time_abundance_map = parse_abundance_profile(abundance_profile)

    chronostrain_database = cfg.database_cfg.get_database()

    for time_point in time_abundance_map.keys():
        results = generate_reads(
            sample_size, read_length, chronostrain_database, output_path,
            time_point + '-' + trial, first_profile, second_profile,
            time_abundance_map[time_point], seed)
        final_read_files.append((time_point, concat_results(results, output_path, time_point)))
        seed += 1
    create_chronostrain_input_map(output_path, final_read_files)

