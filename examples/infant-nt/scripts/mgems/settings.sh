# ================= Themisto + mSWEEP settings
export N_CORES=8

export EUROPE_EFAECALIS_INDEX=/data/local/europe_efaecalis/index.tsv
export INFANT_ISOLATE_INDEX=${DATA_DIR}/database/infant_isolates/index.tsv  # this is a byproduct of ChronoStrain's database construction.

export SPECIES_REF_DIR=/data/cctm/youn/infant_nt/database/mgems/themisto_640k
export SPECIES_REF_INDEX=index
export SPECIES_REF_CLUSTER=index_mSWEEP_indicators.txt

# This is the pre-built index from the ELMC mGEMS analysis authors.
export EFAECALIS_REF_DIR=/data/cctm/youn/infant_nt/database/mgems/Efaecalis_elmc_natcom
export EFAECALIS_REF_INDEX=index_v3/index_v3
export EFAECALIS_REF_CLUSTER=index_v3/index_v3_mSWEEP_indicators.txt

# This is the database that was re-cosntructed using ChronoStrain's granularity.
export EFAECALIS_CHRONO_MIRROR_REF_DIR=/data/cctm/youn/infant_nt/database/mgems/Efaecalis_chrono_mirror
export EFAECALIS_CHRONO_MIRROR_REF_INDEX=ref_idx/ref_idx
export EFAECALIS_CHRONO_MIRROR_CLUSTER=ref_clu.txt

# This is the database that was re-cosntructed using ChronoStrain's granularity (where CS was clustered at 99.99% thresholding)
export EFAECALIS_CHRONO_99_99PCT_MIRROR_REF_DIR=/data/cctm/youn/infant_nt/database/mgems/Efaecalis_chrono_99_99pct_mirror
export EFAECALIS_CHRONO_99_99PCT_MIRROR_REF_INDEX=ref_idx/ref_idx
export EFAECALIS_CHRONO_99_99PCT_MIRROR_CLUSTER=ref_clu.txt


aln_and_compress()
{
	fq1=$1
	fq2=$2
	aln_out1=$3
	aln_out2=$4
	ref_idx=$5
	n_colors=$6
	tmp_dir=$7

  mkdir -p ${tmp_dir}
  input_file=${tmp_dir}/query_files.txt
  output_file=${tmp_dir}/output_files.txt
	aln_raw1=${tmp_dir}/aln1_raw.txt
	aln_raw2=${tmp_dir}/aln2_raw.txt

	# prepare input txt file list
  echo "${fq1}" > "$input_file"
  echo "${fq2}" >> "$input_file"
  echo "${aln_raw1}" > "$output_file"
  echo "${aln_raw2}" >> "$output_file"
	themisto pseudoalign \
    --index-prefix ${ref_idx} --rc --temp-dir ${tmp_dir} --n-threads ${N_CORES} --sort-output-lines \
    --query-file-list "$input_file" \
    --out-file-list "$output_file" \

  n_reads1=$(wc -l < "${aln_raw1}")
  n_reads2=$(wc -l < "${aln_raw2}")
  if [[ $n_reads1 != $n_reads2 ]]; then
    echo "# of reads in pseudoalignments don't match (${n_reads1} vs ${n_reads2})."
    exit 1
  fi
  alignment-writer -n $n_colors -r $n_reads1 -f $aln_raw1 > $aln_out1
  alignment-writer -n $n_colors -r $n_reads2 -f $aln_raw2 > $aln_out2
}

export aln_and_compress
