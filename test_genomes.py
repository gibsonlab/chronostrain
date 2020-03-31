import simulation as sim

genome = sim.parse_fasta("data/NZ_DS981518.1.fasta")[0]
reads = genome.produce_reads_multinomial(
    read_len=250,
    num_reads=100
)
for read in reads:
    print(len(read))
print(len(reads))
