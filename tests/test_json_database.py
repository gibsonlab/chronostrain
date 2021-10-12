from chronostrain.database.parser.json import *


def test_deserialization():
    json_dict = {
        "name": "test_name",
        "accession": "test_accession",
        "markers": [
            {"name": "MARKER_1", "type": "primer", "forward": "FORWARD_PRIMER", "reverse": "REVERSE_PRIMER"},
            {"name": "MARKER_2", "type": "tag", "locus_id": "TEST_LOCUS_ID"}
        ]
    }

    strain_entry = StrainEntry.deserialize(json_dict, 0)
    assert strain_entry.index == 0
    assert strain_entry.id == json_dict["name"]
    assert strain_entry.accession == json_dict["accession"]
    assert len(strain_entry.marker_entries) == 2

    marker_entry = strain_entry.marker_entries[0]
    marker_entry_json = json_dict["markers"][0]
    assert isinstance(marker_entry, PrimerMarkerEntry)
    assert marker_entry.index == 0
    assert marker_entry.name == marker_entry_json["name"]
    assert marker_entry.forward == marker_entry_json["forward"]
    assert marker_entry.reverse == marker_entry_json["reverse"]
    assert marker_entry.parent == strain_entry

    marker_entry = strain_entry.marker_entries[1]
    marker_entry_json = json_dict["markers"][1]
    assert marker_entry.index == 1
    assert isinstance(marker_entry, TagMarkerEntry)
    assert marker_entry.name == marker_entry_json["name"]
    assert marker_entry.locus_tag == marker_entry_json["locus_id"]
    assert marker_entry.parent == strain_entry


def build_regex_test_case(forward_primer, reverse_primer, genome) -> SubsequenceLoader:
    json_dict = {
        "name": "test_name",
        "accession": "test_accession",
        "markers": [
            {"name": "MARKER_1", "type": "primer", "forward": forward_primer, "reverse": reverse_primer}
        ]
    }
    strain_entry = StrainEntry.deserialize(json_dict, 0)

    seq_loader = SubsequenceLoader(
        fasta_filename=Path("NONE"),
        genbank_filename=Path("NONE"),
        marker_entries=strain_entry.marker_entries,
        marker_max_len=500
    )
    seq_loader.full_genome = genome
    return seq_loader


def test_regex():
    genome = "random_string_1_AACCGGTT_test_string_GATATATA_random_string_2"
    forward_primer = "AACCGGTT"
    reverse_primer = "TATATATC"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (16, 45)

    genome = "random_string_1_AACCGGTT_test_string_GATATATA_random_string_2_GATATATA"
    forward_primer = "AACCGGTT"
    reverse_primer = "TATATATC"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (16, 45)

    genome = "AACCGGTT_random_string_1_AACCGGTT_test_string_GATATATA_random_string_2_GATATATA"
    forward_primer = "AACCGGTT"
    reverse_primer = "TATATATC"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (25, 54)

    genome = "random_string_1_AACCGGTT_test_string_GATATATA_random_string_2"
    forward_primer = "RRCCGGTT"
    reverse_primer = "TATATATC"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (16, 45)

    genome = "random_string_1_AACCGGTT_test_string_GATATATA_random_string_2_GATATATA"
    forward_primer = "RRCCGGTT"
    reverse_primer = "TATATAYY"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (16, 45)

    genome = "AACCGGTT_random_string_1_AACCGGTT_test_string_GATATATA_random_string_2_GATATATA"
    forward_primer = "RRCCGGTT"
    reverse_primer = "TATATAYY"
    seq_loader = build_regex_test_case(forward_primer, reverse_primer, genome)
    result = seq_loader._regex_match_primers(forward_primer, reverse_primer)
    assert result == (25, 54)


def test_subsequence_forward():
    # 11..50 gets parsed as FeatureLocation(10,20,+)
    genome = 'AGAGTCAATGAATCGTTTACATTTCAAATTTCCAATGATA'
    s = NucleotideSubsequence(name="test123", id="abcde", start_index=10, end_index=20, complement=False)
    assert s.get_subsequence(genome) == 'AATCGTTTAC'


def test_subsequence_complement():
    # complement(11..50) gets parsed as FeatureLocation(10,20,-)
    genome = 'AGAGTCAATGAATCGTTTACATTTCAAATTTCCAATGATA'
    s = NucleotideSubsequence(name="test123", id="abcde", start_index=10, end_index=20, complement=True)
    assert s.get_subsequence(genome) == 'GTAAACGATT'
