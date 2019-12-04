from nnserver.composite_encoder import CompositeTokenEncoder


def test_roundtrip(sample=None):
    if sample is None:
        sample = "P S-MAIN IP NP-SUBJ pfn_et_nf_p3 /NP-SUBJ /IP /S-MAIN /P"
    default_encoder = CompositeTokenEncoder()
    subtoken_ids = default_encoder.encode(sample)
    decoded_sample = default_encoder.decode(subtoken_ids)
    assert sample == decoded_sample, "Encoding roundtrip does not match, {} - {}".format(sample, decoded_sample)


def test_file(infile):
    o = open(infile)
    for line in o:
        test_roundtrip(line.strip())
