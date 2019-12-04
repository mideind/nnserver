def create_vocab(infile):
    o = open(infile)
    vocab = set()
    for line in o:
        vocab.update(set(line.split('\t')[1].strip().split()))
    for i in vocab:
        print(i)

