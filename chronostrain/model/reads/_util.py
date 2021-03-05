import torch


def mutate_acgt(base):
    # TODO come up with a faster implementation. Maybe represent bases using Z_4, and do mod 4 addition...
    i = torch.randint(low=0, high=3, size=[1]).item()
    bases = {'A', 'C', 'G', 'T'}
    bases.remove(base)
    return list(bases)[i]
