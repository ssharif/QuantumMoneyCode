import numpy as np
from scipy.spatial import distance
from itertools import chain

def eigenform_scaled(N,P):
    """"Give system of scaled eigenvalues for Gamma0(N).

    Give tuple [a_p/sqrt(p)] for primes p < P for each eigenform,
    but discard Eisenstein space."""
    E=numerical_eigenforms(N).systems_of_eigenvalues(P)[:-1]
    prim=primes(P)
    for j,p in enumerate(prim):
        for i in range(len(E)):
            E[i][j] = E[i][j]/float(sqrt(p))
    return E

def mindist(d):
    """Compute minimum distance in list of vectors.

    Given matrix whose rows are v1,..,vm in Rn,
    output smallest |vi - vj|, i != j."""
    y = np.array(d)
    x=distance.pdist(y)
    return x[x.argmin()]

def epsilon_separation(N,P=0):
    """Compute minimum distance between serial numbers.

    Take vector of a_p/sqrt(p) where p are primes up to P
    for eigenforms of Gamma0(N).
    Use Euclidean metric."""
    if P==0:
        P=int(log(N,2).n())
    return mindist(cstors(exxplist(eigenform_scaled(N,P))))

def es_range(a,b):
    """Compute epsilon separation for primes in range.

    Primes vary from a to b.
    Compute prime coeffs up to log_2(N)."""
    for N in primes(a,b):
        print(N,epsilon_separation(N))
    return

def exxp(z):
    return exp(I*z).n()

def exxplist(d):
    return [list(map(exxp,i)) for i in d]

def cstors(D):
    """Separate complex numbers into real and imaginary parts.

    Input list [z1,z2,...] and output 
    [Re(z1),Re(z2),...,Im(z1),Im(z2),..]."""
    return [list(chain(map(real,i),map(imaginary,i))) for i in D]
