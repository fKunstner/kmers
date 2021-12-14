from scipy.sparse import csr_matrix, save_npz  # BSD-3
import numpy as np                             # BSD-3
import fastaparser                             # GPLv3

def simulate_reads(isoform_file, rna_seq_file, N, L):
    isoforms = []
    with open(isoform_file, 'r') as f:
        parser = fastaparser.Reader(f, parse_method='quick')
        pos = 0
        for s in parser:
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            if len(seq) < L:
                continue
            if pos % 10000 == 0:
                print("seqs read = ", pos)
            isoforms.append(seq)
            pos += 1
    K = len(isoforms)
    print("isoforms found = ", K)
    alpha = 0.5 * np.ones(K)
    theta = np.random.dirichlet(alpha)
    print("theta[0:10] =", theta[0:10])
    print("theta[K-10:K] =", theta[K-10:K])
    y = np.random.choice(K, size = N, replace=True, p = theta)
    with open(rna_seq_file, 'w') as out:
        for n in range(N):
            if (n + 1) % 100000 == 0:
                print("sim n = ", n + 1)
            seq = isoforms[y[n]]
            start = np.random.choice(len(seq) - L + 1)
            out.write(">sim-")
            out.write(str(n))
            out.write('\n')
            out.write(seq[start:start+L])
            out.write('\n')
            
