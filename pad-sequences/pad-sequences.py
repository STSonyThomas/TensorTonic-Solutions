import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) ==0:
        L = max_len if max_len is not None else 0
        return np.empty((0,L),dtype=np.int64)
    MAX_SIZE = 0
    for i in range(len(seqs)):
        MAX_SIZE = max(MAX_SIZE,len(seqs[i]))
    MAX_SIZE = max_len if max_len else MAX_SIZE
    padded_seq = [np.pad(seq[:MAX_SIZE],(0,MAX_SIZE-len(seq[:MAX_SIZE])),mode="constant",constant_values=pad_value) for seq in seqs]
    padded_seq = np.asarray(padded_seq,dtype=np.int64)
    return padded_seq
    pass