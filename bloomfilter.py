import hashlib
from bitarray import bitarray
import math
import random
import hashfunctions
import matplotlib.pyplot as plt

class BloomFilter:
    def __init__(self, n, c, k):
        self.n = n                                      # |S| = size of subset
        self.c = c                                      # factor of hash table to subsset size
        self.m = self.c * self.n                        # |T| = size of hash table
        self.bitarray = bitarray(self.m)                # bitarray uses less memory than python array
        self.bitarray.setall(0)                         # initialize at 0
        self.k = k                                      # number of hash functions
        #self.k = math.ceil(self.c * math.log(2))        # round up from optimal # of hash functions
    
    def add(self, x):
        pos1 = hashfunctions.hash1(x, self.m)
        pos2 = hashfunctions.hash2(x, self.m)

        for pos in pos1 + pos2:
            self.bitarray[pos] = 1

    def contains(self, x):
        pos1 = hashfunctions.hash1(x, self.m)
        pos2 = hashfunctions.hash2(x, self.m)

        return all(self.bitarray[pos] == 1 for pos in pos1 + pos2)



# calculates false positive rate for a given size
def false_positve_rate(n, N, c, k, hash_type):
    bf = BloomFilter(n, c, k)                           # create bloom filter
    m = c * n                                           # hash table size

    universe_sample = random.sample(range(N), n)        
    inserted_set = set(universe_sample)                 # used to exclude true positives in test

    # insert elements
    for x in universe_sample:
        if hash_type == 1:
            pos = hashfunctions.hash1(x, m)
        else:
            pos = hashfunctions.hash2(x, m)
        bf.add(x)

    false_pos = 0
    tests_done = 0

    while tests_done < 2 * n:
        x = random.randrange(N)                         # pick a random sample from N
        if x in inserted_set:
            continue                                    # we don't care about true positives, only false
        tests_done += 1
        if bf.contains(x):
            false_pos += 1

    return false_pos / n

# plots the median value from 10 test runs of each k-value for a given c value
def plot_fpr(n, N, c, k_values, hash_type, tests):
    m = c * n
    fprs = []

    for k in range(k_values):
        test_results = []
        for test in range(tests):
            test_results.append(false_positve_rate(n, N, c, k, hash_type))
        test_median = sorted(test_results)[tests // 2]
        fprs.append(test_median)
    
    plt.plot(range(1, k_values + 1), fprs, marker="x")
    plt.xlabel("k (# of Hash Functions)")
    plt.ylabel("False Positve Rates")
    plt.title(f"False Positive Rate vs k (c={c}, hash={'hash1' if hash_type==1 else 'hash2'})")
    plt.grid(True)
    plt.show()

plot_fpr(10000, 10**6, 5, 20, 1, 10)

plot_fpr(10000, 10**6, 10, 20, 1, 10)
plot_fpr(10000, 10**6, 10, 20, 2, 10)

plot_fpr(10000, 10**6, 15, 20, 1, 10)

