import hashlib
from bitarray import bitarray
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------
# Part 1: Parameter Setup
# ------------------------------------------------------------------------------------------------------

p = (2**31 - 1)             # mersenne prime >= N
n = 1000                    # size of subset, |S|
N = 10**6                   # size of universe , |U|
k_1 = 7                     # number of hash functions for par t1
c_1 = 10                    # c value for part 1
m_1 = int(c_1 * n)          # size of hash table for part 1

rng = random.Random()

'''
a = []
while len(a) < k_values:
    candidate = rng.randrange(1, p)
    if math.gcd(candidate, m) == 1:
        a.append(candidate)
'''

a = [rng.randrange(1, p) for _ in range(k_1)]
b = [rng.randrange(0, p) for _ in range(k_1)]
seeds = [rng.randrange(10**9) for _ in range(k_1)]          # seeds for seeded hash function



# ------------------------------------------------------------------------------------------------------
# Part 1: Hash Functions
# ------------------------------------------------------------------------------------------------------

# h1(x) = ((a * x + b) mod p) mod m
def hash1(x, m, k, a, b):
    x = int(__builtins__.hash(x))
    return [((a[i] * x + b[i]) % p) % m for i in range(k)]  # linear modulo hash function

# h2(x) = r(s[i] + x)
def hash2(x, m, k, seeds):
    x = int(__builtins__.hash(x))                           # converts all input types of x into an int
    hash2_results = []  
    for s in seeds: 
        r = random.Random(s + x)                            # add randomized seed to input
        hash2_results.append(r.randrange(m))                # placed into random bucket from 0 to m - 1
    return hash2_results    

# creates the hashtable, uses count because we care about maximum load
def hash_table(data, hash_type, m):
    buckets = [0] * m                                       # initializes all buckets to empty
    for x in data:  
        hashes = hash_type(x, m, k_1)                       # find hashes
        for hash in hashes: 
            buckets[hash] += 1                              # increment load in each bucket

    return buckets

# ------------------------------------------------------------------------------------------------------
# Part 1: Testing on Datasets
# ------------------------------------------------------------------------------------------------------

'''
# Define six datasets (data types)
random_data = [rng.randrange(N) for _ in range(5*m_1)]                     # random data from 0 to N
sequential_data = [n for n in range(5*m_1)]                                # 0, 1, 2, ..., 10*m
even_data = [2*n for n in range(5*m_1)]                                    # 0, 2, 4, ..., 20*m
repeated_data = [42] * (5*m_1)                                             # all same value
few_unique_data = [i % 10 for i in range(5*m_1)]                           # only 10 unique values repeated
alternating_data = [i if i % 2 == 0 else i+1000 for i in range(5*m_1)]     # 1, 1002, 3, 1004, ..., 10*m + 1000


def plot_buckets_on_plotsis(buckets, plot, title):
    plot.bar(range(len(buckets)), buckets)
    plot.set_xlabel("Bucket index")
    plot.set_ylabel("Count")
    plot.set_title(title)



# Prepare datasets list for iteration
datasets = [
    ("Random Data", random_data),
    ("Sequential Data", sequential_data),
    ("Even Data", even_data),
    ("Repeated Data", repeated_data),
    ("Few Unique Data", few_unique_data),
    ("Alternating Data", alternating_data),
]

# Compute hash tables for each dataset with hash1 and hash2
hash1_buckets_list = [hash_table(data, hash1, m_1) for (_name, data) in datasets]
hash2_buckets_list = [hash_table(data, hash2, m_1) for (_name, data) in datasets]

# Plot Hash1 results: one figure with 6 subplots (3x2)
fig1, plotses1 = plt.subplots(3, 2, figsize=(12, 12))
plotses1 = plotses1.ravel()

for i, (name, _data) in enumerate(datasets):
    plot_buckets_on_plotsis(hash1_buckets_list[i], plotses1[i], f"Hash1 - {name}")
plt.tight_layout()
plt.show()

# Plot Hash2 results: one figure with 6 subplots (3x2)
fig2, plotses2 = plt.subplots(3, 2, figsize=(12, 12))
plotses2 = plotses2.ravel()
for i, (name, _data) in enumerate(datasets):
    plot_buckets_on_plotsis(hash2_buckets_list[i], plotses2[i], f"Hash2 - {name}")
plt.tight_layout()
plt.show()
'''

# ------------------------------------------------------------------------------------------------------
# Part 2: Bloom Filter
# ------------------------------------------------------------------------------------------------------


class BloomFilter:
    def __init__(self, n, c, k, hash_type):
        self.n = n                                              # |S| = size of subset
        self.c = c                                              # factor of hash table to subsset size
        self.m = self.c * self.n                                # |T| = size of hash table
        self.bitarray = bitarray(self.m)                        # bitarray uses less memory than python array
        self.bitarray.setall(0)                                 # initialize at 0
        self.k = k                                              # number of hash functions
        #self.k = math.ceil(self.c * math.log(2))                # round up from optimal # of hash functions
        self.hash_type = hash_type

        # for hash functions
        self.a = [rng.randrange(1, p) for _ in range(k)]        # randomized a value
        self.b = [rng.randrange(0, p) for _ in range(k)]        # randomized b value
        self.seeds = [rng.randrange(10**9) for _ in range(k)]   # seeds for seeded hash function


    def add(self, x):
        if (self.hash_type == 1):                                      
            bits = hash1(x, self.m, self.k, self.a, self.b)     # find the hash1 position
        else:       
            bits = hash2(x, self.m, self.k, self.seeds)         # find the hash2 position

        for bit in bits:        
            self.bitarray[bit] = 1                              # cover the bits for both hash positions

    def contains(self, x):      
        if (self.hash_type == 1):                                            
            bits = hash1(x, self.m, self.k, self.a, self.b)     # find the hash1 position
        else:       
            bits = hash2(x, self.m, self.k, self.seeds)         # find the hash2 position

        # checks if every bit of x in the bitarray is set to 1 (meaning its in the maintained set)
        return all(self.bitarray[bit] == 1 for bit in bits)


# calculates false positive rate for a given size
def false_positve_rate(n, N, c, k, hash_type):
    bf = BloomFilter(n, c, k, hash_type)                    # create bloom filter
    m = c * n                                               # hash table size

    universe_sample = random.sample(range(N), n)            # universe of size N = |U|
    inserted_set = set(universe_sample)                     # used to exclude true positives in test

    for x in universe_sample:                               # insert elements
        bf.add(x)

    false_pos = 0                                           # track number of false positives
    tests_done = 0                                          # used to find false positives rate

    while tests_done < n:                                   # runs m tests
        x = random.randrange(N)                             # pick a random sample from N
        if x in inserted_set:                      
            continue                                        # we don't care about true positives, only false
        tests_done += 1                    
        if bf.contains(x):                                  # if the bloom filter says it's there even 
            false_pos += 1                     

    return false_pos / tests_done                           # ratio of false positives tests to total tests


# Plots the median value from 10 test runs of each k-value for a given c value
def plot_fpr(n, N, c, k_values, hash_type, tests, plot):
    m = c * n                                               # size of hash table
    fprs = []                                               # array of false positive rates for each c value

    for k in range(k_values):                               
        test_results = []                                   # store each test run result
        for test in range(tests):
            test_results.append(false_positve_rate(n, N, c, k, hash_type))
        test_median = sorted(test_results)[tests // 2]      # find the median of the test results
        fprs.append(test_median)                            # use the median of the tests as the fpr for that c and k value pair

    ideal_rate = [(1 - math.exp(-k/c))**k for k in range(1, k_values + 1)]
    
    # plotting figure
    plot.plot(range(1, k_values + 1), fprs, marker="x", label=f"c={c}")                     # plots simulated fpr
    plot.plot(range(1, k_values + 1), ideal_rate, marker="x", label=f"c_ideal={c}")         # plots ideal fpr
    plot.set_xlabel("k (# of Hash Functions)")                                              # x-axis label
    plot.set_ylabel("False Positve Rates")                                                  # y-axis label
    plot.set_title(f"False Positive Rate vs k ({'hash1' if hash_type==1 else 'hash2'})")    # title
    plot.grid(True)                                                                         # enable grid lines


# Parameters for plotting all the k and c value pairings
k_values = 20               # Number of hash functions we will test for each hash type and c_value
tests = 10                  # Number of tests ran for each pair of k_value and c_value
c_values = [7, 9, 11]       # Different c values to test
hash_types = [1, 2]         # Different hash functions, hash1 and hash2 implemented in part 1

# Create subplots: one for hash1, one for hash2
fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(14, 5))

for c in c_values:
    for hash_type in hash_types:
        print(f"Plot for c={c}, hash_type={hash_type}")
        plot = plot1 if hash_type == 1 else plot2
        plot_fpr(n, N, c, k_values, hash_type, tests, plot)

# Add legends to both subplots
plot1.legend()
plot2.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
