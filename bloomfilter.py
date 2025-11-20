from bitarray import bitarray
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------
# Part 1: Parameter Setup
# ------------------------------------------------------------------------------------------------------

p = (2**31 - 1)             # mersenne prime >= N
N = 10**6                   # size of universe , |U|
k_1 = 7                     # number of hash functions for par t1
c_1 = 10                    # c value for part 1
m_1 = 10000                 # size of hash table for part 1

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
def hash2(x, m, seeds):
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
        if hash_type == 1:
            hashes = hash1(x, m, k_1, a, b)                 # find hashes
        elif hash_type == 2:
            hashes = hash2(x, m, seeds)                     # find hashes

        for hash in hashes: 
            buckets[hash] += 1                              # increment load in each bucket

    return buckets

# ------------------------------------------------------------------------------------------------------
# Part 1: Testing on Datasets
# ------------------------------------------------------------------------------------------------------


'''
# define two kinds of datasets (data types)
random_data = [rng.randrange(N) for _ in range(5 * m_1)]                   # random data from 0 to N
correlated_data = [2*n for n in range(5 * m_1)]                            # 0, 2, 4, ..., 20*m


def plot_buckets_on_plotsis(buckets, plot, title):                         # plots the buckets on given plot    
    plot.bar(range(len(buckets)), buckets)
    plot.set_xlabel("Bucket index")
    plot.set_ylabel("Count")
    plot.set_title(title)


# prepare datasets list for iteration
datasets = [
    ("Random Data", random_data),
    ("Correlated Data", correlated_data)
]

# compute hash tables for each dataset with hash1 and hash2
hash1_buckets_list = [hash_table(data, 1, m_1) for (_name, data) in datasets]
hash2_buckets_list = [hash_table(data, 2, m_1) for (_name, data) in datasets]

fig1, plotses1 = plt.subplots(1, 2, figsize=(12, 8))                       # create subplots for hash1 results       
plotses1 = plotses1.ravel()                                                # flattens the array of plots for easy looping

for i, (name, _data) in enumerate(datasets):                               # plot each dataset
    plot_buckets_on_plotsis(hash1_buckets_list[i], plotses1[i], f"Hash1 - {name}")
plt.tight_layout()
plt.show()

fig2, plotses2 = plt.subplots(1, 2, figsize=(12, 8))                       # repeat for hash2 results
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
            bits = hash2(x, self.m, self.seeds)                 # find the hash2 position

        for bit in bits:        
            self.bitarray[bit] = 1                              # cover the bits for both hash positions

    def contains(self, x):      
        if (self.hash_type == 1):                                            
            bits = hash1(x, self.m, self.k, self.a, self.b)     # find the hash1 position
        else:       
            bits = hash2(x, self.m, self.seeds)                 # find the hash2 position

        # checks if every bit of x in the bitarray is set to 1 (meaning its in the maintained set)
        return all(self.bitarray[bit] == 1 for bit in bits)


# calculates false positive rate for a given subset size n, universe size N, c value, k value, and hash type
def false_positve_rate(n, N, c, k, hash_type):
    bf = BloomFilter(n, c, k, hash_type)                        # create bloom filter
    m = c * n                                                   # hash table size

    universe_sample = random.sample(range(N), n)                # universe of size N = |U|
    inserted_set = set(universe_sample)                         # used to exclude true positives in test

    for x in universe_sample:                                   # insert elements
        bf.add(x)   

    false_pos = 0                                               # track number of false positives
    tests_done = 0                                              # used to find false positives rate

    while tests_done < n:                                       # runs n tests
        x = random.randrange(N)                                 # pick a random sample from N
        if x in inserted_set:                       
            continue                                            # we don't care about true positives, only false
        tests_done += 1                     
        if bf.contains(x):                                      # if the bloom filter says it's there even 
            false_pos += 1                      

    return false_pos / tests_done                               # ratio of false positives tests to total tests


# plots median value from 10 test runs for of each k-value for a given c value
def plot_fpr(n, N, c, k_values, tests, plot):
    m = c * n                                                   # size of hash table
    hash1_fprs = []                                             # array of false positive rates for each c value
    hash2_fprs = []                                             # array of false positive rates for each c value

    for k in range(k_values):                               
        hash1_test_results = []                                             # store each test run result from hash1 and hash2 separately
        hash2_test_results = []
        for test in range(tests):
            hash1_test_results.append(false_positve_rate(n, N, c, k, 1))    # run false positive rate test for hash1 and hash2
            hash2_test_results.append(false_positve_rate(n, N, c, k, 2))
        hash1_test_median = sorted(hash1_test_results)[tests // 2]          # find the median of the test results for hash1 and hash2
        hash2_test_median = sorted(hash2_test_results)[tests // 2]
        hash1_fprs.append(hash1_test_median)                                # use the median of the tests as the fpr for that c and k value pair for both hash functions
        hash2_fprs.append(hash2_test_median)                               


    ideal_rate = [(1 - math.exp(-k/c))**k for k in range(1, k_values + 1)]                          # ideal false positive rate formula

    # plotting figure, start from 2 because at 1 the fpr jumps very high    
    plot.plot(range(2, k_values + 1), hash1_fprs[1:], marker="x", label=f"c={c}, hash1")            # plots hash1 simulated fpr
    plot.plot(range(2, k_values + 1), hash2_fprs[1:], marker="x", label=f"c={c}, hash2")            # plots hash2 simulated fpr
    plot.plot(range(2, k_values + 1), ideal_rate[1:], marker="x", label=f"c_ideal={c}")             # plots ideal fpr

    hash1_min_idx = hash1_fprs[1:].index(min(hash1_fprs[1:])) + 2                                   # find min indices and values
    hash1_min_val = min(hash1_fprs[1:])                                                             # +2 is to match x-axis (k=2 start)

    hash2_min_idx = hash2_fprs[1:].index(min(hash2_fprs[1:])) + 2                                   # repeat for hash2
    hash2_min_val = min(hash2_fprs[1:])

    ideal_min_idx = ideal_rate[1:].index(min(ideal_rate[1:])) + 2                                   # repeat for ideal              
    ideal_min_val = min(ideal_rate[1:])

    plot.scatter(hash1_min_idx, hash1_min_val, color='blue', s=75, zorder=5, label='min hash1')     # plot min points
    plot.scatter(hash2_min_idx, hash2_min_val, color='orange', s=75, zorder=5, label='min hash2')
    plot.scatter(ideal_min_idx, ideal_min_val, color='green', s=75, zorder=5, label='min ideal')

    plot.set_xlabel("k (# of Hash Functions)")                                                      # x-axis label
    plot.set_ylabel("False Positve Rates")                                                          # y-axis label
    plot.set_title(f"False Positive Rate vs k (c={c})")                                             # title
    plot.grid(True)                                                                                 # enable grid lines


# parameters for plotting all the k and c value pairings
k_values = 15                      # number of hash functions we will test for each hash type and c_value
n = 10000                          # size of subset S
tests = 10                         # number of tests ran for each pair of k_value and c_value
c_values = [5, 7, 9, 11]           # different c values to test
hash_types = [1, 2]                # different hash functions, hash1 and hash2 implemented in part 1

# create subplots: one for each c value, both hash types on same plot
fig, axes = plt.subplots(2, 2, figsize=(14, 5))
axes = axes.ravel()
(plot1, plot2, plot3, plot4) = axes

for c in c_values:                 # run plot_fpr for each c value
    print(f"Plot for c={c}")
    if c == 5:
        plot = plot1
    elif c == 7:
        plot = plot2
    elif c == 9:
        plot = plot3
    else:
        plot = plot4
    plot_fpr(n, N, c, k_values, tests, plot)

# add legends to both subplots
plot1.legend()
plot2.legend()
plot3.legend()
plot4.legend()

# adjust layout and show
plt.tight_layout()
plt.show()

