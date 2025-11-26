from bitarray import bitarray
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------
# Part 1: Parameter Setup
# ------------------------------------------------------------------------------------------------------

p = (2**31 - 1)             # mersenne prime >= N
N = 10**9                   # size of universe , |U|
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
uncomment this section to run part 1 tests and plots

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
def plot_fpr(n, N, c, k_optimal, tests, plot):
    m = c * n                                                                           # size of hash table
    hash_types = [1, 2]                                                                 # hash types to test
    fprs = {}                                                                           # dictionary to store fpr results
    k_min = max(1, k_optimal - 5)                                                       # minimum k value to test, don't go below 1 beacause fpr jumps very high
    k_range = list(range(k_min, k_optimal + 6))                                         # +/- 5 around optimal k

    for hash in hash_types:                                                     
        fprs[hash] = []                                                                 # initialize empty list for each hash type
        for k in k_range:                                                               # test k values +/- 5 from optimal             
            test_reults = [false_positve_rate(n, N, c, k, hash) for _ in range(tests)]  # run tests for each k value
            median_fpr = sorted(test_reults)[tests // 2]                                # find median
            fprs[hash].append(median_fpr)                                               # store median fpr for that k value

    ideal_rate = [(1 - math.exp(-k/c))**k for k in k_range]                             # ideal false positive rate formula

    for hash, color in zip(hash_types, ['blue', 'orange']):                                                                                 # plot for each hash type with different colors                           
        plot.plot(k_range, fprs[hash], color=color, label=f"hash{hash} Functions")                                                          # plots simulated fpr
        min_idx = fprs[hash].index(min(fprs[hash]))                                                                                         # find min indices and values
        plot.scatter(k_range[min_idx], min(fprs[hash]), color=color, s=100, zorder=2, label=f'k={k_range[min_idx]} hash{hash} Ideal K')     # plot min fpr points

    plot.plot(k_range, ideal_rate, color='green', label="Theoretical Function")                                                             # repeat stepes for ideal fpr
    ideal_min_idx = ideal_rate.index(min(ideal_rate))                                                                                                                                            
    plot.scatter(k_range[ideal_min_idx], min(ideal_rate), color='green', s=100, zorder=2, label=f'k={k_range[ideal_min_idx]} Theoretical Ideal K')

    plot.set_xlabel("k (# of Hash Functions)")                                          # x-axis label
    plot.set_ylabel("False Positve Rates")                                              # y-axis label
    plot.set_title(f"False Positive Rate vs k (c={c})")                                 # title
    plot.grid(True)                                                                     # enable grid lines


# parameters for plotting all the k and c value pairings
n = 10000                                           # size of subset S
tests = 10                                          # number of tests ran for each pair of k_value and c_value
c_values = [5, 7, 9, 11]                            # different c values to test
hash_types = [1, 2]                                 # different hash functions, hash1 and hash2 implemented in part 1
k_values = []
for c in c_values:
    k_optimal = math.ceil(c * math.log(2))          # optimal k value for each c value
    k_values.append(k_optimal)                      # store optimal k values for each c value

fig, axes = plt.subplots(2, 2, figsize=(14, 5))     # create subplots: one for each c value, both hash types on same plot
axes = axes.ravel()                                 # flatten the array of plots for easy looping
(plot1, plot2, plot3, plot4) = axes
plots = [plot1, plot2, plot3, plot4]

for i, c in enumerate(c_values):                    # run plot_fpr for each c value
    print(f"Plot for c={c}")
    plot_fpr(n, N, c, k_values[i], tests, plots[i])

plot1.legend()                                      # add legends to both subplots
plot2.legend()
plot3.legend()
plot4.legend()                                  

plt.tight_layout()                                  # adjust layout and show plots
plt.show()

