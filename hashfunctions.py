import random
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------
# Parameter Setup
# ------------------------------------------------------------------------------------------------------

N = 10**6           # |U| = size of universe
m = 997             # |N| = size of hash table
p = (2**31 - 1)     # Mersenne prime >= N
k = 3               # number of hash functions

rng = random.Random()

'''
a = []
while len(a) < k:
    candidate = rng.randrange(1, p)
    if math.gcd(candidate, m) == 1:
        a.append(candidate)
'''

a = [rng.randrange(1, p) for _ in range(k)]
b = [rng.randrange(0, p) for _ in range(k)]


# ------------------------------------------------------------------------------------------------------
# Hash Functions
# ------------------------------------------------------------------------------------------------------

# h1(x) = ((a * x + b) mod p) mod m
def hash1(x, m):
    x = int(hash(x))
    return [((a[i] * x + b[i]) % p) % m for i in range(k)]
    
# Seeds for seeded hash function
seeds = [rng.randrange(10**9) for _ in range(k)]

# h2(x) = r(s[i] + x)
def hash2(x, m):
    x = int(hash(x))
    hash2_results = []
    for s in seeds:
        r = random.Random(s + x)                        # add randomized seed to input
        hash2_results.append(r.randrange(m))            # placed into random bucket from 0 to m - 1
    return hash2_results    

# creates the hashtable, uses count because we care about maximum load
def hash_table(data, hash_func):
    buckets = [0] * m
    for x in data:
        hashes = hash_func(x, m)
        for hash in hashes:
            buckets[hash] += 1
    return buckets

# ------------------------------------------------------------------------------------------------------
# Testing on Datasets
# ------------------------------------------------------------------------------------------------------


# Define six datasets (data types)
random_data = [rng.randrange(N) for _ in range(10*m)]                   # random data from 0 to N
sequential_data = [n for n in range(10*m)]                              # 0, 1, 2, ..., 10*m
even_data = [2*n for n in range(10*m)]                                  # 0, 2, 4, ..., 20*m
repeated_data = [42] * (10*m)                                           # all same value
few_unique_data = [i % 10 for i in range(10*m)]                         # only 10 unique values repeated
alternating_data = [i if i % 2 == 0 else i+1000 for i in range(10*m)]    # 1, 1002, 3, 1004, ..., 10*m + 1000


def plot_buckets_on_axis(buckets, ax, title):
    ax.bar(range(len(buckets)), buckets)
    ax.set_xlabel("Bucket index")
    ax.set_ylabel("Count")
    ax.set_title(title)


def main():
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
    hash1_buckets_list = [hash_table(data, hash1) for (_name, data) in datasets]
    hash2_buckets_list = [hash_table(data, hash2) for (_name, data) in datasets]

    # Plot Hash1 results: one figure with 6 subplots (3x2)
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 12))
    axes1 = axes1.ravel()
    for i, (name, _data) in enumerate(datasets):
        plot_buckets_on_axis(hash1_buckets_list[i], axes1[i], f"Hash1 - {name}")
    plt.tight_layout()
    plt.show()

    # Plot Hash2 results: one figure with 6 subplots (3x2)
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 12))
    axes2 = axes2.ravel()
    for i, (name, _data) in enumerate(datasets):
        plot_buckets_on_axis(hash2_buckets_list[i], axes2[i], f"Hash2 - {name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()