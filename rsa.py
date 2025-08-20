import random
import sys

# This may come in handy...
from fermat import miller_rabin

# If you use a recursive implementation of `mod_exp` or extended-euclid,
# you recurse once for every bit in the number.
# If your number is more than 1000 bits, you'll exceed python's recursion limit.
# Here we raise the limit so the tests can run without any issue.
# Can you implement `mod_exp` and extended-euclid without recursion?
sys.setrecursionlimit(4000)

# When trying to find a relatively prime e for (p-1) * (q-1)
# use this list of 25 primes
# If none of these work, throw an exception (and let the instructors know!)
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


# Implement this function
def ext_euclid(a: int, b: int) -> tuple[int, int, int]:
    """
    The Extended Euclid algorithm
    Returns x, y , d such that:
    - d = GCD(a, b)
    - ax + by = d

    Note: a must be greater than b
    """
    if b == 0:
        return a, 1, 0 # return gcd, x, y   
    else:
        (gcd, x, y) = ext_euclid(b, a % b)
        return gcd, y, x - (a // b) * y

    #return 0, 0, 0
   

# Implement this function
def generate_large_prime(bits=512) -> int:
    """
    Generate a random prime number with the specified bit length.
    Use random.getrandbits(bits) to generate a random number of the
     specified bit length.
    """
    while True:
        Prime = random.getrandbits(bits)
        Prime |= (1 << bits - 1) | 1  
        if miller_rabin(Prime, k = 20) == 'prime':
            return Prime
    #return 5  # Guaranteed random prime number obtained through fair dice roll


# Implement this function
def generate_key_pairs(bits: int) -> tuple[int, int, int]:
    """
    Generate RSA public and private key pairs.
    Return N, e, d
    - N must be the product of two random prime numbers p and q
    - e and d must be multiplicative inverses mod (p-1)(q-1)
    """
    p = generate_large_prime(bits // 2)
    q = generate_large_prime(bits // 2)
    N = p * q
    a = (p - 1) * (q - 1)

    for e in primes:
        gcd, x, y = ext_euclid(e, a)
        if gcd == 1:
            d = x % a
            if d < 0:
                d += a
            return N, e, d

    #return 0, 0, 0
