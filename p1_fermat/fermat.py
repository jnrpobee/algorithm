import argparse
import random

# Explain the time and space complexity of your algorithm by showing and summing up the complexity of each subsection of your code
# Time complexity: O(k) where k is the number of iterations in the loop for both the Fermat and Miller-Rabin tests.  
# Space complexity: O(1) because the space used is constant and does not depend on the input size.  


# This is a convenience function for main(). You don't need to touch it.
def prime_test(N: int, k: int) -> tuple[str, str]:
    return fermat(N, k), miller_rabin(N, k)


# You will need to implement this function and change the return value.
def mod_exp(x: int, y: int, N: int) -> int:
    #return 0
    if y == 0:
        return 1
    # Recursive call to mod_exp function to calculate x^(y//2) mod N. 
    #O(n^2) + O(1)  
    z = mod_exp(x, y // 2, N)  
    # If y is even
    #O(1)
    if y % 2 == 0:
        #O(n^2) + O(n^2) + O(1)
        return z**2 % N
    else:
        # If y is odd
        #O(n^2) + O(n^2) + O(n^2) + O(1)
        return x * z**2 % N



# You will need to implement this function and change the return value.
def fprobability(k: int) -> float:
    return 1 - (1/2)**k


# You will need to implement this function and change the return value.
def mprobability(k: int) -> float:
    return 1 - (1/4)**k


# You will need to implement this function and change the return value, which should be
# either 'prime' or 'composite'.
#
# To generate random values for a, you will most likely want to use
# random.randint(low, hi) which gives a random integer between low and
# hi, inclusive.
def fermat(N: int, k: int) -> str:
    #return "???"
    # If N is even, it is not prime except for 2 itself. the probability of a random number being prime is 1/2. the function runs k times to increase the posibility of N being prime.
    for _ in range(k):
        #in the loop, generate a random number a and check if a^(N-1) mod N != 1, then N is composite.
        a = random.randint(1, N - 1)
        if mod_exp(a, N - 1, N) != 1:
            return 'composite'
    return 'prime'




# You will need to implement this function and change the return value, which should be
# either 'prime' or 'composite'.
#
# To generate random values for a, you will most likely want to use
# random.randint(low, hi) which gives a random integer between low and
# hi, inclusive.
def miller_rabin(N: int, k: int) -> str:
    #return "???"
    # If N is even, it is not prime except for 2 itself 
    for _ in range(k):
        # Generate a random number a
        a = random.randint(1, N-1)
        # If a^(N-1) mod N != 1, then N is composite
        if pow(a, N-1, N) == 1:
            # If N is prime, then N-1 = 2^x * y
            x = N-1
            # If N is prime, then N-1 = 2^x * y
            while pow(a,x,N) == 1 and x % 2 == 0:
                x = x // 2
            if pow(a,x,N) in (N-1,1):
                return 'prime'
            else:
                return 'composite'
        else:
            return 'composite'


def main(number: int, k: int):
    fermat_call, miller_rabin_call = prime_test(number, k)
    fermat_prob = fprobability(k)
    mr_prob = mprobability(k)

    print(f'Is {number} prime?')
    print(f'Fermat: {fermat_call} (prob={fermat_prob})')
    print(f'Miller-Rabin: {miller_rabin_call} (prob={mr_prob})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    main(args.number, args.k)
