# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt

def part_a():
    # ===== Problem 2a =====
    # Define and return Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [0, 2, 2]])
    a = np.array([ 1, 1, 0])
    b = np.array([-1, 2, 5])
    c = np.array([ 0, 2, 3, 2])

    # END YOUR CODE HERE
    return M, a, b, c
    
def part_b(a, b):
    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE
    aDotb = np.dot(np.transpose(a), b)

    # END YOUR CODE HERE
    return aDotb

def part_c(a, b):    
    # ===== Problem 2c =====
    # Find the element-wise product of a and b
    
    aProdb = None
    
    # BEGIN YOUR CODE HERE
    aProdb = np.multiply(a, b)

    # END YOUR CODE HERE
    return aProdb

def part_d(a, b, M):    
    # ===== Problem 2d =====
    # Find (a^T b)Ma
    
    result = None
    
    # BEGIN YOUR CODE HERE
    aDotb = part_b(a, b)
    result = np.dot(aDotb, np.dot(M, a))

    # END YOUR CODE HERE
    return result

def part_e(a, M):
    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function tile() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE
    a_tiled = np.tile(np.transpose(a), (4, 1))
    newM = np.multiply(M, a_tiled)

    # END YOUR CODE HERE
    return newM

def part_f(M):
    # ===== Problem 2f =====
    # Without using a loop, sort all of the values
    # of M in increasing order and plot them.
    
    sortedM = None
    
    # BEGIN YOUR CODE HERE
    sortedM = np.sort(np.reshape(M, -1))
    # END YOUR CODE HERE
    
    plt.bar(range(12), np.squeeze(list(sortedM)))
    plt.savefig('p2f.png')

    return sortedM

if __name__ == '__main__':
    M, a, b, c = part_a()
    aDotb = part_b(a,b)
    print("Problem 2b:%s"%str(aDotb))
    mult = part_c(a,b)
    print("Problem 2c:%s"%str(mult))
    ans = part_d(a,b,M)
    print("Problem 2d:%s"%str(ans))
    newM = part_e(a,M)
    print("Problem 2e:%s"%str(newM))
    part_f(newM)
