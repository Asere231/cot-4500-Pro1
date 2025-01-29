# src/main/assignment_1.py

import numpy as np

def approximation_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Implements the approximation algorithm for finding roots.
    
    Args:
        f (function): The function to find roots for
        a (float): Left endpoint of interval
        b (float): Right endpoint of interval
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    
    Returns:
        float: Approximate root of the function
        int: Number of iterations taken
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    iterations = 0
    while iterations < max_iter:
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c, iterations
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        iterations += 1
    
    return (a + b) / 2, iterations

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Implements the bisection method for finding roots.
    
    Args:
        f (function): The function to find roots for
        a (float): Left endpoint of interval
        b (float): Right endpoint of interval
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    
    Returns:
        float: Approximate root of the function
        int: Number of iterations taken
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    iterations = 0
    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iterations
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        iterations += 1
    
    return (a + b) / 2, iterations

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    Implements the fixed-point iteration method.
    
    Args:
        g (function): The fixed-point function
        x0 (float): Initial guess
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    
    Returns:
        float: Approximate fixed point
        int: Number of iterations taken
    """
    x = x0
    iterations = 0
    
    while iterations < max_iter:
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new, iterations
        x = x_new
        iterations += 1
    
    return x, iterations

def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    Implements the Newton-Raphson method.
    
    Args:
        f (function): The function to find roots for
        f_prime (function): The derivative of f
        x0 (float): Initial guess
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
    
    Returns:
        float: Approximate root of the function
        int: Number of iterations taken
    """
    x = x0
    iterations = 0
    
    while iterations < max_iter:
        if abs(f_prime(x)) < tol:
            raise ValueError("Derivative too close to zero")
        
        x_new = x - f(x) / f_prime(x)
        if abs(x_new - x) < tol:
            return x_new, iterations
        x = x_new
        iterations += 1
    
    return x, iterations

def main():
    # Example usage with test functions
    def f(x): return x**2 - 4  # Example function: x² - 4
    def f_prime(x): return 2*x  # Derivative of x² - 4
    def g(x): return x - (x**2 - 4)/(2*x)  # Fixed-point form of x² - 4
    
    # Test each method
    print("Testing numerical methods to find root of f(x) = x² - 4")
    print("\nApproximation Method:")
    root, iters = approximation_method(f, 0, 3)
    print(f"Root: {root:.6f}, Iterations: {iters}")
    
    print("\nBisection Method:")
    root, iters = bisection_method(f, 0, 3)
    print(f"Root: {root:.6f}, Iterations: {iters}")
    
    print("\nFixed-Point Iteration:")
    root, iters = fixed_point_iteration(g, 1)
    print(f"Root: {root:.6f}, Iterations: {iters}")
    
    print("\nNewton-Raphson Method:")
    root, iters = newton_raphson(f, f_prime, 1)
    print(f"Root: {root:.6f}, Iterations: {iters}")

if __name__ == "__main__":
    main()