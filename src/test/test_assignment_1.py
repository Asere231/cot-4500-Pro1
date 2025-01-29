# src/test/test_assignment_1.py

import pytest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.assignment_1 import (
    approximation_method,
    bisection_method,
    fixed_point_iteration,
    newton_raphson
)

# Test functions
def f(x): return x**2 - 4  # Root at x = ±2
def f_prime(x): return 2*x
def g(x): return x - (x**2 - 4)/(2*x)  # Fixed-point form

def test_approximation_method():
    """Test the approximation method with f(x) = x² - 4"""
    root, iterations = approximation_method(f, 0, 3)
    assert abs(root - 2.0) < 1e-6
    assert iterations < 100

def test_approximation_method_negative_root():
    """Test the approximation method finding negative root"""
    root, iterations = approximation_method(f, -3, 0)
    assert abs(root + 2.0) < 1e-6
    assert iterations < 100

def test_bisection_method():
    """Test the bisection method with f(x) = x² - 4"""
    root, iterations = bisection_method(f, 0, 3)
    assert abs(root - 2.0) < 1e-6
    assert iterations < 100

def test_fixed_point_iteration():
    """Test the fixed-point iteration with g(x) = x - (x² - 4)/(2x)"""
    root, iterations = fixed_point_iteration(g, 1)
    assert abs(root - 2.0) < 1e-6
    assert iterations < 100

def test_newton_raphson():
    """Test the Newton-Raphson method with f(x) = x² - 4"""
    root, iterations = newton_raphson(f, f_prime, 1)
    assert abs(root - 2.0) < 1e-6
    assert iterations < 100

def test_invalid_interval():
    """Test that methods raise ValueError for invalid intervals"""
    with pytest.raises(ValueError):
        approximation_method(f, 1, 1.5)  # Both positive, no root in between
    with pytest.raises(ValueError):
        bisection_method(f, -5, -3)  # Both negative, no root in between

def test_newton_raphson_zero_derivative():
    """Test Newton-Raphson handles zero derivative case"""
    def f_zero_deriv(x): return x**3
    def f_zero_deriv_prime(x): return 3*x**2
    with pytest.raises(ValueError):
        newton_raphson(f_zero_deriv, f_zero_deriv_prime, 0)

def test_convergence_tolerance():
    """Test that methods respect the tolerance parameter"""
    tol = 1e-4
    root, _ = bisection_method(f, 0, 3, tol=tol)
    # Test the actual convergence criterion used by bisection method
    assert abs(root - 2.0) < tol  # Check distance to known root

def test_max_iterations():
    """Test that methods respect the max_iterations parameter"""
    max_iter = 5
    _, iterations = fixed_point_iteration(g, 1, max_iter=max_iter)
    assert iterations <= max_iter

def test_exact_root():
    """Test behavior when exact root is found"""
    def f_exact(x): return x - 2  # Root exactly at x = 2
    root, iterations = newton_raphson(lambda x: x-2, lambda x: 1, 2)
    assert root == 2.0
    assert iterations == 0