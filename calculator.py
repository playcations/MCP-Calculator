# calculator.py
import math
import cmath  # For complex number operations
import numpy as np  # For linear algebra operations
import scipy.linalg  # For advanced linear algebra operations
from typing import List, Tuple, Dict, Union, Optional

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Calculator")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
@mcp.tool()
def power(a: int, b: int) -> int:
    """Raise a number to the power of another"""
    return a ** b
@mcp.tool()
def factorial(n: int) -> int:
    """Calculate the factorial of a number"""
    if n < 0:
        raise ValueError("Cannot calculate factorial of a negative number")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
@mcp.tool()
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number"""
    if n < 0:
        raise ValueError("Cannot calculate Fibonacci of a negative number")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
@mcp.tool()
def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two numbers"""
    while b:
        a, b = b, a % b
    return a
@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of two numbers"""
    return abs(a * b) // gcd(a, b)
@mcp.tool()
def square_root(n: int) -> float:
    """Calculate the square root of a number"""
    if n < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return n ** 0.5
@mcp.tool()
def logarithm(n: int, base: int) -> float:
    """Calculate the logarithm of a number with a given base"""
    if n <= 0 or base <= 1:
        raise ValueError("Invalid input for logarithm")
    return math.log(n, base)
@mcp.tool()
def exponential(n: int) -> float:
    """Calculate the exponential of a number"""
    return math.exp(n)
@mcp.tool()
def sine(n: int) -> float:
    """Calculate the sine of an angle in radians"""
    return math.sin(n)
@mcp.tool()
def cosine(n: int) -> float:
    """Calculate the cosine of an angle in radians"""
    return math.cos(n)
@mcp.tool()
def tangent(n: int) -> float:
    """Calculate the tangent of an angle in radians"""
    return math.tan(n)
@mcp.tool()
def absolute(n: int) -> int:
    """Calculate the absolute value of a number"""
    return abs(n)
@mcp.tool()
def round_number(n: float, digits: int) -> float:
    """Round a number to a given number of decimal places"""
    return round(n, digits)
@mcp.tool()
def ceil(n: float) -> int:
    """Calculate the ceiling of a number"""
    return math.ceil(n)
@mcp.tool()
def floor(n: float) -> int:
    """Calculate the floor of a number"""
    return math.floor(n)
@mcp.tool()
def factorial_iterative(n: int) -> int:
    """Calculate the factorial of a number iteratively"""
    if n < 0:
        raise ValueError("Cannot calculate factorial of a negative number")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
@mcp.tool()
def power_iterative(base: int, exponent: int) -> int:
    """Raise a number to the power of another iteratively"""
    result = 1
    for _ in range(exponent):
        result *= base
    return result
@mcp.tool()
def fibonacci_iterative(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively"""
    if n < 0:
        raise ValueError("Cannot calculate Fibonacci of a negative number")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
@mcp.tool()
def is_prime_iterative(n: int) -> bool:
    """Check if a number is prime iteratively"""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
@mcp.tool()
def gcd_iterative(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two numbers iteratively"""
    while b:
        a, b = b, a % b
    return a
@mcp.tool()
def lcm_iterative(a: int, b: int) -> int:
    """Calculate the least common multiple of two numbers iteratively"""
    return abs(a * b) // gcd_iterative(a, b)
@mcp.tool()
def square_root_iterative(n: int) -> float:
    """Calculate the square root of a number iteratively"""
    if n < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    x = n
    y = (x + n / x) / 2
    while abs(x - y) >= 0.0001:
        x = y
        y = (x + n / x) / 2
    return x
@mcp.tool()
def logarithm_iterative(n: int, base: int) -> float:
    """Calculate the logarithm of a number with a given base iteratively"""
    if n <= 0 or base <= 1:
        raise ValueError("Invalid input for logarithm")
    result = 0
    while n > 1:
        n /= base
        result += 1
    return result
@mcp.tool()
def exponential_iterative(n: int) -> float:
    """Calculate the exponential of a number iteratively"""
    result = 1
    term = 1
    for i in range(1, 100):
        term *= n / i
        result += term
    return result
@mcp.tool()
def sine_iterative(n: int) -> float:
    """Calculate the sine of an angle in radians iteratively"""
    result = 0
    term = n
    for i in range(1, 100):
        result += term
        term *= -n * n / ((2 * i) * (2 * i + 1))
    return result
@mcp.tool()
def cosine_iterative(n: int) -> float:
    """Calculate the cosine of an angle in radians iteratively"""
    result = 0
    term = 1
    for i in range(1, 100):
        result += term
        term *= -n * n / ((2 * i - 1) * (2 * i))
    return result
@mcp.tool()
def tangent_iterative(n: int) -> float:
    """Calculate the tangent of an angle in radians iteratively"""
    return sine_iterative(n) / cosine_iterative(n)
@mcp.tool()
def absolute_iterative(n: int) -> int:
    """Calculate the absolute value of a number iteratively"""
    return n if n >= 0 else -n
@mcp.tool()
def round_number_iterative(n: float, digits: int) -> float:
    """Round a number to a given number of decimal places iteratively"""
    factor = 10 ** digits
    return int(n * factor + 0.5) / factor
@mcp.tool()
def ceil_iterative(n: float) -> int:
    """Calculate the ceiling of a number iteratively"""
    return int(n) + (1 if n % 1 > 0 else 0)
@mcp.tool()
def floor_iterative(n: float) -> int:
    """Calculate the floor of a number iteratively"""
    return int(n) - (1 if n % 1 < 0 else 0)


# Statistics Functions
@mcp.tool()
def mean(numbers: List[float]) -> float:
    """Calculate the mean (average) of a list of numbers"""
    if not numbers:
        raise ValueError("Cannot calculate mean of an empty list")
    return sum(numbers) / len(numbers)

@mcp.tool()
def median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers"""
    if not numbers:
        raise ValueError("Cannot calculate median of an empty list")
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]

@mcp.tool()
def mode(numbers: List[float]) -> List[float]:
    """Calculate the mode(s) of a list of numbers"""
    if not numbers:
        raise ValueError("Cannot calculate mode of an empty list")
    counts = {}
    for num in numbers:
        counts[num] = counts.get(num, 0) + 1
    max_count = max(counts.values())
    return [num for num, count in counts.items() if count == max_count]

@mcp.tool()
def variance(numbers: List[float]) -> float:
    """Calculate the variance of a list of numbers"""
    if len(numbers) < 2:
        raise ValueError("Cannot calculate variance with less than 2 numbers")
    avg = mean(numbers)
    return sum((x - avg) ** 2 for x in numbers) / len(numbers)

@mcp.tool()
def std_deviation(numbers: List[float]) -> float:
    """Calculate the standard deviation of a list of numbers"""
    return math.sqrt(variance(numbers))

# Vector Operations
@mcp.tool()
def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """Add two vectors"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimensions")
    return [v1[i] + v2[i] for i in range(len(v1))]

@mcp.tool()
def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """Subtract second vector from first vector"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimensions")
    return [v1[i] - v2[i] for i in range(len(v1))]

@mcp.tool()
def vector_scalar_multiply(v: List[float], scalar: float) -> List[float]:
    """Multiply a vector by a scalar"""
    return [scalar * x for x in v]

@mcp.tool()
def vector_dot_product(v1: List[float], v2: List[float]) -> float:
    """Calculate the dot product of two vectors"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimensions")
    return sum(v1[i] * v2[i] for i in range(len(v1)))

@mcp.tool()
def vector_cross_product(v1: List[float], v2: List[float]) -> List[float]:
    """Calculate the cross product of two 3D vectors"""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product is only defined for 3D vectors")
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

@mcp.tool()
def vector_magnitude(v: List[float]) -> float:
    """Calculate the magnitude (norm) of a vector"""
    return math.sqrt(sum(x * x for x in v))

@mcp.tool()
def vector_normalize(v: List[float]) -> List[float]:
    """Normalize a vector to unit length"""
    mag = vector_magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize a zero vector")
    return [x / mag for x in v]

@mcp.tool()
def vector_projection(v1: List[float], v2: List[float]) -> List[float]:
    """Project vector v1 onto vector v2"""
    dot_product = vector_dot_product(v1, v2)
    v2_mag_squared = sum(x * x for x in v2)
    if v2_mag_squared == 0:
        raise ValueError("Cannot project onto a zero vector")
    scalar = dot_product / v2_mag_squared
    return [scalar * x for x in v2]

@mcp.tool()
def vector_angle(v1: List[float], v2: List[float]) -> float:
    """Calculate the angle between two vectors in radians"""
    dot_product = vector_dot_product(v1, v2)
    mag_v1 = vector_magnitude(v1)
    mag_v2 = vector_magnitude(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        raise ValueError("Cannot calculate angle with a zero vector")
    # Ensure the value is within the valid range for arccos
    cos_angle = max(min(dot_product / (mag_v1 * mag_v2), 1.0), -1.0)
    return math.acos(cos_angle)

@mcp.tool()
def vector_distance(v1: List[float], v2: List[float]) -> float:
    """Calculate the Euclidean distance between two vectors"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimensions")
    return math.sqrt(sum((v1[i] - v2[i])**2 for i in range(len(v1))))

@mcp.tool()
def vector_outer_product(v1: List[float], v2: List[float]) -> List[List[float]]:
    """Calculate the outer product of two vectors"""
    result = []
    for i in range(len(v1)):
        row = []
        for j in range(len(v2)):
            row.append(v1[i] * v2[j])
        result.append(row)
    return result

@mcp.tool()
def vector_triple_product(a: List[float], b: List[float], c: List[float]) -> List[float]:
    """Calculate the vector triple product a × (b × c)"""
    if len(a) != 3 or len(b) != 3 or len(c) != 3:
        raise ValueError("Triple product is only defined for 3D vectors")
    # Calculate b × c first
    bc_cross = vector_cross_product(b, c)
    # Then calculate a × (b × c)
    return vector_cross_product(a, bc_cross)

@mcp.tool()
def vector_scalar_triple_product(a: List[float], b: List[float], c: List[float]) -> float:
    """Calculate the scalar triple product a · (b × c)"""
    if len(a) != 3 or len(b) != 3 or len(c) != 3:
        raise ValueError("Triple product is only defined for 3D vectors")
    # Calculate b × c first
    bc_cross = vector_cross_product(b, c)
    # Then calculate a · (b × c)
    return vector_dot_product(a, bc_cross)

@mcp.tool()
def vector_projection_matrix(v: List[float]) -> List[List[float]]:
    """Calculate the projection matrix for a unit vector"""
    # Normalize the vector
    unit_v = vector_normalize(v)
    # Calculate the outer product
    return vector_outer_product(unit_v, unit_v)

@mcp.tool()
def vector_reflection(v: List[float], normal: List[float]) -> List[float]:
    """Reflect a vector across a normal vector"""
    # Normalize the normal vector
    unit_normal = vector_normalize(normal)
    # Calculate the dot product
    dot = vector_dot_product(v, unit_normal)
    # Calculate the reflection
    return [v[i] - 2 * dot * unit_normal[i] for i in range(len(v))]

@mcp.tool()
def vector_rejection(v1: List[float], v2: List[float]) -> List[float]:
    """Calculate the rejection of v1 from v2 (orthogonal component)"""
    # Calculate the projection
    proj = vector_projection(v1, v2)
    # Subtract from original vector
    return [v1[i] - proj[i] for i in range(len(v1))]

# Linear Algebra Functions
@mcp.tool()
def matrix_add(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """Add two matrices"""
    if not matrix_a or not matrix_b:
        raise ValueError("Cannot add empty matrices")
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices must have the same dimensions")

    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        result.append(row)
    return result

@mcp.tool()
def matrix_subtract(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """Subtract second matrix from first matrix"""
    if not matrix_a or not matrix_b:
        raise ValueError("Cannot subtract empty matrices")
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices must have the same dimensions")

    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] - matrix_b[i][j])
        result.append(row)
    return result

@mcp.tool()
def matrix_scalar_multiply(matrix: List[List[float]], scalar: float) -> List[List[float]]:
    """Multiply a matrix by a scalar"""
    if not matrix:
        raise ValueError("Cannot multiply an empty matrix")

    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

@mcp.tool()
def matrix_multiply(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices"""
    if not matrix_a or not matrix_b:
        raise ValueError("Cannot multiply empty matrices")
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")

    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_b[0])):
            element = 0
            for k in range(len(matrix_b)):
                element += matrix_a[i][k] * matrix_b[k][j]
            row.append(element)
        result.append(row)
    return result

@mcp.tool()
def matrix_determinant(matrix: List[List[float]]) -> float:
    """Calculate the determinant of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate determinant of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    return float(np.linalg.det(np_matrix))

@mcp.tool()
def matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """Calculate the inverse of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate inverse of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        inverse = np.linalg.inv(np_matrix)
        return inverse.tolist()
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted")

@mcp.tool()
def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Calculate the transpose of a matrix"""
    if not matrix:
        raise ValueError("Cannot calculate transpose of an empty matrix")

    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result

@mcp.tool()
def matrix_trace(matrix: List[List[float]]) -> float:
    """Calculate the trace of a square matrix (sum of diagonal elements)"""
    if not matrix:
        raise ValueError("Cannot calculate trace of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    return sum(matrix[i][i] for i in range(n))

@mcp.tool()
def matrix_rank(matrix: List[List[float]]) -> int:
    """Calculate the rank of a matrix"""
    if not matrix:
        raise ValueError("Cannot calculate rank of an empty matrix")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    return int(np.linalg.matrix_rank(np_matrix))

@mcp.tool()
def matrix_eigenvalues(matrix: List[List[float]]) -> List[complex]:
    """Calculate the eigenvalues of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate eigenvalues of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    eigenvalues = np.linalg.eigvals(np_matrix)

    # Convert complex eigenvalues to tuples of (real, imag)
    result = []
    for val in eigenvalues:
        if val.imag == 0:
            result.append(float(val.real))
        else:
            result.append(complex(val.real, val.imag))

    return result

@mcp.tool()
def matrix_eigenvectors(matrix: List[List[float]]) -> Tuple[List[complex], List[List[complex]]]:
    """Calculate the eigenvalues and eigenvectors of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate eigenvectors of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(np_matrix)

    # Convert eigenvalues to a list
    eigenvalues_list = []
    for val in eigenvalues:
        if val.imag == 0:
            eigenvalues_list.append(float(val.real))
        else:
            eigenvalues_list.append(complex(val.real, val.imag))

    # Convert eigenvectors to a list of lists
    eigenvectors_list = []
    for i in range(n):
        vector = []
        for j in range(n):
            val = eigenvectors[j, i]
            if val.imag == 0:
                vector.append(float(val.real))
            else:
                vector.append(complex(val.real, val.imag))
        eigenvectors_list.append(vector)

    return (eigenvalues_list, eigenvectors_list)

@mcp.tool()
def matrix_lu_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Calculate the LU decomposition of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate LU decomposition of an empty matrix")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        p, l, u = scipy.linalg.lu(np_matrix)
        return p.tolist(), l.tolist(), u.tolist()
    except Exception as e:
        raise ValueError(f"Error in LU decomposition: {str(e)}")

@mcp.tool()
def matrix_qr_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Calculate the QR decomposition of a matrix"""
    if not matrix:
        raise ValueError("Cannot calculate QR decomposition of an empty matrix")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        q, r = np.linalg.qr(np_matrix)
        return q.tolist(), r.tolist()
    except Exception as e:
        raise ValueError(f"Error in QR decomposition: {str(e)}")

@mcp.tool()
def matrix_svd(matrix: List[List[float]]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """Calculate the Singular Value Decomposition (SVD) of a matrix"""
    if not matrix:
        raise ValueError("Cannot calculate SVD of an empty matrix")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        u, s, vh = np.linalg.svd(np_matrix)
        return u.tolist(), s.tolist(), vh.tolist()
    except Exception as e:
        raise ValueError(f"Error in SVD: {str(e)}")

@mcp.tool()
def matrix_condition_number(matrix: List[List[float]]) -> float:
    """Calculate the condition number of a matrix"""
    if not matrix:
        raise ValueError("Cannot calculate condition number of an empty matrix")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        return float(np.linalg.cond(np_matrix))
    except Exception as e:
        raise ValueError(f"Error calculating condition number: {str(e)}")

@mcp.tool()
def matrix_power(matrix: List[List[float]], n: int) -> List[List[float]]:
    """Calculate the power of a square matrix"""
    if not matrix:
        raise ValueError("Cannot calculate power of an empty matrix")
    rows = len(matrix)
    if rows != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Convert to numpy array for calculation
    np_matrix = np.array(matrix)
    try:
        result = np.linalg.matrix_power(np_matrix, n)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"Error calculating matrix power: {str(e)}")

@mcp.tool()
def solve_linear_system(coefficient_matrix: List[List[float]], constant_vector: List[float]) -> List[float]:
    """Solve a system of linear equations Ax = b"""
    if not coefficient_matrix or not constant_vector:
        raise ValueError("Cannot solve with empty inputs")
    if len(coefficient_matrix) != len(constant_vector):
        raise ValueError("Number of equations must match number of constants")

    # Convert to numpy arrays for calculation
    A = np.array(coefficient_matrix)
    b = np.array(constant_vector)
    try:
        x = np.linalg.solve(A, b)
        return x.tolist()
    except np.linalg.LinAlgError:
        raise ValueError("The system does not have a unique solution")

@mcp.tool()
def least_squares_solution(coefficient_matrix: List[List[float]], constant_vector: List[float]) -> List[float]:
    """Find the least squares solution to an overdetermined system Ax = b"""
    if not coefficient_matrix or not constant_vector:
        raise ValueError("Cannot solve with empty inputs")
    if len(coefficient_matrix) < len(coefficient_matrix[0]):
        raise ValueError("System must be overdetermined (more equations than unknowns)")

    # Convert to numpy arrays for calculation
    A = np.array(coefficient_matrix)
    b = np.array(constant_vector)
    try:
        # Ignore the other return values (residuals, rank, singular values)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x.tolist()
    except Exception as e:
        raise ValueError(f"Error finding least squares solution: {str(e)}")

# Calculus Functions
@mcp.tool()
def numerical_derivative(f: str, x: float, h: float = 0.0001) -> float:
    """Calculate the numerical derivative of a function at a point"""
    # We'll use the central difference formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    # The function is passed as a string and evaluated
    def evaluate(expr, val):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        return eval(expr.replace('x', str(val)))

    return (evaluate(f, x + h) - evaluate(f, x - h)) / (2 * h)

@mcp.tool()
def numerical_integration(f: str, a: float, b: float, n: int = 1000) -> float:
    """Calculate the numerical integration of a function over an interval using the trapezoidal rule"""
    if a > b:
        a, b = b, a

    def evaluate(expr, val):
        return eval(expr.replace('x', str(val)))

    h = (b - a) / n
    result = 0.5 * (evaluate(f, a) + evaluate(f, b))

    for i in range(1, n):
        result += evaluate(f, a + i * h)

    return result * h

@mcp.tool()
def taylor_series(f: str, x0: float, x: float, terms: int = 5) -> float:
    """Approximate a function using Taylor series expansion"""
    # This is a simplified version that works for basic functions
    # For a real implementation, symbolic differentiation would be better
    result = 0
    h = 0.0001

    def evaluate(expr, val):
        return eval(expr.replace('x', str(val)))

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    # 0th term (function value)
    result += evaluate(f, x0)

    # Higher order terms
    for n in range(1, terms):
        # Approximate nth derivative using finite differences
        # This is a very simplified approach
        derivative = 0
        if n == 1:
            derivative = (evaluate(f, x0 + h) - evaluate(f, x0 - h)) / (2 * h)
        elif n == 2:
            derivative = (evaluate(f, x0 + h) - 2 * evaluate(f, x0) + evaluate(f, x0 - h)) / (h * h)
        else:
            # For higher derivatives, we'd need more sophisticated methods
            # This is just a placeholder
            derivative = 0

        result += derivative * ((x - x0) ** n) / factorial(n)

    return result

# Complex Number Operations
@mcp.tool()
def complex_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Add two complex numbers represented as tuples (real, imaginary)"""
    return (a[0] + b[0], a[1] + b[1])

@mcp.tool()
def complex_multiply(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Multiply two complex numbers represented as tuples (real, imaginary)"""
    real = a[0] * b[0] - a[1] * b[1]
    imag = a[0] * b[1] + a[1] * b[0]
    return (real, imag)

@mcp.tool()
def complex_divide(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Divide two complex numbers represented as tuples (real, imaginary)"""
    denominator = b[0]**2 + b[1]**2
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    real = (a[0] * b[0] + a[1] * b[1]) / denominator
    imag = (a[1] * b[0] - a[0] * b[1]) / denominator
    return (real, imag)

@mcp.tool()
def complex_modulus(a: Tuple[float, float]) -> float:
    """Calculate the modulus (absolute value) of a complex number"""
    return math.sqrt(a[0]**2 + a[1]**2)

@mcp.tool()
def complex_argument(a: Tuple[float, float]) -> float:
    """Calculate the argument (phase) of a complex number in radians"""
    return math.atan2(a[1], a[0])

# Number Theory Functions
@mcp.tool()
def prime_factorization(n: int) -> Dict[int, int]:
    """Find the prime factorization of a number"""
    if n <= 1:
        raise ValueError("Number must be greater than 1")

    factors = {}
    # Check for divisibility by 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    # Check for divisibility by odd numbers starting from 3
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2

    # If n is a prime number greater than 2
    if n > 2:
        factors[n] = factors.get(n, 0) + 1

    return factors

@mcp.tool()
def euler_totient(n: int) -> int:
    """Calculate Euler's totient function φ(n)"""
    if n <= 0:
        raise ValueError("Number must be positive")

    result = n  # Initialize result as n

    # Consider all prime factors of n and subtract their multiples from result
    p = 2
    while p * p <= n:
        # Check if p is a prime factor
        if n % p == 0:
            # If yes, then update n and result
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    # If n has a prime factor greater than sqrt(n)
    if n > 1:
        result -= result // n

    return result

@mcp.tool()
def modular_exponentiation(base: int, exponent: int, modulus: int) -> int:
    """Calculate (base^exponent) % modulus efficiently"""
    if modulus == 1:
        return 0

    result = 1
    base = base % modulus

    while exponent > 0:
        # If exponent is odd, multiply base with result
        if exponent % 2 == 1:
            result = (result * base) % modulus

        # Exponent must be even now
        exponent = exponent >> 1  # Divide by 2
        base = (base * base) % modulus

    return result

@mcp.tool()
def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Calculate the extended Euclidean algorithm results"""
    if a == 0:
        return (b, 0, 1)

    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return (gcd, x, y)
