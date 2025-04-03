# calculator.py
import math
import cmath  # For complex number operations
import numpy as np  # For linear algebra operations
import scipy.linalg  # For advanced linear algebra operations
import scipy.optimize  # For optimization functions
import scipy.integrate  # For integration functions
import scipy.interpolate  # For interpolation functions
import scipy.special  # For special mathematical functions
from typing import List, Tuple, Dict

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
def complex_subtract(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Subtract second complex number from first complex number"""
    return (a[0] - b[0], a[1] - b[1])

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

@mcp.tool()
def complex_conjugate(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the complex conjugate of a complex number"""
    return (a[0], -a[1])

@mcp.tool()
def complex_exp(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the exponential of a complex number (e^z)"""
    z = complex(a[0], a[1])
    result = cmath.exp(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_log(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the natural logarithm of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.log(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_sqrt(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the square root of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.sqrt(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_sin(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the sine of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.sin(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_cos(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the cosine of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.cos(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_tan(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the tangent of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.tan(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_sinh(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the hyperbolic sine of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.sinh(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_cosh(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the hyperbolic cosine of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.cosh(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_tanh(a: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate the hyperbolic tangent of a complex number"""
    z = complex(a[0], a[1])
    result = cmath.tanh(z)
    return (result.real, result.imag)

@mcp.tool()
def complex_power(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Raise a complex number to a complex power"""
    z1 = complex(a[0], a[1])
    z2 = complex(b[0], b[1])
    result = z1 ** z2
    return (result.real, result.imag)

@mcp.tool()
def complex_rect_to_polar(a: Tuple[float, float]) -> Tuple[float, float]:
    """Convert a complex number from rectangular to polar form (r, theta)"""
    r = complex_modulus(a)
    theta = complex_argument(a)
    return (r, theta)

@mcp.tool()
def complex_polar_to_rect(r: float, theta: float) -> Tuple[float, float]:
    """Convert a complex number from polar form (r, theta) to rectangular form"""
    real = r * math.cos(theta)
    imag = r * math.sin(theta)
    return (real, imag)

# NumPy Array Functions
@mcp.tool()
def create_array(values: List[float]) -> List[float]:
    """Create a NumPy array from a list of values"""
    return np.array(values).tolist()

@mcp.tool()
def create_zeros(shape: int) -> List[float]:
    """Create a NumPy array of zeros with the given shape"""
    return np.zeros(shape).tolist()

@mcp.tool()
def create_ones(shape: int) -> List[float]:
    """Create a NumPy array of ones with the given shape"""
    return np.ones(shape).tolist()

@mcp.tool()
def create_identity(n: int) -> List[List[float]]:
    """Create an identity matrix of size n x n"""
    return np.eye(n).tolist()

@mcp.tool()
def create_linspace(start: float, stop: float, num: int = 50) -> List[float]:
    """Create evenly spaced numbers over a specified interval"""
    return np.linspace(start, stop, num).tolist()

@mcp.tool()
def create_arange(start: float, stop: float, step: float = 1.0) -> List[float]:
    """Create evenly spaced values within a given interval with a step size"""
    return np.arange(start, stop, step).tolist()

@mcp.tool()
def array_reshape(array: List[float], shape: List[int]) -> List[List[float]]:
    """Reshape an array to the given shape"""
    np_array = np.array(array)
    return np_array.reshape(tuple(shape)).tolist()

@mcp.tool()
def array_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Transpose a matrix"""
    return np.transpose(matrix).tolist()

@mcp.tool()
def array_flatten(matrix: List[List[float]]) -> List[float]:
    """Flatten a matrix to a 1D array"""
    return np.array(matrix).flatten().tolist()

@mcp.tool()
def array_concatenate(arrays: List[List[float]], axis: int = 0) -> List[List[float]]:
    """Concatenate arrays along the specified axis"""
    np_arrays = [np.array(arr) for arr in arrays]
    return np.concatenate(np_arrays, axis=axis).tolist()

@mcp.tool()
def array_split(array: List[float], sections: int) -> List[List[float]]:
    """Split an array into multiple sub-arrays"""
    result = np.array_split(array, sections)
    return [arr.tolist() for arr in result]

@mcp.tool()
def array_sort(array: List[float]) -> List[float]:
    """Sort an array"""
    return np.sort(array).tolist()

@mcp.tool()
def array_argsort(array: List[float]) -> List[int]:
    """Return the indices that would sort an array"""
    return np.argsort(array).tolist()

@mcp.tool()
def array_unique(array: List[float]) -> List[float]:
    """Find the unique elements of an array"""
    return np.unique(array).tolist()

@mcp.tool()
def array_where(condition: List[bool], x: List[float], y: List[float]) -> List[float]:
    """Return elements chosen from x or y depending on condition"""
    return np.where(condition, x, y).tolist()

@mcp.tool()
def array_diff(array: List[float]) -> List[float]:
    """Calculate the n-th discrete difference along the array"""
    return np.diff(array).tolist()

@mcp.tool()
def array_cumsum(array: List[float]) -> List[float]:
    """Return the cumulative sum of the elements along a given axis"""
    return np.cumsum(array).tolist()

@mcp.tool()
def array_cumprod(array: List[float]) -> List[float]:
    """Return the cumulative product of the elements along a given axis"""
    return np.cumprod(array).tolist()

@mcp.tool()
def array_histogram(array: List[float], bins: int = 10) -> Tuple[List[float], List[float]]:
    """Compute the histogram of a set of data"""
    hist, bin_edges = np.histogram(array, bins=bins)
    return hist.tolist(), bin_edges.tolist()

@mcp.tool()
def array_percentile(array: List[float], q: float) -> float:
    """Compute the q-th percentile of the data along the specified axis"""
    return float(np.percentile(array, q))

@mcp.tool()
def array_quantile(array: List[float], q: float) -> float:
    """Compute the q-th quantile of the data along the specified axis"""
    return float(np.quantile(array, q))

@mcp.tool()
def array_corrcoef(x: List[float], y: List[float]) -> float:
    """Compute correlation coefficient between two arrays"""
    return float(np.corrcoef(x, y)[0, 1])

@mcp.tool()
def array_convolve(a: List[float], v: List[float], mode: str = 'full') -> List[float]:
    """Convolve two arrays"""
    return np.convolve(a, v, mode=mode).tolist()

@mcp.tool()
def array_fft(array: List[float]) -> List[complex]:
    """Compute the one-dimensional discrete Fourier Transform"""
    result = np.fft.fft(array)
    return [(float(x.real), float(x.imag)) for x in result]

@mcp.tool()
def array_ifft(array: List[Tuple[float, float]]) -> List[complex]:
    """Compute the one-dimensional inverse discrete Fourier Transform"""
    complex_array = [complex(x[0], x[1]) for x in array]
    result = np.fft.ifft(complex_array)
    return [(float(x.real), float(x.imag)) for x in result]

# SciPy Functions
@mcp.tool()
def optimize_minimize(func: str, x0: List[float], method: str = 'BFGS') -> Tuple[List[float], float]:
    """Find the minimum of a function using SciPy's optimize.minimize"""
    def objective(x):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        return eval(func, {"np": np, "x": x})

    result = scipy.optimize.minimize(objective, np.array(x0), method=method)
    return result.x.tolist(), float(result.fun)

@mcp.tool()
def optimize_root(func: str, x0: List[float], method: str = 'hybr') -> List[float]:
    """Find a root of a function using SciPy's optimize.root"""
    def objective(x):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        return eval(func, {"np": np, "x": x})

    result = scipy.optimize.root(objective, np.array(x0), method=method)
    return result.x.tolist()

@mcp.tool()
def optimize_curve_fit(func: str, xdata: List[float], ydata: List[float], p0: List[float] = None) -> List[float]:
    """Fit a curve to data using SciPy's optimize.curve_fit"""
    def model(x, *params):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        param_dict = {f"p{i}": p for i, p in enumerate(params)}
        return eval(func, {"np": np, "x": x, **param_dict})

    if p0 is None:
        p0 = [1.0] * func.count('p')

    popt, _ = scipy.optimize.curve_fit(model, np.array(xdata), np.array(ydata), p0=p0)
    return popt.tolist()

@mcp.tool()
def integrate_quad(func: str, a: float, b: float) -> float:
    """Integrate a function using SciPy's integrate.quad"""
    def integrand(x):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        return eval(func, {"np": np, "math": math, "x": x})

    result, _ = scipy.integrate.quad(integrand, a, b)
    return float(result)

@mcp.tool()
def integrate_odeint(func: str, y0: List[float], t: List[float]) -> List[List[float]]:
    """Solve an ODE using SciPy's integrate.odeint"""
    def system(y, t):
        # Simple expression evaluator - in real code you'd want to use a safer approach
        return eval(func, {"np": np, "math": math, "y": y, "t": t})

    result = scipy.integrate.odeint(system, np.array(y0), np.array(t))
    return result.tolist()

@mcp.tool()
def interpolate_interp1d(x: List[float], y: List[float], xnew: List[float], kind: str = 'linear') -> List[float]:
    """Interpolate a 1D function using SciPy's interpolate.interp1d"""
    f = scipy.interpolate.interp1d(x, y, kind=kind)
    return f(xnew).tolist()

@mcp.tool()
def interpolate_spline(x: List[float], y: List[float], xnew: List[float], k: int = 3) -> List[float]:
    """Interpolate a 1D function using SciPy's interpolate.splrep and splev"""
    tck = scipy.interpolate.splrep(x, y, k=k)
    return scipy.interpolate.splev(xnew, tck).tolist()

@mcp.tool()
def special_gamma(x: float) -> float:
    """Compute the gamma function using SciPy's special.gamma"""
    return float(scipy.special.gamma(x))

@mcp.tool()
def special_beta(a: float, b: float) -> float:
    """Compute the beta function using SciPy's special.beta"""
    return float(scipy.special.beta(a, b))

@mcp.tool()
def special_erf(x: float) -> float:
    """Compute the error function using SciPy's special.erf"""
    return float(scipy.special.erf(x))

@mcp.tool()
def special_erfc(x: float) -> float:
    """Compute the complementary error function using SciPy's special.erfc"""
    return float(scipy.special.erfc(x))

@mcp.tool()
def special_jv(v: float, z: float) -> float:
    """Compute the Bessel function of the first kind using SciPy's special.jv"""
    return float(scipy.special.jv(v, z))

@mcp.tool()
def special_yv(v: float, z: float) -> float:
    """Compute the Bessel function of the second kind using SciPy's special.yv"""
    return float(scipy.special.yv(v, z))

@mcp.tool()
def special_iv(v: float, z: float) -> float:
    """Compute the modified Bessel function of the first kind using SciPy's special.iv"""
    return float(scipy.special.iv(v, z))

@mcp.tool()
def special_kv(v: float, z: float) -> float:
    """Compute the modified Bessel function of the second kind using SciPy's special.kv"""
    return float(scipy.special.kv(v, z))

@mcp.tool()
def special_legendre(n: int, x: float) -> float:
    """Compute the Legendre polynomial using SciPy's special.eval_legendre"""
    return float(scipy.special.eval_legendre(n, x))

@mcp.tool()
def special_hermite(n: int, x: float) -> float:
    """Compute the Hermite polynomial using SciPy's special.eval_hermite"""
    return float(scipy.special.eval_hermite(n, x))

@mcp.tool()
def special_laguerre(n: int, x: float) -> float:
    """Compute the Laguerre polynomial using SciPy's special.eval_laguerre"""
    return float(scipy.special.eval_laguerre(n, x))

# Polynomial Functions
@mcp.tool()
def poly_evaluate(coefficients: List[float], x: float) -> float:
    """Evaluate a polynomial at point x"""
    return float(np.polyval(coefficients, x))

@mcp.tool()
def poly_roots(coefficients: List[float]) -> List[complex]:
    """Find the roots of a polynomial"""
    roots = np.roots(coefficients)
    return [(float(root.real), float(root.imag)) for root in roots]

@mcp.tool()
def poly_fit(x: List[float], y: List[float], degree: int) -> List[float]:
    """Fit a polynomial of specified degree to data"""
    return np.polyfit(x, y, degree).tolist()

@mcp.tool()
def poly_derivative(coefficients: List[float]) -> List[float]:
    """Compute the derivative of a polynomial"""
    return np.polyder(coefficients).tolist()

@mcp.tool()
def poly_integral(coefficients: List[float], k: float = 0.0) -> List[float]:
    """Compute the integral of a polynomial"""
    return np.polyint(coefficients, m=1, k=[k]).tolist()

@mcp.tool()
def poly_add(poly1: List[float], poly2: List[float]) -> List[float]:
    """Add two polynomials"""
    return np.polyadd(poly1, poly2).tolist()

@mcp.tool()
def poly_subtract(poly1: List[float], poly2: List[float]) -> List[float]:
    """Subtract second polynomial from first polynomial"""
    return np.polysub(poly1, poly2).tolist()

@mcp.tool()
def poly_multiply(poly1: List[float], poly2: List[float]) -> List[float]:
    """Multiply two polynomials"""
    return np.polymul(poly1, poly2).tolist()

@mcp.tool()
def poly_divide(poly1: List[float], poly2: List[float]) -> Tuple[List[float], List[float]]:
    """Divide first polynomial by second polynomial"""
    quotient, remainder = np.polydiv(poly1, poly2)
    return quotient.tolist(), remainder.tolist()

# Random Number Generation
@mcp.tool()
def random_uniform(low: float = 0.0, high: float = 1.0, size: int = 1) -> List[float]:
    """Generate random numbers from a uniform distribution"""
    return np.random.uniform(low, high, size).tolist()

@mcp.tool()
def random_normal(mean: float = 0.0, std: float = 1.0, size: int = 1) -> List[float]:
    """Generate random numbers from a normal (Gaussian) distribution"""
    return np.random.normal(mean, std, size).tolist()

@mcp.tool()
def random_poisson(lam: float = 1.0, size: int = 1) -> List[int]:
    """Generate random numbers from a Poisson distribution"""
    return np.random.poisson(lam, size).tolist()

@mcp.tool()
def random_exponential(scale: float = 1.0, size: int = 1) -> List[float]:
    """Generate random numbers from an exponential distribution"""
    return np.random.exponential(scale, size).tolist()

@mcp.tool()
def random_binomial(n: int = 1, p: float = 0.5, size: int = 1) -> List[int]:
    """Generate random numbers from a binomial distribution"""
    return np.random.binomial(n, p, size).tolist()

@mcp.tool()
def random_choice(items: List[float], size: int = 1, replace: bool = True, p: List[float] = None) -> List[float]:
    """Generate a random sample from a given 1-D array"""
    return np.random.choice(items, size=size, replace=replace, p=p).tolist()

@mcp.tool()
def random_shuffle(x: List[float]) -> List[float]:
    """Randomly permute a sequence in place"""
    result = x.copy()
    np.random.shuffle(result)
    return result

@mcp.tool()
def random_permutation(x: int) -> List[int]:
    """Randomly permute a sequence or return a permuted range"""
    return np.random.permutation(x).tolist()

@mcp.tool()
def random_seed(seed: int) -> None:
    """Seed the random number generator"""
    np.random.seed(seed)
    return None

# Mathematical Constants
@mcp.tool()
def get_pi() -> float:
    """Return the mathematical constant π (pi)"""
    return math.pi

@mcp.tool()
def get_e() -> float:
    """Return the mathematical constant e (Euler's number)"""
    return math.e

@mcp.tool()
def get_tau() -> float:
    """Return the mathematical constant τ (tau), equal to 2π"""
    return math.tau

@mcp.tool()
def get_inf() -> float:
    """Return positive infinity"""
    return math.inf

@mcp.tool()
def get_nan() -> float:
    """Return Not a Number (NaN)"""
    return math.nan

# Hyperbolic Functions
@mcp.tool()
def sinh(x: float) -> float:
    """Calculate the hyperbolic sine of x"""
    return math.sinh(x)

@mcp.tool()
def cosh(x: float) -> float:
    """Calculate the hyperbolic cosine of x"""
    return math.cosh(x)

@mcp.tool()
def tanh(x: float) -> float:
    """Calculate the hyperbolic tangent of x"""
    return math.tanh(x)

@mcp.tool()
def asinh(x: float) -> float:
    """Calculate the inverse hyperbolic sine of x"""
    return math.asinh(x)

@mcp.tool()
def acosh(x: float) -> float:
    """Calculate the inverse hyperbolic cosine of x"""
    return math.acosh(x)

@mcp.tool()
def atanh(x: float) -> float:
    """Calculate the inverse hyperbolic tangent of x"""
    return math.atanh(x)

# Inverse Trigonometric Functions
@mcp.tool()
def asin(x: float) -> float:
    """Calculate the arc sine (inverse sine) of x"""
    return math.asin(x)

@mcp.tool()
def acos(x: float) -> float:
    """Calculate the arc cosine (inverse cosine) of x"""
    return math.acos(x)

@mcp.tool()
def atan(x: float) -> float:
    """Calculate the arc tangent (inverse tangent) of x"""
    return math.atan(x)

@mcp.tool()
def atan2(y: float, x: float) -> float:
    """Calculate the arc tangent of y/x, respecting the signs of both arguments"""
    return math.atan2(y, x)

# Special Mathematical Functions
@mcp.tool()
def gamma(x: float) -> float:
    """Calculate the gamma function at x"""
    return math.gamma(x)

@mcp.tool()
def lgamma(x: float) -> float:
    """Calculate the natural logarithm of the absolute value of the gamma function at x"""
    return math.lgamma(x)

@mcp.tool()
def erf(x: float) -> float:
    """Calculate the error function at x"""
    return math.erf(x)

@mcp.tool()
def erfc(x: float) -> float:
    """Calculate the complementary error function at x"""
    return math.erfc(x)

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

# Combinatorial Functions
@mcp.tool()
def permutations_count(n: int, k: int) -> int:
    """Calculate the number of ways to choose k items from n items where order matters (permutations)"""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Invalid input: n and k must be non-negative and k <= n")
    return int(math.factorial(n) / math.factorial(n - k))

@mcp.tool()
def combinations_count(n: int, k: int) -> int:
    """Calculate the number of ways to choose k items from n items where order doesn't matter (combinations)"""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Invalid input: n and k must be non-negative and k <= n")
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))

@mcp.tool()
def binomial_coefficient(n: int, k: int) -> int:
    """Calculate the binomial coefficient (n choose k)"""
    return combinations_count(n, k)

@mcp.tool()
def multinomial_coefficient(ks: List[int]) -> int:
    """Calculate the multinomial coefficient (n choose k1,k2,...,km)"""
    if not all(k >= 0 for k in ks):
        raise ValueError("All values must be non-negative")
    n = sum(ks)
    result = math.factorial(n)
    for k in ks:
        result //= math.factorial(k)
    return result

@mcp.tool()
def stirling_number_first_kind(n: int, k: int) -> int:
    """Calculate the Stirling number of the first kind s(n,k)"""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Invalid input: n and k must be non-negative and k <= n")
    if n == 0 and k == 0:
        return 1
    if n > 0 and k == 0:
        return 0
    if k > n:
        return 0
    return (n - 1) * stirling_number_first_kind(n - 1, k) + stirling_number_first_kind(n - 1, k - 1)

@mcp.tool()
def stirling_number_second_kind(n: int, k: int) -> int:
    """Calculate the Stirling number of the second kind S(n,k)"""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Invalid input: n and k must be non-negative and k <= n")
    if k == 0:
        return 1 if n == 0 else 0
    if k == 1 or k == n:
        return 1
    return k * stirling_number_second_kind(n - 1, k) + stirling_number_second_kind(n - 1, k - 1)

@mcp.tool()
def bell_number(n: int) -> int:
    """Calculate the Bell number B(n), the number of partitions of a set with n elements"""
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n == 0:
        return 1
    result = 0
    for k in range(n):
        result += combinations_count(n - 1, k) * bell_number(k)
    return result

@mcp.tool()
def catalan_number(n: int) -> int:
    """Calculate the Catalan number C(n)"""
    if n < 0:
        raise ValueError("Input must be non-negative")
    return combinations_count(2 * n, n) // (n + 1)

@mcp.tool()
def derangement_count(n: int) -> int:
    """Calculate the number of derangements of n elements"""
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n == 0:
        return 1
    if n == 1:
        return 0
    return (n - 1) * (derangement_count(n - 1) + derangement_count(n - 2))

@mcp.tool()
def factorial_non_integer(x: float) -> float:
    """Calculate the factorial of a non-integer number using the gamma function"""
    if x < 0:
        raise ValueError("Input must be non-negative")
    return math.gamma(x + 1)

# Advanced Statistics Functions
@mcp.tool()
def skewness(data: List[float]) -> float:
    """Calculate the skewness of a dataset"""
    if len(data) < 3:
        raise ValueError("At least 3 data points are required to calculate skewness")
    n = len(data)
    mean_val = mean(data)
    std_dev = std_deviation(data)

    # Calculate the third moment
    third_moment = sum((x - mean_val) ** 3 for x in data) / n

    # Calculate skewness
    return third_moment / (std_dev ** 3)

@mcp.tool()
def kurtosis(data: List[float]) -> float:
    """Calculate the kurtosis of a dataset"""
    if len(data) < 4:
        raise ValueError("At least 4 data points are required to calculate kurtosis")
    n = len(data)
    mean_val = mean(data)
    std_dev = std_deviation(data)

    # Calculate the fourth moment
    fourth_moment = sum((x - mean_val) ** 4 for x in data) / n

    # Calculate kurtosis
    return fourth_moment / (std_dev ** 4) - 3  # Excess kurtosis (normal distribution has kurtosis of 3)

@mcp.tool()
def covariance(x: List[float], y: List[float]) -> float:
    """Calculate the covariance between two datasets"""
    if len(x) != len(y):
        raise ValueError("Both datasets must have the same length")
    if len(x) < 2:
        raise ValueError("At least 2 data points are required to calculate covariance")

    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

@mcp.tool()
def covariance_matrix(data: List[List[float]]) -> List[List[float]]:
    """Calculate the covariance matrix for multivariate data"""
    # Convert to numpy array for calculation
    np_data = np.array(data)
    cov_matrix = np.cov(np_data, rowvar=False)
    return cov_matrix.tolist()

@mcp.tool()
def t_test_one_sample(data: List[float], popmean: float) -> Tuple[float, float]:
    """Perform a one-sample t-test"""
    # Convert to numpy array for calculation
    np_data = np.array(data)
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(np_data, popmean)
    return float(t_stat), float(p_value)

@mcp.tool()
def t_test_independent(data1: List[float], data2: List[float], equal_var: bool = True) -> Tuple[float, float]:
    """Perform an independent samples t-test"""
    # Convert to numpy arrays for calculation
    np_data1 = np.array(data1)
    np_data2 = np.array(data2)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(np_data1, np_data2, equal_var=equal_var)
    return float(t_stat), float(p_value)

@mcp.tool()
def t_test_paired(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """Perform a paired samples t-test"""
    if len(data1) != len(data2):
        raise ValueError("Both datasets must have the same length for a paired t-test")
    # Convert to numpy arrays for calculation
    np_data1 = np.array(data1)
    np_data2 = np.array(data2)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(np_data1, np_data2)
    return float(t_stat), float(p_value)

@mcp.tool()
def chi_square_test(observed: List[float], expected: List[float] = None) -> Tuple[float, float]:
    """Perform a chi-square test for goodness of fit"""
    if len(observed) < 2:
        raise ValueError("At least 2 categories are required for chi-square test")
    # Convert to numpy arrays for calculation
    np_observed = np.array(observed)
    from scipy import stats

    if expected is None:
        # Test for uniform distribution
        expected = [sum(observed) / len(observed)] * len(observed)

    np_expected = np.array(expected)
    chi2_stat, p_value = stats.chisquare(np_observed, np_expected)
    return float(chi2_stat), float(p_value)

@mcp.tool()
def confidence_interval_mean(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the confidence interval for the mean"""
    if len(data) < 2:
        raise ValueError("At least 2 data points are required to calculate a confidence interval")
    # Convert to numpy array for calculation
    np_data = np.array(data)
    from scipy import stats

    mean_val = np.mean(np_data)
    std_err = stats.sem(np_data)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, len(np_data) - 1)

    return float(mean_val - interval), float(mean_val + interval)

@mcp.tool()
def kernel_density_estimation(data: List[float], points: List[float], bandwidth: float = None) -> List[float]:
    """Perform kernel density estimation on a dataset"""
    # Convert to numpy arrays for calculation
    np_data = np.array(data)
    np_points = np.array(points)
    from scipy import stats

    kde = stats.gaussian_kde(np_data, bw_method=bandwidth)
    density = kde(np_points)

    return density.tolist()

# Advanced Numerical Methods
@mcp.tool()
def newton_raphson_root(func: str, dfunc: str, x0: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Find a root of a function using the Newton-Raphson method"""
    def f(x):
        return eval(func, {"math": math, "np": np, "x": x})

    def df(x):
        return eval(dfunc, {"math": math, "np": np, "x": x})

    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero, cannot continue")
        x = x - fx / dfx

    return x

@mcp.tool()
def bisection_root(func: str, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Find a root of a function using the bisection method"""
    def f(x):
        return eval(func, {"math": math, "np": np, "x": x})

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at the interval endpoints")

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2

@mcp.tool()
def secant_root(func: str, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Find a root of a function using the secant method"""
    def f(x):
        return eval(func, {"math": math, "np": np, "x": x})

    f0 = f(x0)
    f1 = f(x1)

    for i in range(max_iter):
        if abs(f1) < tol:
            return x1

        if f1 == f0:
            raise ValueError("Division by zero in secant method")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
        f0, f1 = f1, f(x2)

    return x1

@mcp.tool()
def numerical_derivative_higher_order(func: str, x: float, order: int = 1, h: float = 0.0001) -> float:
    """Calculate higher-order numerical derivatives of a function at a point"""
    def f(x):
        return eval(func, {"math": math, "np": np, "x": x})

    if order == 1:
        # Central difference formula for first derivative
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        # Central difference formula for second derivative
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    elif order == 3:
        # Central difference formula for third derivative
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)
    elif order == 4:
        # Central difference formula for fourth derivative
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    else:
        raise ValueError("Order must be between 1 and 4")

# Financial Mathematics Functions
@mcp.tool()
def present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate the present value of a future sum"""
    if rate <= -1:
        raise ValueError("Rate must be greater than -1")
    return future_value / ((1 + rate) ** periods)

@mcp.tool()
def future_value(present_value: float, rate: float, periods: int) -> float:
    """Calculate the future value of a present sum"""
    if rate <= -1:
        raise ValueError("Rate must be greater than -1")
    return present_value * ((1 + rate) ** periods)

@mcp.tool()
def pmt(principal: float, rate: float, periods: int) -> float:
    """Calculate the payment for a loan based on constant payments and a constant interest rate"""
    if rate == 0:
        return principal / periods
    return principal * rate * (1 + rate) ** periods / ((1 + rate) ** periods - 1)

@mcp.tool()
def fv_annuity(payment: float, rate: float, periods: int) -> float:
    """Calculate the future value of an annuity"""
    if rate == 0:
        return payment * periods
    return payment * ((1 + rate) ** periods - 1) / rate

@mcp.tool()
def pv_annuity(payment: float, rate: float, periods: int) -> float:
    """Calculate the present value of an annuity"""
    if rate == 0:
        return payment * periods
    return payment * (1 - 1 / ((1 + rate) ** periods)) / rate

@mcp.tool()
def amortization_schedule(principal: float, rate: float, periods: int) -> List[Dict[str, float]]:
    """Generate an amortization schedule for a loan"""
    if rate <= 0 or periods <= 0 or principal <= 0:
        raise ValueError("Rate, periods, and principal must be positive")

    monthly_rate = rate / 12
    monthly_payment = pmt(principal, monthly_rate, periods)

    schedule = []
    balance = principal

    for period in range(1, periods + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment

        schedule.append({
            "period": period,
            "payment": monthly_payment,
            "principal": principal_payment,
            "interest": interest_payment,
            "balance": max(0, balance)  # Ensure balance doesn't go below zero due to rounding
        })

    return schedule

@mcp.tool()
def black_scholes_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Calculate the price of a European option using the Black-Scholes model"""
    from scipy.stats import norm

    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")

    if T <= 0 or sigma <= 0:
        raise ValueError("Time to maturity and volatility must be positive")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Signal Processing Functions
@mcp.tool()
def fft_real(signal: List[float]) -> Tuple[List[float], List[float]]:
    """Compute the Fast Fourier Transform of a real signal"""
    # Convert to numpy array for calculation
    np_signal = np.array(signal)
    fft_result = np.fft.rfft(np_signal)

    # Return magnitude and phase
    magnitude = np.abs(fft_result).tolist()
    phase = np.angle(fft_result).tolist()

    return magnitude, phase

@mcp.tool()
def ifft_real(magnitude: List[float], phase: List[float]) -> List[float]:
    """Compute the Inverse Fast Fourier Transform from magnitude and phase"""
    # Reconstruct complex FFT result
    fft_result = np.array([mag * np.exp(1j * ph) for mag, ph in zip(magnitude, phase)])

    # Compute inverse FFT
    signal = np.fft.irfft(fft_result)

    return signal.tolist()

@mcp.tool()
def create_window(window_type: str, length: int) -> List[float]:
    """Create a window function for signal processing"""
    if window_type not in ['hamming', 'hanning', 'blackman', 'bartlett', 'kaiser']:
        raise ValueError("Window type must be one of: hamming, hanning, blackman, bartlett, kaiser")

    if window_type == 'hamming':
        window = np.hamming(length)
    elif window_type == 'hanning':
        window = np.hanning(length)
    elif window_type == 'blackman':
        window = np.blackman(length)
    elif window_type == 'bartlett':
        window = np.bartlett(length)
    elif window_type == 'kaiser':
        window = np.kaiser(length, beta=14)  # Beta parameter controls sidelobe level

    return window.tolist()

@mcp.tool()
def apply_window(signal: List[float], window_type: str) -> List[float]:
    """Apply a window function to a signal"""
    # Create window of the same length as the signal
    window = create_window(window_type, len(signal))

    # Apply window
    windowed_signal = [s * w for s, w in zip(signal, window)]

    return windowed_signal

@mcp.tool()
def filter_signal_lowpass(signal: List[float], cutoff_freq: float, sampling_rate: float) -> List[float]:
    """Apply a low-pass filter to a signal"""
    from scipy import signal as sp_signal

    # Normalize cutoff frequency
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist

    # Create a Butterworth filter
    b, a = sp_signal.butter(4, normal_cutoff, btype='low', analog=False)

    # Apply the filter
    filtered_signal = sp_signal.filtfilt(b, a, signal)

    return filtered_signal.tolist()

@mcp.tool()
def filter_signal_highpass(signal: List[float], cutoff_freq: float, sampling_rate: float) -> List[float]:
    """Apply a high-pass filter to a signal"""
    from scipy import signal as sp_signal

    # Normalize cutoff frequency
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist

    # Create a Butterworth filter
    b, a = sp_signal.butter(4, normal_cutoff, btype='high', analog=False)

    # Apply the filter
    filtered_signal = sp_signal.filtfilt(b, a, signal)

    return filtered_signal.tolist()

@mcp.tool()
def filter_signal_bandpass(signal: List[float], lowcut: float, highcut: float, sampling_rate: float) -> List[float]:
    """Apply a band-pass filter to a signal"""
    from scipy import signal as sp_signal

    # Normalize cutoff frequencies
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create a Butterworth filter
    b, a = sp_signal.butter(4, [low, high], btype='band', analog=False)

    # Apply the filter
    filtered_signal = sp_signal.filtfilt(b, a, signal)

    return filtered_signal.tolist()

@mcp.tool()
def compute_spectrogram(signal: List[float], sampling_rate: float, window_size: int = 256, overlap: int = 128) -> Tuple[List[List[float]], List[float], List[float]]:
    """Compute the spectrogram of a signal"""
    from scipy import signal as sp_signal

    # Compute spectrogram
    frequencies, times, spectrogram = sp_signal.spectrogram(
        signal, fs=sampling_rate, window='hanning',
        nperseg=window_size, noverlap=overlap, detrend=False
    )

    # Convert to dB scale
    spectrogram_db = 10 * np.log10(spectrogram + 1e-10)  # Add small value to avoid log(0)

    return spectrogram_db.tolist(), frequencies.tolist(), times.tolist()

@mcp.tool()
def resample_signal(signal: List[float], original_rate: float, target_rate: float) -> List[float]:
    """Resample a signal to a new sampling rate"""
    from scipy import signal as sp_signal

    # Calculate the number of samples in the resampled signal
    num_samples = int(len(signal) * target_rate / original_rate)

    # Resample the signal
    resampled_signal = sp_signal.resample(signal, num_samples)

    return resampled_signal.tolist()

# Geometry Functions
@mcp.tool()
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate the Euclidean distance between two points"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

@mcp.tool()
def manhattan_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate the Manhattan (L1) distance between two points"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

@mcp.tool()
def chebyshev_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate the Chebyshev (L∞) distance between two points"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    return max(abs(p1 - p2) for p1, p2 in zip(point1, point2))

@mcp.tool()
def minkowski_distance(point1: List[float], point2: List[float], p: float) -> float:
    """Calculate the Minkowski distance between two points"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")
    if p <= 0:
        raise ValueError("p must be positive")

    return (sum(abs(p1 - p2) ** p for p1, p2 in zip(point1, point2))) ** (1/p)

@mcp.tool()
def point_line_distance(point: List[float], line_point1: List[float], line_point2: List[float]) -> float:
    """Calculate the distance from a point to a line defined by two points"""
    if len(point) != len(line_point1) or len(point) != len(line_point2):
        raise ValueError("All points must have the same dimensions")
    if len(point) != 2 and len(point) != 3:
        raise ValueError("This function only supports 2D and 3D points")

    # Convert to numpy arrays for calculation
    p = np.array(point)
    l1 = np.array(line_point1)
    l2 = np.array(line_point2)

    # Calculate the distance
    if len(point) == 2:  # 2D case
        # Formula: d = |Ax + By + C| / sqrt(A^2 + B^2)
        # where Ax + By + C = 0 is the line equation
        A = l2[1] - l1[1]  # y2 - y1
        B = l1[0] - l2[0]  # x1 - x2
        C = l2[0] * l1[1] - l1[0] * l2[1]  # x2*y1 - x1*y2

        return abs(A * p[0] + B * p[1] + C) / math.sqrt(A**2 + B**2)
    else:  # 3D case
        # Calculate the cross product and divide by the length of the line segment
        line_vec = l2 - l1
        point_vec = p - l1
        cross_product = np.cross(point_vec, line_vec)

        return np.linalg.norm(cross_product) / np.linalg.norm(line_vec)

@mcp.tool()
def point_plane_distance(point: List[float], plane_point: List[float], plane_normal: List[float]) -> float:
    """Calculate the distance from a point to a plane defined by a point and normal vector"""
    if len(point) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("All points and vectors must be 3D")

    # Convert to numpy arrays for calculation
    p = np.array(point)
    plane_p = np.array(plane_point)
    normal = np.array(plane_normal)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Calculate the distance
    return abs(np.dot(normal, p - plane_p))

@mcp.tool()
def triangle_area(point1: List[float], point2: List[float], point3: List[float]) -> float:
    """Calculate the area of a triangle defined by three points"""
    if len(point1) != len(point2) or len(point1) != len(point3):
        raise ValueError("All points must have the same dimensions")

    # Convert to numpy arrays for calculation
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Calculate vectors from p1 to p2 and p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1

    if len(point1) == 2:  # 2D case
        # Use the cross product formula for 2D vectors
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
    else:  # 3D case
        # Use the cross product for 3D vectors
        cross = np.cross(v1, v2)
        return np.linalg.norm(cross) / 2

@mcp.tool()
def convex_hull_2d(points: List[List[float]]) -> List[List[float]]:
    """Compute the convex hull of a set of 2D points"""
    if not points:
        raise ValueError("Input must contain at least one point")
    if any(len(p) != 2 for p in points):
        raise ValueError("All points must be 2D")

    # Convert to numpy array for calculation
    np_points = np.array(points)

    # Compute the convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(np_points)

    # Extract the vertices of the convex hull in order
    hull_points = [points[i] for i in hull.vertices]

    return hull_points

@mcp.tool()
def rotate_point_2d(point: List[float], center: List[float], angle_degrees: float) -> List[float]:
    """Rotate a 2D point around a center point by a given angle in degrees"""
    if len(point) != 2 or len(center) != 2:
        raise ValueError("Both point and center must be 2D")

    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)

    # Translate point to origin
    x = point[0] - center[0]
    y = point[1] - center[1]

    # Rotate point
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    # Translate back
    x_new = x_rot + center[0]
    y_new = y_rot + center[1]

    return [x_new, y_new]

@mcp.tool()
def scale_point(point: List[float], center: List[float], scale_factor: float) -> List[float]:
    """Scale a point relative to a center point by a given factor"""
    if len(point) != len(center):
        raise ValueError("Point and center must have the same dimensions")

    # Scale the point
    scaled_point = [center[i] + scale_factor * (point[i] - center[i]) for i in range(len(point))]

    return scaled_point
