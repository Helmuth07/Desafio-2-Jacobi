import numpy as np

# Matriz A y vector B
A = np.array([[52, 20, 25],
              [30, 50, 20],
              [18, 30, 55]], dtype=float)

B = np.array([[4800],
              [5210],
              [5690]], dtype=float)

# Tolerancia y número máximo de iteraciones
tolerancia = 1e-10
max_iteraciones = 1000

# Verificación de la diagonal dominante
diagonal = np.abs(np.diag(A))  # Términos diagonales
suma_filas = np.sum(np.abs(A), axis=1) - diagonal  # Suma de los valores fuera de la diagonal
if np.all(diagonal > suma_filas):
    print("La matriz es diagonal dominante, por lo tanto, el método de Jacobi debería converger.")
else:
    print("La matriz no es diagonal dominante, el método de Jacobi puede no converger.")

# Inicialización del vector solución
x = np.zeros_like(B)

# Implementación del método de Jacobi
for iteracion in range(max_iteraciones):
    x_nuevo = np.zeros_like(x)
    
    # Actualización de cada variable
    for i in range(A.shape[0]):
        suma = np.dot(A[i, :], x) - A[i, i] * x[i]
        x_nuevo[i] = (B[i] - suma) / A[i, i]
    
    # Verificación de la convergencia
    if np.linalg.norm(x_nuevo - x, ord=np.inf) < tolerancia:
        print(f"El método de Jacobi ha convergido en {iteracion + 1} iteraciones.")
        break
    
    x = x_nuevo

print("Solución aproximada:", x)
