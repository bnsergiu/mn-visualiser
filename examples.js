// Starter snippets for the "Load an example…" menu. Any other code can be pasted
// into the editor; these only show what the visualiser does.
window.EXAMPLES = [
  {
    title: 'Lab 1 · Ex 1 — Frobenius norm (notebook version)',
    stdin: '3\n3',
    code: `import numpy as np
def frobenius():
  rng = np.random.default_rng()
  n = input("Nr. linii")
  m = input("Nr. coloane")
  a = rng.random((int(n), int(m)))
  fr = 0;
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      fr += a[i,j]
  fr = np.sqrt(fr)
  if fr == np.linalg.norm(a):
    print("da")
  else:
    print("nu")
  print(fr)
  print( np.linalg.norm(a))
frobenius()
`,
  },
  {
    title: 'Lab 1 · Ex 2 — A·B with nested for',
    stdin: '3',
    code: `import numpy as np
n = int(input())
rng = np.random.default_rng()
a = rng.random((n, n))
b = rng.random((n, n))
c = np.zeros((n, n))
for i in range(n):
  for j in range(n):
    for k in range(n):
      c[i,j] = c[i,j] + a[i, k] * b[k,j]

print(c)
print("\\n")
c = np.dot(a, b)
print(c)
`,
  },
  {
    title: 'Lab 1 — Selecting rows, columns and elements',
    stdin: '',
    code: `import numpy as np
data = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])
print(data[1, 0], data[1], data[:, 0])

M = np.arange(1, 17).reshape(4, 4)
first_row = M[0]
last_col = M[:, -1]
block = M[1:3, 1:3]
diag = np.diag(M)
upper = np.triu(M)
lower = np.tril(M, -1)
M[2, :] = 0
`,
  },
  {
    title: 'Gaussian elimination + back substitution',
    stdin: '',
    code: `import numpy as np

A = np.array([[2., 1., -1.],
              [-3., -1., 2.],
              [-2., 1., 2.]])
b = np.array([8., -11., -3.])
n = len(b)
A0, b0 = A.copy(), b.copy()

# forward elimination: zero out everything below the diagonal
for k in range(n - 1):
    for i in range(k + 1, n):
        m = A[i, k] / A[k, k]
        A[i, k:] = A[i, k:] - m * A[k, k:]
        b[i] = b[i] - m * b[k]

# back substitution on the upper triangular system
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

print(x)
print(np.linalg.solve(A0, b0))
`,
  },
];
