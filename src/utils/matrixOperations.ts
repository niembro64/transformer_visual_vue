/**
 * Utility functions for matrix operations in the transformer visualization
 */

/**
 * Matrix multiplication: A x B
 * @param a - First matrix of shape [m, n]
 * @param b - Second matrix of shape [n, p]
 * @returns Result matrix of shape [m, p]
 */
export function matrixMultiply(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0) return [[]];
  if (a[0].length !== b.length) {
    throw new Error(
      `Matrix dimensions don't match for multiplication: ${a[0].length} != ${b.length}`
    );
  }

  const result: number[][] = [];
  const m = a.length;
  const n = a[0].length;
  const p = b[0].length;

  for (let i = 0; i < m; i++) {
    result[i] = [];
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}

/**
 * Matrix transpose
 * @param a - Matrix to transpose
 * @returns Transposed matrix
 */
export function transpose(a: number[][]): number[][] {
  if (a.length === 0) return [[]];

  const rows = a.length;
  const cols = a[0].length;
  const result: number[][] = [];

  for (let j = 0; j < cols; j++) {
    result[j] = [];
    for (let i = 0; i < rows; i++) {
      result[j][i] = a[i][j];
    }
  }

  return result;
}

/**
 * Apply softmax to each row of a matrix
 * @param matrix - Input matrix
 * @returns Matrix with softmax applied to each row
 */
export function softmax(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    // Find the maximum value for numerical stability
    const max = Math.max(...row);

    // Calculate exp(x - max) for each element
    const expValues = row.map((val) => Math.exp(val - max));

    // Sum of all exp values
    const sumExp = expValues.reduce((a, b) => a + b, 0);

    // Normalize by dividing each by the sum
    return expValues.map((exp) => exp / sumExp);
  });
}

/**
 * Scale a matrix by a scalar value
 * @param a - Matrix to scale
 * @param scalar - Scalar value
 * @returns Scaled matrix
 */
export function scaleMatrix(a: number[][], scalar: number): number[][] {
  return a.map((row) => row.map((val) => val * scalar));
}

/**
 * Add a vector to each row of a matrix (broadcasting)
 * @param a - Matrix
 * @param b - Vector to add to each row
 * @returns Result matrix
 */
export function addBias(a: number[][], b: number[]): number[][] {
  if (a.length === 0) return [[]];
  if (a[0].length !== b.length) {
    throw new Error(
      `Dimensions don't match for bias addition: ${a[0].length} != ${b.length}`
    );
  }

  return a.map((row) => row.map((val, i) => val + b[i]));
}

/**
 * Element-wise addition of two matrices
 * @param a - First matrix
 * @param b - Second matrix
 * @returns Result matrix
 */
export function matrixAdd(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0) return [[]];
  if (a.length !== b.length || a[0].length !== b[0].length) {
    throw new Error(
      `Matrix dimensions don't match for addition: [${a.length},${a[0].length}] != [${b.length},${b[0].length}]`
    );
  }

  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }

  return result;
}

/**
 * Apply a function element-wise to a matrix
 * @param a - Input matrix
 * @param fn - Function to apply to each element
 * @returns Transformed matrix
 */
export function applyFn(a: number[][], fn: (x: number) => number): number[][] {
  return a.map((row) => row.map(fn));
}

/**
 * ReLU activation function
 * @param x - Input value
 * @returns max(0, x)
 */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/**
 * Generate a random value from normal distribution using Box-Muller transform
 * @param mean - Mean of the distribution (default: 0)
 * @param stdDev - Standard deviation (default: 1)
 * @returns Random value from normal distribution
 */
export function randomNormal(mean = 0, stdDev = 1): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

/**
 * Generate a random matrix with values sampled from normal distribution
 * @param rows - Number of rows
 * @param cols - Number of columns
 * @param scale - Scaling factor for initialization
 * @returns Random matrix
 */
export function randomMatrix(
  rows: number,
  cols: number,
  scale: number = 0.1
): number[][] {
  const matrix: number[][] = [];
  const stdDev = scale / Math.sqrt(cols);

  for (let i = 0; i < rows; i++) {
    matrix[i] = [];
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = randomNormal(0, stdDev);
    }
  }

  return matrix;
}

/**
 * Generate a random vector with values sampled from normal distribution
 * @param size - Length of vector
 * @param scale - Scaling factor
 * @returns Random vector
 */
export function randomVector(size: number, scale: number = 0.01): number[] {
  return Array.from({ length: size }, () => randomNormal(0, scale));
}

/**
 * Generate sample embeddings that are distinct from each other
 * @param numTokens - Number of tokens
 * @param embeddingDim - Embedding dimension
 * @returns Matrix where each row is a token embedding
 */
export function generateSampleEmbeddings(
  numTokens: number,
  embeddingDim: number
): number[][] {
  const embeddings: number[][] = [];
  const magnitude = 0.5;

  for (let i = 0; i < numTokens; i++) {
    const embedding = new Array(embeddingDim).fill(0);

    // Create distinct patterns for each token
    for (let d = 0; d < embeddingDim; d++) {
      const angle = (2 * Math.PI * i) / numTokens + (d * Math.PI) / embeddingDim;
      embedding[d] = Math.sin(angle) * magnitude + Math.cos(angle * 1.5) * magnitude * 0.5;
    }

    embeddings.push(embedding);
  }

  return embeddings;
}
