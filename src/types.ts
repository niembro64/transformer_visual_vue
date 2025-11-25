/**
 * Type definitions for the transformer visualization
 */

/**
 * Represents a single token with its embedding
 */
export interface Token {
  id: number;
  text: string;
  embedding: number[];
}

/**
 * Attention weights for Q, K, V projections
 */
export interface AttentionWeights {
  weightQ: number[][];
  weightK: number[][];
  weightV: number[][];
}

/**
 * Feed-forward (MLP) network weights
 */
export interface MLPWeights {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
}

/**
 * Complete transformer layer configuration
 */
export interface TransformerLayer {
  attention: AttentionWeights;
  mlp: MLPWeights;
}

/**
 * Attention computation intermediate results
 */
export interface AttentionOutput {
  Q: number[][];
  K: number[][];
  V: number[][];
  scores: number[][];
  attentionWeights: number[][];
  output: number[][];
}

/**
 * MLP computation intermediate results
 */
export interface MLPOutput {
  hidden: number[][];
  activated: number[][];
  output: number[][];
}

/**
 * Selected element in a matrix for highlighting
 */
export interface SelectedElement {
  matrixType: 'Q' | 'K' | 'V' | 'attention' | 'embedding' | 'mlp' | null;
  row: number;
  col: number;
}

/**
 * Matrix dimensions configuration
 */
export interface DimensionConfig {
  numTokens: number;
  embeddingDim: number;
  attentionHeadDim: number;
  mlpHiddenDim: number;
}
