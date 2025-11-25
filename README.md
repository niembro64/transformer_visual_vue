# Transformer Visual Vue

A Vue 3 TypeScript visualization of how a transformer attention layer works.

## Features

- **10 tokens** with **6-dimensional embeddings**
- Single transformer layer showing:
  - Self-attention mechanism (QKV)
  - Feed-forward (MLP) layer
- Fully typed TypeScript
- Vue 3 Composition API with `<script setup>`
- Tailwind CSS styling
- Interactive controls

## Architecture Flow

```
Input Tokens (10 × 6)
    ↓
Input Embeddings Matrix
    ↓
Attention Layer
  - QKV Weight Matrices (6 × 6 each)
  - Q, K, V Values
  - Attention Scores (Q × K^T / √d_k)
  - Softmax Attention Weights
  - Attention Output (weights × V)
    ↓
Feed-Forward (MLP) Layer
  - First Layer: Input × W1 + b1 (6 → 8)
  - ReLU Activation
  - Second Layer: Hidden × W2 + b2 (8 → 6)
    ↓
Final Output (10 × 6)
```

## Project Structure

```
src/
├── types.ts                    # TypeScript type definitions
├── utils/
│   └── matrixOperations.ts    # Matrix math utilities
├── components/
│   ├── Token.vue              # Token display component
│   ├── MatrixDisplay.vue      # Reusable matrix visualization
│   ├── AttentionLayer.vue     # Attention mechanism visualization
│   └── FeedForward.vue        # MLP layer visualization
└── App.vue                    # Main application
```

## Configuration

Edit the constants in `src/App.vue` to change dimensions:

```typescript
const NUM_TOKENS = 10;           // Number of input tokens
const EMBEDDING_DIM = 6;         // Token embedding dimension
const ATTENTION_HEAD_DIM = 6;    // Attention head output dimension
const MLP_HIDDEN_DIM = 8;        // MLP hidden layer dimension
```

## Interactive Controls

- **Randomize Weights**: Generate new random weight matrices for Q, K, V, and MLP layers
- **Reset Embeddings**: Regenerate token embeddings with new random values

## Development

```bash
# Install dependencies
npm install

# Start development server
npm start
# or
npm run dev

# Build for production
npm run build
```

## Type Definitions

All functions and variables are fully typed. See `src/types.ts` for:
- `Token`, `AttentionWeights`, `MLPWeights`
- `AttentionOutput`, `MLPOutput`
- `TransformerLayer`, `DimensionConfig`

## Matrix Operations

Located in `src/utils/matrixOperations.ts`:
- Matrix multiplication, transpose, addition
- Softmax activation
- ReLU activation
- Random matrix/vector generation
- Sample embedding generation

## Based On

"Attention Is All You Need" by Vaswani et al., 2017

## Technology Stack

- Vue 3.5
- TypeScript 5.9
- Vite 5.4
- Tailwind CSS 3.3
