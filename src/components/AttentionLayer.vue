<script setup lang="ts">
import { computed } from 'vue';
import MatrixDisplay from './MatrixDisplay.vue';
import type { AttentionWeights, AttentionOutput } from '../types';
import { matrixMultiply, transpose, softmax, scaleMatrix } from '../utils/matrixOperations';

interface Props {
  embeddings: number[][];
  weights: AttentionWeights;
  tokenLabels?: string[];
  showSteps?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  tokenLabels: () => [],
  showSteps: true,
});

const emit = defineEmits<{
  outputComputed: [output: number[][]];
}>();

// Compute attention output
const attentionOutput = computed<AttentionOutput>(() => {
  // Step 1: Compute Q, K, V
  const Q = matrixMultiply(props.embeddings, props.weights.weightQ);
  const K = matrixMultiply(props.embeddings, props.weights.weightK);
  const V = matrixMultiply(props.embeddings, props.weights.weightV);

  // Step 2: Compute attention scores (Q * K^T)
  const scores = matrixMultiply(Q, transpose(K));

  // Step 3: Scale by sqrt(d_k)
  const d_k = props.weights.weightQ[0].length;
  const scaledScores = scaleMatrix(scores, 1 / Math.sqrt(d_k));

  // Step 4: Apply softmax to get attention weights
  const attentionWeights = softmax(scaledScores);

  // Step 5: Compute output (attention_weights * V)
  const output = matrixMultiply(attentionWeights, V);

  // Emit output to parent
  emit('outputComputed', output);

  return {
    Q,
    K,
    V,
    scores: scaledScores,
    attentionWeights,
    output,
  };
});

const tokenLabels = computed(() =>
  props.tokenLabels.length > 0
    ? props.tokenLabels
    : Array.from({ length: props.embeddings.length }, (_, i) => `T${i}`)
);

const embeddingDimLabels = computed(() =>
  Array.from({ length: props.embeddings[0].length }, (_, i) => `e${i}`)
);

const headDimLabels = computed(() =>
  Array.from({ length: props.weights.weightQ[0].length }, (_, i) => `h${i}`)
);
</script>

<template>
  <div class="attention-layer bg-white p-6 rounded-lg shadow-lg">
    <h2 class="text-xl font-bold mb-4 text-gray-800">Attention Layer</h2>

    <div v-if="showSteps" class="steps space-y-6">
      <!-- Step 1: Input Embeddings -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Input Embeddings</h3>
        <MatrixDisplay
          :matrix="embeddings"
          :row-labels="tokenLabels"
          :col-labels="embeddingDimLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="sm"
        />
      </div>

      <!-- Step 2: QKV Weights -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">QKV Weight Matrices</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MatrixDisplay
            :matrix="weights.weightQ"
            label="Weight Q"
            :row-labels="embeddingDimLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="0.5"
            cell-size="sm"
          />
          <MatrixDisplay
            :matrix="weights.weightK"
            label="Weight K"
            :row-labels="embeddingDimLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="0.5"
            cell-size="sm"
          />
          <MatrixDisplay
            :matrix="weights.weightV"
            label="Weight V"
            :row-labels="embeddingDimLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="0.5"
            cell-size="sm"
          />
        </div>
      </div>

      <!-- Step 3: Q, K, V values -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Q, K, V Values (Embeddings × Weights)</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MatrixDisplay
            :matrix="attentionOutput.Q"
            label="Q = Embeddings × W_Q"
            :row-labels="tokenLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="3"
            cell-size="sm"
          />
          <MatrixDisplay
            :matrix="attentionOutput.K"
            label="K = Embeddings × W_K"
            :row-labels="tokenLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="3"
            cell-size="sm"
          />
          <MatrixDisplay
            :matrix="attentionOutput.V"
            label="V = Embeddings × W_V"
            :row-labels="tokenLabels"
            :col-labels="headDimLabels"
            :precision="2"
            :max-abs-value="3"
            cell-size="sm"
          />
        </div>
      </div>

      <!-- Step 4: Attention Scores and Weights -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Attention Scores (Q × K^T / √d_k)</h3>
        <MatrixDisplay
          :matrix="attentionOutput.scores"
          :row-labels="tokenLabels"
          :col-labels="tokenLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="sm"
        />
      </div>

      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Attention Weights (Softmax of Scores)</h3>
        <MatrixDisplay
          :matrix="attentionOutput.attentionWeights"
          :row-labels="tokenLabels"
          :col-labels="tokenLabels"
          :precision="3"
          :max-abs-value="1.0"
          cell-size="sm"
        />
        <p class="text-sm text-gray-600 mt-2">
          Each row shows how much each token attends to other tokens (sums to 1.0)
        </p>
      </div>

      <!-- Step 5: Output -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Attention Output (Weights × V)</h3>
        <MatrixDisplay
          :matrix="attentionOutput.output"
          :row-labels="tokenLabels"
          :col-labels="headDimLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="sm"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.step {
  padding: 1rem;
  background-color: #f9fafb;
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
}
</style>
