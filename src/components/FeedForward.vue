<script setup lang="ts">
import { computed } from 'vue';
import MatrixDisplay from './MatrixDisplay.vue';
import type { MLPWeights, MLPOutput } from '../types';
import { matrixMultiply, addBias, applyFn, relu } from '../utils/matrixOperations';

interface Props {
  inputs: number[][];
  weights: MLPWeights;
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

// Compute MLP output
const mlpOutput = computed<MLPOutput>(() => {
  // Step 1: First linear layer (inputs × W1 + b1)
  const hidden = addBias(matrixMultiply(props.inputs, props.weights.W1), props.weights.b1);

  // Step 2: Apply ReLU activation
  const activated = applyFn(hidden, relu);

  // Step 3: Second linear layer (activated × W2 + b2)
  const output = addBias(matrixMultiply(activated, props.weights.W2), props.weights.b2);

  // Emit output to parent
  emit('outputComputed', output);

  return {
    hidden,
    activated,
    output,
  };
});

const tokenLabels = computed(() =>
  props.tokenLabels.length > 0
    ? props.tokenLabels
    : Array.from({ length: props.inputs.length }, (_, i) => `T${i}`)
);

const inputDimLabels = computed(() =>
  Array.from({ length: props.inputs[0].length }, (_, i) => `d${i}`)
);

const hiddenDimLabels = computed(() =>
  Array.from({ length: props.weights.W1[0].length }, (_, i) => `h${i}`)
);

const outputDimLabels = computed(() =>
  Array.from({ length: props.weights.W2[0].length }, (_, i) => `d${i}`)
);
</script>

<template>
  <div class="feedforward-layer bg-white p-6 rounded-lg shadow-lg">
    <h2 class="text-xl font-bold mb-4 text-gray-800">Feed-Forward (MLP) Layer</h2>

    <div v-if="showSteps" class="steps space-y-6">
      <!-- Step 1: Input from Attention -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Input (from Attention)</h3>
        <MatrixDisplay
          :matrix="inputs"
          :row-labels="tokenLabels"
          :col-labels="inputDimLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="sm"
        />
      </div>

      <!-- Step 2: First Layer Weights -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">First Layer: W1 and b1</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <MatrixDisplay
            :matrix="weights.W1"
            label="W1 (Input → Hidden)"
            :row-labels="inputDimLabels"
            :col-labels="hiddenDimLabels"
            :precision="2"
            :max-abs-value="0.5"
            cell-size="xs"
          />
          <div>
            <p class="text-sm font-semibold mb-2 text-gray-600">b1 (Bias)</p>
            <div class="flex flex-wrap gap-2">
              <span
                v-for="(val, idx) in weights.b1"
                :key="`b1-${idx}`"
                class="px-3 py-1 bg-gray-100 rounded text-xs font-mono"
              >
                {{ val.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 3: Hidden Layer (before activation) -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Hidden Layer (Input × W1 + b1)</h3>
        <MatrixDisplay
          :matrix="mlpOutput.hidden"
          :row-labels="tokenLabels"
          :col-labels="hiddenDimLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="xs"
        />
      </div>

      <!-- Step 4: After ReLU Activation -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">After ReLU Activation</h3>
        <MatrixDisplay
          :matrix="mlpOutput.activated"
          :row-labels="tokenLabels"
          :col-labels="hiddenDimLabels"
          :precision="2"
          :max-abs-value="3"
          cell-size="xs"
        />
        <p class="text-sm text-gray-600 mt-2">ReLU(x) = max(0, x) - negative values become zero</p>
      </div>

      <!-- Step 5: Second Layer Weights -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">Second Layer: W2 and b2</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <MatrixDisplay
            :matrix="weights.W2"
            label="W2 (Hidden → Output)"
            :row-labels="hiddenDimLabels"
            :col-labels="outputDimLabels"
            :precision="2"
            :max-abs-value="0.5"
            cell-size="xs"
          />
          <div>
            <p class="text-sm font-semibold mb-2 text-gray-600">b2 (Bias)</p>
            <div class="flex flex-wrap gap-2">
              <span
                v-for="(val, idx) in weights.b2"
                :key="`b2-${idx}`"
                class="px-3 py-1 bg-gray-100 rounded text-xs font-mono"
              >
                {{ val.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 6: Final Output -->
      <div class="step">
        <h3 class="text-lg font-semibold mb-2 text-gray-700">MLP Output (Activated × W2 + b2)</h3>
        <MatrixDisplay
          :matrix="mlpOutput.output"
          :row-labels="tokenLabels"
          :col-labels="outputDimLabels"
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
