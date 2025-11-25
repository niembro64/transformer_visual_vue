<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue';
import MatrixDisplay from './components/MatrixDisplay.vue';
import type { AttentionWeights } from './types';
import {
  generateSampleEmbeddings,
  randomMatrix,
  matrixMultiply,
  transpose,
  softmax,
  scaleMatrix,
} from './utils/matrixOperations';

// Configuration - matching mobile view from React
const NUM_TOKENS = 4;
const EMBEDDING_DIM = 4;
const ATTENTION_HEAD_DIM = 4; // Same as embedding dim to preserve shape

// Token vocabulary
const vocabulary = ['ai', 'bot', 'go', 'brr'];

// Generate initial embeddings
const embeddings = ref<number[][]>(generateSampleEmbeddings(NUM_TOKENS, EMBEDDING_DIM));

// Initialize attention weights
const attentionWeights = ref<AttentionWeights>({
  weightQ: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
  weightK: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
  weightV: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
});

// Selected cell state
type SelectedCell = {
  type: 'embeddings' | 'weightQ' | 'weightK' | 'weightV';
  row: number;
  col: number;
} | null;

// Default to editing d2 of "cat" token (row 1, col 1)
const selectedCell = ref<SelectedCell>({
  type: 'embeddings',
  row: 1,
  col: 1,
});

// Toggle for showing cell values
const showCellValues = ref(false);

// Toggle for showing labels
const showLabels = ref(false);

// Auto-wiggle state
const autoWiggle = ref(true);

// Compute Q, K, V
const Q = computed(() => matrixMultiply(embeddings.value, attentionWeights.value.weightQ));
const K = computed(() => matrixMultiply(embeddings.value, attentionWeights.value.weightK));
const V = computed(() => matrixMultiply(embeddings.value, attentionWeights.value.weightV));

// Compute attention scores (pre-softmax)
const attentionScores = computed(() => {
  const scores = matrixMultiply(Q.value, transpose(K.value));
  return scaleMatrix(scores, 1 / Math.sqrt(ATTENTION_HEAD_DIM));
});

// Apply softmax to get attention weights
const attentionWeightsMatrix = computed(() => softmax(attentionScores.value));

// Compute attention output
const attentionOutput = computed(() => matrixMultiply(attentionWeightsMatrix.value, V.value));

// Get current selected value
const selectedValue = computed(() => {
  if (!selectedCell.value) return 0;
  const { type, row, col } = selectedCell.value;
  if (type === 'embeddings') return embeddings.value[row][col];
  if (type === 'weightQ') return attentionWeights.value.weightQ[row][col];
  if (type === 'weightK') return attentionWeights.value.weightK[row][col];
  if (type === 'weightV') return attentionWeights.value.weightV[row][col];
  return 0;
});

// Format value in scientific notation with explicit signs
function formatScientific(value: number): string {
  if (value === 0) return '+0.00e+0';

  const scientificNotation = value.toExponential(2);
  const [coef, exp] = scientificNotation.split('e');

  // Add + prefix for positive coefficient
  const formattedCoef = value >= 0 ? `+${coef}` : coef;

  // Ensure exponent has explicit sign
  const expNum = parseInt(exp);
  const formattedExp = expNum >= 0 ? `e+${expNum}` : `e${expNum}`;

  return `${formattedCoef}${formattedExp}`;
}

// Update selected value
function updateSelectedValue(newValue: number) {
  if (!selectedCell.value) return;
  const { type, row, col } = selectedCell.value;

  if (type === 'embeddings') {
    const newEmbeddings = embeddings.value.map((r, i) =>
      i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
    );
    embeddings.value = newEmbeddings;
  } else if (type === 'weightQ') {
    const newWeights = attentionWeights.value.weightQ.map((r, i) =>
      i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
    );
    attentionWeights.value = { ...attentionWeights.value, weightQ: newWeights };
  } else if (type === 'weightK') {
    const newWeights = attentionWeights.value.weightK.map((r, i) =>
      i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
    );
    attentionWeights.value = { ...attentionWeights.value, weightK: newWeights };
  } else if (type === 'weightV') {
    const newWeights = attentionWeights.value.weightV.map((r, i) =>
      i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
    );
    attentionWeights.value = { ...attentionWeights.value, weightV: newWeights };
  }
}

// Handle cell click
function handleCellClick(type: 'embeddings' | 'weightQ' | 'weightK' | 'weightV', row: number, col: number) {
  if (selectedCell.value?.type === type && selectedCell.value.row === row && selectedCell.value.col === col) {
    selectedCell.value = null;
  } else {
    selectedCell.value = { type, row, col };
  }
}

// Randomize everything
function randomizeAll() {
  embeddings.value = generateSampleEmbeddings(NUM_TOKENS, EMBEDDING_DIM);
  attentionWeights.value = {
    weightQ: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
    weightK: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
    weightV: randomMatrix(EMBEDDING_DIM, ATTENTION_HEAD_DIM, 1.0),
  };
}

// Token labels
const tokenLabels = computed(() => vocabulary.slice(0, NUM_TOKENS));
const embDimLabels = Array.from({ length: EMBEDDING_DIM }, (_, i) => `d${i}`);
const headDimLabels = Array.from({ length: ATTENTION_HEAD_DIM }, (_, i) => `h${i}`);

// Auto-wiggle implementation
let wiggleInterval: number | null = null;
let wiggleStartTime = Date.now();
const wiggleAmplitude = 4.0; // Range of wiggle (-4 to +4)
const wiggleFrequency = 0.5; // Hz (oscillations per second)

function startWiggle() {
  if (wiggleInterval !== null) return;

  wiggleStartTime = Date.now();

  wiggleInterval = window.setInterval(() => {
    if (!autoWiggle.value || !selectedCell.value) {
      stopWiggle();
      return;
    }

    const elapsed = (Date.now() - wiggleStartTime) / 1000; // seconds
    const angle = 2 * Math.PI * wiggleFrequency * elapsed;
    const wiggleValue = Math.sin(angle) * wiggleAmplitude;

    updateSelectedValue(wiggleValue);
  }, 16); // ~60fps
}

function stopWiggle() {
  if (wiggleInterval !== null) {
    clearInterval(wiggleInterval);
    wiggleInterval = null;
  }
}

// Watch autoWiggle to start/stop animation
watch(autoWiggle, (newValue) => {
  if (newValue) {
    startWiggle();
  } else {
    stopWiggle();
  }
});

// Start wiggle on mount if autoWiggle is true
onMounted(() => {
  if (autoWiggle.value) {
    startWiggle();
  }
});

// Clean up on unmount
onUnmounted(() => {
  stopWiggle();
});
</script>

<template>
  <div class="app min-h-screen bg-gray-50 py-2 px-1 overflow-x-hidden">
    <div class="container mx-auto max-w-5xl overflow-x-hidden">
      <!-- Header -->
      <header class="mb-2 text-center">
        <h1 class="text-xl sm:text-2xl font-bold text-gray-800 mb-1">Transformer Attention</h1>
        <p class="text-gray-600 text-[0.65rem] sm:text-xs">Single-Head Self-Attention Mechanism</p>
        <p class="text-[0.6rem] text-gray-500 mt-0.5">
          {{ NUM_TOKENS }} tokens × {{ EMBEDDING_DIM }} dimensions
        </p>
      </header>

      <!-- Controls -->
      <div class="controls mb-3 flex justify-center gap-2 flex-wrap">
        <button
          @click="randomizeAll"
          class="px-3 sm:px-4 py-2 text-xs sm:text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md font-semibold"
        >
          Randomize
        </button>
        <button
          @click="showLabels = !showLabels"
          :class="showLabels ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-400 hover:bg-gray-500'"
          class="px-3 sm:px-4 py-2 text-xs sm:text-sm text-white rounded-lg transition-colors shadow-md font-semibold"
        >
          {{ showLabels ? 'Hide' : 'Show' }} Labels
        </button>
      </div>

      <!-- Value Editor - Fixed sticky footer -->
      <div v-if="selectedCell" class="fixed bottom-0 left-0 right-0 bg-fuchsia-50 border-t-2 border-fuchsia-500 shadow-2xl z-50" style="padding: 0.5rem; padding-bottom: max(0.5rem, env(safe-area-inset-bottom));">
        <div class="flex flex-col sm:flex-row items-center justify-center gap-2 mb-2">
          <h3 class="text-xs font-semibold text-fuchsia-700 text-center">
            Editing: {{ selectedCell.type }}[{{ selectedCell.row }}, {{ selectedCell.col }}]
          </h3>
          <button
            @click="autoWiggle = !autoWiggle"
            :class="autoWiggle ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-400 hover:bg-gray-500'"
            class="px-2 py-1 text-xs text-white rounded-lg transition-colors shadow-md font-semibold whitespace-nowrap"
          >
            {{ autoWiggle ? 'Auto' : 'Manual' }} Wiggle
          </button>
        </div>
        <div class="flex items-center gap-2 sm:gap-4 justify-center">
          <span class="text-xs text-gray-600">-10</span>
          <input
            type="range"
            min="-10"
            max="10"
            step="0.01"
            :value="selectedValue"
            @input="updateSelectedValue(parseFloat(($event.target as HTMLInputElement).value))"
            :disabled="autoWiggle"
            class="w-32 h-2 bg-gray-200 rounded-lg cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          />
          <span class="text-xs text-gray-600">+10</span>
          <span class="text-xs sm:text-sm font-mono font-bold text-fuchsia-700">{{ formatScientific(selectedValue) }}</span>
        </div>
      </div>

      <!-- Main Content -->
      <div class="flex flex-col gap-3 p-1 sm:p-2 bg-white rounded-lg shadow-lg mb-24">
        <!-- Step 1: Input Embeddings and Weight Matrices -->
        <div class="flex flex-col gap-2">
          <h4 class="text-sm font-semibold text-center text-blue-600">
            Step 1: Input Embeddings & Weight Matrices
          </h4>
          <div class="flex justify-center items-start gap-1 overflow-x-auto">
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">Input</h5>
              <MatrixDisplay
                :matrix="embeddings"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? embDimLabels : []"
                :max-abs-value="3"
                cell-size="xs"
                :precision="2"
                :selectable="true"
                :selected-row="selectedCell?.type === 'embeddings' ? selectedCell.row : -1"
                :selected-col="selectedCell?.type === 'embeddings' ? selectedCell.col : -1"
                :show-values="showCellValues"
                @cell-click="(row, col) => handleCellClick('embeddings', row, col)"
              />
            </div>
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">W<sup>Q</sup></h5>
              <MatrixDisplay
                :matrix="attentionWeights.weightQ"
                :row-labels="showLabels ? embDimLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.5"
                cell-size="xs"
                :precision="2"
                :selectable="true"
                :selected-row="selectedCell?.type === 'weightQ' ? selectedCell.row : -1"
                :selected-col="selectedCell?.type === 'weightQ' ? selectedCell.col : -1"
                :show-values="showCellValues"
                @cell-click="(row, col) => handleCellClick('weightQ', row, col)"
              />
            </div>
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">W<sup>K</sup></h5>
              <MatrixDisplay
                :matrix="attentionWeights.weightK"
                :row-labels="showLabels ? embDimLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.5"
                cell-size="xs"
                :precision="2"
                :selectable="true"
                :selected-row="selectedCell?.type === 'weightK' ? selectedCell.row : -1"
                :selected-col="selectedCell?.type === 'weightK' ? selectedCell.col : -1"
                :show-values="showCellValues"
                @cell-click="(row, col) => handleCellClick('weightK', row, col)"
              />
            </div>
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">W<sup>V</sup></h5>
              <MatrixDisplay
                :matrix="attentionWeights.weightV"
                :row-labels="showLabels ? embDimLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.5"
                cell-size="xs"
                :precision="2"
                :selectable="true"
                :selected-row="selectedCell?.type === 'weightV' ? selectedCell.row : -1"
                :selected-col="selectedCell?.type === 'weightV' ? selectedCell.col : -1"
                :show-values="showCellValues"
                @cell-click="(row, col) => handleCellClick('weightV', row, col)"
              />
            </div>
          </div>
        </div>

        <!-- Step 2: Q, K, V Vectors -->
        <div class="flex flex-col gap-2">
          <h4 class="text-sm font-semibold text-center text-blue-600">
            Step 2: Q, K, V Vectors
          </h4>
          <div class="flex justify-center items-start gap-1 overflow-x-auto">
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">Q</h5>
              <MatrixDisplay
                :matrix="Q"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.3"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">K</h5>
              <MatrixDisplay
                :matrix="K"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.3"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
            <div class="flex-shrink-0">
              <h5 class="text-[0.6rem] text-center mb-1 font-semibold text-gray-700">V</h5>
              <MatrixDisplay
                :matrix="V"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.3"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
          </div>
        </div>

        <!-- Step 3a: Attention Scores (Pre-Softmax) -->
        <div class="flex flex-col gap-2">
          <h4 class="text-sm font-semibold text-center text-blue-600">
            Step 3a: Attention Scores = Q × K<sup>T</sup> / √d<sub>k</sub>
          </h4>
          <div class="flex justify-center items-start">
            <MatrixDisplay
              :matrix="attentionScores"
              :row-labels="showLabels ? tokenLabels : []"
              :col-labels="showLabels ? tokenLabels : []"
              :max-abs-value="3"
              cell-size="sm"
              :precision="2"
              :show-values="showCellValues"
            />
          </div>
          <p class="text-xs text-gray-600 text-center">
            Raw scores showing similarity between Q and K vectors
          </p>
        </div>

        <!-- Step 3b: Attention Weights (Post-Softmax) -->
        <div class="flex flex-col gap-2">
          <h4 class="text-sm font-semibold text-center text-blue-600">
            Step 3b: Attention Weights = softmax(Scores)
          </h4>
          <div class="flex justify-center">
            <MatrixDisplay
              :matrix="attentionWeightsMatrix"
              :row-labels="showLabels ? tokenLabels : []"
              :col-labels="showLabels ? tokenLabels : []"
              :max-abs-value="1.0"
              cell-size="sm"
              :precision="3"
              :show-values="showCellValues"
            />
          </div>
          <p class="text-xs text-gray-600 text-center">
            Each row shows how much each token "attends to" others (sums to 1.0)
          </p>
        </div>

        <!-- Step 4: Output Computation -->
        <div class="flex flex-col gap-2">
          <h4 class="text-sm font-semibold text-center text-blue-600">
            Step 4: Output = Attention Weights × V
          </h4>
          <div class="flex justify-center items-start gap-2">
            <!-- Attention Weights on the left -->
            <div class="flex flex-col items-center">
              <div class="text-center mb-1 h-4 flex items-center justify-center">
                <span class="text-xs font-semibold text-gray-600">Attn →</span>
              </div>
              <MatrixDisplay
                :matrix="attentionWeightsMatrix"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? tokenLabels : []"
                :max-abs-value="1.0"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
            <!-- V in center -->
            <div class="flex flex-col">
              <div class="text-center mb-1 h-4 flex items-center justify-center">
                <span class="text-xs font-semibold text-gray-600">V ↓</span>
              </div>
              <MatrixDisplay
                :matrix="V"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? headDimLabels : []"
                :max-abs-value="0.3"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
            <!-- Output on right -->
            <div class="flex flex-col items-center">
              <div class="text-center mb-1 h-4 flex items-center justify-center">
                <span class="text-xs font-semibold text-gray-600">= Output</span>
              </div>
              <MatrixDisplay
                :matrix="attentionOutput"
                :row-labels="showLabels ? tokenLabels : []"
                :col-labels="showLabels ? embDimLabels : []"
                :max-abs-value="3"
                cell-size="xs"
                :precision="2"
                :show-values="showCellValues"
              />
            </div>
          </div>
        </div>

        <!-- Final Summary -->
        <div class="text-center p-3 bg-green-50 rounded-lg text-sm text-gray-700 font-semibold">
          ✓ Shape preserved: Input {{ NUM_TOKENS }}×{{ EMBEDDING_DIM }} → Output {{ NUM_TOKENS }}×{{ EMBEDDING_DIM }}
        </div>
      </div>

      <!-- Footer -->
      <footer class="mt-4 text-center text-gray-500 text-[0.6rem]">
        <p>Vue TypeScript Transformer Visualization</p>
        <p class="mt-0.5">
          Based on "Attention Is All You Need" (Vaswani et al., 2017)
        </p>
      </footer>
    </div>
  </div>
</template>

<style>
/* Global styles to prevent horizontal scroll */
html, body {
  overflow-x: hidden;
  width: 100%;
  position: relative;
}
</style>

<style scoped>
.app {
  font-family: system-ui, -apple-system, sans-serif;
  width: 100%;
  max-width: 100vw;
}

button {
  transition: all 0.2s ease-in-out;
}

button:active {
  transform: scale(0.98);
}

input[type="range"] {
  touch-action: pan-x;
  -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #d946ef;
  cursor: pointer;
}

input[type="range"]::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #d946ef;
  cursor: pointer;
  border: none;
}
</style>
