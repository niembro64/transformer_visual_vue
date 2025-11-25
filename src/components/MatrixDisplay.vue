<script setup lang="ts">
import { computed } from 'vue';
import EmbeddingCell from './EmbeddingCell.vue';

interface Props {
  matrix: number[][];
  label?: string;
  rowLabels?: string[];
  colLabels?: string[];
  highlightRow?: number;
  highlightCol?: number;
  precision?: number;
  maxAbsValue?: number;
  cellSize?: 'xs' | 'sm' | 'md' | 'lg';
  selectable?: boolean;
  selectedRow?: number;
  selectedCol?: number;
  showValues?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  label: '',
  rowLabels: () => [],
  colLabels: () => [],
  highlightRow: -1,
  highlightCol: -1,
  precision: 2,
  maxAbsValue: 0.5,
  cellSize: 'sm',
  selectable: false,
  selectedRow: -1,
  selectedCol: -1,
  showValues: true,
});

const emit = defineEmits<{
  cellClick: [row: number, col: number];
}>();

const showRowLabels = computed(() => props.rowLabels.length === props.matrix.length);
const showColLabels = computed(() => props.colLabels.length > 0 && props.matrix.length > 0 && props.colLabels.length === props.matrix[0].length);

// Calculate grid template
const gridTemplateColumns = computed(() => {
  const cols = props.matrix[0]?.length || 0;
  if (showRowLabels.value) {
    return `1.5rem repeat(${cols}, auto)`;
  }
  return `repeat(${cols}, auto)`;
});

const gridTemplateRows = computed(() => {
  const rows = props.matrix.length;
  if (showColLabels.value) {
    return `0.7rem repeat(${rows}, auto)`;
  }
  return `repeat(${rows}, auto)`;
});

const isHighlighted = (rowIdx: number, colIdx: number): boolean => {
  return rowIdx === props.highlightRow || colIdx === props.highlightCol;
};

const isSelected = (rowIdx: number, colIdx: number): boolean => {
  return rowIdx === props.selectedRow && colIdx === props.selectedCol;
};

function handleCellClick(row: number, col: number) {
  if (props.selectable) {
    emit('cellClick', row, col);
  }
}
</script>

<template>
  <div class="matrix-display flex flex-col items-center justify-center mb-4">
    <h3 v-if="label" class="text-[0.6rem] font-semibold mb-1 text-gray-700 text-center">
      {{ label }}
    </h3>
    <div
      class="grid mx-auto"
      :style="{
        gridTemplateColumns,
        gridTemplateRows,
        gap: '0',
        justifyItems: 'center',
        alignItems: 'center',
      }"
    >
      <!-- Empty cell in top-left corner when both labels shown -->
      <div v-if="showRowLabels && showColLabels" class="col-start-1 row-start-1"></div>

      <!-- Column labels -->
      <template v-if="showColLabels">
        <div
          v-for="(label, idx) in colLabels"
          :key="`col-${idx}`"
          class="text-center text-[0.5rem] text-gray-600 font-medium flex items-center justify-center"
          :style="{
            gridColumn: showRowLabels ? idx + 2 : idx + 1,
            gridRow: 1,
          }"
        >
          {{ label }}
        </div>
      </template>

      <!-- Matrix rows -->
      <template v-for="(row, rowIdx) in matrix" :key="`row-${rowIdx}`">
        <!-- Row label -->
        <div
          v-if="showRowLabels"
          class="text-center text-[0.5rem] text-gray-600 font-medium flex items-center justify-center"
          :style="{
            gridColumn: 1,
            gridRow: showColLabels ? rowIdx + 2 : rowIdx + 1,
          }"
        >
          {{ rowLabels[rowIdx] }}
        </div>

        <!-- Matrix cells -->
        <div
          v-for="(value, colIdx) in row"
          :key="`cell-${rowIdx}-${colIdx}`"
          class="flex items-center justify-center"
          :style="{
            gridColumn: showRowLabels ? colIdx + 2 : colIdx + 1,
            gridRow: showColLabels ? rowIdx + 2 : rowIdx + 1,
          }"
        >
          <EmbeddingCell
            :value="value"
            :max-abs-value="maxAbsValue"
            :size="cellSize"
            :precision="precision"
            :is-highlighted="isHighlighted(rowIdx, colIdx)"
            :is-selectable="selectable"
            :is-selected="isSelected(rowIdx, colIdx)"
            :show-value="showValues"
            @click="handleCellClick(rowIdx, colIdx)"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.matrix-display {
  overflow-x: auto;
  max-width: 100%;
  width: fit-content;
  margin: 0 auto;
}
</style>
