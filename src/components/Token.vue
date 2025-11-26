<script setup lang="ts">
interface Props {
  text: string;
  embedding: number[];
  index: number;
  showEmbedding?: boolean;
  precision?: number;
}

withDefaults(defineProps<Props>(), {
  showEmbedding: false,
  precision: 2,
});
</script>

<template>
  <div class="token-container inline-block m-2">
    <div
      class="token-text px-4 py-2 bg-blue-500 text-white rounded-lg font-semibold text-center shadow-md"
    >
      {{ text }}
    </div>
    <div v-if="showEmbedding" class="token-embedding mt-2 text-xs">
      <div class="text-gray-600 font-semibold mb-1">Embedding:</div>
      <div class="flex flex-wrap gap-1">
        <span
          v-for="(val, idx) in embedding"
          :key="`emb-${index}-${idx}`"
          class="px-2 py-1 bg-gray-100 rounded text-gray-700 font-mono"
        >
          {{ val.toFixed(precision) }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.token-container {
  transition: transform 0.2s;
}

.token-container:hover {
  transform: scale(1.05);
}
</style>
