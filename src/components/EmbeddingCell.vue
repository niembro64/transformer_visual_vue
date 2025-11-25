<script setup lang="ts">
import { computed } from 'vue';

interface Props {
  value: number;
  maxAbsValue?: number;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  precision?: number;
  isHighlighted?: boolean;
  isSelectable?: boolean;
  isSelected?: boolean;
  showValue?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  maxAbsValue: 0.5,
  size: 'sm',
  precision: 2,
  isHighlighted: false,
  isSelectable: false,
  isSelected: false,
  showValue: true,
});

const emit = defineEmits<{
  click: [];
}>();

// Calculate color based on arctangent mapping (like React version)
const cellStyle = computed(() => {
  const { value, maxAbsValue } = props;

  // Use arctangent for smooth, non-linear color mapping
  const steepness = maxAbsValue === 1.0 ? 2.0 : 0.3;
  const normalizedValue = Math.atan(value * steepness) / (Math.PI / 2);

  // Base colors - make selectable items slightly more vibrant
  const neutralColor = props.isSelectable ? [230, 240, 230] : [240, 240, 240];
  const maxBlueColor = [20, 20, 255];
  const maxRedColor = [255, 20, 20];

  let red: number, green: number, blue: number, textColor: string;

  if (normalizedValue < 0) {
    // Negative: interpolate to red
    const t = -normalizedValue;
    red = neutralColor[0] * (1 - t) + maxRedColor[0] * t;
    green = neutralColor[1] * (1 - t) + maxRedColor[1] * t;
    blue = neutralColor[2] * (1 - t) + maxRedColor[2] * t;
    textColor = normalizedValue < -0.7 ? 'white' : 'black';
  } else if (normalizedValue > 0) {
    // Positive: interpolate to blue
    const t = normalizedValue;
    red = neutralColor[0] * (1 - t) + maxBlueColor[0] * t;
    green = neutralColor[1] * (1 - t) + maxBlueColor[1] * t;
    blue = neutralColor[2] * (1 - t) + maxBlueColor[2] * t;
    textColor = normalizedValue > 0.7 ? 'white' : 'black';
  } else {
    // Zero: neutral gray
    [red, green, blue] = neutralColor;
    textColor = 'black';
  }

  return {
    backgroundColor: `rgb(${Math.round(red)}, ${Math.round(green)}, ${Math.round(blue)})`,
    color: textColor,
  };
});

// Format value in scientific notation
const formattedValue = computed(() => {
  const { value, precision } = props;

  if (value === 0) {
    return { coefficient: '0.00', exponent: 'e+0' };
  }

  const scientificNotation = value.toExponential(precision);
  const [coef, exp] = scientificNotation.split('e');

  // Add + prefix for positive values
  let formattedCoef = value > 0 ? `+${coef}` : coef;

  // Ensure coefficient is 5 characters for consistency
  if (formattedCoef.length < 5) {
    const parts = formattedCoef.split('.');
    if (parts.length === 2) {
      while (formattedCoef.length < 5) {
        parts[1] += '0';
        formattedCoef = parts[0] + '.' + parts[1];
      }
    }
  } else if (formattedCoef.length > 5) {
    const sign = formattedCoef[0];
    const firstDigit = formattedCoef[1];
    formattedCoef = sign + firstDigit + '.' + formattedCoef.slice(3, 5);
  }

  return {
    coefficient: formattedCoef,
    exponent: `e${exp}`,
  };
});

// Size-based styling
const sizeClasses = computed(() => {
  // Compact sizes when not showing values
  const compactSizeMap = {
    xs: {
      container: 'w-[0.8rem] h-[0.8rem]',
      coefficient: 'text-[0.45rem]',
      exponent: 'text-[0.45rem]',
    },
    sm: {
      container: 'w-[1.0rem] h-[1.0rem]',
      coefficient: 'text-[0.5rem]',
      exponent: 'text-[0.5rem]',
    },
    md: {
      container: 'w-[1.2rem] h-[1.2rem]',
      coefficient: 'text-[0.55rem]',
      exponent: 'text-[0.55rem]',
    },
    lg: {
      container: 'w-[1.5rem] h-[1.5rem]',
      coefficient: 'text-[0.6rem]',
      exponent: 'text-[0.6rem]',
    },
  };

  // Full sizes when showing values
  const fullSizeMap = {
    xs: {
      container: 'w-[1.7rem] h-[1.5rem]',
      coefficient: 'text-[0.45rem]',
      exponent: 'text-[0.45rem]',
    },
    sm: {
      container: 'w-[2.1rem] h-[1.9rem]',
      coefficient: 'text-[0.5rem]',
      exponent: 'text-[0.5rem]',
    },
    md: {
      container: 'w-[2.7rem] h-[2.4rem]',
      coefficient: 'text-[0.55rem]',
      exponent: 'text-[0.55rem]',
    },
    lg: {
      container: 'w-[3.4rem] h-[3.0rem]',
      coefficient: 'text-[0.6rem]',
      exponent: 'text-[0.6rem]',
    },
  };

  const sizeMap = props.showValue ? fullSizeMap : compactSizeMap;
  return sizeMap[props.size];
});

function handleClick() {
  if (props.isSelectable) {
    emit('click');
  }
}
</script>

<template>
  <div
    class="font-mono flex flex-col justify-evenly items-center py-0.5 px-0.5 relative"
    :class="[
      sizeClasses.container,
      isSelectable ? 'cursor-pointer hover:opacity-80' : '',
      isSelected ? 'ring-2 ring-fuchsia-500 z-10 rounded-full' : 'z-0',
    ]"
    :style="cellStyle"
    @click="handleClick"
  >
    <div
      v-if="showValue"
      :class="sizeClasses.coefficient"
      class="flex justify-center items-center w-full tracking-tight leading-none"
    >
      <span class="text-center">{{ formattedValue.coefficient }}</span>
    </div>
    <div
      v-if="showValue"
      :class="sizeClasses.exponent"
      class="flex justify-center items-center w-full tracking-tight leading-none"
    >
      <span class="text-center">{{ formattedValue.exponent }}</span>
    </div>
  </div>
</template>
