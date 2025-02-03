<!-- src/App.vue -->
<template>
  <transition name="fade" mode="out-in">
    <router-view />
  </transition>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
export default defineComponent({
  name: 'App'
})
</script>

<style>
/* Global styles can go here if needed */
</style>
