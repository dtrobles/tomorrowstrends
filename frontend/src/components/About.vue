<!-- src/components/About.vue -->
<template>
    <div style="padding: 20px;">
      <h1>About Trending Keywords Map</h1>
      <p>
        This project predicts the next trending keywords based on Google Trends data using machine learning.
      </p>
      <p>
        The homepage features an interactive world map where you can click on a country to see predictions specific to that country.

        I used a random forest colassifier because of limited dataset size, and it works well with tabular data.
      </p>
      <button @click="goBack">Go Back</button>
    </div>
  </template>
  
  <script lang="ts">
  import { defineComponent } from 'vue'
  import { useRouter } from 'vue-router'
  
  export default defineComponent({
    name: 'About',
    setup() {
      const router = useRouter()
      const goBack = () => {
        router.push('/')
      }
      return { goBack }
    }
  })
  </script>
  
  <style scoped>
  /* Additional styling for the About page */
  </style>
  