<template>
  <div class="map-container">
    <div class="header">
      <router-link to="/about">
        <button class="about">About</button>
      </router-link>
      <h1 class="title">Tomorrow's Trends</h1>
      <a href="https://github.com/dtrobles/tomorrowstrends" target="_blank" rel="noopener noreferrer">
        <img src="/src/assets/images/github-icon.png" alt="GitHub" class="github-icon" />
      </a>
    </div>
    <div class="date">December 16, 2025</div>
    <!-- Insert the scrollable, zoomable world map here -->
    <WorldMap />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import '../assets/styles/home.css';
import WorldMap from './WorldMap.vue';

export default defineComponent({
  name: "HomePage",
  components: {
    WorldMap,
  },
});
</script>
