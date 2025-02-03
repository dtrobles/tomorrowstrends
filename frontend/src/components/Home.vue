<template>
  <div class="map-container">
    <div class="header">
      <h1 class="title">Tomorrow's Trends</h1>
      <div class="date">December 16, 2025</div>
    </div>
    <div class="about-button">
      <button class="about">About</button>
    </div>
  </div>
</template>

<script>
export default {
  name: "HomePage",
};
</script>

<style scoped>
.map-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: #1a1a1a;
  color: white;
  height: 100vh;
  padding: 20px;
}

.header {
  text-align: center;
}

.title {
  font-size: 2.5rem;
  font-weight: bold;
  color: orange;
  margin: 10px 0;
}

.date {
  font-size: 1rem;
  color: #e0e0e0;
  background-color: #333;
  padding: 5px 10px;
  border-radius: 5px;
}

.map {
  margin: 20px 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #333;
  border-radius: 10px;
  padding: 20px;
  width: 80%;
  height: 60%;
}

.map-placeholder {
  color: #888;
  font-size: 1.5rem;
  text-align: center;
  line-height: 60%;
}

.about-button {
  position: fixed;
  top: 20px;
  left: 20px;
}

.about {
  font-size: 1rem;
  padding: 10px 20px;
  background-color: #444;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.about:hover {
  background-color: #666;
}
</style>
