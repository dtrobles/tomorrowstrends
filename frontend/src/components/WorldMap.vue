<template>
  <div class="world-map">
    <!-- Map container -->
    <div id="map" ref="mapContainer"></div>

    <!-- Modal overlay: only visible when a country is selected -->
    <div v-if="selectedCountry" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <!-- Title & Horizontal Rule -->
        <h2 class="modalTitle">News Prediction</h2>
        <hr />

        <!-- Country and Date Dropdown -->
        <p class="country-date-line">
          {{ selectedCountry }} |
          <select v-model="selectedDateOption" class="date-dropdown">
            <option v-for="opt in dateOptions" :key="opt.value" :value="opt.value">
              {{ opt.label }}
            </option>
          </select>
        </p>

        <!-- Articles Section (3-Bar Layout) -->
        <div class="news-articles">
          <div class="prediction-section">
            <div class="bar-container">
              <div 
                v-for="(trend, index) in displayedArticles" 
                :key="index" 
                class="trend-bar"
              >
                <img 
                  :src="trend.imageUrl" 
                  alt="Article Image" 
                  class="trend-image"
                />
                <div class="trend-content">
                  <h4>{{ trend.title }}</h4>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                </div>
              </div>
            </div>

            <!-- Pagination Buttons -->
            <div class="pagination-buttons">
              <button 
                class="prev-button" 
                @click.stop="prevPage" 
                :disabled="pageIndex === 0"
              >
                Previous
              </button>
              <button 
                class="next-button" 
                @click.stop="nextPage" 
                :disabled="pageIndex >= totalPages - 1"
              >
                Next
              </button>
            </div>
          </div>
        </div>

        <button class="close-modal" @click="closeModal">Close</button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onBeforeUnmount, watch, computed } from 'vue';
import mapboxgl from 'mapbox-gl';
import throttle from 'lodash.throttle';
import '../assets/styles/worldMap.css';
import 'mapbox-gl/dist/mapbox-gl.css';

export default defineComponent({
  name: 'WorldMap',
  setup() {
    const mapContainer = ref<HTMLElement | null>(null);
    const map = ref<mapboxgl.Map | null>(null);
    const selectedCountry = ref<string | null>(null);

    // Updated predictions: now each article is an object with a title and an imageUrl
    const predictions = ref({
      tomorrow: [] as { title: string, imageUrl: string }[],
      threeDays: [] as { title: string, imageUrl: string }[],
      fiveDays: [] as { title: string, imageUrl: string }[],
    });

    // -------------------------------------------
    // DATE DROPDOWN & PAGINATION
    // -------------------------------------------
    const dateOptions = ref([
      { value: 1, label: formatDatePlusDays(1) },
      { value: 3, label: formatDatePlusDays(3) },
      { value: 5, label: formatDatePlusDays(5) },
    ]);
    const selectedDateOption = ref<number>(1);
    const pageIndex = ref(0);
    const trendsPerPage = 3;

    const selectedArticles = computed(() => {
      if (selectedDateOption.value === 1) {
        return predictions.value.tomorrow;
      } else if (selectedDateOption.value === 3) {
        return predictions.value.threeDays;
      } else {
        return predictions.value.fiveDays;
      }
    });

    const displayedArticles = computed(() => {
      const start = pageIndex.value * trendsPerPage;
      return selectedArticles.value.slice(start, start + trendsPerPage);
    });

    const totalPages = computed(() => {
      return Math.ceil(selectedArticles.value.length / trendsPerPage);
    });

    const nextPage = () => {
      if (pageIndex.value < totalPages.value - 1) {
        pageIndex.value++;
      }
    };

    const prevPage = () => {
      if (pageIndex.value > 0) {
        pageIndex.value--;
      }
    };

    watch(selectedDateOption, () => {
      pageIndex.value = 0;
    });

    function formatDatePlusDays(days: number) {
      const d = new Date();
      d.setDate(d.getDate() + days);
      return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric'
      });
    }
    // -------------------------------------------

    let hoveredStateId: number | string | null = null;
    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;

    onMounted(() => {
      if (!mapContainer.value) return;
      map.value = new mapboxgl.Map({
        style: 'mapbox://styles/notaspect0/cm6yfnj5a00oa01sb3y1pfflk',
        container: mapContainer.value,
        center: [0, 20],
        zoom: 2,
        maxZoom: 5,
        minZoom: 1,
      });

      map.value.on('load', () => {
        map.value!.addSource('countries', {
          type: 'vector',
          url: 'http://localhost:8080/data/output_hi.json',
        });

        map.value!.addLayer({
          id: 'country-fills',
          type: 'fill',
          source: 'countries',
          'source-layer': 'hidef',
          paint: {
            'fill-color': 'orange',
            'fill-opacity': ['coalesce', ['feature-state', 'opacity'], 0],
          },
        });

        map.value!.on('click', (e: mapboxgl.MapMouseEvent) => {
          const features = map.value!.queryRenderedFeatures(e.point, {
            layers: ['country-fills'],
          });
          if (features && features.length) {
            const countryName =
              features[0].properties?.ADMIN ||
              features[0].properties?.name ||
              'Unknown country';
            selectedCountry.value = countryName;
          }
        });

        map.value!.on(
          'mousemove',
          'country-fills',
          throttle((e: any) => {
            if (!e.features || !e.features.length) return;
            map.value!.getCanvas().style.cursor = 'pointer';
            const newFeatureId = e.features[0].id;
            if (hoveredStateId !== null && hoveredStateId !== newFeatureId) {
              map.value!.setFeatureState(
                { source: 'countries', sourceLayer: 'hidef', id: hoveredStateId },
                { opacity: 0 }
              );
              hoveredStateId = newFeatureId;
              map.value!.setFeatureState(
                { source: 'countries', sourceLayer: 'hidef', id: newFeatureId },
                { opacity: 0.6 }
              );
            } else if (hoveredStateId === null) {
              hoveredStateId = newFeatureId;
              map.value!.setFeatureState(
                { source: 'countries', sourceLayer: 'hidef', id: newFeatureId },
                { opacity: 0.6 }
              );
            }
          }, 100)
        );

        map.value!.on('mouseleave', 'country-fills', () => {
          if (hoveredStateId !== null) {
            map.value!.setFeatureState(
              { source: 'countries', sourceLayer: 'hidef', id: hoveredStateId },
              { opacity: 0 }
            );
          }
          hoveredStateId = null;
          map.value!.getCanvas().style.cursor = '';
        });
      });
    });

    onBeforeUnmount(() => {
      if (map.value) {
        map.value.remove();
      }
    });

    // New helper function to fetch an image for a given article title.
    async function fetchArticleImage(articleTitle: string): Promise<string> {
      const query = encodeURIComponent(`${articleTitle} article`);
      // Replace with your own API key and Custom Search Engine ID
      const apiKey = 'YOUR_GOOGLE_API_KEY';
      const cx = 'YOUR_CUSTOM_SEARCH_ENGINE_ID';
      const url = `https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=${cx}&searchType=image&q=${query}`;
      try {
        const response = await fetch(url);
        const data = await response.json();
        if (data.items && data.items.length) {
          return data.items[0].link;
        }
      } catch (error) {
        console.error("Error fetching image:", error);
      }
      // Fallback image
      return 'https://via.placeholder.com/120x80';
    }

    // Updated fetchPredictions to also fetch article images.
    async function fetchPredictions(country: string) {
      const query = `
        query($country: String) {
          predictions(country: $country) {
            tomorrow
            threeDays
            fiveDays
          }
        }
      `;
      const variables = { country };
      try {
        const response = await fetch('http://127.0.0.1:5000/graphql', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, variables }),
        });
        const result = await response.json();
        if (result.data && result.data.predictions) {
          const fetched = result.data.predictions;
          // Transform each article title into an object with title and imageUrl
          predictions.value.tomorrow = await Promise.all(
            fetched.tomorrow.map(async (title: string) => ({
              title,
              imageUrl: await fetchArticleImage(title)
            }))
          );
          predictions.value.threeDays = await Promise.all(
            fetched.threeDays.map(async (title: string) => ({
              title,
              imageUrl: await fetchArticleImage(title)
            }))
          );
          predictions.value.fiveDays = await Promise.all(
            fetched.fiveDays.map(async (title: string) => ({
              title,
              imageUrl: await fetchArticleImage(title)
            }))
          );
        }
      } catch (error) {
        console.error("Error fetching predictions:", error);
      }
    }

    // When a new country is selected, reset page and fetch predictions
    watch(selectedCountry, (newCountry) => {
      if (newCountry) {
        pageIndex.value = 0;
        fetchPredictions(newCountry);
      } else {
        predictions.value = { tomorrow: [], threeDays: [], fiveDays: [] };
      }
    });

    const closeModal = () => {
      selectedCountry.value = null;
    };

    return {
      mapContainer,
      selectedCountry,
      predictions,
      closeModal,
      dateOptions,
      selectedDateOption,
      pageIndex,
      displayedArticles,
      totalPages,
      nextPage,
      prevPage,
    };
  },
});
</script>

<style scoped>
/* ... (unchanged styles) ... */
.country-date-line {
  text-align: center;
  margin: 0.75rem auto;
  font-size: 1.1rem;
}
.date-dropdown {
  background-color: rgba(0, 0, 0, 0.7);
  color: #fff;
  border: none;
  padding: 0.4rem 0.8rem;
  margin-left: 0.4rem;
  border-radius: 4px;
  cursor: pointer;
}
.bar-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}
.trend-bar {
  display: flex;
  background-color: #1c1c1c;
  padding: 1rem;
  border-radius: 8px;
  align-items: center;
}
.trend-image {
  width: 120px;
  height: 80px;
  object-fit: cover;
  margin-right: 1rem;
}
.trend-content {
  display: flex;
  flex-direction: column;
}
.pagination-buttons {
  display: flex;
  justify-content: space-between;
  margin-top: 1rem;
}
.prev-button,
.next-button {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  background-color: #444;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.prev-button:disabled,
.next-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
