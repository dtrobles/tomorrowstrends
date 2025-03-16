<template>
  <div class="world-map">
    <!-- Map container -->
    <div id="map" ref="mapContainer"></div>

    <!-- Modal overlay: only visible when a country is selected -->
    <div v-if="selectedCountry" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <h2 class="modalTitle">News Prediction</h2>
        <hr />

        <!-- COUNTRY & DATE DROPDOWN -->
        <p>{{ selectedCountry }}</p>
        <label for="dateSelect"><strong>Select Date: </strong></label>
        <select id="dateSelect" v-model="selectedDateOption">
          <option v-for="opt in dateOptions" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
        </select>

        <!-- ARTICLES SECTION -->
        <div class="news-articles">
          <div class="prediction-section">
            <h3>Articles for {{ currentDateLabel }}</h3>
            
            <!-- 3-Bar layout (page-based) -->
            <div class="bar-container">
              <div 
                v-for="(trend, index) in displayedArticles" 
                :key="index" 
                class="trend-bar"
              >
                <img 
                  src="https://via.placeholder.com/120x80" 
                  alt="Article Image" 
                  class="trend-image"
                />
                <div class="trend-content">
                  <h4>{{ trend }}</h4>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                </div>
              </div>
            </div>

            <!-- PAGINATION BUTTONS -->
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

    // Holds the predictions fetched from backend
    const predictions = ref({
      tomorrow: [] as string[],    // +1
      threeDays: [] as string[],  // +3
      fiveDays: [] as string[],   // +5
    });

    // -------------------------------------------
    //  DATE DROPDOWN & PAGINATION
    // -------------------------------------------
    /**
     * For the dropdown, we build an array of date options:
     *   value = 1, 3, or 5   -> corresponds to tomorrow/3Days/5Days
     *   label = e.g. "Mar 16, 2025" (today +1) – or any format you like
     */
    const dateOptions = ref([
      { value: 1, label: formatDatePlusDays(1) },
      { value: 3, label: formatDatePlusDays(3) },
      { value: 5, label: formatDatePlusDays(5) },
    ]);

    // The user’s selection from the dropdown
    const selectedDateOption = ref<number>(1);

    // For pagination (3 items per page)
    const pageIndex = ref(0);
    const trendsPerPage = 3;

    // This computed picks the correct array from "predictions"
    // based on the user’s dropdown selection.
    const selectedArticles = computed(() => {
      if (selectedDateOption.value === 1) {
        return predictions.value.tomorrow;
      } else if (selectedDateOption.value === 3) {
        return predictions.value.threeDays;
      } else {
        return predictions.value.fiveDays;
      }
    });

    // Sliced array: only 3 items based on current pageIndex
    const displayedArticles = computed(() => {
      const start = pageIndex.value * trendsPerPage;
      const end = start + trendsPerPage;
      return selectedArticles.value.slice(start, end);
    });

    // Total pages for the chosen set of articles
    const totalPages = computed(() => {
      return Math.ceil(selectedArticles.value.length / trendsPerPage);
    });

    // Show date in heading: e.g. "Mar 16, 2025"
    const currentDateLabel = computed(() => {
      const option = dateOptions.value.find(o => o.value === selectedDateOption.value);
      return option ? option.label : '';
    });

    // Move to the next page if possible
    const nextPage = () => {
      if (pageIndex.value < totalPages.value - 1) {
        pageIndex.value++;
      }
    };

    // Move to the previous page if possible
    const prevPage = () => {
      if (pageIndex.value > 0) {
        pageIndex.value--;
      }
    };

    // If the user switches date from the dropdown, reset pageIndex
    watch(selectedDateOption, () => {
      pageIndex.value = 0;
    });

    // Simple date formatter for "today + N days"
    function formatDatePlusDays(days: number) {
      const d = new Date();
      d.setDate(d.getDate() + days);
      return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric'
      });
    }
    // -------------------------------------------

    let hoveredStateId: number | string | null = null;

    // Mapbox access token
    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
    console.log('Mapbox token:', import.meta.env.VITE_MAPBOX_TOKEN);

    // Initialize map on mount
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
        // Add your vector source
        map.value!.addSource('countries', {
          type: 'vector',
          url: 'http://localhost:8080/data/output_hi.json',
        });

        // Add fill layer
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

        // Click event to select a country
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

        // Hover effect
        map.value!.on(
          'mousemove',
          'country-fills',
          throttle((e: any) => {
            if (!e.features || !e.features.length) return;
            map.value!.getCanvas().style.cursor = 'pointer';
            const newFeatureId = e.features[0].id;
            if (hoveredStateId !== null && hoveredStateId !== newFeatureId) {
              // Reset previous hover
              map.value!.setFeatureState(
                { source: 'countries', sourceLayer: 'hidef', id: hoveredStateId },
                { opacity: 0 }
              );
              hoveredStateId = newFeatureId;
              // Set new hover
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

        // Mouse leave
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

    // Remove map on unmount
    onBeforeUnmount(() => {
      if (map.value) {
        map.value.remove();
      }
    });

    // Fetch predictions from your Flask GraphQL backend
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
        console.log("[DEBUG] GraphQL response received:", result);
        if (result.data && result.data.predictions) {
          predictions.value = result.data.predictions;
        }
      } catch (error) {
        console.error("Error fetching predictions:", error);
      }
    }

    // When a new country is selected, reset the page and fetch predictions
    watch(selectedCountry, (newCountry) => {
      if (newCountry) {
        pageIndex.value = 0;
        fetchPredictions(newCountry);
      } else {
        predictions.value = { tomorrow: [], threeDays: [], fiveDays: [] };
      }
    });

    // Close modal
    const closeModal = () => {
      selectedCountry.value = null;
    };

    return {
      mapContainer,
      selectedCountry,
      predictions,
      closeModal,
      // date dropdown
      dateOptions,
      selectedDateOption,
      currentDateLabel,
      // pagination
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
/* Minimal example styling – adjust as needed */

/* Container for the 3 "bars" (articles) */
.bar-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

/* Each individual article bar */
.trend-bar {
  display: flex;
  background-color: #1c1c1c;
  padding: 1rem;
  border-radius: 8px;
  align-items: center;
}

/* Example for image sizing */
.trend-image {
  width: 120px;
  height: 80px;
  object-fit: cover;
  margin-right: 1rem;
}

/* Text content of each article */
.trend-content {
  display: flex;
  flex-direction: column;
}

/* Pagination buttons side by side, inside the modal */
.pagination-buttons {
  display: flex;
  justify-content: space-between;
  margin-top: 1rem;
}

/* Semi-large rectangle buttons */
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

/* Disabled state for pagination buttons */
.prev-button:disabled,
.next-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
