<template> 
  <div class="world-map">
    <!-- Map container -->
    <div id="map" ref="mapContainer"></div>

    <!-- Modal overlay: only visible when a country is selected -->
    <div v-if="selectedCountry" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <h2 class="modalTitle">News Prediction</h2>
        <hr />
        <p>{{ selectedCountry }} | Dec 15, 2026</p>
        <div class="news-articles">
          <div class="prediction-section">
            <h3>Tomorrow's Trends</h3>
            <ul>
              <li v-for="(trend, index) in predictions.tomorrow" :key="index">{{ trend }}</li>
            </ul>
          </div>
          <div class="prediction-section">
            <h3>3 Days Trends</h3>
            <ul>
              <li v-for="(trend, index) in predictions.threeDays" :key="index">{{ trend }}</li>
            </ul>
          </div>
          <div class="prediction-section">
            <h3>5 Days Trends</h3>
            <ul>
              <li v-for="(trend, index) in predictions.fiveDays" :key="index">{{ trend }}</li>
            </ul>
          </div>
        </div>
        <button class="close-modal" @click="closeModal">Close</button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onBeforeUnmount, watch } from 'vue';
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
    const predictions = ref({
      tomorrow: [] as string[],
      threeDays: [] as string[],
      fiveDays: [] as string[],
    });
    let hoveredStateId: number | string | null = null;

    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
    console.log('Mapbox token:', import.meta.env.VITE_MAPBOX_TOKEN);

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
          url: 'http://localhost:8080/data/output_hi.json'
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

        map.value!.on('mousemove', 'country-fills', throttle((e: any) => {
          if (!e.features || !e.features.length) return;
          map.value!.getCanvas().style.cursor = 'pointer';
          const newFeatureId = e.features[0].id;
          if (hoveredStateId !== null && hoveredStateId !== newFeatureId) {
            // Reset previous country immediately
            map.value!.setFeatureState(
              { source: 'countries', sourceLayer: 'hidef', id: hoveredStateId },
              { opacity: 0 }
            );
            hoveredStateId = newFeatureId;
            // Set new hovered country immediately
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
        }, 100));

        map.value!.on('mouseleave', 'country-fills', () => {
          if (hoveredStateId !== null) {
            // Reset hovered country immediately
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

    // Fetch predictions from the Flask GraphQL backend
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

    // Watch for changes in selectedCountry to fetch predictions
    watch(selectedCountry, (newCountry) => {
      if (newCountry) {
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
    };
  },
});
</script>
