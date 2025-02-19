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
          <!-- Div for adding news articles later -->
        </div>
        <button class="close-modal" @click="closeModal">Close</button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onBeforeUnmount } from 'vue';
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
    let hoveredStateId: number | string | null = null;

    // Animate fade in from 0 to 1 over 1 second
    function animateHover(featureId: number | string) {
      let start: number | null = null;
      function step(timestamp: number) {
        if (start === null) start = timestamp;
        const progress = Math.min((timestamp - start) / 200, 0.6); // 0 to 1
        map.value!.setFeatureState(
          { source: 'countries', sourceLayer: 'countriesgeo', id: featureId },
          { opacity: progress }
        );
        if (progress < 0.6) {
          requestAnimationFrame(step);
        }
      }
      requestAnimationFrame(step);
    }

    // Animate fade out from 1 to 0 over 1 second
    function animateFadeOut(featureId: number | string) {
      let start: number | null = null;
      function step(timestamp: number) {
        if (start === null) start = timestamp;
        const progress = Math.min((timestamp - start) / 200, 0.6); // 0 to 1
        const newOpacity = 0.6 - progress;
        map.value!.setFeatureState(
          { source: 'countries', sourceLayer: 'countriesgeo', id: featureId },
          { opacity: newOpacity }
        );
        if (progress < 0.6) {
          requestAnimationFrame(step);
        }
      }
      requestAnimationFrame(step);
    }

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
        // Add your vector tile source
        map.value!.addSource('countries', {
          type: 'vector',
          url: 'http://localhost:8080/data/output_low.json'
        });

        // (Optional) Debug layer to see features
        map.value!.addLayer({
          id: 'debug-fill',
          type: 'fill',
          source: 'countries',
          'source-layer': 'countriesgeo',
          paint: {
            'fill-color': '#FFFFFF',
            'fill-opacity': 0.5
          }
        });

        // Fill layer using a numeric opacity from feature state
        map.value!.addLayer({
          id: 'country-fills',
          type: 'fill',
          source: 'countries',
          'source-layer': 'countriesgeo', // update if needed
          paint: {
            'fill-color': 'orange',
            'fill-opacity': ['coalesce', ['feature-state', 'opacity'], 0],
          },
        });

        // Show modal on click
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

        // Throttle mousemove event
        map.value!.on('mousemove', 'country-fills', throttle((e: any) => {
          if (!e.features || !e.features.length) return;
          map.value!.getCanvas().style.cursor = 'pointer';

          // const hoveredCountry =
          //   e.features[0].properties?.ADMIN ||
          //   e.features[0].properties?.name ||
          //   'Unknown country';
          // console.log('Hovering over country:', hoveredCountry);

          const newFeatureId = e.features[0].id;

          // If a different feature is now hovered, fade out the previous one.
          if (hoveredStateId !== null && hoveredStateId !== newFeatureId) {
            animateFadeOut(hoveredStateId);
            hoveredStateId = newFeatureId;
            animateHover(newFeatureId);
          } else if (hoveredStateId === null) {
            hoveredStateId = newFeatureId;
            animateHover(newFeatureId);
          }
        }, 5));

        map.value!.on('mouseleave', 'country-fills', () => {
          if (hoveredStateId !== null) {
            animateFadeOut(hoveredStateId);
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

    const closeModal = () => {
      selectedCountry.value = null;
    };

    return {
      mapContainer,
      selectedCountry,
      closeModal,
    };
  },
});
</script>
