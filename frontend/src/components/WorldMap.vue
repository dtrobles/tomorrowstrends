<template>
  <div class="world-map">
    <!-- Map container -->
    <div id="map" ref="mapContainer"></div>
    <!-- Projection toggle and reset view buttons -->
    <div class="projection-toggle">
      <button @click="resetView">
        Reset View
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onBeforeUnmount } from 'vue';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

export default defineComponent({
  name: 'WorldMap',
  setup() {
    const mapContainer = ref<HTMLElement | null>(null);
    const map = ref<mapboxgl.Map | null>(null);
    const projection = ref<'globe'>('globe');

    // Set your Mapbox access token (ensure your .env uses the VITE_ prefix)
    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
    console.log('Mapbox token:', import.meta.env.VITE_MAPBOX_TOKEN);

    // Function to reset the map view (center, zoom, pitch, and bearing)
    const resetView = () => {
      if (map.value) {
        map.value.flyTo({
          center: [0, 20],
          zoom: 1.5,
          pitch: 0,
          bearing: 0,
        });
      }
    };

    // Variable to store the currently hovered feature's id
    let hoveredStateId: number | string | null = null;

    onMounted(() => {
      if (!mapContainer.value) return;

      map.value = new mapboxgl.Map({
        style: 'mapbox://styles/notaspect0/cm6yfnj5a00oa01sb3y1pfflk', // base style
        container: mapContainer.value,
        center: [0, 20],
        zoom: 1.5,
        maxZoom: 5,
        minZoom: 1,

      });

      map.value.on('load', () => {

        // Add your custom GeoJSON source for country borders and generate unique IDs
        map.value.addSource('countries', {
          type: 'geojson',
          data: '/countries.geo.json', // Adjust this path if needed.
          generateId: true,
        });
        
        // Add a fill layer that is invisible by default but will be styled on hover.
        // The paint uses feature-state "hover" to determine its fill color and opacity.
        map.value.addLayer({
          id: 'country-fills',
          type: 'fill',
          source: 'countries',
          layout: {},
          paint: {
            'fill-color': [
              'case',
              ['boolean', ['feature-state', 'hover'], false],
              'orange',
              '#000000'
            ],
            'fill-opacity': [
              'case',
              ['boolean', ['feature-state', 'hover'], false],
              0.5,
              0
            ],
          },
        });

        // Add a line layer to display country borders.
        map.value.addLayer({
          id: 'country-borders',
          type: 'line',
          source: 'countries',
          layout: {},
          paint: {
            'line-color': '#E4801D',
            'line-width': 0.5,
          },
        });

        // When clicking on the map, query the invisible fill layer for country features.
        map.value.on('click', (e: mapboxgl.MapMouseEvent) => {
          const features = map.value!.queryRenderedFeatures(e.point, {
            layers: ['country-fills'],
          });
          if (features && features.length) {
            const countryName =
              features[0].properties?.ADMIN ||
              features[0].properties?.name ||
              'Unknown country';
            console.log(countryName);
          }
        });

        // Use mousemove on the fill layer to update feature state and style hovered country.
        map.value.on('mousemove', 'country-fills', (e: any) => {
          if (!e.features || !e.features.length) return;

          // Change the cursor to pointer
          map.value!.getCanvas().style.cursor = 'pointer';

          // Remove hover state from the previously hovered feature, if any.
          if (hoveredStateId !== null) {
            map.value!.setFeatureState(
              { source: 'countries', id: hoveredStateId },
              { hover: false }
            );
          }

          hoveredStateId = e.features[0].id;
          map.value!.setFeatureState(
            { source: 'countries', id: hoveredStateId },
            { hover: true }
          );
        });

        // Reset the hover state when the mouse leaves the fill layer.
        map.value.on('mouseleave', 'country-fills', () => {
          if (hoveredStateId !== null) {
            map.value!.setFeatureState(
              { source: 'countries', id: hoveredStateId },
              { hover: false }
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

    const setProjection = (mode: 'globe') => {
      projection.value = mode;
      if (map.value) {
        if (mode === 'globe') {
          map.value.setProjection('globe');
        }
      }
    };

    return {
      mapContainer,
      setProjection,
      resetView,
      projection,
    };
  },
});
</script>

<style scoped>
.world-map {
  position: relative;
  width: 100%;
  height: 100%;
}

#map {
  width: 100%;
  height: 100%;
}

/* Style for the projection toggle and reset view buttons */
.projection-toggle {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 1;
  display: flex;
  flex-direction: column;
}

.projection-toggle button {
  background-color: #0D1017;
  color: white;
  border: 1px solid #fff;
  padding: 5px 10px;
  margin-bottom: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease, color 0.3s ease;
}

.projection-toggle button.active {
  background-color: orange;
  color: black;
}
</style>
