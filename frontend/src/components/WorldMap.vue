<template>
  <div class="world-map">
    <!-- Map container -->
    <div id="map" ref="mapContainer"></div>
    <!-- Projection toggle buttons -->
    <div class="projection-toggle">
      <button @click="setProjection('mercator')" :class="{ active: projection === 'mercator' }">
        Mercator
      </button>
      <button @click="setProjection('globe')" :class="{ active: projection === 'globe' }">
        Globe
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
        // Track the current projection mode ('mercator' or 'globe')
    const projection = ref<'mercator' | 'globe'>('mercator');
    // Set your Mapbox access token here
    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
    mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
    console.log('Mapbox token:', import.meta.env.VITE_MAPBOX_TOKEN);


    onMounted(() => {
      if (!mapContainer.value) return;

      // Initialize the map
      map.value = new mapboxgl.Map({
        container: mapContainer.value,
        style: 'mapbox://styles/mapbox/light-v10', // a clean base style
        center: [0, 20],
        zoom: 1.5,
        maxZoom: 3,
        minZoom: 1, // Prevents unnecessary deep zooms
        antialias: true, // Improves anti-aliasing\
        renderWorldCopies: true, // Prevents "jumps" at 180° longitude
        failIfMajorPerformanceCaveat: false, // Allows fallback to software rendering
        preserveDrawingBuffer: false, // Prevents unnecessary memory usage
      });

      map.value.on('load', () => {
        // If the current projection is globe, switch to globe mode
        if (projection.value === 'globe') {
          // For Mapbox GL JS v2.9+ you can set the projection to 'globe'
          map.value!.setProjection('globe');
        }

        // Add a GeoJSON source for country borders.
        // This example uses an external dataset of country boundaries.
        map.value!.addSource('countries', {
          type: 'geojson',
          data: '/countries.geo.json',
          // countries.geo.json 
          // world_countries_geojson.geojson
        });
        map.value!.addLayer({
          id: 'country-fills',
          type: 'fill',
          source: 'countries',
          layout: {},
          paint: {
            'fill-color': '#000000',
            'fill-opacity': 0  // Invisible fill layer
          }
        });
        // Add a layer to display country borders only.
        map.value!.addLayer({
          id: 'country-borders',
          type: 'line',
          source: 'countries',
          layout: {},
          paint: {
            'line-color': '#FF6600', // orange borders
            'line-width': 1,
          },
        });
        map.value!.on('render', () => {
          map.value!.resize(); // Resize only when necessary
        });

        // When clicking on the map, check if a country was clicked
        map.value!.on('click', (e: mapboxgl.MapMouseEvent) => {
        const features = map.value!.queryRenderedFeatures(e.point, {
          layers: ['country-fills']
        });
        if (features && features.length) {
          const countryName =
            features[0].properties?.ADMIN ||
            features[0].properties?.name ||
            'Unknown country';
          console.log(countryName);
        }
      });

        // Change the cursor to a pointer when hovering over a country border.
        map.value!.on('mouseenter', 'country-borders', () => {
          map.value!.getCanvas().style.cursor = 'pointer';
        });
        map.value!.on('mouseleave', 'country-borders', () => {
          map.value!.getCanvas().style.cursor = '';
        });
      });
            
    });

    // Clean up the map instance when the component is unmounted
    onBeforeUnmount(() => {
      if (map.value) {
        map.value.remove();
      }
    });

    // Function to change the map projection mode
    const setProjection = (mode: 'mercator' | 'globe') => {
      projection.value = mode;
      if (map.value) {
        if (mode === 'globe') {
          map.value.setProjection('globe');
        } else {
          map.value.setProjection('mercator');
        }
      }
    };

    return {
      mapContainer,
      setProjection,
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

/* Style for the projection toggle buttons */
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
