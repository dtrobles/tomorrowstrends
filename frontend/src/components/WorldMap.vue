<template>
  <div class="map-wrapper">
    <svg ref="svgRef" :width="width" :height="height"></svg>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, ref } from 'vue';
import * as d3 from 'd3';

export default defineComponent({
  name: 'WorldMap',
  setup() {
    const svgRef = ref<SVGSVGElement | null>(null);
    const width = 960;
    const height = 500;

    onMounted(async () => {
      console.log('WorldMap component mounted');
      console.log('SVG element:', svgRef.value);
      if (!svgRef.value) return;

      const svg = d3.select(svgRef.value);
      svg.attr("viewBox", `0 0 ${width} ${height}`);
      const g = svg.append('g');

      const projection = d3.geoMercator()
        .scale(150)
        .translate([width / 2, height / 1.5]);
      const path = d3.geoPath().projection(projection);

      try {
        const url = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson';
        console.log('Fetching GeoJSON data from:', url);

        // Option 1: Using d3.json
        const geoData = await d3.json(url) as any;
        console.log("Loaded geoData:", geoData);

        // Option 2: If d3.json fails, try this instead:
        // const response = await fetch(url);
        // if (!response.ok) {
        //   throw new Error(`Network response was not ok: ${response.statusText}`);
        // }
        // const geoData = await response.json();
        // console.log("Loaded geoData (via fetch):", geoData);

        if (!geoData || !geoData.features) {
          console.error("No features found in the GeoJSON data");
          return;
        }

        g.selectAll('path')
          .data(geoData.features)
          .enter()
          .append('path')
          .attr('d', path)
          .attr('fill', 'rgba(255,255,255,0.1)') // temporary fill for debugging
          .attr('stroke', 'white')
          .attr('stroke-width', 0.5)
          .on('click', (_event: MouseEvent, d: any) => {
            const countryName = d.properties.ADMIN || d.properties.name || 'Unknown Country';
            console.log(countryName);
          });
      } catch (error) {
        console.error("Error loading GeoJSON:", error);
      }

      svg.call(
        d3.zoom()
          .scaleExtent([1, 8])
          .on('zoom', (event: any) => {
            g.attr('transform', event.transform);
          })
      );
    });

    return {
      svgRef,
      width,
      height,
    };
  },
});
</script>

<style scoped>
.map-wrapper {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

svg {
  background-color: #0D1017; /* Debug background blue , real #0D1017 */ 
  cursor: move;
}
</style>
