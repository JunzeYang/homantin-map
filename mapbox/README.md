# Mapbox-ready result layers

These are WGS84 GeoJSON layers exported from the *saved* atlas result bundle. No analysis code was run to produce them.

| File | Geometry | Features | Suggested Studio layer |
| --- | --- | ---: | --- |
| `critical_links.geojson` | Directed road lines | 2,849 | Line |
| `critical_nodes.geojson` | Road nodes | 1,733 | Circle |
| `hospital_grid.geojson` | 100 m grid squares | 3,599 | Fill |
| `hospitals.geojson` | Hospital candidates | 14 | Circle / symbol |
| `study_tpusb.geojson` | Simplified context zones | 1,200 | Outline |

In Mapbox Studio, create a style, choose **Add new layer → Upload data**, and import the GeoJSON files. Colour roads or nodes with the `od_eff_loss`, `odbc`, `lscc_loss`, `td_loss`, or `reach_loss` property. For percentages, multiply each fractional loss by 100 in the style expression or label; `reach_loss` × 100 gives **percentage points**. Colour grid squares using `distance_km` or `reachable`, and inspect hospital candidates via `institution_eng` and `with_ae`.

Mapbox Studio can share the saved map style as a map link, but it cannot reproduce the atlas's result cards, removal slider and network/hospital curves as Studio style layers. Keep the full atlas webpage for those interactions. The GeoJSON geometry is the same browser-display geometry as the atlas, including lightly simplified TPUSB boundaries. Opposite-direction and parallel road links may overlap on the map; separate `link_id` values are preserved.

The network layer is the full supplied road-network envelope, not an administrative Ho Man Tin boundary. Hospital distance is directed road distance from a snapped network node, excluding the off-network snap leg; it is not travel time. Grid shares are equal-weight cells, not population shares. Use the source CSVs or atlas for full metric precision and scenario interpretations.
