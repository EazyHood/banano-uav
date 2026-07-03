# Changelog

Todas las novedades notables de este proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [1.0.0] — 2026-07-03

Primera versión pública, utilizable de punta a punta con un dron (flujo post-vuelo).

### Añadido
- **Pipeline geoespacial completo**: lectura de ortomosaicos **GeoTIFF** con detección
  automática del GSD y la georreferencia (`banano/geo.py`), procesado por **tiles con
  deduplicación en bordes** (`banano/ortho.py`).
- **CLI `banano-detect`**: ortomosaico → conteo + capa GIS + informe, en un comando.
- **Entregables**: informe HTML autocontenido, mapa PNG, **GeoJSON** (capa GIS con lon/lat),
  CSV y resumen JSON (`banano/report.py`).
- **Detección híbrida sin datos etiquetados**: corrección de iluminación, dosel por ExGR
  adaptativo + textura + morfología, **marco de siembra por autocorrelación FFT**, centros
  por **transformada de distancia + watershed** fusionados con **Fast Radial Symmetry
  Transform**, agrupamiento en macollas (DBSCAN).
- **Reporte honesto**: conteo fiable a nivel de macolla + **rango** de pseudotallos +
  cobertura de dosel + avisos (guardarraíl de GSD, dosel cerrado).
- Índice **TGI** (Neupane 2019) y camino de deep learning (YOLOv8-seg / ALSS-YOLO-Seg).
- Generador de plantación sintética + GeoTIFF de ejemplo; pruebas; documentación de estado
  del arte con referencias reales.

### Corregido
- Umbrales de picos ahora son **robustos al tamaño del tile** (percentil/absoluto en vez de
  máximo global), evitando que un blob fuerte suprima plantas débiles en tiles grandes.

### Licencia
- Publicado bajo **AGPL-3.0** para garantizar que permanezca libre y abierto.
