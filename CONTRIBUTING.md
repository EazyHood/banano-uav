# Contribuir a banano-drone

¡Gracias por tu interés! Este proyecto es libre (AGPL-3.0) y toda mejora es bienvenida.

## Cómo empezar

```bash
git clone https://github.com/EazyHood/banano-uav
cd banano-uav
python -m venv .venv && source .venv/Scripts/activate   # o .venv\Scripts\Activate.ps1
pip install -e .[geo,dev]
pytest            # deben pasar todas las pruebas
python scripts/demo.py   # demo con métricas sobre datos sintéticos
```

## Qué ayuda más

1. **Validación con imágenes reales de banano.** Lo más valioso: correr el programa sobre
   ortomosaicos reales y reportar el error contra conteo manual. Si puedes compartir datos
   (aunque sea un recorte), abre un issue.
2. **Etiquetado y modelo entrenado** (camino `deep/`): tiles anotados de macollas/pseudotallos
   para entrenar YOLOv8-seg y publicar pesos abiertos.
3. **Robustez de la línea base:** malezas de hoja ancha (platanillo/*Heliconia*), dosel
   cerrado, sombras, GSD alto.
4. **Formatos GIS** (Shapefile/GeoPackage), soporte multiespectral (NDVI/NDRE), mejoras del
   informe.

## Estilo

- Código y comentarios en español (coherente con el resto del repo).
- Añade una prueba en `tests/` para cada cambio de lógica.
- Cambios pequeños y enfocados; describe el *porqué* en el PR.

## Reportar un problema

Abre un issue con: comando exacto, tamaño/GSD de la imagen, salida de error y qué esperabas.
Si es un problema de detección, adjunta el `informe.html` o el `mapa.png` si puedes.
