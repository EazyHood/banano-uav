# Política de seguridad

## Versiones con soporte

Se da soporte a la última versión publicada (rama `main`).

## Reportar una vulnerabilidad

Si encuentras un problema de seguridad, por favor **no** abras un issue público.
Repórtalo de forma privada a través de los
[Security Advisories](https://github.com/EazyHood/banano-uav/security/advisories/new)
de GitHub. Intentaremos responder en un plazo razonable.

## Alcance

`banano-drone` procesa imágenes locales y no realiza conexiones de red en su flujo
principal. Las áreas de mayor interés son:

- Lectura de archivos raster (rutas, archivos malformados).
- Deserialización de configuración YAML (se usa `yaml.safe_load`).
- Carga de pesos de modelo (`.pt`): **carga solo pesos de fuentes en las que confíes**,
  ya que los checkpoints de PyTorch pueden ejecutar código al deserializarse.
