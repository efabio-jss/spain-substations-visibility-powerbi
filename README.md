
# Substations Visibility — Data Pipeline & Power BI Dashboard

End‑to‑end toolkit to:
- Normalize and geocode **substation datasets** from multiple DSOs/TSO,
- Export geospatial products (**KMZ**, **GeoPackage**),
- Produce **standardized Excel** outputs for analytics,
- Power a **Power BI dashboard** for visibility and trend analysis **by month** and **by substation**.

> Includes two Python pipelines:
> 1) **Substations pipeline** (multi-source ingest, standardization, geocoding, KMZ/GPKG generation)  
> 2) **Storage pipeline** (storage siting dataset → KMZ/GPKG)

And one **Power BI report** (substations visibility & capacity tracking).

---

## Prerequisites

### Python
- Python **3.9+** recommended

### Python packages
Core:
```bash
pip install pandas pyproj openpyxl


Geo (optional for GPKG exports but recommended):
pip install geopandas shapely fiona
