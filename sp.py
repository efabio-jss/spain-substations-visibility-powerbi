import html
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pyproj import Transformer


ICON_ON = "ON.png"
ICON_OFF = "of.png"


DATASETS = [
    # (source_name, filename, flags)
    ("E-Distribución", "edistribuicion.xlsx", {}),
    ("E-REDES",        "eredes.xlsx",         {}),
    ("IDE",            "ide.xlsx",            {}),
    ("UFD",            "ufd.xlsx",            {}),
    ("Viesgo",         "viesgo.xlsx",         {}),
    ("REE",            "ree.xlsx",            {"tso": True}),  # Rede de Transporte (assume Availability=Y se faltar)

    
    ("Begasa",  "begasa.xlsx",  {"utm_cols": ("Coordenada UTM X", "Coordenada UTM Y")}),
    ("Cuerva",  "cuerva.xlsx",  {"utm_cols": ("Coordenada UTM X", "Coordenada UTM Y")}),
    ("Anselmo", "anselmo.xlsx", {"utm_cols": ("Coordenada UTM X", "Coordenada UTM Y")}),
    ("Pitarch", "pitarch.xlsx", {
        "utm_cols": ("Coordenada X UTM H30", "Coordenada Y UTM H30"),
        "force_epsg": "25830",  # UTM H30 → ETRS89 / UTM zone 30N
    }),
]


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s

def _safe(v):
    if pd.isna(v): return ""
    return html.escape(str(v))

def detect_lat_lon(df: pd.DataFrame):
    lat_col = lon_col = None
    for col in df.columns:
        s = _norm(col)
        if lat_col is None and any(k in s for k in ("lat", "latitude", "latitud")):
            lat_col = col
        if lon_col is None and any(k in s for k in ("lon", "long", "longitude", "longitud")):
            lon_col = col
    # Pequenos fallbacks (X/Y)
    if lat_col is None:
        for col in df.columns:
            if _norm(col) in ("y",):
                lat_col = col; break
    if lon_col is None:
        for col in df.columns:
            if _norm(col) in ("x",):
                lon_col = col; break
    return lat_col, lon_col

def detect_utm_columns(columns):
   
    x_col = y_col = None
    for col in columns:
        s = _norm(col)
        if 'utm' in s and ' x ' in s:
            x_col = col
        if 'utm' in s and ' y ' in s:
            y_col = col
    if x_col is None:
        for col in columns:
            s = _norm(col)
            if 'utm' in s and 'x' in s:
                x_col = col; break
    if y_col is None:
        for col in columns:
            s = _norm(col)
            if 'utm' in s and 'y' in s:
                y_col = col; break
    return x_col, y_col

def detect_nudo_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None; best_score = -1
    for col in df.columns:
        s = _norm(col)
        score = 0
        if "nudo" in s: score += 3
        if "afeccion" in s or "afección" in s: score += 2
        if "rdt" in s or "r d t" in s: score += 1
        if score > best_score:
            best_score = score; best_col = col
    return best_col if best_score > 0 else None

def get_by_header_or_index(df: pd.DataFrame, possible_headers, fallback_idx=None):
    cols = list(df.columns)
    for h in possible_headers or []:
        if h in df.columns: return h
        for col in df.columns:
            if _norm(col) == _norm(h):
                return col
    if fallback_idx is not None and 0 <= fallback_idx < len(cols):
        return cols[fallback_idx]
    return None


REGION_RULES = {
    "galicia": {"bbox": (-9.8, 41.7, -6.3, 43.9), "epsg": ["25829","32629","23029","25830"]},
    "extremadura": {"bbox": (-7.7, 38.0, -4.5, 41.0), "epsg": ["25829","25830","32629","23029"]},
    "andalucia": {"bbox": (-7.7, 35.2, -1.2, 38.9), "epsg": ["25830","25829","32630","23030"]},
    "asturias":  {"bbox": (-7.5, 42.7, -4.7, 43.7), "epsg": ["25830","25829","32630"]},
    "cantabria": {"bbox": (-4.9, 42.7, -3.0, 43.7), "epsg": ["25830","32630"]},
    "pais vasco": {"bbox": (-3.9, 42.6, -1.4, 43.6), "epsg": ["25830","32630"]},
    "euskadi": {"bbox": (-3.9, 42.6, -1.4, 43.6), "epsg": ["25830","32630"]},
    "navarra": {"bbox": (-2.9, 41.7, -0.5, 43.4), "epsg": ["25830","32630"]},
    "la rioja": {"bbox": (-3.5, 41.7, -1.4, 43.1), "epsg": ["25830","32630"]},
    "aragon": {"bbox": (-1.8, 39.8, 1.0, 42.9), "epsg": ["25830","25831","32630"]},
    "castilla y leon": {"bbox": (-7.3, 40.1, -1.4, 43.5), "epsg": ["25830","25829","32630"]},
    "castilla y león": {"bbox": (-7.3, 40.1, -1.4, 43.5), "epsg": ["25830","25829","32630"]},
    "castilla-la mancha": {"bbox": (-5.3, 37.7, -0.5, 41.5), "epsg": ["25830","32630"]},
    "madrid": {"bbox": (-4.2, 40.1, -3.2, 41.1), "epsg": ["25830","32630"]},
    "murcia": {"bbox": (-2.7, 37.3, -0.5, 38.9), "epsg": ["25830","32630"]},
    "valenciana": {"bbox": (-1.9, 38.2, 0.3, 41.1), "epsg": ["25830","32630","25831"]},
    "cataluna": {"bbox": (0.0, 40.4, 3.9, 43.5), "epsg": ["25831","32631","23031","25830"]},
    "catalunya": {"bbox": (0.0, 40.4, 3.9, 43.5), "epsg": ["25831","32631","23031","25830"]},
    "balears": {"bbox": (0.7, 38.6, 4.6, 40.2), "epsg": ["25831","32631"]},
    "canarias": {"bbox": (-18.5, 27.1, -13.0, 29.6), "epsg": ["32628","23028"]},
}
GENERIC_EPSG = ["25830","25831","25829","32630","32631","32629","23030","23031","23029","32628","23028"]

@lru_cache(None)
def get_tr(epsg: str) -> Transformer:
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

def inside(lat, lon, bbox, pad=0.25):
    lo_min, la_min, lo_max, la_max = bbox
    return (lo_min - pad) <= lon <= (lo_max + pad) and (la_min - pad) <= lat <= (la_max + pad)

def convert_by_region(easting, northing, region):
    r = _norm(region)
    rule = REGION_RULES.get(r)
    epsgs = (rule["epsg"] if rule else []) + GENERIC_EPSG
    bbox = rule["bbox"] if rule else None
    for epsg in epsgs:
        tr = get_tr(epsg)
        lon, lat = tr.transform(float(easting), float(northing))
        if bbox and inside(lat, lon, bbox):
            return lat, lon, epsg, "by region"
        if not bbox and (-19.5 <= lon <= 5.5) and (27.0 <= lat <= 44.8):
            return lat, lon, epsg, "by country"
    
    lon, lat = get_tr("25830").transform(float(easting), float(northing))
    return lat, lon, "25830", "fallback"


def _convert_coords(df: pd.DataFrame, flags: dict) -> Tuple[pd.Series, pd.Series, str]:
    """
    Devolve (lat, lon, method). Se converter UTM, preenche 'Chosen EPSG' e 'Method' no df.
    Respeita:
      - flags['utm_cols'] -> (x_col, y_col)
      - flags['force_epsg'] -> usa EPSG fixo
    """
    lat_col, lon_col = detect_lat_lon(df)
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        if "Chosen EPSG" not in df.columns:
            df["Chosen EPSG"] = ""
        df["Method"] = "latlon"
        return lat, lon, "latlon"

    # UTM
    if "utm_cols" in flags:
        x_col, y_col = flags["utm_cols"]
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas UTM não encontradas (esperado: {x_col}/{y_col}).")
    else:
        x_col, y_col = detect_utm_columns(df.columns)
        if x_col is None or y_col is None:
            raise ValueError("Não encontrei Latitude/Longitude nem colunas UTM X/Y.")

    easting = pd.to_numeric(df[x_col], errors='coerce')
    northing = pd.to_numeric(df[y_col], errors='coerce')
    valid = easting.notna() & northing.notna()

    if "Chosen EPSG" not in df.columns:
        df['Chosen EPSG'] = ''
    df['Method'] = ''

    lats, lons, epsgs, methods = [], [], [], []

    if flags.get("force_epsg"):
        epsg = str(flags["force_epsg"])
        tr = get_tr(epsg)
        for x, y in zip(easting[valid], northing[valid]):
            lo, la = tr.transform(float(x), float(y))
            lats.append(la); lons.append(lo); epsgs.append(epsg); methods.append("forced epsg")
    else:
        reg_series = df.loc[valid, 'Region'] if 'Region' in df.columns else pd.Series(['']*int(valid.sum()), index=easting[valid].index)
        for x, y, r in zip(easting[valid], northing[valid], reg_series):
            la, lo, ep, how = convert_by_region(x, y, r)
            lats.append(la); lons.append(lo); epsgs.append(ep); methods.append(how)

    lat = pd.Series(index=df.index, dtype=float)
    lon = pd.Series(index=df.index, dtype=float)
    lat.loc[easting[valid].index] = lats
    lon.loc[northing[valid].index] = lons
    df.loc[easting[valid].index, 'Chosen EPSG'] = epsgs
    df.loc[easting[valid].index, 'Method'] = methods
    return lat, lon, "utm->wgs84"


def load_and_standardize(source: str, xlsx_path: Path, flags: dict):
    """Lê o Excel e devolve dataframe normalizado (inclui conversão de coordenadas)."""
    df = pd.read_excel(xlsx_path)

    
    lat, lon, method = _convert_coords(df, flags)

    
    tension_col = get_by_header_or_index(df, [
        "Tension (kV)", "Tension", "Tensión (kV)", "Nivel de Tensión (kV)", "Tension kV"
    ], fallback_idx=1 if flags.get("tso") else None)

    capacity_col = get_by_header_or_index(df, [
        "Available Capacity (MW)", "Available Capacity MW",
        "Capacidad Disponible (MW)", "Capacidad disponible (MW)", "Capacity (MW)"
    ])

    region_col   = get_by_header_or_index(df, ["Region", "Região", "Región", "Zona"])
    sub_col      = get_by_header_or_index(df, ["Substation", "Subestacao", "Subestación", "Substacion", "Name", "Nombre"])
    owner_col    = get_by_header_or_index(df, ["Grid Owner", "Owner", "Operador", "TSO"])
    type_col     = get_by_header_or_index(df, ["Grid Type", "Type"])
    avail_col    = get_by_header_or_index(df, ["Availability", "Disponibilidade", "Disponibilidad"], fallback_idx=3 if flags.get("tso") else None)

  
    def parse_av(v):
        s = str(v).strip().lower()
        if s in ("y","yes","on","1","true","sim","si"): return "Y"
        if s in ("n","no","off","0","false","nao","não"): return "N"
        return "Y" if flags.get("tso") else ""

    availability = df[avail_col].map(parse_av) if (avail_col and avail_col in df.columns) \
        else pd.Series(["Y" if flags.get("tso") else "" ]*len(df))

    
    nudo_col = None if flags.get("tso") else detect_nudo_column(df)
    nudo_vals = df[nudo_col] if (nudo_col and nudo_col in df.columns) else pd.Series([""]*len(df))

    out = pd.DataFrame({
        "Source": source,
        "Substation": df[sub_col] if (sub_col and sub_col in df.columns) else "",
        "Region": df[region_col] if (region_col and region_col in df.columns) else "",
        "Tension (kV)": df[tension_col] if (tension_col and tension_col in df.columns) else "",
        "Available Capacity (MW)": df[capacity_col] if (capacity_col and capacity_col in df.columns) else "",
        "Nudo Afección RdT": nudo_vals,
        "Availability": availability,
        "Grid Owner": df[owner_col] if (owner_col and owner_col in df.columns) else source,
        "Grid Type": df[type_col] if (type_col and type_col in df.columns) else "",
        "Latitude": lat.values,
        "Longitude": lon.values,
        "Method": method,
    })

   
    if "Chosen EPSG" in df.columns:
        out["Chosen EPSG"] = df["Chosen EPSG"]
    else:
        out["Chosen EPSG"] = ""

    return out


def build_kmz(df: pd.DataFrame, kmz_path: Path, name: str, include_nudo: bool = True):
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">', "<Document>",
        f"<name>{name}</name>",
        '<Style id="onStyle"><IconStyle><scale>1.0</scale><Icon><href>ON.png</href></Icon></IconStyle></Style>',
        '<Style id="offStyle"><IconStyle><scale>1.0</scale><Icon><href>of.png</href></Icon></IconStyle></Style>',
    ]

    def row_desc(row):
        rows = [
            ("Substation", _safe(row.get("Substation"))),
            ("Region", _safe(row.get("Region"))),
            ("Tension (kV)", _safe(row.get("Tension (kV)"))),
            ("Available Capacity (MW)", _safe(row.get("Available Capacity (MW)"))),
        ]
        if include_nudo and "Nudo Afección RdT" in row.index:
            rows.append(("Nudo Afección RdT", _safe(row.get("Nudo Afección RdT"))))
        rows += [
            ("Availability", "Y" if str(row.get("Availability")).upper() == "Y" else "N"),
            ("Grid Owner", _safe(row.get("Grid Owner"))),
            ("Grid Type", _safe(row.get("Grid Type"))),
            ("Lat, Lon", f"{float(row['Latitude']):.6f}, {float(row['Longitude']):.6f}"),
        ]
        table = "<table border='1' cellpadding='3' cellspacing='0'>" + "".join(
            f"<tr><th align='left'>{k}</th><td>{v}</td></tr>" for k, v in rows
        ) + "</table>"
        return f"<![CDATA[{table}]]>"

   
    for region, g in df.groupby("Region"):
        kml.append(f"<Folder><name>{html.escape(str(region))}</name>")
        for _, row in g.iterrows():
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
            style = "#onStyle" if str(row.get("Availability","")).upper() == "Y" else "#offStyle"
            name_pm = str(row.get("Substation",""))
            kml.append(
                f"<Placemark><name>{html.escape(name_pm)}</name><styleUrl>{style}</styleUrl>"
                f"<description>{row_desc(row)}</description>"
                f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>"
            )
        kml.append("</Folder>")
    kml.append("</Document></kml>")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", "\n".join(kml))
        zf.write(ICON_ON, "ON.png")
        zf.write(ICON_OFF, "of.png")

def build_kmz_all(df_all: pd.DataFrame, kmz_path: Path):
  
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">', "<Document>",
        "<name>all_substations</name>",
        '<Style id="onStyle"><IconStyle><scale>1.0</scale><Icon><href>ON.png</href></Icon></IconStyle></Style>',
        '<Style id="offStyle"><IconStyle><scale>1.0</scale><Icon><href>of.png</href></Icon></IconStyle></Style>',
    ]

    def row_desc(row):
        rows = [
            ("Source", _safe(row.get("Source"))),
            ("Substation", _safe(row.get("Substation"))),
            ("Region", _safe(row.get("Region"))),
            ("Tension (kV)", _safe(row.get("Tension (kV)"))),
            ("Available Capacity (MW)", _safe(row.get("Available Capacity (MW)"))),
        ]
        if "Nudo Afección RdT" in row.index and str(row.get("Nudo Afección RdT","")).strip():
            rows.append(("Nudo Afección RdT", _safe(row.get("Nudo Afección RdT"))))
        rows += [
            ("Availability", "Y" if str(row.get("Availability","")).upper() == "Y" else "N"),
            ("Grid Owner", _safe(row.get("Grid Owner"))),
            ("Grid Type", _safe(row.get("Grid Type"))),
            ("Lat, Lon", f"{float(row['Latitude']):.6f}, {float(row['Longitude']):.6f}"),
        ]
        table = "<table border='1' cellpadding='3' cellspacing='0'>" + "".join(
            f"<tr><th align='left'>{k}</th><td>{v}</td></tr>" for k, v in rows
        ) + "</table>"
        return f"<![CDATA[{table}]]>"

  
    for source, gsrc in df_all.groupby("Source"):
        kml.append(f"<Folder><name>{html.escape(str(source))}</name>")
        for region, g in gsrc.groupby("Region"):
            kml.append(f"<Folder><name>{html.escape(str(region))}</name>")
            for _, row in g.iterrows():
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
                style = "#onStyle" if str(row.get("Availability","")).upper() == "Y" else "#offStyle"
                name_pm = str(row.get("Substation",""))
                kml.append(
                    f"<Placemark><name>{html.escape(name_pm)}</name><styleUrl>{style}</styleUrl>"
                    f"<description>{row_desc(row)}</description>"
                    f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>"
                )
            kml.append("</Folder>")
        kml.append("</Folder>")
    kml.append("</Document></kml>")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", "\n".join(kml))
        zf.write(ICON_ON, "ON.png")
        zf.write(ICON_OFF, "of.png")


def build_gpkg(df_all: pd.DataFrame, gpkg_path: Path):
    """Exporta um GeoPackage (EPSG:4326) com camada combinada e camadas por Source."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        print(f"⚠️ GeoPackage não gerado (instale geopandas e shapely): {e}")
        return

    def as_gdf(df: pd.DataFrame):
        return gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
            crs="EPSG:4326",
        )

    gdf_all = as_gdf(df_all)
    gdf_all.to_file(gpkg_path, layer="all_substations", driver="GPKG")

    import re as _re
    for source, g in df_all.groupby("Source"):
        layer = _re.sub(r"[^A-Za-z0-9_]+", "_", str(source)).lower().strip("_")
        as_gdf(g).to_file(gpkg_path, layer=layer, driver="GPKG")

    print(f"✅ GeoPackage gerado: {gpkg_path}")


def run():
    base = Path(".")
    if not (base / ICON_ON).exists() or not (base / ICON_OFF).exists():
        raise SystemExit("❌ Ícones ON.png e/of.png não encontrados na pasta de trabalho.")

    frames = []
    for source, fname, flags in DATASETS:
        path = base / fname
        if not path.exists():
            print(f"• {source}: ficheiro não encontrado ({fname}) — a ignorar.")
            continue
        print(f"🔄 {source}: a processar {fname} ...")
        try:
            df_std = load_and_standardize(source, path, flags)
        except Exception as e:
            print(f"   ❌ Erro em {source}: {e}")
            continue
        frames.append(df_std)

       
        kmz_name = f"{Path(fname).stem}.kmz" if source != "REE" else "REE.kmz"
        include_nudo = not flags.get("tso")
        build_kmz(df_std, base / kmz_name, name=Path(kmz_name).stem, include_nudo=include_nudo)
        print(f"   ✅ KMZ: {kmz_name}  |  {len(df_std)} pontos")

        
        df_std.to_excel(base / f"{Path(fname).stem}_standardized.xlsx", index=False)

    if not frames:
        raise SystemExit("❌ Nenhum dataset encontrado na pasta.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_excel(base / "all_substations.xlsx", index=False)
    build_kmz_all(df_all, base / "all_substations.kmz")
    build_gpkg(df_all, base / "all_substations.gpkg")
    print(f"\n🎉 Concluído! all_substations.kmz e all_substations.gpkg com {len(df_all)} pontos.")

if __name__ == "__main__":
    run()
