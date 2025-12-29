import argparse
import html
import re
import unicodedata
import zipfile
from pathlib import Path
import pandas as pd


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s


def _safe(v):
    if pd.isna(v): return ""
    return html.escape(str(v))


def _find_col(possible, df: pd.DataFrame):
    for target in possible:
       
        for col in df.columns:
            if _norm(col) == _norm(target):
                return col
        
        if target in df.columns:
            return target
    return None


def load_storage(input_xlsx: Path) -> pd.DataFrame:
    """Carrega o Excel e devolve dataframe padronizado com colunas esperadas."""
    df = pd.read_excel(input_xlsx)

    sub  = _find_col(["Substation","Subestacao","Subestación","Name","Nombre"], df)
    ten  = _find_col(["Tension","Tension (kV)","Tensión (kV)","KV"], df)
    reg  = _find_col(["Region","Região","Región"], df)
    capT = _find_col(["Available Capacity MW Transport","Available Capacity MW  Transport","Available Capacity (MW) Transport","MW Transport"], df)
    capD = _find_col(["Available Capacity MW Distribution","Available Capacity (MW) Distribution","MW Distribution"], df)
    gtyp = _find_col(["Grid Type","Type"], df)
    gown = _find_col(["Grid Owner","Owner"], df)
    lat  = _find_col(["Latitude","Lat","Latitud"], df)
    lon  = _find_col(["Longitude","Lon","Long","Longitud"], df)

    required_names = ["Substation","Tension","Region","Available Capacity MW Transport",
                      "Available Capacity MW Distribution","Grid Type","Grid Owner","Latitude","Longitude"]
    required_cols  = [sub, ten, reg, capT, capD, gtyp, gown, lat, lon]
    missing = [name for name, col in zip(required_names, required_cols) if col is None]
    if missing:
        raise SystemExit("❌ Colunas em falta no Excel: " + ", ".join(missing))


    for c in [ten, capT, capD, lat, lon]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "Substation": df[sub],
        "Tension (kV)": df[ten],
        "Region": df[reg],
        "Available Capacity MW Transport": df[capT],
        "Available Capacity MW Distribution": df[capD],
        "Grid Type": df[gtyp],
        "Grid Owner": df[gown],
        "Latitude": df[lat],
        "Longitude": df[lon],
    })
    
    out = out.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    return out


def build_kmz(df: pd.DataFrame, kmz_path: Path, icon_on: Path, icon_off: Path):
    """Cria KMZ agrupado por Region. Availability = ON para todos."""
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Storage</name>",
        '<Style id="onStyle"><IconStyle><scale>1.0</scale><Icon><href>ON.png</href></Icon></IconStyle></Style>',
        '<Style id="offStyle"><IconStyle><scale>1.0</scale><Icon><href>of.png</href></Icon></IconStyle></Style>',
    ]

   
    for region, g in df.groupby("Region"):
        kml.append(f"<Folder><name>{html.escape(str(region))}</name>")
        for _, row in g.iterrows():
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
            name = str(row.get("Substation",""))
            desc_rows = [
                ("Substation", _safe(row["Substation"])),
                ("Region", _safe(row["Region"])),
                ("Tension (kV)", _safe(row["Tension (kV)"])),
                ("Available Capacity MW Transport", _safe(row["Available Capacity MW Transport"])),
                ("Available Capacity MW Distribution", _safe(row["Available Capacity MW Distribution"])),
                ("Grid Type", _safe(row["Grid Type"])),
                ("Grid Owner", _safe(row["Grid Owner"])),
                ("Lat, Lon", f"{lat:.6f}, {lon:.6f}"),
            ]
            table = "<table border='1' cellpadding='3' cellspacing='0'>" + "".join(
                f"<tr><th align='left'>{k}</th><td>{v}</td></tr>" for k,v in desc_rows
            ) + "</table>"
            desc = f"<![CDATA[{table}]]>"
            kml.append(
                f"<Placemark><name>{html.escape(name)}</name><styleUrl>#onStyle</styleUrl>"
                f"<description>{desc}</description>"
                f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>"
            )
        kml.append("</Folder>")

    kml.append("</Document></kml>")

    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", "\n".join(kml))
        if icon_on.exists():
            zf.write(icon_on, "ON.png")
        if icon_off.exists():
            zf.write(icon_off, "of.png")


def build_gpkg(df: pd.DataFrame, gpkg_path: Path):
    """Escreve GeoPackage (EPSG:4326) com camada 'storage'."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        raise SystemExit("❌ Para gerar GPKG instale: geopandas shapely fiona") from e

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg_path, layer="storage", driver="GPKG")


def main():
    ap = argparse.ArgumentParser(description="Gera KMZ e GeoPackage para o dataset Storage (Lat/Lon).")
    ap.add_argument("--input", default="Storage.xlsx", help="Excel de entrada (default: Storage.xlsx)")
    ap.add_argument("--output_kmz", default="storage.kmz", help="Ficheiro KMZ de saída (default: storage.kmz)")
    ap.add_argument("--output_gpkg", default="storage.gpkg", help="Ficheiro GeoPackage de saída (default: storage.gpkg)")
    ap.add_argument("--on_icon", default="ON.png", help="Ícone ON (default: ON.png)")
    ap.add_argument("--off_icon", default="of.png", help="Ícone OFF (default: of.png)")
    args = ap.parse_args()

    input_xlsx = Path(args.input)
    if not input_xlsx.exists():
        raise SystemExit(f"❌ Excel não encontrado: {input_xlsx}")

    df = load_storage(input_xlsx)
    build_kmz(df, Path(args.output_kmz), Path(args.on_icon), Path(args.off_icon))
    print(f"✅ KMZ criado: {Path(args.output_kmz).resolve()}")

    try:
        build_gpkg(df, Path(args.output_gpkg))
        print(f"✅ GeoPackage criado: {Path(args.output_gpkg).resolve()}")
    except SystemExit as e:
      
        print(str(e))


if __name__ == "__main__":
    main()
