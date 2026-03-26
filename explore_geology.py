"""explore_geology.py — Exploration BDCharm-50 Isère."""
from __future__ import annotations

import os
from pathlib import Path

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import geopandas as gpd
from shapely.geometry import box

import config

SHP = Path("data/GEO050K_HARM_038_S_FGEOL_2154.shp")

print(f"Fichier : {SHP}")
print(f"Existe  : {SHP.exists()}")

if not SHP.exists():
    # Essayer chemin alternatif
    alt = list(Path(".").rglob("*S_FGEOL*2154.shp"))
    if alt:
        SHP_FOUND = alt[0]
        print(f"Trouvé  : {SHP_FOUND}")
    else:
        print("❌ Fichier non trouvé. Placer dans data/ ou ajuster le chemin.")
        raise SystemExit(1)
else:
    SHP_FOUND = SHP

# ── Chargement complet ──
gdf_full = gpd.read_file(SHP_FOUND)
print(f"\nTotal polygones : {len(gdf_full)}")
print(f"CRS : {gdf_full.crs}")
print(f"Colonnes : {list(gdf_full.columns)}")
print(f"Bounds : {gdf_full.total_bounds}")

# ── Clip sur bbox Grenoble ──
bbox = dict(config.BBOX)
clip = box(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])

if gdf_full.crs and gdf_full.crs.to_epsg() != 2154:
    gdf_full = gdf_full.to_crs(epsg=2154)

gdf = gdf_full[gdf_full.intersects(clip)].copy()
print(f"\nDans bbox Grenoble : {len(gdf)} polygones")

# ── Colonnes descriptives ──
for col in gdf.columns:
    if col == "geometry":
        continue
    n_unique = gdf[col].nunique()
    n_null = int(gdf[col].isna().sum())
    print(f"\n{'═'*60}")
    print(f"  {col}  ({n_unique} uniques, {n_null} null)")
    print(f"{'═'*60}")
    if n_unique <= 50:
        for val, cnt in gdf[col].value_counts().head(30).items():
            pct = cnt / len(gdf) * 100
            print(f"    {val!s:55s} : {cnt:4d} ({pct:5.1f}%)")
    else:
        print(f"  (trop de valeurs, top 15)")
        for val, cnt in gdf[col].value_counts().head(15).items():
            pct = cnt / len(gdf) * 100
            print(f"    {val!s:55s} : {cnt:4d} ({pct:5.1f}%)")

# ── Test du mapping existant ──
print(f"\n{'═'*60}")
print("  TEST resolve_geology() sur DESCR")
print(f"{'═'*60}")

if "DESCR" in gdf.columns:
    gdf["_resolved"] = gdf["DESCR"].fillna("").apply(config.resolve_geology)
    gdf["_score"] = gdf["_resolved"].apply(
        lambda c: config.GEOLOGY_SCORES.get(c, config.GEOLOGY_SCORES.get("unknown", 0.3))
    )

    for cat, cnt in gdf["_resolved"].value_counts().items():
        sc = config.GEOLOGY_SCORES.get(cat, "?")
        print(f"    {cat:25s} : {cnt:4d} polys  (score={sc})")

    n_unk = int((gdf["_resolved"] == "unknown").sum())
    print(f"\n  → {len(gdf) - n_unk}/{len(gdf)} résolus ({(len(gdf)-n_unk)/len(gdf)*100:.1f}%)")
    
    if n_unk > 0:
        print(f"\n  DESCR non résolus (→ unknown) :")
        unk = gdf[gdf["_resolved"] == "unknown"]
        for val, cnt in unk["DESCR"].value_counts().head(20).items():
            print(f"    {val!s:55s} : {cnt:4d}")

# ── Test sur NOTATION aussi ──
if "NOTATION" in gdf.columns:
    print(f"\n  TEST resolve_geology() sur NOTATION")
    gdf["_resolved_n"] = gdf["NOTATION"].fillna("").apply(config.resolve_geology)
    n_unk_n = int((gdf["_resolved_n"] == "unknown").sum())
    n_better = int(
        ((gdf["_resolved"] == "unknown") & (gdf["_resolved_n"] != "unknown")).sum()
    )
    print(f"  → NOTATION résout {n_better} polygones supplémentaires vs DESCR")

print("\n✅ Exploration terminée")