#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geocode / interpolate addresses (no geometry) using streets + parcels shapefiles.
- INPUTS:
    * Parquet of addresses (no geometry). Expected columns (flexible, see --map-* options):
        - id: unique ID (string/int)           [--map-id]
        - address_text: raw free text           [--map-addr]
        - street_name: street name (optional)   [--map-street]
        - street_type: utca/út/körút/etc (opt)  [--map-type]
        - housenumber: "12", "12/A", "12-16"    [--map-hn]
        - hrsz: helyrajzi szám (optional)       [--map-hrsz]
        - settlement: település (optional)      [--map-settlement]
    * Shapefile of streets (LineString/MultiLineString) with 'name' attribute (can be customized).
    * Shapefile of parcels (Polygon/MultiPolygon) with 'hrsz' attribute (can be customized).
- OUTPUT:
    * Shapefile of geocoded points (EPSG same as streets), with quality scoring & explanation.
- APPROACH:
    1) Normalize text (lowercase, diacritics, alias map) and parse housenumbers (num, suffix, range).
    2) Fuzzy match addresses to street components (by name) within the most likely settlement (if any).
    3) If address has HRSZ and parcel dataset has matching HRSZ -> snap to parcel "front point" (nearest point to street).
    4) Else: group by matched street, split by parity (even/odd), place points by rank interpolation along street length;
       choose left/right side for parity by comparing parcel-side densities; offset points to side by small distance.
    5) Compute quality score and write output.
- NOTES:
    * The script does not require any database.
    * If scikit-learn is installed, isotonic regression is used per street-side to regularize positions.
"""
import sys
import os
import re
import argparse
import warnings
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union, linemerge, nearest_points
from shapely.affinity import translate
from shapely import wkb

# Fuzzy matching (rapidfuzz preferred; fallback to difflib)
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False

# Optional isotonic regression
HAVE_ISO = False
try:
    from sklearn.isotonic import IsotonicRegression
    HAVE_ISO = True
except Exception:
    pass

def strip_diacritics(s: str) -> str:
    if s is None:
        return ""
    import unicodedata
    s_norm = unicodedata.normalize('NFD', s)
    s_noacc = ''.join(ch for ch in s_norm if unicodedata.category(ch) != 'Mn')
    return s_noacc

# Basic alias map for Hungarian street types
TYPE_ALIASES = {
    'u': 'utca', 'u.': 'utca', 'ut': 'ut', 'út': 'ut', 'út.': 'ut', 'ut.': 'ut',
    'krt': 'korut', 'krt.': 'korut', 'körút': 'korut', 'körut': 'korut',
    'krt.': 'korut', 'krt': 'korut', 'körút.': 'korut',
    'tér.': 'ter', 'tér': 'ter',
    'sétány': 'setany', 'stny': 'setany',
    'köz': 'koz', 'köz.': 'koz',
    'útja': 'utja', 'útja.': 'utja',
    'sor': 'sor'
}

def normalize_street_name(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = strip_diacritics(s)
    s = re.sub(r'[\.\,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # split last token for type
    toks = s.split(' ')
    if len(toks) >= 2 and toks[-1] in TYPE_ALIASES:
        toks[-1] = TYPE_ALIASES[toks[-1]]
    return ' '.join(toks)

def parse_housenumber(hn: str) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int,int]]]:
    """
    Returns (base_number, suffix, range_tuple)
    Examples:
      "12" -> (12, None, None)
      "12/A" -> (12, 'A', None)
      "12-16" -> (12, None, (12,16))
      "12–16" -> same as above
      None/"" -> (None, None, None)
    """
    if hn is None:
        return (None, None, None)
    s = str(hn).strip()
    if s == "":
        return (None, None, None)
    s = s.replace('–', '-').replace('−', '-')
    # range?
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return (a, None, (a, b))
    # number + optional suffix
    m = re.match(r'^\s*(\d+)\s*(?:/|\s+)?\s*([A-Za-z])?\s*$', s)
    if m:
        base = int(m.group(1))
        suf = m.group(2).upper() if m.group(2) else None
        return (base, suf, None)
    # fallback: digits anywhere
    m = re.search(r'(\d+)', s)
    if m:
        base = int(m.group(1))
        return (base, None, None)
    return (None, None, None)

def is_even(n: Optional[int]) -> Optional[bool]:
    if n is None:
        return None
    return (n % 2 == 0)

def fuzzy_match_one(query: str, choices: List[str]) -> Tuple[Optional[str], float]:
    if not query or not choices:
        return (None, 0.0)
    if HAVE_RAPIDFUZZ:
        match = rf_process.extractOne(query, choices, scorer=rf_fuzz.token_set_ratio)
        if match is None:
            return (None, 0.0)
        return (match[0], float(match[1]) / 100.0)
    # fallback difflib
    best = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)
    if not best:
        return (None, 0.0)
    # rough score
    seq = difflib.SequenceMatcher(None, query, best[0])
    return (best[0], seq.ratio())

def line_total_length(line):
    if isinstance(line, LineString):
        return line.length
    elif isinstance(line, MultiLineString):
        return sum(ls.length for ls in line.geoms)
    return 0.0

def line_interpolate_point(line, frac: float) -> Point:
    # frac in [0,1] along total length
    L = line_total_length(line)
    t = max(0.0, min(1.0, frac))
    d = L * t
    if isinstance(line, LineString):
        return line.interpolate(d)
    elif isinstance(line, MultiLineString):
        acc = 0.0
        for ls in line.geoms:
            if acc + ls.length >= d:
                return ls.interpolate(d - acc)
            acc += ls.length
        return line.geoms[-1].interpolate(line.geoms[-1].length)  # end
    else:
        return Point()

def point_side_of_segment(p: Point, a: Point, b: Point) -> float:
    # >0 left, <0 right relative to segment a->b
    return (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x)

def nearest_segment(line: LineString, p: Point) -> Tuple[Point, Tuple[float,float]]:
    # returns (projection point, (ax, ay, bx, by)) for the closest segment of a LineString
    if isinstance(line, MultiLineString):
        # choose the nearest LS
        min_d = 1e18
        best = None
        for ls in line.geoms:
            proj = ls.interpolate(ls.project(p))
            d = proj.distance(p)
            if d < min_d:
                min_d = d; best = (proj, ls)
        line = best[1]
    # now single LineString
    coords = list(line.coords)
    proj = line.interpolate(line.project(p))
    min_d = 1e18; seg = (coords[0], coords[1])
    for i in range(len(coords)-1):
        a = Point(coords[i]); b = Point(coords[i+1])
        # approximate distance to segment by distance to its mid projection
        # (we already know the projection on whole line is proj)
        d = proj.distance(a) + proj.distance(b)  # cheap proxy
        if d < min_d:
            min_d = d; seg = (a, b)
    return proj, (seg[0].x, seg[0].y, seg[1].x, seg[1].y)

def offset_point_along_normal(line, p: Point, distance: float, favor_left: bool) -> Point:
    # Offset p by +/- distance perpendicular to the nearest segment direction.
    proj, (ax, ay, bx, by) = nearest_segment(line if isinstance(line, LineString) else linemerge(line), p)
    vx, vy = (bx - ax, by - ay)
    # unit normal to the left of segment
    norm_len = (vx**2 + vy**2)**0.5 + 1e-9
    nx, ny = (-vy / norm_len, vx / norm_len)
    if not favor_left:
        nx, ny = -nx, -ny
    return Point(p.x + nx * distance, p.y + ny * distance)

def build_street_index(streets_gdf, name_field: str) -> Tuple[dict, dict]:
    # Returns: name_norm -> [indices], and comp_id -> geometry
    name_to_idx = {}
    comp_geom = {}
    for idx, row in streets_gdf.iterrows():
        nm = normalize_street_name(str(row.get(name_field, "")))
        if nm not in name_to_idx:
            name_to_idx[nm] = []
        name_to_idx[nm].append(idx)
        comp_geom[idx] = row.geometry
    return name_to_idx, comp_geom

def choose_side_for_parity(street_geom, parcels_gdf) -> Tuple[str, str]:
    """
    Heuristic: decide which side is 'odd' vs 'even' by comparing parcel density left/right.
    Returns ('odd_is', 'even_is') with values in {'left','right'}.
    """
    # Sample 50 points along line, count parcels on each side by centroid orientation.
    n_samples = 50
    left_count, right_count = 0, 0
    if parcels_gdf.empty:
        # fallback
        return ('left','right')
    if isinstance(street_geom, MultiLineString):
        line = linemerge(street_geom)
    else:
        line = street_geom
    coords = list(line.coords)
    if len(coords) < 2:
        return ('left','right')
    # Precompute parcel centroids
    cents = parcels_gdf.geometry.centroid
    # Sample midpoints of segments
    for i in range(len(coords)-1):
        a = Point(coords[i]); b = Point(coords[i+1])
        mid = Point((a.x+b.x)/2.0, (a.y+b.y)/2.0)
        vec = Point(b.x - a.x, b.y - a.y)
        # Count nearest K parcels and their side sign
        # Here K=20 in a small radius
        subset = parcels_gdf[cents.distance(mid) < 80]  # 80 m radius window
        for _, prow in subset.iterrows():
            p = prow.geometry.centroid
            sgn = point_side_of_segment(p, a, b)
            if sgn > 0:
                left_count += 1
            elif sgn < 0:
                right_count += 1
    if left_count >= right_count:
        return ('left','right')
    else:
        return ('right','left')

def isotonic_positions(indices: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """
    Given ordinal indices (0..n-1) and target housenumbers (monotone ascending),
    compute smoothed positions (0..1). If sklearn not available, return normalized ranks.
    """
    n = len(indices)
    if n == 0:
        return np.array([])
    # Normalize indices to [0,1]
    x = (indices - indices.min()) / max(1e-9, (indices.max() - indices.min()))
    y = numbers.astype(float)
    # Safeguard: strictly increasing y
    y = pd.Series(y).fillna(method='ffill').fillna(method='bfill').to_numpy()
    if HAVE_ISO:
        try:
            iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
            y_fit = iso.fit_transform(x, y)
            # Return positions proportional to rank of y_fit
            order = np.argsort(y_fit)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n)
            pos = ranks / max(1, n-1)
            return pos
        except Exception:
            pass
    # Fallback: simple rank
    order = np.argsort(y)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)
    pos = ranks / max(1, n-1)
    return pos

def main():
    ap = argparse.ArgumentParser(description="Interpolate/geocode addresses from Parquet + Shapefiles (no DB).")
    ap.add_argument('--addresses-parquet', required=True, help='Parquet file with addresses (no geometry).')
    ap.add_argument('--streets-shp', required=True, help='Shapefile of streets (LineString) with a name field.')
    ap.add_argument('--parcels-shp', required=True, help='Shapefile of parcels (Polygon) with HRSZ field (optional but recommended).')
    ap.add_argument('--out-shp', required=True, help='Output Shapefile path (points).')
    ap.add_argument('--streets-name-field', default='name', help='Field name in streets for street name (default: name).')
    ap.add_argument('--parcels-hrsz-field', default='hrsz', help='Field name in parcels for HRSZ (default: hrsz).')
    ap.add_argument('--crs-epsg', default='23700', help='EPSG code to use for processing (default: 23700 HD72/EOV).')
    # Address column mapping
    ap.add_argument('--map-id', default='id')
    ap.add_argument('--map-addr', default='address')
    ap.add_argument('--map-street', default='street')
    ap.add_argument('--map-type', default='type')
    ap.add_argument('--map-hn', default='housenumber')
    ap.add_argument('--map-hrsz', default='hrsz')
    ap.add_argument('--map-settlement', default='settlement')
    # Matching thresholds
    ap.add_argument('--min-street-sim', type=float, default=0.55, help='Minimum fuzzy similarity to accept street match.')
    ap.add_argument('--side-offset', type=float, default=1.5, help='Offset distance (meters) to place points on side of street.')
    args = ap.parse_args()

    epsg = f"EPSG:{args.crs_epsg}"

    # Load data
    df = pd.read_parquet(args.addresses_parquet)
    # Ensure needed columns exist
    for c in [args.map_id, args.map_addr, args.map_street, args.map_hn, args.map_hrsz, args.map_settlement]:
        if c not in df.columns:
            # Create missing as empty
            df[c] = None

    gdf_streets = gpd.read_file(args.streets_shp)
    gdf_parcels = gpd.read_file(args.parcels_shp)

    # Reproject to EPSG if needed
    if gdf_streets.crs is None:
        warnings.warn("Streets CRS unknown; assuming EPSG:%s" % args.crs_epsg)
        gdf_streets.set_crs(epsg, inplace=True)
    else:
        gdf_streets = gdf_streets.to_crs(epsg)
    if gdf_parcels.crs is None:
        warnings.warn("Parcels CRS unknown; assuming EPSG:%s" % args.crs_epsg)
        gdf_parcels.set_crs(epsg, inplace=True)
    else:
        gdf_parcels = gdf_parcels.to_crs(epsg)

    # Normalized street name index for streets
    gdf_streets['__name_norm'] = gdf_streets[args.streets_name_field].astype(str).apply(normalize_street_name)
    name_to_idx, comp_geom = build_street_index(gdf_streets, '__name_norm')

    # Prepare parcels centroids (for side decision)
    gdf_parcels['__centroid'] = gdf_parcels.geometry.centroid

    # Normalize address-side fields
    df['__street_raw'] = df[args.map_street].astype(str).where(df[args.map_street].notna(), df[args.map_addr])
    df['__street_norm'] = df['__street_raw'].astype(str).apply(normalize_street_name)
    df['__hn_raw'] = df[args.map_hn].astype(str).where(df[args.map_hn].notna(), df[args.map_addr])
    parsed = df['__hn_raw'].apply(parse_housenumber)
    df['__hn_num'] = parsed.apply(lambda x: x[0])
    df['__hn_suffix'] = parsed.apply(lambda x: x[1])
    df['__hn_range_lo'] = parsed.apply(lambda x: x[2][0] if x[2] else None)
    df['__hn_range_hi'] = parsed.apply(lambda x: x[2][1] if x[2] else None)
    df['__parity'] = df['__hn_num'].apply(lambda n: 'even' if (n is not None and n % 2 == 0) else ('odd' if n is not None else None))

    # Fuzzy match street names
    street_names = list(name_to_idx.keys())
    matches = df['__street_norm'].apply(lambda q: fuzzy_match_one(q, street_names))
    df['__matched_name'] = matches.apply(lambda m: m[0])
    df['__sim'] = matches.apply(lambda m: m[1])
    df.loc[(df['__matched_name'].isna()) | (df['__sim'] < args.min_street_sim), '__matched_name'] = None

    # Prepare output containers
    out_rows = []
    # Precompute per-street geometry and side mapping
    per_street_cache = {}

    # Quick helper: get geometry for a matched street (merge all comps with that name)
    def get_street_geom_for_name(nm: str):
        idxs = name_to_idx.get(nm, [])
        if not idxs:
            return None, []
        geoms = [gdf_streets.loc[i, 'geometry'] for i in idxs]
        merged = unary_union(geoms)
        # Flatten to merged multiline if possible
        if isinstance(merged, (MultiLineString, LineString)):
            geom = merged
        else:
            # attempt linemerge on union of LineStrings
            all_lines = []
            for g in geoms:
                if isinstance(g, LineString):
                    all_lines.append(g)
                elif isinstance(g, MultiLineString):
                    all_lines.extend(list(g.geoms))
            if all_lines:
                geom = linemerge(MultiLineString(all_lines))
            else:
                geom = merged
        return geom, idxs

    # Build per-street groups from addresses
    grouped = df.groupby('__matched_name', dropna=False)
    for street_name, sdf in grouped:
        if street_name is None:
            # Can't match street; skip (will output with low quality at settlement centroid if available later)
            continue
        street_geom, idxs = get_street_geom_for_name(street_name)
        if street_geom is None:
            continue

        # Restrict parcels to a buffer around this street to speed up
        buf = gpd.GeoSeries([street_geom], crs=gdf_streets.crs).buffer(40).iloc[0]
        parcels_near = gdf_parcels[gdf_parcels.intersects(buf)].copy()
        odd_side, even_side = choose_side_for_parity(street_geom, parcels_near)

        # SPLIT addresses by parity
        for parity in ['odd','even']:
            asub = sdf[sdf['__parity'] == parity].copy()
            if asub.empty:
                continue
            # Sort by housenumber (base); handle ranges by using low end as key
            asub['__num_for_sort'] = asub['__hn_num']
            mask_range = asub['__hn_range_lo'].notna()
            asub.loc[mask_range, '__num_for_sort'] = asub.loc[mask_range, '__hn_range_lo']
            asub = asub.sort_values('__num_for_sort')
            # Ordinal index
            asub['__ord'] = np.arange(len(asub))

            # Derive positions along the street [0..1]
            nums = asub['__num_for_sort'].fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)
            idxs_ord = asub['__ord'].to_numpy(dtype=float)
            pos = isotonic_positions(idxs_ord, nums)  # 0..1
            asub['__m'] = pos

            # Create points along line, offset to chosen side
            side_flag = True if (parity == 'odd' and odd_side == 'left') or (parity == 'even' and even_side == 'left') else False
            points = []
            for m in asub['__m']:
                p = line_interpolate_point(street_geom, m)
                p2 = offset_point_along_normal(street_geom, p, args.side_offset, favor_left=side_flag)
                points.append(p2)
            asub['geometry'] = points

            # Quality score (heuristic): similarity + spread
            sim = asub['__sim'].fillna(0.0).to_numpy()
            # If isotonic used, slight boost
            iso_used = 1.0 if HAVE_ISO else 0.8
            q = 0.6*sim + 0.4*iso_used
            asub['quality'] = np.clip(q, 0, 1)

            # Explain
            asub['explain'] = asub.apply(
                lambda r: f"street='{street_name}', parity={parity}, sim={r['__sim']:.2f}, method={'iso' if HAVE_ISO else 'rank'}",
                axis=1
            )

            # Append to output
            out_rows.append(asub)

    # Addresses that failed to match any street: place at centroid of nearest parcel by HRSZ (if available),
    # else drop with very low quality.
    unmatched = df[df['__matched_name'].isna()].copy()
    if not unmatched.empty:
        # Try by HRSZ
        hrsz_field = args.map_hrsz
        if hrsz_field in df.columns and args.parcels_hrsz_field in gdf_parcels.columns:
            # Normalize HRSZ strings for join
            def norm_hrsz(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                s = str(x).strip().replace(' ', '')
                s = s.replace('/', '-')
                return s
            gdf_parcels['__hrsz_norm'] = gdf_parcels[args.parcels_hrsz_field].astype(str).apply(norm_hrsz)
            unmatched['__hrsz_norm'] = unmatched[hrsz_field].astype(str).apply(norm_hrsz)
            um = unmatched.merge(gdf_parcels[['__hrsz_norm','geometry']], left_on='__hrsz_norm', right_on='__hrsz_norm', how='left')
            # Place at parcel "front point" (nearest point to nearest street)
            placed_rows = []
            for _, r in um.iterrows():
                geom = r.get('geometry', None)
                if geom is None or geom is np.nan:
                    continue
                # pick nearest street from full set
                # (for speed, could spatial index; here simple nearest via centroid)
                cent = geom.centroid
                # naive nearest: iterate few (OK for moderate data)
                min_d = 1e18; best = None
                for idx, srow in gdf_streets.iterrows():
                    d = srow.geometry.distance(cent)
                    if d < min_d:
                        min_d = d; best = srow.geometry
                # project centroid to street and then offset slightly
                proj = nearest_points(best, cent)[0]
                pt = offset_point_along_normal(best, proj, 1.0, favor_left=True)
                row = r.copy()
                row['geometry'] = pt
                row['quality'] = 0.55
                row['explain'] = "unmatched street; HRSZ snap to nearest street"
                placed_rows.append(row)
            if placed_rows:
                out_rows.append(pd.DataFrame(placed_rows))
        # else: ignore

    # Concatenate outputs
    if not out_rows:
        print("No output rows produced. Check input columns and thresholds.", file=sys.stderr)
        sys.exit(2)
    out_df = pd.concat(out_rows, ignore_index=True, sort=False)

    # Assemble GeoDataFrame
    out_gdf = gpd.GeoDataFrame(out_df, geometry='geometry', crs=gdf_streets.crs)

    # Keep selected columns in output
    keep_cols = [args.map_id, args.map_addr, args.map_street, args.map_hn, args.map_hrsz, args.map_settlement,
                 '__matched_name', '__sim', 'quality', 'explain']
    keep_cols = [c for c in keep_cols if c in out_gdf.columns]
    out_gdf = out_gdf[keep_cols + ['geometry']]

    # Write Shapefile
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.out_shp), exist_ok=True)
    out_gdf.to_file(args.out_shp)

    print(f"Done. Wrote {len(out_gdf)} features to {args.out_shp}")

if __name__ == "__main__":
    main()
