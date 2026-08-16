"""
Metrik ketimpangan spasial yang memisahkan dua sumber ketimpangan yang
berbeda secara konseptual:
- Ketimpangan ANTAR wilayah (cluster GMM, bisa berjarak ratusan km) --
  wajar terjadi karena perbedaan demand antar wilayah.
- Ketimpangan LOKAL antar SPKLU yang benar-benar berdekatan secara fisik
  (default <= 5 km) -- ini yang relevan untuk klaim riset "2 SPKLU
  berdekatan tapi timpang", karena tidak bisa dijelaskan oleh perbedaan
  demand wilayah, hanya oleh dinamika lokal (herding, habit, dst).
"""
import math
import itertools


def theil_t(values):
    """Indeks Theil T atas nilai positif (nol/negatif diabaikan). Dipakai untuk
    dekomposisi bertingkat antar-klaster vs dalam-klaster (lihat
    `theil_nested_by_proximity`), konsisten dengan metodologi
    `Analisis_Kausalitas_Ketimpangan.ipynb` §12."""
    x = [float(v) for v in values if v > 0]
    if len(x) == 0:
        return 0.0
    mu = sum(x) / len(x)
    return sum((v / mu) * math.log(v / mu) for v in x) / len(x)


def calculate_gini(values):
    x = sorted(float(v) for v in values)
    n = len(x)
    total = sum(x)
    if n == 0 or total == 0:
        return 0.0
    weighted_sum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(x))
    return weighted_sum / (n * total)


def _distance_km(loc_a, loc_b):
    return math.dist(loc_a, loc_b)


def cluster_gini(spklus, served_counts):
    """
    Gini DALAM tiap cluster GMM (mengikut field 'cluster_id' tiap SPKLU).

    Returns:
    - dict {cluster_id: gini}
    - float: rata-rata gini antar cluster (yang relevan = ketimpangan lokal)
    """
    groups = {}
    for s in spklus:
        cid = s.get("cluster_id")
        groups.setdefault(cid, []).append(served_counts.get(s["id"], 0))

    per_cluster = {cid: calculate_gini(counts) for cid, counts in groups.items()}
    mean_gini = sum(per_cluster.values()) / len(per_cluster) if per_cluster else 0.0
    return per_cluster, mean_gini


def _proximity_groups_chained(spklus, threshold_km=5.0):
    """
    [Internal -- fallback bila 'cluster_id' tak tersedia] Mengelompokkan
    SPKLU via CONNECTED COMPONENTS: dua SPKLU 'berdekatan' jika jarak
    Euclidean-nya (km) <= threshold_km, ditransitifkan lewat perantara.

    ⚠ PERINGATAN METODOLOGI (ditemukan saat evaluasi gate 2.1d): connected-
    components bisa MERANTAI banyak klaster substitusi asli yang tidak
    saling berdekatan langsung, selama ada rantai perantara <= threshold_km
    di antaranya. Pada dataset produksi (50 SPKLU/10 klaster generator),
    metode ini merantai 5 klaster generator terpisah (yang masing-masing
    dijamin berdiameter <=5km) menjadi SATU grup evaluasi 30 SPKLU yang
    membentang jauh lebih luas -- secara artifisial menggelembungkan
    "disparitas dalam-klaster" karena membandingkan SPKLU dari wilayah yang
    sebenarnya berbeda. Gunakan `proximity_groups` (bukan fungsi ini
    langsung) yang otomatis memakai ground-truth `cluster_id` generator
    (dijamin complete-linkage <= threshold_km, TIDAK merantai) bila tersedia.
    """
    n = len(spklus)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _distance_km(spklus[i]["location"], spklus[j]["location"]) <= threshold_km:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(spklus[i])
    return list(groups.values())


def proximity_groups(spklus, threshold_km=5.0):
    """
    Mengelompokkan SPKLU ke klaster substitusi (>= 1 anggota).

    Prioritas: jika SEMUA record punya field 'cluster_id' (ground-truth dari
    generator sintetik `enforce_local_proximity`, dijamin complete-linkage
    -- diameter internal <= threshold_km, TIDAK bisa merantai keluar), grup
    dibentuk langsung dari situ -- lebih presisi & jauh lebih murah daripada
    menghitung ulang jarak berpasangan.

    Fallback ke connected-components jarak aktual (`_proximity_groups_chained`)
    HANYA bila 'cluster_id' tidak tersedia di semua record (mis. dataset lama
    atau data riil tanpa kolom klaster) -- lihat peringatan metodologi di
    docstring fallback tsb (chaining bisa menggelembungkan disparitas).
    """
    if spklus and all("cluster_id" in s and s["cluster_id"] is not None for s in spklus):
        groups = {}
        for s in spklus:
            groups.setdefault(s["cluster_id"], []).append(s)
        return list(groups.values())
    return _proximity_groups_chained(spklus, threshold_km)


def proximity_gini(spklus, served_counts, threshold_km=5.0):
    """
    Gini LOKAL antar SPKLU yang benar-benar berjarak <= threshold_km satu
    sama lain (bukan sekadar satu cluster GMM yang sama).

    Hanya grup beranggota >= 2 SPKLU yang dihitung gini-nya (grup tunggal
    tidak punya makna ketimpangan).

    Returns:
    - list of dict: [{"spklu_ids": [...], "counts": [...], "gini": ...}, ...]
    - float: rata-rata gini antar grup berdekatan (metrik utama)
    """
    groups = proximity_groups(spklus, threshold_km)
    results = []
    for group in groups:
        if len(group) < 2:
            continue
        counts = [served_counts.get(s["id"], 0) for s in group]
        results.append({
            "spklu_ids": [s["id"] for s in group],
            "counts": counts,
            "gini": calculate_gini(counts),
        })

    mean_gini = sum(r["gini"] for r in results) / len(results) if results else 0.0
    return results, mean_gini


def pairwise_ratio_ge_threshold(spklus, values, threshold_km=5.0, ratio_threshold=3.0):
    """
    Fraksi PASANGAN SPKLU dalam-klaster substitusi (jarak <= threshold_km,
    grup >= 2 anggota, metode connected-components sama dengan
    `proximity_groups`) yang rasio nilainya (max/min) >= ratio_threshold.

    Ini adalah padanan simulasi dari temuan empiris riil (§12
    `Analisis_Kausalitas_Ketimpangan.ipynb`): 42.5% pasangan dalam-radius-
    substitusi punya rasio utilisasi >= 3x MESKI sudah dalam jangkauan
    substitusi geografis yang sama -- dipakai sebagai gate S0 (bukan
    sekadar informasional) karena inilah fenomena inti yang scope tesis
    revisi (Rancangan §1.1) fokuskan.

    Pasangan dengan salah satu nilai <= 0 dilewati (rasio tak terdefinisi).

    Returns:
    - list of dict pasangan {"a", "b", "dist_km", "ratio"}
    - float: fraksi pasangan dengan ratio >= ratio_threshold (NaN bila tak ada pasangan valid)
    """
    groups = proximity_groups(spklus, threshold_km)
    pairs = []
    for group in groups:
        if len(group) < 2:
            continue
        for a, b in itertools.combinations(group, 2):
            va, vb = values.get(a["id"], 0), values.get(b["id"], 0)
            if va <= 0 or vb <= 0:
                continue
            ratio = max(va, vb) / min(va, vb)
            pairs.append({
                "a": a["id"], "b": b["id"],
                "dist_km": _distance_km(a["location"], b["location"]),
                "ratio": ratio,
            })

    if not pairs:
        return pairs, float("nan")
    frac_ge = sum(1 for p in pairs if p["ratio"] >= ratio_threshold) / len(pairs)
    return pairs, frac_ge


def theil_nested_by_proximity(spklus, values, threshold_km=5.0):
    """
    Dekomposisi Theil T bertingkat: total (atas SEMUA SPKLU aktif) -> antar-
    klaster (non-actionable, di luar radius substitusi) vs dalam-klaster
    (actionable oleh rekomendasi, karena SPKLU-nya saling menggantikan
    secara fisik). Metodologi sama dengan
    `Analisis_Kausalitas_Ketimpangan.ipynb` §12b, tapi memakai grup
    connected-components (`proximity_groups`) yang sudah dipakai gate ini,
    bukan hierarchical clustering complete-linkage notebook (catatan: bisa
    sedikit lebih longgar karena connected-components dapat "merantai"
    SPKLU lewat perantara, meski secara praktik jarang terjadi pada radius
    5 km yang ketat).

    Returns dict: {"T_total", "T_between", "T_within", "pct_within"}
    (pct_within = NaN bila tak ada SPKLU aktif untuk didekomposisi)
    """
    groups = proximity_groups(spklus, threshold_km)
    id_to_group = {}
    for gi, group in enumerate(groups):
        for s in group:
            id_to_group[s["id"]] = gi

    active_ids = [sid for sid, v in values.items() if v > 0 and sid in id_to_group]
    if not active_ids:
        return {"T_total": 0.0, "T_between": 0.0, "T_within": 0.0, "pct_within": float("nan")}

    all_vals = [values[sid] for sid in active_ids]
    T_total = theil_t(all_vals)

    total_sum = sum(all_vals)
    n_total = len(all_vals)
    by_group = {}
    for sid in active_ids:
        by_group.setdefault(id_to_group[sid], []).append(values[sid])

    T_between, T_within = 0.0, 0.0
    for g_vals in by_group.values():
        n_g = len(g_vals)
        s_g = sum(g_vals) / total_sum
        p_g = n_g / n_total
        if s_g > 0:
            T_between += s_g * math.log(s_g / p_g)
        if n_g >= 2:
            T_within += s_g * theil_t(g_vals)

    denom = T_between + T_within
    pct_within = (T_within / denom * 100.0) if denom > 0 else float("nan")
    return {"T_total": T_total, "T_between": T_between, "T_within": T_within, "pct_within": pct_within}
