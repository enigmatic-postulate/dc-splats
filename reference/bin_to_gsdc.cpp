// bin_to_gsdc.cpp
// Converts old 44-byte splat binary to GSDC format, selecting the N most
// significant splats for maximum spatial coverage.
//
// Scoring: opacity * area of the two largest scale axes (screen-filling metric)
// Spatial pass: scene is divided into a 3D grid; within each cell the best
// splats are kept first, then remaining budget is filled globally by score.
// This prevents big splats in one region from consuming the entire budget.
//
// Build (MSVC):  cl /O2 /std:c++17 bin_to_gsdc.cpp
// Build (MinGW): g++ -O2 -std=c++17 -o bin_to_gsdc bin_to_gsdc.cpp
// Run:           bin_to_gsdc <input.bin> <output_gsdc.bin> [max_splats=25000]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_map>

// ---------- Input format (cactus.bin) ----------
#pragma pack(push, 1)
struct InputSplat {
    float x, y, z;
    float qx, qy, qz, qw;
    float sx, sy, sz;   // log-space scales
    uint8_t r, g, b, a;
};
struct InputHeader {
    uint32_t num_opaque;
    uint32_t num_translucent;
};
#pragma pack(pop)
static_assert(sizeof(InputSplat) == 44);

// ---------- Output format (GSDC v1) ----------
#pragma pack(push, 1)
struct GSDCSplat {
    float x, y, z;
    float r, g, b, a;       // normalised 0-1
    float scale_x, scale_y; // billboard half-size in world units
};
static const uint32_t GSDC_MAGIC   = 0x43445347u; // "GSDC"
static const uint32_t GSDC_VERSION = 1u;
#pragma pack(pop)
static_assert(sizeof(GSDCSplat) == 36);

// ---------- Scoring ----------
static float splat_score(const InputSplat &s) {
    float opacity = s.a / 255.0f;
    if (opacity < 0.02f) return 0.0f;

    float scales[3] = { expf(s.sx), expf(s.sy), expf(s.sz) };
    // Sort descending to get the two largest axes
    if (scales[0] < scales[1]) std::swap(scales[0], scales[1]);
    if (scales[1] < scales[2]) std::swap(scales[1], scales[2]);
    if (scales[0] < scales[1]) std::swap(scales[0], scales[1]);

    // Screen-filling metric: area of the billboard footprint * opacity
    return opacity * scales[0] * scales[1];
}

// ---------- Convert one splat ----------
static GSDCSplat convert(const InputSplat &s) {
    GSDCSplat g;
    g.x = s.x; g.y = s.y; g.z = s.z;
    g.r = s.r / 255.0f;
    g.g = s.g / 255.0f;
    g.b = s.b / 255.0f;
    g.a = s.a / 255.0f;

    float scales[3] = { expf(s.sx), expf(s.sy), expf(s.sz) };
    if (scales[0] < scales[1]) std::swap(scales[0], scales[1]);
    if (scales[1] < scales[2]) std::swap(scales[1], scales[2]);
    if (scales[0] < scales[1]) std::swap(scales[0], scales[1]);

    g.scale_x = scales[0];
    g.scale_y = scales[1];
    return g;
}

// ---------- Spatial cell key ----------
static uint64_t cell_key(float x, float y, float z, float cell_size) {
    int64_t ix = (int64_t)floorf(x / cell_size);
    int64_t iy = (int64_t)floorf(y / cell_size);
    int64_t iz = (int64_t)floorf(z / cell_size);
    // Pack into uint64 (bias by 0x1FFF to handle negatives in 14 bits each)
    return ((uint64_t)(ix + 0x1FFF) & 0x3FFF) |
           (((uint64_t)(iy + 0x1FFF) & 0x3FFF) << 14) |
           (((uint64_t)(iz + 0x1FFF) & 0x3FFF) << 28);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output_gsdc.bin> [max_splats=25000]\n", argv[0]);
        return 1;
    }

    uint32_t max_splats = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 25000u;

    // ---------- Read input ----------
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    InputHeader ihdr;
    fread(&ihdr, sizeof(ihdr), 1, fin);
    uint32_t total = ihdr.num_opaque + ihdr.num_translucent;
    printf("Input: %u opaque + %u translucent = %u splats\n",
           ihdr.num_opaque, ihdr.num_translucent, total);

    std::vector<InputSplat> splats(total);
    fread(splats.data(), sizeof(InputSplat), total, fin);
    fclose(fin);

    // ---------- Score all splats ----------
    struct Scored { float score; uint32_t idx; };
    std::vector<Scored> scored(total);
    for (uint32_t i = 0; i < total; i++) {
        scored[i] = { splat_score(splats[i]), i };
    }

    // Global sort descending by score
    std::sort(scored.begin(), scored.end(),
              [](const Scored &a, const Scored &b) { return a.score > b.score; });

    // Drop zero-opacity splats
    while (!scored.empty() && scored.back().score <= 0.0f)
        scored.pop_back();

    printf("Non-trivial splats: %zu\n", scored.size());

    // ---------- Spatial coverage pass ----------
    // Compute scene bounds from scored splats (top half by score)
    float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
    for (size_t i = 0; i < scored.size(); i++) {
        const auto &s = splats[scored[i].idx];
        mn[0] = std::min(mn[0], s.x); mx[0] = std::max(mx[0], s.x);
        mn[1] = std::min(mn[1], s.y); mx[1] = std::max(mx[1], s.y);
        mn[2] = std::min(mn[2], s.z); mx[2] = std::max(mx[2], s.z);
    }
    float extent = std::max({mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]});
    printf("Scene extent: %.3f  (%.2f,%.2f,%.2f) - (%.2f,%.2f,%.2f)\n",
           extent, mn[0],mn[1],mn[2], mx[0],mx[1],mx[2]);

    // Target ~8 splats per cell on average across the volume
    // cell_size chosen so grid cells^3 ≈ max_splats / 8
    float cells_per_axis = cbrtf((float)max_splats / 8.0f);
    float cell_size = (cells_per_axis > 0.0f) ? (extent / cells_per_axis) : extent;
    cell_size = std::max(cell_size, 1e-4f);
    printf("Grid cell size: %.4f\n", cell_size);

    // Per-cell quota: how many splats each cell may contribute in the first pass.
    // We use sqrt scaling so large cells don't hog the budget.
    // First pass: give each occupied cell up to `cell_quota` splats, highest score first.
    // Second pass: fill remaining budget globally.
    const uint32_t cell_quota = 16;

    std::vector<bool> selected(total, false);
    std::unordered_map<uint64_t, uint32_t> cell_count;

    std::vector<uint32_t> first_pass, second_pass;
    first_pass.reserve(max_splats);
    second_pass.reserve(scored.size());

    // First pass — per-cell quota (splats already sorted by score, so first seen = best)
    for (const auto &sc : scored) {
        const auto &s = splats[sc.idx];
        uint64_t key = cell_key(s.x, s.y, s.z, cell_size);
        uint32_t &cnt = cell_count[key];
        if (cnt < cell_quota) {
            selected[sc.idx] = true;
            first_pass.push_back(sc.idx);
            cnt++;
            if (first_pass.size() >= max_splats) break;
        } else {
            second_pass.push_back(sc.idx);
        }
    }

    printf("First pass (spatial): %zu splats\n", first_pass.size());

    // Second pass — fill remaining budget by global score rank
    uint32_t remaining = max_splats - (uint32_t)first_pass.size();
    for (uint32_t idx : second_pass) {
        if (remaining == 0) break;
        if (!selected[idx]) {
            first_pass.push_back(idx);
            remaining--;
        }
    }

    printf("Total selected: %zu / %u requested\n", first_pass.size(), max_splats);

    // Sort selected by score descending for consistent output ordering
    std::sort(first_pass.begin(), first_pass.end(), [&](uint32_t a, uint32_t b) {
        return splat_score(splats[a]) > splat_score(splats[b]);
    });

    // ---------- Write GSDC output ----------
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { fprintf(stderr, "Cannot open %s\n", argv[2]); return 1; }

    uint32_t count = (uint32_t)first_pass.size();
    fwrite(&GSDC_MAGIC,   4, 1, fout);
    fwrite(&count,        4, 1, fout);
    fwrite(&GSDC_VERSION, 4, 1, fout);

    for (uint32_t idx : first_pass) {
        GSDCSplat g = convert(splats[idx]);
        fwrite(&g, sizeof(g), 1, fout);
    }
    fclose(fout);

    size_t out_bytes = 12 + (size_t)count * sizeof(GSDCSplat);
    printf("Output: %u splats → %zu bytes (%.1f KB)\n",
           count, out_bytes, out_bytes / 1024.0);

    return 0;
}
