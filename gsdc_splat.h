/*
 * gsdc_splat.h — Gaussian Splat loader for Dreamcast / KallistiOS
 *
 * Binary format v1 (cactus_dc.bin):
 *   [0..3]   magic   "GSDC"
 *   [4..7]   count   uint32
 *   [8..11]  version uint32 (1)
 *   [12..]   splats  GSDCSplat[count], 36 bytes each
 *
 * Each GSDCSplat:
 *   float x, y, z          — world position
 *   float r, g, b, a       — base colour + opacity (0-1)
 *   float scale_x, scale_y — billboard half-size
 */

#ifndef GSDC_SPLAT_H
#define GSDC_SPLAT_H

#include <stdint.h>
#include <string.h>
#include <kos.h>
#include <dc/pvr.h>

#define GSDC_MAGIC   0x43445347u  /* "GSDC" little-endian */
#define GSDC_VERSION 1

typedef struct {
    float x, y, z;
    float r, g, b, a;
    float scale_x, scale_y;
} __attribute__((packed)) GSDCSplat;

typedef struct {
    uint32_t   count;
    GSDCSplat *splats;
} GSDCScene;

static inline int gsdc_load(GSDCScene *scene, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t header[3];
    fread(header, 4, 3, f);

    if (header[0] != GSDC_MAGIC || header[2] != GSDC_VERSION) {
        fclose(f);
        return 0;
    }

    scene->count  = header[1];
    scene->splats = (GSDCSplat *)malloc(scene->count * sizeof(GSDCSplat));
    if (!scene->splats) { fclose(f); return 0; }

    fread(scene->splats, sizeof(GSDCSplat), scene->count, f);
    fclose(f);
    return 1;
}

/* Depth-sort scratch buffers */
#define GSDC_SORT_BINS  256
#define GSDC_MAX_SPLATS 65536
static float    _gsdc_depths[GSDC_MAX_SPLATS];
static uint16_t _gsdc_order[GSDC_MAX_SPLATS];
static uint16_t _gsdc_bin_pos[GSDC_SORT_BINS];

/* O(n) counting sort — back-to-front */
static inline void gsdc_sort(GSDCScene *scene,
                              float cam_fx, float cam_fy, float cam_fz) {
    uint32_t n = scene->count;

    float dmin = 1e30f, dmax = -1e30f;
    for (uint32_t i = 0; i < n; i++) {
        GSDCSplat *s = &scene->splats[i];
        float d = s->x * cam_fx + s->y * cam_fy + s->z * cam_fz;
        _gsdc_depths[i] = d;
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
    }

    float scale = (dmax > dmin) ? ((float)(GSDC_SORT_BINS - 1) / (dmax - dmin)) : 0.0f;

    uint16_t hist[GSDC_SORT_BINS];
    memset(hist, 0, sizeof(hist));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t b = (uint32_t)((_gsdc_depths[i] - dmin) * scale);
        hist[b]++;
    }

    uint16_t pos = 0;
    for (int b = GSDC_SORT_BINS - 1; b >= 0; b--) {
        _gsdc_bin_pos[b] = pos;
        pos += hist[b];
    }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t b = (uint32_t)((_gsdc_depths[i] - dmin) * scale);
        _gsdc_order[_gsdc_bin_pos[b]++] = (uint16_t)i;
    }
}

/*
 * Combined header+sprite — static BSS placement satisfies pvr_prim()'s
 * 32-byte alignment requirement. Compiled once per texture handle.
 */
typedef struct {
    pvr_sprite_hdr_t hdr;  /* 32 bytes */
    pvr_sprite_txr_t spr;  /* 64 bytes */
} _gsdc_prim_t;            /* 96 bytes = 3×32 */

static _gsdc_prim_t _gsdc_prim;
static pvr_ptr_t    _gsdc_last_txr = NULL;

static inline void gsdc_submit(GSDCScene *scene,
                                pvr_ptr_t gauss_txr,
                                float *view_proj_16f) {
    if (gauss_txr != _gsdc_last_txr) {
        pvr_sprite_cxt_t cxt;
        pvr_sprite_cxt_txr(&cxt, PVR_LIST_TR_POLY,
                           PVR_TXRFMT_ARGB4444 | PVR_TXRFMT_TWIDDLED | PVR_TXRFMT_VQ_ENABLE,
                           64, 64, gauss_txr, PVR_FILTER_BILINEAR);
        pvr_sprite_compile(&_gsdc_prim.hdr, &cxt);
        _gsdc_prim.spr.flags = PVR_CMD_VERTEX_EOL;
        _gsdc_prim.spr.dummy = 0;
        _gsdc_prim.spr.auv = PVR_PACK_16BIT_UV(0.0f, 0.0f);
        _gsdc_prim.spr.buv = PVR_PACK_16BIT_UV(1.0f, 0.0f);
        _gsdc_prim.spr.cuv = PVR_PACK_16BIT_UV(0.0f, 1.0f);
        _gsdc_last_txr = gauss_txr;
    }

    float *m = view_proj_16f;
    float proj_scale = m[5] * 240.0f;

    const float m0=m[0], m4=m[4], m8=m[8],  m12=m[12];
    const float m1=m[1], m5=m[5], m9=m[9],  m13=m[13];
    const float m3=m[3], m7=m[7], m11=m[11],m15=m[15];

    for (uint32_t oi = 0; oi < scene->count; oi++) {
        GSDCSplat *s = &scene->splats[_gsdc_order[oi]];

        /* Y negated: scene uses Y-down (photogrammetry convention) */
        float x = s->x, y = -s->y, z = s->z;
        float cx = m0*x + m4*y + m8*z  + m12;
        float cy = m1*x + m5*y + m9*z  + m13;
        float cw = m3*x + m7*y + m11*z + m15;

        if (cw <= 0.01f) continue;

        float inv_w = 1.0f / cw;
        float sx   = (cx * inv_w *  0.5f + 0.5f) * 640.0f;
        float sy   = (cy * inv_w * -0.5f + 0.5f) * 480.0f;
        float sz   = inv_w;
        float half = (s->scale_x + s->scale_y) * 0.5f * proj_scale * inv_w;

        if (half < 1.0f) continue;

        _gsdc_prim.hdr.argb = ((uint32_t)(s->a * 255.0f) << 24) |
                               ((uint32_t)(s->r * 255.0f) << 16) |
                               ((uint32_t)(s->g * 255.0f) <<  8) |
                               ((uint32_t)(s->b * 255.0f));

        _gsdc_prim.spr.ax = sx - half;  _gsdc_prim.spr.ay = sy - half;  _gsdc_prim.spr.az = sz;
        _gsdc_prim.spr.bx = sx + half;  _gsdc_prim.spr.by = sy - half;  _gsdc_prim.spr.bz = sz;
        _gsdc_prim.spr.cx = sx - half;  _gsdc_prim.spr.cy = sy + half;  _gsdc_prim.spr.cz = sz;
        _gsdc_prim.spr.dx = sx + half;  _gsdc_prim.spr.dy = sy + half;

        pvr_prim(&_gsdc_prim, sizeof(_gsdc_prim));
    }
}

static inline void gsdc_free(GSDCScene *scene) {
    if (scene->splats) free(scene->splats);
    scene->splats = NULL;
    scene->count  = 0;
}

#endif /* GSDC_SPLAT_H */
