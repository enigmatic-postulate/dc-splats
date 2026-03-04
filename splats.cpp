#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#include <dc/pvr.h>
#include <dc/maple.h>
#include <dc/fmath.h>
#include <dc/maple/controller.h>

#include <kos/init.h>

#include <cmath>
#include <vector>
#include <cassert>

#define PVR_HDR_SIZE 16

#pragma pack(push, 1)
struct SplatVertex {
    float x, y, z;
    float qx, qy, qz, qw;
    float sx, sy, sz;
    uint8_t r, g, b, a;
};

struct SplatHeader {
    uint32_t num_opaque;
    uint32_t num_translucent;
};
#pragma pack(pop)

static_assert(sizeof(SplatVertex) == 44);


struct Vec3 { float x, y, z; };
struct Mat4 { float m[16]; };

static Mat4 mat4_perspective(float fovy, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fovy * 0.5f);
    Mat4 r = {};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

static Mat4 mat4_lookdir(Vec3 pos, Vec3 fwd, Vec3 up) {
    Vec3 right = {
        fwd.y * up.z - fwd.z * up.y,
        fwd.z * up.x - fwd.x * up.z,
        fwd.x * up.y - fwd.y * up.x
    };
    float rlen = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right.x /= rlen; right.y /= rlen; right.z /= rlen;
    Vec3 u = {
        right.y * fwd.z - right.z * fwd.y,
        right.z * fwd.x - right.x * fwd.z,
        right.x * fwd.y - right.y * fwd.x
    };
    Mat4 r = {};
    r.m[0] = right.x; r.m[4] = right.y; r.m[8]  = right.z;
    r.m[1] = u.x;     r.m[5] = u.y;     r.m[9]  = u.z;
    r.m[2] = -fwd.x;  r.m[6] = -fwd.y;  r.m[10] = -fwd.z;
    r.m[15] = 1.0f;
    r.m[12] = -(right.x * pos.x + right.y * pos.y + right.z * pos.z);
    r.m[13] = -(u.x * pos.x + u.y * pos.y + u.z * pos.z);
    r.m[14] = (fwd.x * pos.x + fwd.y * pos.y + fwd.z * pos.z);
    return r;
}

static Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 r = {};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                r.m[j * 4 + i] += a.m[k * 4 + i] * b.m[j * 4 + k];
    return r;
}

static void mat4_transform(const Mat4 &m, float x, float y, float z,
                           float &ox, float &oy, float &oz) {
    float w = m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15];
    float inv_w = 1.0f / w;
    ox = (m.m[0] * x + m.m[4] * y + m.m[8]  * z + m.m[12]) * inv_w;
    oy = (m.m[1] * x + m.m[5] * y + m.m[9]  * z + m.m[13]) * inv_w;
    oz = inv_w; // PVR expects 1/w for depth
}

static void quat_rotate(float qx, float qy, float qz, float qw,
                         float vx, float vy, float vz,
                         float &ox, float &oy, float &oz) {
    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

pvr_ptr_t loadtxr(const char *fname) {
    FILE *tex = NULL;
    unsigned char *texBuf;
    uint8_t HDR[PVR_HDR_SIZE];
    unsigned int texSize;
    
    tex = fopen(fname, "rb");

    if(tex == NULL) {
        fprintf(stderr, "FILE READ ERROR: %s\n", fname);
        return NULL;
    }

    fseek(tex, 0, SEEK_END);
    texSize = ftell(tex) - PVR_HDR_SIZE;
    fseek(tex, 0, SEEK_SET);

    /* Read in the PVR texture file header */
    fread(HDR, 1, PVR_HDR_SIZE, tex);

    texBuf = (unsigned char*)malloc(texSize);
    fread(texBuf, 1, texSize, tex); /* Read in the PVR texture data */
    fclose(tex);

    int texW = HDR[PVR_HDR_SIZE - 4] | HDR[PVR_HDR_SIZE - 3] << 8;
    int texH = HDR[PVR_HDR_SIZE - 2] | HDR[PVR_HDR_SIZE - 1] << 8;
    uint32_t color = (uint32_t)HDR[PVR_HDR_SIZE - 8];
    uint32_t format = (uint32_t)HDR[PVR_HDR_SIZE - 7];
    bool twiddled = format == 0x01;
    bool compressed = (format == 0x10 || format == 0x03);

    auto rv = pvr_mem_malloc(texSize);

    if(rv == NULL) {
        fprintf(stderr, "PVR MEM ALLOC ERROR: %s\n", fname);
        free(texBuf);
        return NULL;
    }

    /* Load the texture into PVR memory */
    pvr_txr_load(texBuf, rv, texSize);
    free(texBuf);
    printf("Loaded texture %s: %dx%d, color format 0x%02x, %s, %s\n",
           fname, texW, texH, color,
           twiddled ? "twiddled" : "linear",
           compressed ? "compressed" : "uncompressed");
    return rv;
}

static pvr_sprite_txr_t sprites;
static pvr_sprite_hdr_t shdr;
static pvr_ptr_t splat;
std::vector<SplatVertex> splats;

static Vec3 camPos = {0.0f, 0.0f, 2.0f};
static float camYaw = -3.14159f;
static float camPitch = 0.0f;

static const float QUAD_SCALE = 6.0f; // +-3 sigma

static void setup(void) {
    
    
    // ---- Load splat data ----
    FILE *f = fopen("/rd/cactus.bin", "rb");
    if (!f) {
        fprintf(stderr, "Cannot open /rd/cactus.bin\n");
        exit(EXIT_FAILURE);
    }

    SplatHeader hdr;
    assert(fread(&hdr, sizeof(hdr), 1, f) == 1);
    uint32_t total = hdr.num_opaque + hdr.num_translucent;
    splats.resize(total);

    assert(fread(splats.data(), sizeof(SplatVertex), total, f) == total);
    fclose(f);

    printf("Loaded %lu splats (%lu opaque, %lu translucent)\n", total, hdr.num_opaque, hdr.num_translucent);

    if(!(splat = loadtxr("/rd/splat_texture.pvr"))) {
        exit(EXIT_FAILURE);
    }
    
    pvr_sprite_cxt_t cxt;
    pvr_sprite_cxt_txr(&cxt, PVR_LIST_TR_POLY, PVR_TXRFMT_ARGB4444 | PVR_TXRFMT_TWIDDLED | PVR_TXRFMT_VQ_ENABLE, 64, 64, splat, PVR_FILTER_BILINEAR);
    cxt.gen.culling = PVR_CULLING_NONE;
    cxt.depth.comparison = PVR_DEPTHCMP_ALWAYS;
    pvr_sprite_compile(&shdr, &cxt);

    /* Set up the two sprites. */

    sprites.dummy = 0;
    sprites.auv = PVR_PACK_16BIT_UV(0.0f, 1.0f);
    sprites.buv = PVR_PACK_16BIT_UV(0.0f, 0.0f);
    sprites.cuv = PVR_PACK_16BIT_UV(1.0f, 0.0f);
}

static int check_start(void) {
    maple_device_t *cont;
    cont_state_t *state;

    cont = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);

    if(cont) {
        state = (cont_state_t *)maple_dev_status(cont);

        if(!state)
            return 0;

        if(state->buttons & CONT_START)
            return 1;

        camPitch += (state->joyy / 64.0f) * 0.02f;
        camYaw += (state->joyx / 64.0f) * 0.02f;

        if (state->buttons & CONT_DPAD_UP) {
            camPos.x += sinf(camYaw) * 0.1f;
            camPos.z += cosf(camYaw) * 0.1f;
        }
        if (state->buttons & CONT_DPAD_DOWN) {
            camPos.x -= sinf(camYaw) * 0.1f;
            camPos.z -= cosf(camYaw) * 0.1f;
        }
        if (state->buttons & CONT_DPAD_LEFT) {
            camPos.x += cosf(camYaw) * 0.1f;
            camPos.z -= sinf(camYaw) * 0.1f;
        }
        if (state->buttons & CONT_DPAD_RIGHT) {
            camPos.x -= cosf(camYaw) * 0.1f;
            camPos.z += sinf(camYaw) * 0.1f;
        }
    }

    return 0;
}

static const float quadOffsets[4][2] = {
    {-0.5f, +0.5f}, { -0.5f, -0.5f}, {+0.5f,  -0.5f}, { 0.5f,  0.5f}
};

static void do_frame(void) {

     Vec3 fwd = {
        cosf(camPitch) * sinf(camYaw),
        sinf(camPitch),
        cosf(camPitch) * cosf(camYaw)
    };
    Vec3 right = { sinf(camYaw - 1.5708f), 0.0f, cosf(camYaw - 1.5708f) };

    Mat4 view = mat4_lookdir(camPos, fwd, {0.0f, 1.0f, 0.0f});
    Mat4 proj = mat4_perspective(1.0f, 640.0f / 480.0f, 0.01f, 100.0f);
    Mat4 vp = mat4_mul(proj, view);

    float projScale = proj.m[5] * 480 * 0.5f; // world-size to pixels factor (at depth=1)

    pvr_wait_ready();
    pvr_scene_begin();

    pvr_list_begin(PVR_LIST_TR_POLY);

    for (uint32_t si = 0; si < splats.size(); si++) {
        const SplatVertex &s = splats[si];

        // Edge-on culling: compute splat normal (local Z rotated by quaternion)
        float nx, ny, nz;
        quat_rotate(s.qx, s.qy, s.qz, s.qw, 0.0f, 0.0f, 1.0f, nx, ny, nz);

        // View direction from camera to splat
        float vdx = s.x - camPos.x;
        float vdy = s.y - camPos.y;
        float vdz = s.z - camPos.z;
        float vlen = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
        if (vlen > 0.0f) { vdx /= vlen; vdy /= vlen; vdz /= vlen; }

        float facing = fabsf(nx*vdx + ny*vdy + nz*vdz);
        if (facing < 0.05f) continue;

        // Scale: exp(log-space)
        float scx = expf(s.sx);
        float scy = expf(s.sy);
        float scz = expf(s.sz);

        // Screen-size LOD: approximate pixel size from largest scale axis
        float sizeFade = 1.0f;
        if (1) {
            float maxScale = std::max(scx, std::max(scy, scz));
            float dx = splats[si].x - camPos.x;
            float dy = splats[si].y - camPos.y;
            float dz = splats[si].z - camPos.z;
            float depth = view.m[2]*dx + view.m[6]*dy + view.m[10]*dz;

            float viewDist = fabsf(depth);
            float pixelSize = (viewDist > 0.001f) ? (maxScale * projScale / viewDist) : 10000.0f;

            if (pixelSize < 4.0f) continue;

            if (1 && pixelSize < 16.0f)
                sizeFade = (pixelSize - 4.0f) / (16.0f - 4.0f);
        }
        
        shdr.argb = (int32_t)(s.a * sizeFade) << 24 | s.r << 16 | s.g << 8 | s.b;

        sprites.flags = (si == splats.size() - 1) ? PVR_CMD_VERTEX_EOL : PVR_CMD_VERTEX;
        
        // Transform all 4 vertices: local -> world -> clip -> screen
        float screenVerts[4][3];
        for (int vi = 0; vi < 4; vi++) {
            float lx = quadOffsets[vi][0] * QUAD_SCALE * scx;
            float ly = quadOffsets[vi][1] * QUAD_SCALE * scy;
            float lz = 0.0f;

            // Rotate by quaternion into world space
            float wx, wy, wz;
            quat_rotate(s.qx, s.qy, s.qz, s.qw, lx, ly, lz, wx, wy, wz);
            wx += s.x;
            wy += s.y;
            wz += s.z;

            // View-projection transform (includes perspective divide)
            float cx, cy, cz;
            mat4_transform(vp, wx, wy, wz, cx, cy, cz);

            // NDC to screen coordinates
            screenVerts[vi][0] = (cx * 0.5f + 0.5f) * 640.0f;
            screenVerts[vi][1] = (1.0f - (cy * 0.5f + 0.5f)) * 480.0f;
            screenVerts[vi][2] = cz;
        }

        sprites.ax = screenVerts[0][0];
        sprites.ay = screenVerts[0][1];
        sprites.az = screenVerts[0][2];

        sprites.bx = screenVerts[1][0];
        sprites.by = screenVerts[1][1];
        sprites.bz = screenVerts[1][2];

        sprites.cx = screenVerts[2][0];
        sprites.cy = screenVerts[2][1];
        sprites.cz = screenVerts[2][2];

        sprites.dx = screenVerts[3][0];
        sprites.dy = screenVerts[3][1];

        // printf("Splat %u: screen coords = (%f, %f, %f), (%f, %f, %f), (%f, %f, %f), (%f, %f)\n",
        //        si,
        //        sprites.ax, sprites.ay, sprites.az,
        //        sprites.bx, sprites.by, sprites.bz,
        //        sprites.cx, sprites.cy, sprites.cz,
        //        sprites.dx, sprites.dy);

        pvr_prim(&shdr, sizeof(pvr_sprite_hdr_t));
        pvr_prim(&sprites, sizeof(pvr_sprite_txr_t));
    }
    
    pvr_list_finish();

    pvr_scene_finish();
}

static pvr_init_params_t pvr_params = {
    /* Enable only opaque and punchthru polygons. */
    {
        PVR_BINSIZE_0, PVR_BINSIZE_0, PVR_BINSIZE_16, PVR_BINSIZE_0,
        PVR_BINSIZE_0
    },
    2 * 1024 * 1024 + 512 * 1024, 0, 0, 1, 35, 0
};

int main(int argc, char *argv[]) {

    pvr_init(&pvr_params);

    setup();

    /* Go as long as the user hasn't pressed start on controller 1. */
    while(!check_start()) {
        do_frame();
    }

    pvr_mem_free(splat);

    return 0;
}
