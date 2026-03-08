#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dc/pvr.h>
#include <dc/maple.h>
#include <dc/maple/controller.h>

#include <kos/init.h>

#include <cmath>

#include "gsdc_splat.h"

#define PVR_HDR_SIZE 16

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

pvr_ptr_t loadtxr(const char *fname) {
    FILE *tex = fopen(fname, "rb");
    if (!tex) {
        fprintf(stderr, "FILE READ ERROR: %s\n", fname);
        return NULL;
    }

    uint8_t HDR[PVR_HDR_SIZE];
    fseek(tex, 0, SEEK_END);
    unsigned int texSize = ftell(tex) - PVR_HDR_SIZE;
    fseek(tex, 0, SEEK_SET);
    fread(HDR, 1, PVR_HDR_SIZE, tex);

    unsigned char *texBuf = (unsigned char *)malloc(texSize);
    fread(texBuf, 1, texSize, tex);
    fclose(tex);

    pvr_ptr_t rv = pvr_mem_malloc(texSize);
    if (!rv) {
        fprintf(stderr, "PVR MEM ALLOC ERROR: %s\n", fname);
        free(texBuf);
        return NULL;
    }

    pvr_txr_load(texBuf, rv, texSize);
    free(texBuf);

    int texW = HDR[PVR_HDR_SIZE - 4] | HDR[PVR_HDR_SIZE - 3] << 8;
    int texH = HDR[PVR_HDR_SIZE - 2] | HDR[PVR_HDR_SIZE - 1] << 8;
    printf("Loaded texture %s: %dx%d\n", fname, texW, texH);
    return rv;
}

static GSDCScene scene;
static pvr_ptr_t gauss_txr;

static Vec3 camPos  = {0.0f, 0.0f, 2.0f};
static float camYaw   = -3.14159f;
static float camPitch = 0.0f;

static void setup(void) {
    if (!gsdc_load(&scene, "/rd/cactus_dc.bin")) {
        fprintf(stderr, "Cannot open /rd/cactus_dc.bin\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded %u splats\n", scene.count);

    gauss_txr = loadtxr("/rd/splat_texture.pvr");
    if (!gauss_txr)
        exit(EXIT_FAILURE);
}

static int check_start(void) {
    maple_device_t *cont = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
    if (!cont) return 0;

    cont_state_t *state = (cont_state_t *)maple_dev_status(cont);
    if (!state) return 0;

    if (state->buttons & CONT_START) return 1;

    camPitch += (state->joyy / 64.0f) * 0.02f;
    camYaw   += (state->joyx / 64.0f) * 0.02f;

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

    return 0;
}

static void do_frame(void) {
    Vec3 fwd = {
        cosf(camPitch) * sinf(camYaw),
        -sinf(camPitch),
        cosf(camPitch) * cosf(camYaw)
    };

    Mat4 view = mat4_lookdir(camPos, fwd, {0.0f, 1.0f, 0.0f});
    Mat4 proj = mat4_perspective(1.0f, 640.0f / 480.0f, 0.01f, 100.0f);
    Mat4 vp   = mat4_mul(proj, view);

    gsdc_sort(&scene, fwd.x, fwd.y, fwd.z);

    pvr_wait_ready();
    pvr_scene_begin();

    pvr_list_begin(PVR_LIST_TR_POLY);
    gsdc_submit(&scene, gauss_txr, vp.m);
    pvr_list_finish();

    pvr_scene_finish();
}

static pvr_init_params_t pvr_params = {
    {PVR_BINSIZE_0, PVR_BINSIZE_0, PVR_BINSIZE_16, PVR_BINSIZE_0, PVR_BINSIZE_0},
    29000 * 96, 0, 0, 1, 35, 0   /* 2.78MB: constrained by texture_base = 2*(VB+OPB+TM+FB); leaves ~190KB for textures */
};

int main(int argc, char *argv[]) {
    pvr_init(&pvr_params);
    setup();

    while (!check_start())
        do_frame();

    gsdc_free(&scene);
    pvr_mem_free(gauss_txr);
    return 0;
}
