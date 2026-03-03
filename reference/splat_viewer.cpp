// Reference OpenGL Gaussian Splat Renderer
// Renders binary splat files produced by tools/splat_converter.cpp
//
// Build: g++ -O2 -std=c++17 -o build/splat_viewer reference/splat_viewer.cpp $(pkg-config --cflags --libs sdl2) -lGLEW -lGL -lm
// Usage: ./build/splat_viewer splats/cactus.bin splats/splat_texture.png

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <SDL.h>
#include <SDL_opengl.h>

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

// ---- Binary format (must match splat_converter output) ----

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

// ---- Simple math ----

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

// ---- CPU vertex format (pre-computed world positions) ----

struct RenderVertex {
    float x, y, z;    // world-space position
    float u, v;        // UV
    float r, g, b, a;  // color
};

// ---- CPU math helpers ----

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

// ---- Shaders ----

static const char *vs_src = R"(
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProj;

out vec2 vUV;
out vec4 vColor;

void main() {
    vUV = aUV;
    vColor = aColor;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)";

static const char *fs_src = R"(
#version 330 core

in vec2 vUV;
in vec4 vColor;

out vec4 fragColor;

void main() {
    vec2 d = vUV * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float g = exp(-4.5 * r2);
    float alpha = vColor.a * g;
    if (alpha < 1.0/255.0) discard;
    fragColor = vec4(vColor.rgb, alpha);
}
)";

// ---- GL helpers ----

static GLuint compileShader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader error: %s\n", log);
        exit(1);
    }
    return s;
}

static GLuint buildProgram(const char *vs, const char *fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "Link error: %s\n", log);
        exit(1);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ---- Camera state ----

static Vec3 camPos = {0.0f, 0.0f, 2.0f};
static float camYaw = -3.14159f;
static float camPitch = 0.0f;
static bool leftDrag = false;   // left click = rotate
static bool rightDrag = false;  // right click = move

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <splat.bin> <texture.png> [--fade] [--cull]\n", argv[0]);
        return 1;
    }

    bool enableFade = false;
    bool enableCull = false;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--fade") == 0) enableFade = true;
        if (strcmp(argv[i], "--cull") == 0) enableCull = true;
    }

    // ---- Load splat data ----
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    SplatHeader hdr;
    assert(fread(&hdr, sizeof(hdr), 1, f) == 1);
    uint32_t total = hdr.num_opaque + hdr.num_translucent;

    std::vector<SplatVertex> splats(total);
    assert(fread(splats.data(), sizeof(SplatVertex), total, f) == total);
    fclose(f);

    printf("Loaded %u splats (%u opaque, %u translucent)\n", total, hdr.num_opaque, hdr.num_translucent);

    // ---- Load texture ----
    int tw, th, tc;
    stbi_set_flip_vertically_on_load(1);
    unsigned char *texData = stbi_load(argv[2], &tw, &th, &tc, 4);
    if (!texData) { fprintf(stderr, "Cannot load %s\n", argv[2]); return 1; }
    printf("Texture: %dx%d (%d channels)\n", tw, th, tc);

    // ---- Init SDL + GL ----
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *win = SDL_CreateWindow("Splat Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "Window failed: %s\n", SDL_GetError()); return 1; }

    SDL_GLContext glctx = SDL_GL_CreateContext(win);
    if (!glctx) { fprintf(stderr, "GL context failed: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(glewErr));
        return 1;
    }
    printf("OpenGL: %s\n", glGetString(GL_VERSION));

    // ---- Upload texture ----
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    stbi_image_free(texData);

    // ---- Compile shaders ----
    GLuint prog = buildProgram(vs_src, fs_src);
    GLint uView = glGetUniformLocation(prog, "uView");
    GLint uProj = glGetUniformLocation(prog, "uProj");

    // ---- Create VAO for CPU-generated vertices ----
    // Each splat = 4 vertices (triangle strip), generated on CPU each frame
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Allocate max size: 4 verts per splat
    glBufferData(GL_ARRAY_BUFFER, total * 4 * sizeof(RenderVertex), nullptr, GL_STREAM_DRAW);

    // aPos (location 0): vec3
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void *)0);
    // aUV (location 1): vec2
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void *)(3 * sizeof(float)));
    // aColor (location 2): vec4
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void *)(5 * sizeof(float)));

    // Index buffer for triangle strips -> triangles (6 indices per quad)
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    {
        std::vector<uint32_t> indices(total * 6);
        for (uint32_t i = 0; i < total; i++) {
            uint32_t base = i * 4;
            indices[i * 6 + 0] = base + 0;
            indices[i * 6 + 1] = base + 1;
            indices[i * 6 + 2] = base + 2;
            indices[i * 6 + 3] = base + 2;
            indices[i * 6 + 4] = base + 1;
            indices[i * 6 + 5] = base + 3;
        }
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);
    }

    glBindVertexArray(0);

    // ---- CPU buffers ----
    std::vector<float> depths(total);
    std::vector<uint32_t> sortIdx(total);
    std::vector<RenderVertex> cpuVerts(total * 4);

    // Quad corner offsets in local 2D space (before scale/rotation)
    static const float quadOffsets[4][2] = {
        {-0.5f, -0.5f}, { 0.5f, -0.5f}, {-0.5f,  0.5f}, { 0.5f,  0.5f}
    };
    static const float quadUVs[4][2] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };
    static const float QUAD_SCALE = 6.0f; // +-3 sigma

    // ---- Main loop ----
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearColor(0.15f, 0.15f, 0.2f, 1.0f);

    Uint64 lastTick = SDL_GetPerformanceCounter();
    Uint64 freq = SDL_GetPerformanceFrequency();
    bool running = true;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) {
                running = false;
            } else if (ev.type == SDL_MOUSEBUTTONDOWN) {
                if (ev.button.button == SDL_BUTTON_LEFT) leftDrag = true;
                if (ev.button.button == SDL_BUTTON_RIGHT) rightDrag = true;
            } else if (ev.type == SDL_MOUSEBUTTONUP) {
                if (ev.button.button == SDL_BUTTON_LEFT) leftDrag = false;
                if (ev.button.button == SDL_BUTTON_RIGHT) rightDrag = false;
            } else if (ev.type == SDL_MOUSEMOTION) {
                float dx = ev.motion.xrel * 0.0003f;
                float dy = ev.motion.yrel * 0.0003f;
                if (leftDrag) {
                    camYaw += dx;
                    camPitch -= dy;
                    if (camPitch > 1.5f) camPitch = 1.5f;
                    if (camPitch < -1.5f) camPitch = -1.5f;
                }
                if (rightDrag) {
                    Vec3 right = { sinf(camYaw - 1.5708f), 0.0f, cosf(camYaw - 1.5708f) };
                    Vec3 up = { 0.0f, 1.0f, 0.0f };
                    camPos.x -= right.x * dx * 5.0f + up.x * dy * 5.0f;
                    camPos.y -= right.y * dx * 5.0f + up.y * dy * 5.0f;
                    camPos.z -= right.z * dx * 5.0f + up.z * dy * 5.0f;
                }
            }
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = (float)(now - lastTick) / (float)freq;
        lastTick = now;

        // Camera
        Vec3 fwd = {
            cosf(camPitch) * sinf(camYaw),
            sinf(camPitch),
            cosf(camPitch) * cosf(camYaw)
        };
        Vec3 right = { sinf(camYaw - 1.5708f), 0.0f, cosf(camYaw - 1.5708f) };

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);
        float speed = 1.0f * dt;
        if (keys[SDL_SCANCODE_LSHIFT]) speed *= 5.0f;
        if (keys[SDL_SCANCODE_W]) { camPos.x += fwd.x*speed; camPos.y += fwd.y*speed; camPos.z += fwd.z*speed; }
        if (keys[SDL_SCANCODE_S]) { camPos.x -= fwd.x*speed; camPos.y -= fwd.y*speed; camPos.z -= fwd.z*speed; }
        if (keys[SDL_SCANCODE_A]) { camPos.x -= right.x*speed; camPos.z -= right.z*speed; }
        if (keys[SDL_SCANCODE_D]) { camPos.x += right.x*speed; camPos.z += right.z*speed; }
        if (keys[SDL_SCANCODE_Q]) camPos.y -= speed;
        if (keys[SDL_SCANCODE_E]) camPos.y += speed;

        int ww, wh;
        SDL_GL_GetDrawableSize(win, &ww, &wh);
        if (ww == 0 || wh == 0) continue;
        glViewport(0, 0, ww, wh);

        Mat4 view = mat4_lookdir(camPos, fwd, {0.0f, 1.0f, 0.0f});
        Mat4 proj = mat4_perspective(1.0f, (float)ww / (float)wh, 0.01f, 100.0f);

        // Sort all splats back-to-front by view-space depth
        for (uint32_t i = 0; i < total; i++) {
            float dx = splats[i].x - camPos.x;
            float dy = splats[i].y - camPos.y;
            float dz = splats[i].z - camPos.z;
            depths[i] = view.m[2]*dx + view.m[6]*dy + view.m[10]*dz;
        }
        std::iota(sortIdx.begin(), sortIdx.end(), 0);
        std::sort(sortIdx.begin(), sortIdx.end(), [&](uint32_t a, uint32_t b) {
            return depths[a] < depths[b];
        });

        // CPU: generate 4 world-space vertices per visible splat
        // Screen-size fade: proj.m[5] = focal length in Y clip units
        float projScale = proj.m[5] * (float)wh * 0.5f; // world-size to pixels factor (at depth=1)

        uint32_t numVisible = 0;
        for (uint32_t si = 0; si < total; si++) {
            uint32_t idx = sortIdx[si];
            const SplatVertex &s = splats[idx];

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
            if (enableCull) {
                float maxScale = std::max({scx, scy, scz});
                float viewDist = fabsf(depths[idx]);
                float pixelSize = (viewDist > 0.001f) ? (maxScale * projScale / viewDist) : 10000.0f;

                if (pixelSize < 4.0f) continue;

                if (enableFade && pixelSize < 16.0f)
                    sizeFade = (pixelSize - 4.0f) / (16.0f - 4.0f);
            }

            // Color
            float cr = s.r / 255.0f;
            float cg = s.g / 255.0f;
            float cb = s.b / 255.0f;
            float ca = s.a / 255.0f * sizeFade;

            // Generate 4 quad corners
            uint32_t base = numVisible * 4;
            for (int q = 0; q < 4; q++) {
                // Local position: quad offset * quad_scale * scale
                float lx = quadOffsets[q][0] * QUAD_SCALE * scx;
                float ly = quadOffsets[q][1] * QUAD_SCALE * scy;
                float lz = 0.0f;

                // Rotate by quaternion into world space
                float wx, wy, wz;
                quat_rotate(s.qx, s.qy, s.qz, s.qw, lx, ly, lz, wx, wy, wz);

                cpuVerts[base + q].x = wx + s.x;
                cpuVerts[base + q].y = wy + s.y;
                cpuVerts[base + q].z = wz + s.z;
                cpuVerts[base + q].u = quadUVs[q][0];
                cpuVerts[base + q].v = quadUVs[q][1];
                cpuVerts[base + q].r = cr;
                cpuVerts[base + q].g = cg;
                cpuVerts[base + q].b = cb;
                cpuVerts[base + q].a = ca;
            }
            numVisible++;
        }

        printf("\rSplats: %u / %u  ", numVisible, total);
        fflush(stdout);

        // Upload CPU vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numVisible * 4 * sizeof(RenderVertex), cpuVerts.data());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(uView, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProj, 1, GL_FALSE, proj.m);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(vao);

        // All splats sorted back-to-front, alpha blend
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawElements(GL_TRIANGLES, numVisible * 6, GL_UNSIGNED_INT, nullptr);
        glDepthMask(GL_TRUE);

        glBindVertexArray(0);
        SDL_GL_SwapWindow(win);
    }

    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    SDL_GL_DeleteContext(glctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
