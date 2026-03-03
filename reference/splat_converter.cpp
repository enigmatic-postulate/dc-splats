// splat_converter.cpp
// Converts SuperSplat compressed PLY to a flat binary format with DC SH only.
// Output: two lists (opaque then translucent), each sorted by decreasing splat size.
//
// Build: g++ -O2 -o splat_converter tools/splat_converter.cpp -std=c++17

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

// ---------- Output splat format ----------
// Per splat (40 bytes):
//   float x, y, z        (12 bytes) - position
//   float qx, qy, qz, qw (16 bytes) - rotation quaternion
//   float sx, sy, sz     (12 bytes) - scale (log space)
//   uint8_t r, g, b, a   (4 bytes)  - DC color + opacity [0..255]
//                        Total: 44 bytes
//
// File layout:
//   Header:
//     uint32_t num_opaque
//     uint32_t num_translucent
//   Data:
//     Splat[num_opaque]       - opaque splats, sorted by decreasing size
//     Splat[num_translucent]  - translucent splats, sorted by decreasing size

#pragma pack(push, 1)
struct OutputSplat {
    float x, y, z;
    float qx, qy, qz, qw;
    float sx, sy, sz;
    uint8_t r, g, b, a;
};

struct OutputHeader {
    uint32_t num_opaque;
    uint32_t num_translucent;
};
#pragma pack(pop)

static_assert(sizeof(OutputSplat) == 44, "OutputSplat must be 44 bytes");
static_assert(sizeof(OutputHeader) == 8, "OutputHeader must be 8 bytes");

// ---------- PLY parsing ----------

struct PlyElement {
    std::string name;
    int count;
    struct Prop {
        std::string type;
        std::string name;
    };
    std::vector<Prop> props;
};

static float unpackUnorm(uint32_t value, int bits) {
    uint32_t t = (1u << bits) - 1u;
    return (float)(value & t) / (float)t;
}

static float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

// Decode packed_position: 11-10-11 bits for x,y,z
static void unpackPosition(uint32_t packed, float &nx, float &ny, float &nz) {
    nx = unpackUnorm(packed >> 21, 11);
    ny = unpackUnorm(packed >> 11, 10);
    nz = unpackUnorm(packed, 11);
}

// Decode packed_scale: 11-10-11 bits for x,y,z
static void unpackScale(uint32_t packed, float &nx, float &ny, float &nz) {
    nx = unpackUnorm(packed >> 21, 11);
    ny = unpackUnorm(packed >> 11, 10);
    nz = unpackUnorm(packed, 11);
}

// Decode packed_color: 8-8-8-8 bits for r,g,b,a
static void unpackColor(uint32_t packed, float &r, float &g, float &b, float &a) {
    r = unpackUnorm(packed >> 24, 8);
    g = unpackUnorm(packed >> 16, 8);
    b = unpackUnorm(packed >> 8, 8);
    a = unpackUnorm(packed, 8);
}

// Decode packed_rotation: 2-10-10-10 bits (largest component omitted)
static void unpackRotation(uint32_t packed, float &qx, float &qy, float &qz, float &qw) {
    const float norm = std::sqrt(2.0f);
    float a = (unpackUnorm(packed >> 20, 10) - 0.5f) * norm;
    float b = (unpackUnorm(packed >> 10, 10) - 0.5f) * norm;
    float c = (unpackUnorm(packed, 10) - 0.5f) * norm;
    float m2 = 1.0f - (a * a + b * b + c * c);
    float m = (m2 > 0.0f) ? std::sqrt(m2) : 0.0f;

    switch (packed >> 30) {
        case 0: qx = a; qy = b; qz = c; qw = m; break;
        case 1: qx = m; qy = b; qz = c; qw = a; break;
        case 2: qx = b; qy = m; qz = c; qw = a; break;
        case 3: qx = b; qy = c; qz = m; qw = a; break;
    }
}

// Compute "splat size" as product of exponentials of scale values
// Scale in the compressed format is stored as log-space values after chunk interpolation
static float splatSize(const OutputSplat &s) {
    return std::exp(s.sx) * std::exp(s.sy) * std::exp(s.sz);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.ply> <output.bin>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    // ---------- Parse PLY header ----------
    std::vector<PlyElement> elements;
    bool binary_le = false;

    char line[1024];
    while (fgets(line, sizeof(line), fin)) {
        // Strip newline
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        if (strcmp(line, "end_header") == 0)
            break;

        if (strncmp(line, "format ", 7) == 0) {
            if (strstr(line, "binary_little_endian"))
                binary_le = true;
        } else if (strncmp(line, "element ", 8) == 0) {
            PlyElement el;
            char name[256];
            int count;
            sscanf(line, "element %s %d", name, &count);
            el.name = name;
            el.count = count;
            elements.push_back(el);
        } else if (strncmp(line, "property ", 9) == 0) {
            if (!elements.empty()) {
                PlyElement::Prop prop;
                char type[64], pname[256];
                sscanf(line, "property %s %s", type, pname);
                prop.type = type;
                prop.name = pname;
                elements.back().props.push_back(prop);
            }
        }
    }

    if (!binary_le) {
        fprintf(stderr, "Error: only binary_little_endian PLY supported\n");
        fclose(fin);
        return 1;
    }

    // Identify elements
    PlyElement *chunkEl = nullptr, *vertexEl = nullptr, *shEl = nullptr;
    for (auto &el : elements) {
        if (el.name == "chunk") chunkEl = &el;
        else if (el.name == "vertex") vertexEl = &el;
        else if (el.name == "sh") shEl = &el;
    }

    if (!chunkEl || !vertexEl) {
        fprintf(stderr, "Error: missing 'chunk' or 'vertex' element\n");
        fclose(fin);
        return 1;
    }

    int numChunks = chunkEl->count;
    int numVertices = vertexEl->count;
    int chunkFloats = (int)chunkEl->props.size(); // should be 18 (with color) or 12 (without)

    printf("Chunks: %d (%d floats each)\n", numChunks, chunkFloats);
    printf("Vertices: %d\n", numVertices);
    if (shEl)
        printf("SH entries: %d (%d uchars each)\n", shEl->count, (int)shEl->props.size());

    // ---------- Read chunk data ----------
    std::vector<float> chunkData(numChunks * chunkFloats);
    if (fread(chunkData.data(), sizeof(float), numChunks * chunkFloats, fin) != (size_t)(numChunks * chunkFloats)) {
        fprintf(stderr, "Error: failed to read chunk data\n");
        fclose(fin);
        return 1;
    }

    // ---------- Read vertex data ----------
    struct PackedVertex {
        uint32_t packed_position;
        uint32_t packed_rotation;
        uint32_t packed_scale;
        uint32_t packed_color;
    };
    std::vector<PackedVertex> packedVerts(numVertices);
    if (fread(packedVerts.data(), sizeof(PackedVertex), numVertices, fin) != (size_t)numVertices) {
        fprintf(stderr, "Error: failed to read vertex data\n");
        fclose(fin);
        return 1;
    }

    // ---------- Skip SH data (we only want DC) ----------
    if (shEl) {
        int shBytes = shEl->count * (int)shEl->props.size();
        fseek(fin, shBytes, SEEK_CUR);
    }

    fclose(fin);

    // ---------- Decode splats ----------
    std::vector<OutputSplat> opaque, translucent;
    opaque.reserve(numVertices);
    translucent.reserve(numVertices / 4);

    bool hasChunkColor = (chunkFloats >= 18);

    for (int i = 0; i < numVertices; i++) {
        int ci = (i / 256) * chunkFloats;
        const auto &pv = packedVerts[i];

        OutputSplat s;

        // Position
        float nx, ny, nz;
        unpackPosition(pv.packed_position, nx, ny, nz);
        s.x = lerp(chunkData[ci + 0], chunkData[ci + 3], nx);
        s.y = lerp(chunkData[ci + 1], chunkData[ci + 4], ny);
        s.z = lerp(chunkData[ci + 2], chunkData[ci + 5], nz);

        // Rotation
        unpackRotation(pv.packed_rotation, s.qx, s.qy, s.qz, s.qw);

        // Scale (chunk stores log-space scale min/max)
        float snx, sny, snz;
        unpackScale(pv.packed_scale, snx, sny, snz);
        s.sx = lerp(chunkData[ci + 6], chunkData[ci + 9], snx);
        s.sy = lerp(chunkData[ci + 7], chunkData[ci + 10], sny);
        s.sz = lerp(chunkData[ci + 8], chunkData[ci + 11], snz);

        // Color (DC SH only)
        float cr, cg, cb, ca;
        unpackColor(pv.packed_color, cr, cg, cb, ca);

        if (hasChunkColor) {
            cr = lerp(chunkData[ci + 12], chunkData[ci + 15], cr);
            cg = lerp(chunkData[ci + 13], chunkData[ci + 16], cg);
            cb = lerp(chunkData[ci + 14], chunkData[ci + 17], cb);
        }

        // The compressed PLY decompress round-trip is identity:
        //   f_dc = (c - 0.5) / SH_C0, then render does sigmoid(f_dc * SH_C0 + 0.5) = sigmoid(c)
        //   But sigmoid(c) where c is in ~[0,1] is just a mild S-curve.
        //   Actually: the decompress stores f_dc, and the renderer does:
        //     color = SH_C0 * f_dc + 0.5 = c.xyz  (the original value)
        //     then clamps to [0,1]. So c.xyz IS the final color.
        //   For opacity: logit = -log(1/c.w-1), renderer does sigmoid(logit) = c.w.
        // So the interpolated values ARE the final color/opacity directly.
        s.r = (uint8_t)(std::clamp(cr, 0.0f, 1.0f) * 255.0f + 0.5f);
        s.g = (uint8_t)(std::clamp(cg, 0.0f, 1.0f) * 255.0f + 0.5f);
        s.b = (uint8_t)(std::clamp(cb, 0.0f, 1.0f) * 255.0f + 0.5f);
        s.a = (uint8_t)(std::clamp(ca, 0.0f, 1.0f) * 255.0f + 0.5f);

        // Split by opacity threshold: >0.95 = opaque
        if (ca > 0.95f) {
            opaque.push_back(s);
        } else {
            translucent.push_back(s);
        }
    }

    // Sort both lists by decreasing splat size
    auto cmpSize = [](const OutputSplat &a, const OutputSplat &b) {
        return splatSize(a) > splatSize(b);
    };
    std::sort(opaque.begin(), opaque.end(), cmpSize);
    std::sort(translucent.begin(), translucent.end(), cmpSize);

    printf("Opaque: %zu, Translucent: %zu\n", opaque.size(), translucent.size());

    // ---------- Write output ----------
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "Error: cannot open '%s' for writing\n", argv[2]);
        return 1;
    }

    OutputHeader hdr;
    hdr.num_opaque = (uint32_t)opaque.size();
    hdr.num_translucent = (uint32_t)translucent.size();
    fwrite(&hdr, sizeof(hdr), 1, fout);
    fwrite(opaque.data(), sizeof(OutputSplat), opaque.size(), fout);
    fwrite(translucent.data(), sizeof(OutputSplat), translucent.size(), fout);

    fclose(fout);

    size_t totalBytes = sizeof(hdr) + (opaque.size() + translucent.size()) * sizeof(OutputSplat);
    printf("Output: %zu bytes (%.1f KB)\n", totalBytes, totalBytes / 1024.0);

    return 0;
}
