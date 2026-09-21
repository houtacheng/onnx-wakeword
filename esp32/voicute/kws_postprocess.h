// Generic float head + mel-quant helpers (same for every keyword).
// Pipeline: mel[98][32] -> quantize int8 -> [INT8 backbone] -> int8[256]
//           -> dequant -> KWS float head -> prob.
#pragma once
#include <math.h>
#include <stdint.h>
#include "head.h"   // KWS_HEAD_K, KWS_HEAD_D, KWS_ABS_TEMP, KWS_FC_B, KWS_FC_W[], KWS_PROTO_NORM[][]

static inline float kws_postprocess(const int8_t *out_i8, float out_scale, int out_zero) {
    float feat[KWS_HEAD_D]; float norm = 0.0f;
    for (int i = 0; i < KWS_HEAD_D; i++) {
        feat[i] = ((float)out_i8[i] - (float)out_zero) * out_scale;
        norm += feat[i] * feat[i];
    }
    norm = sqrtf(norm) + 1e-12f;
    for (int i = 0; i < KWS_HEAD_D; i++) feat[i] /= norm;
    float logit = KWS_FC_B;
    for (int k = 0; k < KWS_HEAD_K; k++) {
        float cos = 0.0f;
        for (int i = 0; i < KWS_HEAD_D; i++) cos += feat[i] * KWS_PROTO_NORM[k][i];
        logit += KWS_FC_W[k] * (cos / KWS_ABS_TEMP);
    }
    return 1.0f / (1.0f + expf(-logit));
}

static inline void kws_quantize_mel(const float *mel, int8_t *out, int n, float in_scale, int in_zero) {
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(mel[i] / in_scale) + in_zero;
        out[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
}
