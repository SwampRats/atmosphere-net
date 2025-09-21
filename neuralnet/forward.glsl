#version 300 es
precision highp float;

uniform sampler2D uWeights;
uniform int uTexWidth;

float getWeight(int idx) {
    int x = idx;
    return texelFetch(uWeights, ivec2(x, 0), 0).r;
}

vecN dense1(vec3 x) {
    vecN y = vecN(0.0);
    for (int i = 0; i < 128; i++) {
        float sum = 0.0;
        for (int j = 0; j < 3; j++) {
            sum += x[j] * getWeight(0 + i*3 + j);
        }
        sum += getWeight(384 + i);
        y[i] = sum;
    }
    return y;
}

vecN dense2(vec128 x) {
    vecN y = vecN(0.0);
    for (int i = 0; i < 256; i++) {
        float sum = 0.0;
        for (int j = 0; j < 128; j++) {
            sum += x[j] * getWeight(512 + i*128 + j);
        }
        sum += getWeight(33280 + i);
        y[i] = sum;
    }
    return y;
}

vecN dense3(vec256 x) {
    vecN y = vecN(0.0);
    for (int i = 0; i < 256; i++) {
        float sum = 0.0;
        for (int j = 0; j < 256; j++) {
            sum += x[j] * getWeight(33536 + i*256 + j);
        }
        sum += getWeight(99072 + i);
        y[i] = sum;
    }
    return y;
}

vecN dense4(vec256 x) {
    vecN y = vecN(0.0);
    for (int i = 0; i < 128; i++) {
        float sum = 0.0;
        for (int j = 0; j < 256; j++) {
            sum += x[j] * getWeight(99328 + i*256 + j);
        }
        sum += getWeight(132096 + i);
        y[i] = sum;
    }
    return y;
}

vec3 dense5(vec128 x) {
    vec3 y = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float sum = 0.0;
        for (int j = 0; j < 128; j++) {
            sum += x[j] * getWeight(132224 + i*128 + j);
        }
        sum += getWeight(132608 + i);
        y[i] = sum;
    }
    return y;
}

vec3 forward(vec3 input) {{
    vec128 h1 = max(dense1(input), vec128(0.0));
    vec256 h2 = max(dense2(h1), vec256(0.0));
    vec256 h3 = max(dense3(h2), vec256(0.0));
    vec128 h4 = max(dense4(h3), vec128(0.0));
    vec3 out = dense5(h4);
    return clamp(out, 0.0, 1.0);
}}
