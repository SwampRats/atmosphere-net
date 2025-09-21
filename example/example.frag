#version 300 es
#define SHADER_NAME quad.frag

precision highp float;

uniform sampler2D uWeights;
uniform int uTexWidth;
uniform vec3 uSunPos;

in vec3 vPosition;          // fragment shader input
out vec4 FragColor;         // fragment shader output

float getWeight(int idx) {
    int x = idx % uTexWidth;
    int y = idx / uTexWidth;
    return texelFetch(uWeights, ivec2(x, y), 0).r;
}

// Layer 1: input vec3 -> 128 hidden
void dense1(vec3 x, out float y[128]) {
    for (int i = 0; i < 128; i++) {
        float sum = 0.0;
        for (int j = 0; j < 3; j++) {
            sum += x[j] * getWeight(i*3 + j);
        }
        sum += getWeight(384 + i);
        y[i] = sum;
    }
}

// Layer 2: 128 -> 256
void dense2(float x[128], out float y[256]) {
    for (int i = 0; i < 256; i++) {
        float sum = 0.0;
        for (int j = 0; j < 128; j++) {
            sum += x[j] * getWeight(512 + i*128 + j);
        }
        sum += getWeight(33280 + i);
        y[i] = sum;
    }
}

// Layer 3: 256 -> 256
void dense3(float x[256], out float y[256]) {
    for (int i = 0; i < 256; i++) {
        float sum = 0.0;
        for (int j = 0; j < 256; j++) {
            sum += x[j] * getWeight(33536 + i*256 + j);
        }
        sum += getWeight(99072 + i);
        y[i] = sum;
    }
}

// Layer 4: 256 -> 128
void dense4(float x[256], out float y[128]) {
    for (int i = 0; i < 128; i++) {
        float sum = 0.0;
        for (int j = 0; j < 256; j++) {
            sum += x[j] * getWeight(99328 + i*256 + j);
        }
        sum += getWeight(132096 + i);
        y[i] = sum;
    }
}

// Layer 5: 128 -> 3
void dense5(float x[128], out vec3 y) {
    for (int i = 0; i < 3; i++) {
        float sum = 0.0;
        for (int j = 0; j < 128; j++) {
            sum += x[j] * getWeight(132224 + i*128 + j);
        }
        sum += getWeight(132608 + i);
        y[i] = sum;
    }
}

vec3 forward(vec3 inPos) {
    float h1[128];
    float h2[256];
    float h3[256];
    float h4[128];
    vec3 outColor;

    dense1(inPos, h1);
    for(int i=0;i<128;i++) h1[i] = max(h1[i], 0.0); // ReLU

    dense2(h1, h2);
    for(int i=0;i<256;i++) h2[i] = max(h2[i], 0.0); // ReLU

    dense3(h2, h3);
    for(int i=0;i<256;i++) h3[i] = max(h3[i], 0.0); // ReLU

    dense4(h3, h4);
    for(int i=0;i<128;i++) h4[i] = max(h4[i], 0.0); // ReLU

    dense5(h4, outColor);

    return clamp(outColor, 0.0, 1.0);
}

void main() {
  vec3 color = forward(normalize(vPosition));
  color = 1.0 - exp(-1.0 * color);
  FragColor = vec4(color, 1.0);
}
