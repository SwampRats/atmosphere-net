import torch
import numpy as np

from atmosphere_net import AtmosphereNet

# Load model
model = AtmosphereNet()
model.load_state_dict(torch.load("best_atmosphere_model.pth", map_location="cpu"))
model.eval()

weights = []
shapes = []

# flatten parameters
for name, param in model.named_parameters():
    arr = param.detach().cpu().numpy().astype(np.float32).ravel()
    weights.append(arr)
    shapes.append((name, param.shape, len(arr)))

flat_weights = np.concatenate(weights)
flat_weights.tofile("weights.bin")

# GLSL forward generator
layers = [
    ("dense1", 3, 128),
    ("dense2", 128, 256),
    ("dense3", 256, 256),
    ("dense4", 256, 128),
    ("dense5", 128, 3),
]

offset = 0
glsl_funcs = []


def gen_dense(name, in_dim, out_dim, offset):
    w_offset = offset
    b_offset = w_offset + in_dim * out_dim
    next_offset = b_offset + out_dim

    func = f"""
vec{out_dim if out_dim <= 4 else 'N'} {name}(vec{in_dim} x) {{
    vec{out_dim if out_dim <= 4 else 'N'} y = vec{out_dim if out_dim <= 4 else 'N'}(0.0);
    for (int i = 0; i < {out_dim}; i++) {{
        float sum = 0.0;
        for (int j = 0; j < {in_dim}; j++) {{
            sum += x[j] * getWeight({w_offset} + i*{in_dim} + j);
        }}
        sum += getWeight({b_offset} + i);
        y[i] = sum;
    }}
    return y;
}}"""
    return func, next_offset


for idx, (name, in_dim, out_dim) in enumerate(layers):
    func, offset = gen_dense(name, in_dim, out_dim, offset)
    glsl_funcs.append(func)

forward = """
vec3 forward(vec3 input) {{
    vec128 h1 = max(dense1(input), vec128(0.0));
    vec256 h2 = max(dense2(h1), vec256(0.0));
    vec256 h3 = max(dense3(h2), vec256(0.0));
    vec128 h4 = max(dense4(h3), vec128(0.0));
    vec3 out = dense5(h4);
    return clamp(out, 0.0, 1.0);
}}
"""

with open("forward.glsl", "w") as f:
    f.write(
        """#version 300 es
precision highp float;

uniform sampler2D uWeights;
uniform int uTexWidth;

float getWeight(int idx) {
    int x = idx;
    return texelFetch(uWeights, ivec2(x, 0), 0).r;
}
"""
    )
    for func in glsl_funcs:
        f.write(func + "\n")
    f.write(forward)
