var Geometry = require("gl-geometry");
var glShader = require("gl-shader");
var glslify = require("glslify");

var canvas = document.createElement("canvas");
canvas.width = canvas.height = 128;
document.body.appendChild(canvas);

var gl = canvas.getContext("webgl2");
gl.clearColor(1, 0, 1, 1);

var quad = Geometry(gl).attr(
  "aPosition",
  [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1],
);

var program = glShader(
  gl,
  glslify("./example.vert"),
  glslify("./example.frag"),
);

function reflow() {
  document.body.style.background = "#333";
  document.body.style.margin = "0px";
  canvas.style.position = "fixed";
  var s = Math.min(window.innerHeight, window.innerWidth) * 0.95;
  canvas.style.height = canvas.style.width = s + "px";
  canvas.style.left = window.innerWidth / 2 - s / 2 + "px";
  canvas.style.top = window.innerHeight / 2 - s / 2 + "px";
}

// Fixed loadWeights function
async function loadWeights(gl, program, url) {
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const weights = new Float32Array(buf);

  const texWidth = 512;
  const texHeight = Math.ceil(weights.length / texWidth);

  const padded = new Float32Array(texWidth * texHeight);
  padded.set(weights);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);

  if (!gl.getExtension("EXT_color_buffer_float")) {
    console.warn("EXT_color_buffer_float not supported");
  }

  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R32F,
    texWidth,
    texHeight,
    0,
    gl.RED,
    gl.FLOAT,
    padded,
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);

  // Use getUniformLocation + uniform* for gl-shader
  gl.useProgram(program.program);

  const uWeightsLoc = gl.getUniformLocation(program.program, "uWeights");
  const uTexWidthLoc = gl.getUniformLocation(program.program, "uTexWidth");
  const uSunPosLoc = gl.getUniformLocation(program.program, "uSunPos");

  gl.uniform1i(uWeightsLoc, 0); // texture unit 0
  gl.uniform1i(uTexWidthLoc, texWidth);
  gl.uniform3f(uSunPosLoc, 1, 0.5, -1);

  return { tex, texWidth, texHeight };
}

// Async render
async function main() {
  await loadWeights(gl, program, "weights.bin");

  reflow();
  gl.clear(gl.COLOR_BUFFER_BIT);

  program.bind();
  quad.bind(program);
  quad.draw();
}

main();
