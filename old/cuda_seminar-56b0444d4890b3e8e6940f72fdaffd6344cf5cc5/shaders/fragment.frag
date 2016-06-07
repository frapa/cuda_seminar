#version 330

uniform sampler2D tex;
in vec2 texCoord;
out vec4 colorOut;

void main() {
    colorOut = texture(tex, texCoord);
}
