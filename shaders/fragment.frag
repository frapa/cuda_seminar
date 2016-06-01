#version 330

uniform sampler2D texture;
in vec2 texCoord;
out vec4 colorOut;

void main() {
    colorOut = texture(texture, texCoord);
}
