#version 330

out vec4 colorOut;
uniform sampler2D texture;
in vec2 texCoord;

void main() {
    colorOut = vec4(1.0, 0.0, 0.0, 1.0); //texture2D(texture, texCoord)
}
