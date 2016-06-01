#version 330

layout(location = 0) in vec2 position;
out vec2 texCoord;

void main() {
    texCoord = vec2(position.x/2 + 0.5, position.y/2 + 0.5);
    gl_Position = vec4(position.x, position.y, 1.0, 1.0);
}
