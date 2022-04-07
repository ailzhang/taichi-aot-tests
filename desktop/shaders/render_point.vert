#version 460

layout (location = 0) in vec3 position;

layout (set = 0, binding = 0) uniform Constants {
    mat4 proj;
    mat4 view;
};

void main() {
    vec4 pos = vec4(position, 1.0);
    pos = view * pos;
    pos = proj * pos;

    gl_Position = pos;
}
