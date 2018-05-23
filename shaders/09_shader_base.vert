#version 450
#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex {
    vec4 gl_Position;
};

layout(binding =0)uniform UniformBufferObj{
	mat4 model;
	mat4 view;
	mat4 project;
};

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoord;

layout(location = 0) in vec2 positions;
layout(location = 1) in vec3 colors;
layout(location = 2) in vec2 texCoord;

void main() {
    gl_Position = project * view * model * vec4(positions, 0.0, 1.0);
    fragColor = colors;
	fragCoord = texCoord;
}