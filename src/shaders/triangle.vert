#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    mat4 viewMatrix; 
    mat4 projectionMatrix;
    vec4 cameraPosition;
} pushConstants;

struct Vertex {
    vec3 position;
    vec3 color;
};

struct Object {
    vec4 position;
    vec4 color;
    uint meshIndex;
    uint materialIndex;
    uint padding1;
    uint padding2;
};

layout(set = 0, binding = 0) readonly buffer VertexBuffer {
    Vertex vertices[];
} vertexBuffer;

layout(set = 0, binding = 1) readonly buffer ObjectBuffer {
    Object objects[];
} objectBuffer;

layout(location = 0) out vec3 oColor;

void main() {
    Vertex vertex = vertexBuffer.vertices[gl_VertexIndex];
    
    Object obj = objectBuffer.objects[gl_InstanceIndex];
    
    oColor = obj.color.rgb;
    
    vec4 localPosition = vec4(vertex.position + obj.position.xyz, 1.0);
    
    vec4 worldPosition = pushConstants.modelMatrix * localPosition;
    vec4 viewPosition = pushConstants.viewMatrix * worldPosition;
    vec4 clipPosition = pushConstants.projectionMatrix * viewPosition;
    
    gl_Position = clipPosition;
}