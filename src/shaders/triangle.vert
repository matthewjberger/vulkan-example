#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_ARB_shader_draw_parameters : require

struct Vertex {
    vec4 position;
    vec4 color;
};

struct Object {
    vec4 position;
    vec4 color;
    uint meshIndex;
    uint materialIndex;
    uint padding1;
    uint padding2;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer ObjectBuffer {
    Object objects[];
};

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 cameraPosition;
    uvec2 vertexBufferAddress;
    uvec2 objectBufferAddress;
};

layout(location = 0) out vec3 inColor;

void main() {
    uint64_t vertexAddr = packUint2x32(vertexBufferAddress);
    uint64_t objectAddr = packUint2x32(objectBufferAddress);

    VertexBuffer vertexBuffer = VertexBuffer(vertexAddr);
    ObjectBuffer objectBuffer = ObjectBuffer(objectAddr);

    uint drawIndex = gl_DrawID;
    
    Vertex vertex = vertexBuffer.vertices[gl_VertexIndex];
    
    Object obj = objectBuffer.objects[gl_BaseInstance + gl_InstanceIndex];

    inColor = obj.color.rgb;

    vec4 localPosition = vec4(vertex.position.xyz + obj.position.xyz, 1.0);

    vec4 worldPosition = modelMatrix * localPosition;
    vec4 viewPosition = viewMatrix * worldPosition;
    vec4 clipPosition = projectionMatrix * viewPosition;

    gl_Position = clipPosition;
}
