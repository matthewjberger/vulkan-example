use ash::{ext::debug_utils, khr::swapchain, vk};
use bytemuck::{Pod, Zeroable};
use nalgebra_glm as glm;
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut Context::default())?;
    Ok(())
}

#[derive(Default)]
struct Context {
    pub window_handle: Option<winit::window::Window>,
    pub renderer: Option<Renderer>,
    pub egui_state: Option<egui_winit::State>,
    pub start_time: Option<std::time::Instant>,
}

impl winit::application::ApplicationHandler for Context {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.start_time = Some(std::time::Instant::now());

        let mut attributes = winit::window::Window::default_attributes();
        attributes.title = "Rust + Vulkan Multi-Mesh BDA Instancing".to_string();
        if let Ok(window) = event_loop.create_window(attributes) {
            if let Ok(mut renderer) = create_renderer(&window, 800, 600) {
                setup_sample_scene(&mut renderer);
                self.renderer = Some(renderer);
            }

            let gui_context = egui::Context::default();
            gui_context.set_pixels_per_point(window.scale_factor() as _);

            let viewport_id = gui_context.viewport_id();
            let gui_state = egui_winit::State::new(
                gui_context,
                viewport_id,
                &window,
                Some(window.scale_factor() as _),
                Some(winit::window::Theme::Dark),
                None,
            );

            egui_extras::install_image_loaders(gui_state.egui_ctx());

            self.window_handle = Some(window);
            self.egui_state = Some(gui_state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(window) = self.window_handle.as_mut() {
            if let Some(gui_state) = &mut self.egui_state {
                let _consumed_event = gui_state.on_window_event(window, &event).consumed;
            }
        }

        if matches!(event, winit::event::WindowEvent::CloseRequested) {
            event_loop.exit();
            return;
        }

        if let winit::event::WindowEvent::Resized(winit::dpi::PhysicalSize { width, height }) =
            event
        {
            if width > 0 && height > 0 {
                if let Some(gui_state) = &mut self.egui_state {
                    if let Some(window) = self.window_handle.as_ref() {
                        gui_state
                            .egui_ctx()
                            .set_pixels_per_point(window.scale_factor() as _);
                    }
                }

                if let Some(renderer) = &mut self.renderer {
                    renderer.is_swapchain_dirty = true;
                }
            }
            return;
        }
        if matches!(event, winit::event::WindowEvent::RedrawRequested) {
            let Self {
                window_handle: Some(window_handle),
                renderer: Some(renderer),
                egui_state: Some(egui_state),
                ..
            } = self
            else {
                return;
            };

            if renderer.is_swapchain_dirty {
                let dimension = window_handle.inner_size();
                if dimension.width > 0 && dimension.height > 0 {
                    match recreate_swapchain(renderer, dimension.width, dimension.height) {
                        Ok(()) => {
                            renderer.is_swapchain_dirty = false;
                        }
                        Err(error) => {
                            log::error!("Failed to recreate swapchain: {error}");
                            return;
                        }
                    }
                } else {
                    return;
                }
            }

            let gui_input = egui_state.take_egui_input(window_handle);
            egui_state.egui_ctx().begin_pass(gui_input);
            let egui_ctx = egui_state.egui_ctx().clone();
            egui::Window::new("Multi-Mesh BDA Demo").show(&egui_ctx, |ui| {
                ui.heading("Buffer Device Address Multi-Mesh Rendering");
                ui.label("Using BDA for efficient GPU draw command generation");
                ui.label(format!("Total objects: {}", renderer.objects.len()));
                ui.label(format!("Mesh types: {}", renderer.meshes.len()));

                ui.separator();
                let mut mesh_counts = std::collections::HashMap::new();
                for obj in &renderer.objects {
                    *mesh_counts.entry(obj.mesh_id.0).or_insert(0) += 1;
                }

                for (mesh_idx, count) in mesh_counts {
                    let mesh_name = match mesh_idx {
                        0 => "Triangles",
                        1 => "Cubes",
                        2 => "Pyramids",
                        _ => "Unknown",
                    };
                    ui.label(format!("{}: {}", mesh_name, count));
                }
            });
            let output = egui_state.egui_ctx().end_pass();
            egui_state.handle_platform_output(window_handle, output.platform_output.clone());
            let paint_jobs = egui_ctx.tessellate(output.shapes.clone(), output.pixels_per_point);
            let ui_frame_output = Some((output, paint_jobs));

            let should_render =
                window_handle.inner_size().width > 0 && window_handle.inner_size().height > 0;
            if should_render {
                let elapsed_time = self
                    .start_time
                    .map(|start| start.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                if let Err(error) = render_frame(renderer, ui_frame_output, elapsed_time) {
                    log::error!("Failed to draw frame: {error}");
                } else {
                    renderer.is_swapchain_dirty = false;
                }
            }

            if let Some(window_handle) = self.window_handle.as_mut() {
                window_handle.request_redraw();
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(renderer) = self.renderer.take() {
            drop(renderer);
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Object {
    position: [f32; 4],
    color: [f32; 4],
    mesh_index: u32,
    material_index: u32,
    padding1: u32,
    padding2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuMeshData {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ComputePushConstants {
    object_count: u32,
    mesh_count: u32,
    vertex_buffer_address: [u32; 2],
    object_buffer_address: [u32; 2],
    mesh_buffer_address: [u32; 2],
    indirect_buffer_address: [u32; 2],
    count_buffer_address: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GraphicsPushConstants {
    model_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    camera_position: [f32; 4],
    vertex_buffer_address: [u32; 2],
    object_buffer_address: [u32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObjectId(usize);

#[derive(Debug, Clone)]
struct MeshData {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
}

#[derive(Debug, Clone)]
struct ObjectData {
    mesh_id: MeshId,
    position: [f32; 3],
    color: [f32; 3],
    material_index: u32,
}

struct BufferAllocation {
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    device_address: Option<u64>,
}

struct ImageAllocation {
    image: vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,
}

struct Renderer {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: ash::khr::surface::Instance,
    pub surface_khr: vk::SurfaceKHR,
    pub debug_utils: debug_utils::Instance,
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub present_queue_family_index: u32,
    pub graphics_queue_family_index: u32,
    pub _transfer_queue_family_index: Option<u32>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub _transfer_queue: Option<vk::Queue>,
    pub command_pool: vk::CommandPool,
    pub allocator: Option<Arc<Mutex<gpu_allocator::vulkan::Allocator>>>,
    pub swapchain: Swapchain,
    pub depth_image: Option<ImageAllocation>,
    pub depth_image_view: Option<vk::ImageView>,
    pub depth_format: vk::Format,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub compute_pipeline: vk::Pipeline,
    pub compute_pipeline_layout: vk::PipelineLayout,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub vertex_buffer: Option<BufferAllocation>,
    pub index_buffer: Option<BufferAllocation>,
    pub indirect_buffer: Option<BufferAllocation>,
    pub count_buffer: Option<BufferAllocation>,
    pub object_buffer: Option<BufferAllocation>,
    pub mesh_buffer: Option<BufferAllocation>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
    pub current_frame: usize,
    pub frames_in_flight: usize,
    pub is_swapchain_dirty: bool,
    pub egui_renderer: Option<egui_ash_renderer::Renderer>,

    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub meshes: Vec<MeshData>,
    pub objects: Vec<ObjectData>,
    pub next_mesh_id: usize,
    pub next_object_id: usize,
    pub buffers_dirty: bool,
}

impl Renderer {
    pub fn create_triangle_mesh(&mut self) -> Result<MeshId, Box<dyn std::error::Error>> {
        let vertices = [
            Vertex {
                position: [0.0, 0.5, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];
        let indices = [0, 1, 2];
        self.add_mesh(&vertices, &indices)
    }

    pub fn create_cube_mesh(&mut self) -> Result<MeshId, Box<dyn std::error::Error>> {
        let vertices = [
            Vertex {
                position: [-0.5, -0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, -0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let indices = [
            0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 4, 0, 3, 3, 7, 4, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ];

        self.add_mesh(&vertices, &indices)
    }

    pub fn create_pyramid_mesh(&mut self) -> Result<MeshId, Box<dyn std::error::Error>> {
        let vertices = [
            Vertex {
                position: [-0.5, -0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, -0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.5, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.0, 0.5, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let indices = [0, 2, 1, 0, 3, 2, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];

        self.add_mesh(&vertices, &indices)
    }

    pub fn add_mesh(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<MeshId, Box<dyn std::error::Error>> {
        let vertex_offset = self.vertices.len() as u32;
        let vertex_count = vertices.len() as u32;
        let index_offset = self.indices.len() as u32;
        let index_count = indices.len() as u32;

        self.vertices.extend_from_slice(vertices);
        self.indices.extend_from_slice(indices);

        let mesh_id = MeshId(self.next_mesh_id);
        self.next_mesh_id += 1;

        let mesh_data = MeshData {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
        };

        self.meshes.push(mesh_data);
        self.buffers_dirty = true;

        Ok(mesh_id)
    }

    pub fn add_object(
        &mut self,
        mesh_id: MeshId,
        position: [f32; 3],
        color: [f32; 3],
        material_index: u32,
    ) -> Result<ObjectId, Box<dyn std::error::Error>> {
        if mesh_id.0 >= self.meshes.len() {
            return Err("Invalid mesh ID".into());
        }

        let object_id = ObjectId(self.next_object_id);
        self.next_object_id += 1;

        let object_data = ObjectData {
            mesh_id,
            position,
            color,
            material_index,
        };

        self.objects.push(object_data);
        self.buffers_dirty = true;

        Ok(object_id)
    }

    fn update_buffers_if_dirty(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.buffers_dirty {
            return Ok(());
        }

        if !self.vertices.is_empty() {
            self.update_vertex_buffer()?;
        }

        if !self.indices.is_empty() {
            self.update_index_buffer()?;
        }

        if !self.objects.is_empty() {
            self.update_object_buffer()?;
        }

        if !self.meshes.is_empty() {
            self.update_mesh_buffer()?;
        }

        self.buffers_dirty = false;
        Ok(())
    }

    fn update_vertex_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.vertices.is_empty() {
            return Ok(());
        }

        let buffer_size = (std::mem::size_of::<Vertex>() * self.vertices.len()) as vk::DeviceSize;

        if let Some(ref buffer) = self.vertex_buffer {
            let current_size = unsafe {
                self.device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .size
            };
            if current_size < buffer_size {
                self.recreate_vertex_buffer(buffer_size)?;
            }
        } else {
            self.recreate_vertex_buffer(buffer_size)?;
        }

        if let Some(ref buffer) = self.vertex_buffer {
            unsafe {
                let data_ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let vertex_bytes = bytemuck::cast_slice(&self.vertices);
                std::ptr::copy_nonoverlapping(vertex_bytes.as_ptr(), data_ptr, vertex_bytes.len());
            }
        }

        Ok(())
    }

    fn recreate_vertex_buffer(
        &mut self,
        size: vk::DeviceSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(old_buffer) = self.vertex_buffer.take() {
            if let Some(allocator_arc) = &self.allocator {
                if let Ok(mut alloc) = allocator_arc.lock() {
                    unsafe {
                        self.device.destroy_buffer(old_buffer.buffer, None);
                    }
                    let _ = alloc.free(old_buffer.allocation);
                }
            }
        }

        let allocator = self.allocator.as_ref().ok_or("Allocator not available")?;
        let new_buffer = create_vertex_buffer_with_size(&self.device, allocator, size)?;
        self.vertex_buffer = Some(new_buffer);

        Ok(())
    }

    fn update_index_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.indices.is_empty() {
            return Ok(());
        }

        let buffer_size = (std::mem::size_of::<u32>() * self.indices.len()) as vk::DeviceSize;

        if let Some(ref buffer) = self.index_buffer {
            let current_size = unsafe {
                self.device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .size
            };
            if current_size < buffer_size {
                self.recreate_index_buffer(buffer_size)?;
            }
        } else {
            self.recreate_index_buffer(buffer_size)?;
        }

        if let Some(ref buffer) = self.index_buffer {
            unsafe {
                let data_ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let index_bytes = bytemuck::cast_slice(&self.indices);
                std::ptr::copy_nonoverlapping(index_bytes.as_ptr(), data_ptr, index_bytes.len());
            }
        }

        Ok(())
    }

    fn recreate_index_buffer(
        &mut self,
        size: vk::DeviceSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(old_buffer) = self.index_buffer.take() {
            if let Some(allocator_arc) = &self.allocator {
                if let Ok(mut alloc) = allocator_arc.lock() {
                    unsafe {
                        self.device.destroy_buffer(old_buffer.buffer, None);
                    }
                    let _ = alloc.free(old_buffer.allocation);
                }
            }
        }

        let allocator = self.allocator.as_ref().ok_or("Allocator not available")?;
        let new_buffer = create_index_buffer_with_size(&self.device, allocator, size)?;
        self.index_buffer = Some(new_buffer);

        Ok(())
    }

    fn update_object_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.objects.is_empty() {
            return Ok(());
        }

        let mut sorted_objects = self.objects.clone();
        sorted_objects.sort_by_key(|obj| obj.mesh_id.0);

        let gpu_objects: Vec<Object> = sorted_objects
            .iter()
            .enumerate()
            .map(|(_, obj_data)| Object {
                position: [
                    obj_data.position[0],
                    obj_data.position[1],
                    obj_data.position[2],
                    1.0,
                ],
                color: [obj_data.color[0], obj_data.color[1], obj_data.color[2], 1.0],
                mesh_index: obj_data.mesh_id.0 as u32,
                material_index: obj_data.material_index,
                padding1: 0,
                padding2: 0,
            })
            .collect();

        let buffer_size = (std::mem::size_of::<Object>() * gpu_objects.len()) as vk::DeviceSize;

        if let Some(ref buffer) = self.object_buffer {
            let current_size = unsafe {
                self.device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .size
            };
            if current_size < buffer_size {
                self.recreate_object_buffer(buffer_size)?;
            }
        } else {
            self.recreate_object_buffer(buffer_size)?;
        }

        if let Some(ref buffer) = self.object_buffer {
            unsafe {
                let data_ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let object_bytes = bytemuck::cast_slice(&gpu_objects);
                std::ptr::copy_nonoverlapping(object_bytes.as_ptr(), data_ptr, object_bytes.len());
            }
        }

        self.objects = sorted_objects;

        Ok(())
    }

    fn recreate_object_buffer(
        &mut self,
        size: vk::DeviceSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(old_buffer) = self.object_buffer.take() {
            if let Some(allocator_arc) = &self.allocator {
                if let Ok(mut alloc) = allocator_arc.lock() {
                    unsafe {
                        self.device.destroy_buffer(old_buffer.buffer, None);
                    }
                    let _ = alloc.free(old_buffer.allocation);
                }
            }
        }

        let allocator = self.allocator.as_ref().ok_or("Allocator not available")?;
        let new_buffer = create_object_buffer_with_size(&self.device, allocator, size)?;
        self.object_buffer = Some(new_buffer);

        Ok(())
    }

    fn update_mesh_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.meshes.is_empty() {
            return Ok(());
        }

        let gpu_meshes: Vec<GpuMeshData> = self
            .meshes
            .iter()
            .map(|mesh| GpuMeshData {
                vertex_offset: mesh.vertex_offset,
                vertex_count: mesh.vertex_count,
                index_offset: mesh.index_offset,
                index_count: mesh.index_count,
            })
            .collect();

        let buffer_size = (std::mem::size_of::<GpuMeshData>() * gpu_meshes.len()) as vk::DeviceSize;

        if let Some(ref buffer) = self.mesh_buffer {
            let current_size = unsafe {
                self.device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .size
            };
            if current_size < buffer_size {
                self.recreate_mesh_buffer(buffer_size)?;
            }
        } else {
            self.recreate_mesh_buffer(buffer_size)?;
        }

        if let Some(ref buffer) = self.mesh_buffer {
            unsafe {
                let data_ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                let mesh_bytes = bytemuck::cast_slice(&gpu_meshes);
                std::ptr::copy_nonoverlapping(mesh_bytes.as_ptr(), data_ptr, mesh_bytes.len());
            }
        }

        Ok(())
    }

    fn recreate_mesh_buffer(
        &mut self,
        size: vk::DeviceSize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(old_buffer) = self.mesh_buffer.take() {
            if let Some(allocator_arc) = &self.allocator {
                if let Ok(mut alloc) = allocator_arc.lock() {
                    unsafe {
                        self.device.destroy_buffer(old_buffer.buffer, None);
                    }
                    let _ = alloc.free(old_buffer.allocation);
                }
            }
        }

        let allocator = self.allocator.as_ref().ok_or("Allocator not available")?;
        let new_buffer = create_mesh_buffer_with_size(&self.device, allocator, size)?;
        self.mesh_buffer = Some(new_buffer);

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            self.egui_renderer = None;

            for fence in self.in_flight_fences.iter() {
                self.device.destroy_fence(*fence, None);
            }

            for semaphore in self.image_available_semaphores.iter() {
                self.device.destroy_semaphore(*semaphore, None);
            }

            for semaphore in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(*semaphore, None);
            }

            if !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
                self.command_buffers.clear();
            }

            if let Some(depth_image_view) = self.depth_image_view.take() {
                self.device.destroy_image_view(depth_image_view, None);
            }

            if let Some(allocator_arc) = &self.allocator {
                match allocator_arc.lock() {
                    Ok(mut alloc) => {
                        if let Some(depth_image) = self.depth_image.take() {
                            self.device.destroy_image(depth_image.image, None);
                            let _ = alloc.free(depth_image.allocation);
                        }

                        if let Some(vertex_buffer) = self.vertex_buffer.take() {
                            self.device.destroy_buffer(vertex_buffer.buffer, None);
                            let _ = alloc.free(vertex_buffer.allocation);
                        }

                        if let Some(index_buffer) = self.index_buffer.take() {
                            self.device.destroy_buffer(index_buffer.buffer, None);
                            let _ = alloc.free(index_buffer.allocation);
                        }

                        if let Some(indirect_buffer) = self.indirect_buffer.take() {
                            self.device.destroy_buffer(indirect_buffer.buffer, None);
                            let _ = alloc.free(indirect_buffer.allocation);
                        }

                        if let Some(count_buffer) = self.count_buffer.take() {
                            self.device.destroy_buffer(count_buffer.buffer, None);
                            let _ = alloc.free(count_buffer.allocation);
                        }

                        if let Some(object_buffer) = self.object_buffer.take() {
                            self.device.destroy_buffer(object_buffer.buffer, None);
                            let _ = alloc.free(object_buffer.allocation);
                        }

                        if let Some(mesh_buffer) = self.mesh_buffer.take() {
                            self.device.destroy_buffer(mesh_buffer.buffer, None);
                            let _ = alloc.free(mesh_buffer.allocation);
                        }
                    }
                    Err(_) => log::error!("Failed to lock allocator"),
                }
            }

            for image_view in self.swapchain.image_views.iter() {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain.image_views.clear();

            self.swapchain
                .swapchain
                .destroy_swapchain(self.swapchain.swapchain_khr, None);

            let _ = self.device.device_wait_idle();

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.compute_pipeline_layout, None);

            self.allocator = None;
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Swapchain {
    pub swapchain: swapchain::Device,
    pub swapchain_khr: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub format: vk::SurfaceFormatKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

fn setup_sample_scene(renderer: &mut Renderer) {
    let triangle_mesh = renderer.create_triangle_mesh().unwrap();
    let cube_mesh = renderer.create_cube_mesh().unwrap();
    let pyramid_mesh = renderer.create_pyramid_mesh().unwrap();

    let triangles = [
        ([-2.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
        ([-1.0, 0.0, -2.0], [0.0, 1.0, 0.0]),
        ([-2.0, 1.0, -3.0], [0.0, 0.0, 1.0]),
    ];

    for (position, color) in triangles.iter() {
        let _ = renderer.add_object(triangle_mesh, *position, *color, 0);
    }

    let cubes = [
        ([0.0, 0.0, -1.0], [1.0, 1.0, 0.0]),
        ([1.0, 0.0, -2.0], [1.0, 0.0, 1.0]),
        ([0.0, 1.0, -3.0], [0.0, 1.0, 1.0]),
        ([1.0, 1.0, -1.0], [0.5, 0.7, 0.3]),
    ];

    for (position, color) in cubes.iter() {
        let _ = renderer.add_object(cube_mesh, *position, *color, 0);
    }

    let pyramids = [
        ([2.0, 0.0, -1.0], [0.8, 0.2, 0.8]),
        ([3.0, 0.0, -2.0], [0.2, 0.8, 0.4]),
        ([2.0, 1.0, -3.0], [0.9, 0.6, 0.1]),
    ];

    for (position, color) in pyramids.iter() {
        let _ = renderer.add_object(pyramid_mesh, *position, *color, 0);
    }
}

fn find_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format, Box<dyn std::error::Error>> {
    let candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    for format in candidates {
        let format_props =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };

        if format_props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(format);
        }
    }

    Err("Failed to find supported depth format".into())
}

fn create_depth_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    extent: vk::Extent2D,
    depth_format: vk::Format,
) -> Result<(ImageAllocation, vk::ImageView), Box<dyn std::error::Error>> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(depth_format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&image_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Depth Buffer",
        requirements: unsafe { device.get_image_memory_requirements(image) },
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
    }

    let image_allocation = ImageAllocation { image, allocation };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image_allocation.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(depth_format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let image_view = unsafe { device.create_image_view(&view_info, None)? };

    Ok((image_allocation, image_view))
}

fn create_renderer<W>(
    window_handle: W,
    initial_width: u32,
    initial_height: u32,
) -> std::result::Result<Renderer, Box<dyn std::error::Error>>
where
    W: raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle,
{
    let entry = unsafe { ash::Entry::load()? };

    let instance = create_instance(&window_handle, &entry)?;

    let (surface, surface_khr) = create_surface(window_handle, &entry, &instance)?;

    let (debug_utils, debug_utils_messenger) = create_debug_utils(&entry, &instance)?;

    let (
        physical_device,
        present_queue_family_index,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ) = find_vulkan_physical_device(&instance, &surface, surface_khr)?;

    let depth_format = find_depth_format(&instance, physical_device)?;

    let mut queue_indices = vec![present_queue_family_index, graphics_queue_family_index];
    if let Some(transfer_queue_index) = transfer_queue_family_index {
        queue_indices.push(transfer_queue_index);
    }
    queue_indices.dedup();
    let device = create_device(&instance, physical_device, &queue_indices)?;

    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_queue_family_index, 0) };
    let transfer_queue =
        transfer_queue_family_index.map(|index| unsafe { device.get_device_queue(index, 0) });

    let command_pool = create_command_pool(graphics_queue_family_index, &device)?;

    let allocator = create_allocator(&instance, physical_device, &device)?;
    let allocator = Arc::new(Mutex::new(allocator));

    let swapchain = create_swapchain(
        &instance,
        &device,
        &surface,
        surface_khr,
        physical_device,
        initial_width,
        initial_height,
        graphics_queue_family_index,
        present_queue_family_index,
    )?;

    let (depth_image, depth_image_view) =
        create_depth_buffer(&device, &allocator, swapchain.extent, depth_format)?;

    let vertex_buffer = create_vertex_buffer_with_size(&device, &allocator, 1024 * 1024)?;
    let index_buffer = create_index_buffer_with_size(&device, &allocator, 256 * 1024)?;
    let mesh_buffer = create_mesh_buffer_with_size(&device, &allocator, 64 * 1024)?;

    const MAX_DRAW_COMMANDS: usize = 64;
    let indirect_buffer = create_indirect_buffer(&device, &allocator, MAX_DRAW_COMMANDS)?;
    let count_buffer = create_count_buffer(&device, &allocator)?;
    let object_buffer = create_object_buffer_with_size(&device, &allocator, 64 * 1024)?;

    let (pipeline_layout, pipeline) = create_pipeline(&device, &swapchain, depth_format)?;
    let (compute_pipeline_layout, compute_pipeline) = create_compute_pipeline(&device)?;

    let egui_renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
        allocator.clone(),
        device.clone(),
        egui_ash_renderer::DynamicRendering {
            color_attachment_format: swapchain.format.format,
            depth_attachment_format: Some(depth_format),
        },
        egui_ash_renderer::Options {
            in_flight_frames: FRAMES_IN_FLIGHT,
            ..Default::default()
        },
    )?;

    const FRAMES_IN_FLIGHT: usize = 3;

    let image_available_semaphores = {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        (0..FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) })
            .collect::<Result<Vec<_>, _>>()?
    };

    let render_finished_semaphores = {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        (0..swapchain.images.len())
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) })
            .collect::<Result<Vec<_>, _>>()?
    };

    let in_flight_fences = {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        (0..FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_fence(&fence_info, None) })
            .collect::<Result<Vec<_>, _>>()?
    };

    let images_in_flight = vec![vk::Fence::null(); swapchain.images.len()];

    let mut renderer = Renderer {
        _entry: entry,
        instance,
        surface,
        surface_khr,
        debug_utils,
        debug_utils_messenger,
        physical_device,
        device,
        present_queue_family_index,
        graphics_queue_family_index,
        _transfer_queue_family_index: transfer_queue_family_index,
        graphics_queue,
        present_queue,
        _transfer_queue: transfer_queue,
        command_pool,
        allocator: Some(allocator),
        swapchain,
        depth_image: Some(depth_image),
        depth_image_view: Some(depth_image_view),
        depth_format,
        pipeline,
        pipeline_layout,
        compute_pipeline,
        compute_pipeline_layout,
        command_buffers: vec![],
        vertex_buffer: Some(vertex_buffer),
        index_buffer: Some(index_buffer),
        indirect_buffer: Some(indirect_buffer),
        count_buffer: Some(count_buffer),
        object_buffer: Some(object_buffer),
        mesh_buffer: Some(mesh_buffer),
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,
        current_frame: 0,
        frames_in_flight: FRAMES_IN_FLIGHT,
        is_swapchain_dirty: false,
        egui_renderer: Some(egui_renderer),

        vertices: Vec::new(),
        indices: Vec::new(),
        meshes: Vec::new(),
        objects: Vec::new(),
        next_mesh_id: 0,
        next_object_id: 0,
        buffers_dirty: false,
    };

    renderer.command_buffers = create_and_record_command_buffers(&renderer)?;

    Ok(renderer)
}

fn create_vertex_buffer_with_size(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    size: vk::DeviceSize,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Vertex Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_index_buffer_with_size(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    size: vk::DeviceSize,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Index Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_object_buffer_with_size(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    size: vk::DeviceSize,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Object Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_mesh_buffer_with_size(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    size: vk::DeviceSize,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Mesh Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_indirect_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    max_commands: usize,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_size = (std::mem::size_of::<DrawCommand>() * max_commands) as vk::DeviceSize;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(
            vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Indirect Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_count_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
    let buffer_size = std::mem::size_of::<u32>() as vk::DeviceSize;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(
            vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Count Buffer",
        requirements: unsafe { device.get_buffer_memory_requirements(buffer) },
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    };

    let mut allocator = allocator.lock().unwrap();
    let allocation = allocator.allocate(&allocation_create_desc)?;

    unsafe {
        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }

    let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

    Ok(BufferAllocation {
        buffer,
        allocation,
        device_address: Some(device_address),
    })
}

fn create_pipeline(
    device: &ash::Device,
    swapchain: &Swapchain,
    depth_format: vk::Format,
) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn std::error::Error + 'static>> {
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<GraphicsPushConstants>() as u32);

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;
    let entry_point_name = c"main";
    let vertex_source = read_shader(&include_bytes!("shaders/triangle.vert.spv")[..])?;
    let vertex_create_info = vk::ShaderModuleCreateInfo::default().code(&vertex_source);
    let vertex_module = unsafe { device.create_shader_module(&vertex_create_info, None)? };
    let fragment_source = read_shader(&include_bytes!("shaders/triangle.frag.spv")[..])?;
    let fragment_create_info = vk::ShaderModuleCreateInfo::default().code(&fragment_source);
    let fragment_module = unsafe { device.create_shader_module(&fragment_create_info, None)? };
    let shader_states_infos = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(entry_point_name),
    ];
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let extent = swapchain.extent;
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as _,
        height: extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    }];
    let viewport_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);
    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);
    let color_attachment_formats = [swapchain.format.format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_attachment_formats)
        .depth_attachment_format(depth_format);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .depth_stencil_state(&depth_stencil_info)
        .color_blend_state(&color_blending_info)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);
    let pipeline = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|e| e.1)?[0]
    };
    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
    Ok((pipeline_layout, pipeline))
}

fn create_push_constants(
    time: f32,
    vertex_buffer_address: u64,
    object_buffer_address: u64,
) -> GraphicsPushConstants {
    let camera_pos = glm::vec3(4.0, 4.0, 4.0);
    let target = glm::vec3(0.0, 0.0, 0.0);
    let up = glm::vec3(0.0, 1.0, 0.0);

    let view_matrix = glm::look_at(&camera_pos, &target, &up);

    let fov = 45.0_f32.to_radians();
    let aspect = 4.0 / 3.0;
    let near = 0.1;
    let far = 100.0;
    let projection_matrix = glm::perspective(aspect, fov, near, far);

    let rotation_speed = 1.0;
    let angle = time * rotation_speed;
    let model_matrix = glm::rotate(&glm::Mat4::identity(), angle, &glm::vec3(0.0, 1.0, 0.0));

    let camera_position = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];

    let vertex_buffer_address_split = [
        vertex_buffer_address as u32,
        (vertex_buffer_address >> 32) as u32,
    ];
    let object_buffer_address_split = [
        object_buffer_address as u32,
        (object_buffer_address >> 32) as u32,
    ];

    GraphicsPushConstants {
        model_matrix: model_matrix.into(),
        view_matrix: view_matrix.into(),
        projection_matrix: projection_matrix.into(),
        camera_position,
        vertex_buffer_address: vertex_buffer_address_split,
        object_buffer_address: object_buffer_address_split,
    }
}

fn create_compute_pipeline(
    device: &ash::Device,
) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn std::error::Error>> {
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<ComputePushConstants>() as u32);

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let compute_shader_code = read_shader(&include_bytes!("shaders/generate_draws.comp.spv")[..])?;
    let compute_shader_module = {
        let shader_module_create_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&compute_shader_code));
        unsafe { device.create_shader_module(&shader_module_create_info, None)? }
    };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module)
        .name(c"main");

    let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let compute_pipeline = unsafe {
        let pipelines = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&compute_pipeline_info),
                None,
            )
            .map_err(|e| e.1)?;
        pipelines[0]
    };

    unsafe {
        device.destroy_shader_module(compute_shader_module, None);
    }

    Ok((pipeline_layout, compute_pipeline))
}

fn render_frame(
    renderer: &mut Renderer,
    ui_frame_output: Option<(egui::FullOutput, Vec<egui::ClippedPrimitive>)>,
    time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    renderer.update_buffers_if_dirty()?;

    if renderer.objects.is_empty() {
        return Ok(());
    }

    let current_fence = renderer.in_flight_fences[renderer.current_frame];
    unsafe {
        renderer
            .device
            .wait_for_fences(&[current_fence], true, u64::MAX)?;
    }

    let next_image_result = unsafe {
        renderer.swapchain.swapchain.acquire_next_image(
            renderer.swapchain.swapchain_khr,
            u64::MAX,
            renderer.image_available_semaphores[renderer.current_frame],
            vk::Fence::null(),
        )
    };
    let image_index = match next_image_result {
        Ok((image_index, _)) => image_index,
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            renderer.is_swapchain_dirty = true;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    if renderer.images_in_flight[image_index as usize] != vk::Fence::null() {
        unsafe {
            renderer.device.wait_for_fences(
                &[renderer.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }
    }

    renderer.images_in_flight[image_index as usize] = current_fence;
    unsafe {
        renderer.device.reset_fences(&[current_fence])?;
    }

    let (textures_delta, clipped_primitives, pixels_per_point) =
        if let Some((full_output, primitives)) = ui_frame_output {
            if !full_output.textures_delta.set.is_empty() {
                if let Some(egui_renderer) = &mut renderer.egui_renderer {
                    egui_renderer.set_textures(
                        renderer.graphics_queue,
                        renderer.command_pool,
                        full_output.textures_delta.set.as_slice(),
                    )?;
                }
            }

            (
                Some(full_output.textures_delta),
                Some(primitives),
                full_output.pixels_per_point,
            )
        } else {
            (None, None, 1.0)
        };

    let command_buffer = renderer.command_buffers[image_index as usize];

    unsafe {
        renderer
            .device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
    }

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    unsafe {
        renderer
            .device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
    }

    unsafe {
        renderer.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            renderer.compute_pipeline,
        );

        let compute_push_constants = ComputePushConstants {
            object_count: renderer.objects.len() as u32,
            mesh_count: renderer.meshes.len() as u32,
            vertex_buffer_address: [
                renderer
                    .vertex_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap() as u32,
                (renderer
                    .vertex_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap()
                    >> 32) as u32,
            ],
            object_buffer_address: [
                renderer
                    .object_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap() as u32,
                (renderer
                    .object_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap()
                    >> 32) as u32,
            ],
            mesh_buffer_address: [
                renderer
                    .mesh_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap() as u32,
                (renderer
                    .mesh_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap()
                    >> 32) as u32,
            ],
            indirect_buffer_address: [
                renderer
                    .indirect_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap() as u32,
                (renderer
                    .indirect_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap()
                    >> 32) as u32,
            ],
            count_buffer_address: [
                renderer
                    .count_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap() as u32,
                (renderer
                    .count_buffer
                    .as_ref()
                    .unwrap()
                    .device_address
                    .unwrap()
                    >> 32) as u32,
            ],
        };

        renderer.device.cmd_push_constants(
            command_buffer,
            renderer.compute_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&[compute_push_constants]),
        );

        renderer
            .device
            .cmd_dispatch(command_buffer, renderer.objects.len() as u32, 1, 1);

        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
            .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ);

        let dependency_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        renderer
            .device
            .cmd_pipeline_barrier2(command_buffer, &dependency_info);
    }

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::empty())
        .old_layout(vk::ImageLayout::UNDEFINED)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
        .image(renderer.swapchain.images[image_index as usize])
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            layer_count: 1,
            level_count: 1,
            ..Default::default()
        });

    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));

    unsafe {
        renderer
            .device
            .cmd_pipeline_barrier2(command_buffer, &dependency_info);
    }

    let color_attachment_info = vk::RenderingAttachmentInfo::default()
        .image_view(renderer.swapchain.image_views[image_index as usize])
        .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        });

    let depth_attachment_info = vk::RenderingAttachmentInfo::default()
        .image_view(renderer.depth_image_view.unwrap())
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        });

    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: renderer.swapchain.extent,
        })
        .layer_count(1)
        .color_attachments(std::slice::from_ref(&color_attachment_info))
        .depth_attachment(&depth_attachment_info);

    unsafe {
        renderer
            .device
            .cmd_begin_rendering(command_buffer, &rendering_info);

        renderer.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            renderer.pipeline,
        );

        renderer.device.cmd_bind_index_buffer(
            command_buffer,
            renderer.index_buffer.as_ref().unwrap().buffer,
            0,
            vk::IndexType::UINT32,
        );

        let graphics_push_constants = create_push_constants(
            time,
            renderer
                .vertex_buffer
                .as_ref()
                .unwrap()
                .device_address
                .unwrap(),
            renderer
                .object_buffer
                .as_ref()
                .unwrap()
                .device_address
                .unwrap(),
        );

        renderer.device.cmd_push_constants(
            command_buffer,
            renderer.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::cast_slice(&[graphics_push_constants]),
        );

        renderer.device.cmd_draw_indexed_indirect_count(
            command_buffer,
            renderer.indirect_buffer.as_ref().unwrap().buffer,
            0,
            renderer.count_buffer.as_ref().unwrap().buffer,
            0,
            renderer.objects.len() as u32,
            std::mem::size_of::<DrawCommand>() as u32,
        );

        if let Some(primitives) = clipped_primitives {
            if let Some(egui_renderer) = &mut renderer.egui_renderer {
                egui_renderer.cmd_draw(
                    command_buffer,
                    renderer.swapchain.extent,
                    pixels_per_point,
                    &primitives,
                )?;
            }
        }

        renderer.device.cmd_end_rendering(command_buffer);
    }

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::empty())
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(renderer.swapchain.images[image_index as usize])
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            layer_count: 1,
            level_count: 1,
            ..Default::default()
        });

    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));

    unsafe {
        renderer
            .device
            .cmd_pipeline_barrier2(command_buffer, &dependency_info);
        renderer.device.end_command_buffer(command_buffer)?;
    }

    let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(renderer.image_available_semaphores[renderer.current_frame])
        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

    let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(renderer.render_finished_semaphores[image_index as usize])
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

    let cmd_buffer_submit_info =
        vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);

    let submit_info = vk::SubmitInfo2::default()
        .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
        .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
        .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

    unsafe {
        renderer.device.queue_submit2(
            renderer.graphics_queue,
            std::slice::from_ref(&submit_info),
            current_fence,
        )?;
    }

    let signal_semaphores = [renderer.render_finished_semaphores[image_index as usize]];
    let swapchains = [renderer.swapchain.swapchain_khr];
    let images_indices = [image_index];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&images_indices);

    let present_result = unsafe {
        renderer
            .swapchain
            .swapchain
            .queue_present(renderer.present_queue, &present_info)
    };
    match present_result {
        Ok(_) => {}
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
            renderer.is_swapchain_dirty = true;
        }
        Err(error) => return Err(error.into()),
    }

    renderer.current_frame = (renderer.current_frame + 1) % renderer.frames_in_flight;

    if let Some(textures_delta) = textures_delta {
        if !textures_delta.free.is_empty() {
            if let Some(egui_renderer) = &mut renderer.egui_renderer {
                egui_renderer.free_textures(&textures_delta.free)?;
            }
        }
    }

    Ok(())
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_indices: &[u32],
) -> Result<ash::Device, Box<dyn std::error::Error + 'static>> {
    let queue_create_info_list = queue_indices
        .iter()
        .map(|index| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*index)
                .queue_priorities(&[1.0f32])
        })
        .collect::<Vec<_>>();
    let device_extensions_ptrs = [
        swapchain::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ];
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .draw_indirect_count(true)
        .buffer_device_address(true);
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .features(
            vk::PhysicalDeviceFeatures::default()
                .multi_draw_indirect(true)
                .shader_int64(true),
        )
        .push_next(&mut features12)
        .push_next(&mut features13);
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_info_list)
        .enabled_extension_names(&device_extensions_ptrs)
        .push_next(&mut features);
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    Ok(device)
}

fn create_surface<W>(
    window_handle: W,
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<(ash::khr::surface::Instance, vk::SurfaceKHR), Box<dyn std::error::Error + 'static>>
where
    W: raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle,
{
    let surface = ash::khr::surface::Instance::new(entry, instance);
    let surface_khr = unsafe {
        ash_window::create_surface(
            entry,
            instance,
            window_handle.display_handle()?.as_raw(),
            window_handle.window_handle()?.as_raw(),
            None,
        )?
    };
    Ok((surface, surface_khr))
}

fn create_instance(
    window_handle: &impl raw_window_handle::HasDisplayHandle,
    entry: &ash::Entry,
) -> Result<ash::Instance, Box<dyn std::error::Error + 'static>> {
    let app_name = c"Rust + Vulkan Multi-Mesh BDA Instancing";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(app_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 4, 0));

    let layer_names = [c"VK_LAYER_KHRONOS_validation"];
    let layers_names_raw: Vec<*const std::os::raw::c_char> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut extension_names =
        ash_window::enumerate_required_extensions(window_handle.display_handle()?.as_raw())?
            .to_vec();
    extension_names.push(ash::ext::debug_utils::NAME.as_ptr());

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let create_flags = if cfg!(any(target_os = "macos")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    Ok(unsafe { entry.create_instance(&create_info, None) }?)
}

fn create_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<(debug_utils::Instance, vk::DebugUtilsMessengerEXT), Box<dyn std::error::Error + 'static>>
{
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_utils_messenger =
        unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
    Ok((debug_utils, debug_utils_messenger))
}

extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_flag: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
    let message = unsafe { std::ffi::CStr::from_ptr((*p_callback_data).p_message) };
    match flag {
        Flag::VERBOSE => log::trace!("{type_flag:?} - {message:?}"),
        Flag::INFO => log::info!("{type_flag:?} - {message:?}"),
        Flag::WARNING => log::warn!("{type_flag:?} - {message:?}"),
        _ => log::error!("{type_flag:?} - {message:?}"),
    }
    vk::FALSE
}

struct PhysicalDeviceFeatures {
    device_features: vk::PhysicalDeviceFeatures,
    vulkan12_features: vk::PhysicalDeviceVulkan12Features<'static>,
    vulkan13_features: vk::PhysicalDeviceVulkan13Features<'static>,
}

#[allow(clippy::type_complexity)]
fn find_vulkan_physical_device(
    instance: &ash::Instance,
    surface: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32, Option<u32>), Box<dyn std::error::Error>> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let mut physical_device = devices.iter().find(|device| {
        let properties = unsafe { instance.get_physical_device_properties(**device) };
        matches!(properties.device_type, vk::PhysicalDeviceType::DISCRETE_GPU)
    });
    if physical_device.is_none() {
        physical_device = devices.iter().find(|device| {
            let properties = unsafe { instance.get_physical_device_properties(**device) };
            matches!(
                properties.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU
            )
        });
        if physical_device.is_some() {
            log::warn!("No discrete GPU is available, using integrated GPU");
        }
    }
    let Some(physical_device) = physical_device else {
        return Err("No discrete GPU or integrated GPU is available".into());
    };

    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

    let (present_queue_family_index, graphics_queue_family_index, transfer_queue_family_index) =
        find_queue_families(
            surface,
            surface_khr,
            physical_device,
            queue_family_properties,
        )?;

    if !swapchain_supported(instance, physical_device)? {
        return Err("Physical device does not support swapchains".into());
    }

    let formats =
        unsafe { surface.get_physical_device_surface_formats(*physical_device, surface_khr) }?;
    if formats.is_empty() {
        return Err("Physical device does not have any surface formats".into());
    }

    let present_modes = unsafe {
        surface.get_physical_device_surface_present_modes(*physical_device, surface_khr)
    }?;
    if present_modes.is_empty() {
        return Err("Physical device does not have any present modes".into());
    }

    let features = get_physical_device_features(instance, physical_device);
    let supports_dynamic_rendering = features.vulkan13_features.dynamic_rendering == vk::TRUE;
    if !supports_dynamic_rendering {
        return Err("Physical device does not support dynamic rendering".into());
    }
    let supports_synchronization = features.vulkan13_features.synchronization2 == vk::TRUE;
    if !supports_synchronization {
        return Err("Physical device does not support synchronization".into());
    }
    let supports_draw_indirect_count = features.vulkan12_features.draw_indirect_count == vk::TRUE;
    if !supports_draw_indirect_count {
        return Err("Physical device does not support draw indirect count".into());
    }
    let supports_buffer_device_address =
        features.vulkan12_features.buffer_device_address == vk::TRUE;
    if !supports_buffer_device_address {
        return Err("Physical device does not support buffer device address".into());
    }

    if features.device_features.multi_draw_indirect == vk::FALSE {
        return Err("Physical device does not support multi-draw indirect".into());
    }

    Ok((
        *physical_device,
        present_queue_family_index,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ))
}

fn get_physical_device_features(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> PhysicalDeviceFeatures {
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan12_features)
        .push_next(&mut vulkan13_features);

    unsafe { instance.get_physical_device_features2(*physical_device, &mut features2) };

    PhysicalDeviceFeatures {
        device_features: features2.features,
        vulkan12_features,
        vulkan13_features,
    }
}

fn swapchain_supported(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> Result<bool, Box<dyn std::error::Error>> {
    let extension_props =
        unsafe { instance.enumerate_device_extension_properties(*physical_device) }?;
    let supports_swapchain = extension_props.iter().any(|ext| {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        swapchain::NAME == name
    });
    Ok(supports_swapchain)
}

fn find_queue_families(
    surface: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: &vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
) -> Result<(u32, u32, Option<u32>), Box<dyn std::error::Error>> {
    let mut graphics_queue_family_index = None;
    let mut present_queue_family_index = None;
    let mut transfer_queue_family_index = None;

    for (queue_family_index, queue_family) in queue_family_properties
        .iter()
        .filter(|family| family.queue_count > 0)
        .enumerate()
    {
        let has_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let has_compute = queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let has_transfer = queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER);

        if graphics_queue_family_index.is_none() && has_graphics && has_compute {
            graphics_queue_family_index = Some(queue_family_index as u32);
        }

        if transfer_queue_family_index.is_none() && has_transfer && !has_graphics && !has_compute {
            transfer_queue_family_index = Some(queue_family_index as u32);
        }

        let present_support = unsafe {
            surface.get_physical_device_surface_support(
                *physical_device,
                queue_family_index as u32,
                surface_khr,
            )
        }?;
        if present_queue_family_index.is_none() && present_support {
            present_queue_family_index = Some(queue_family_index as u32);
        }
    }

    let present_queue_family_index =
        present_queue_family_index.ok_or("No present queue family found")?;
    let graphics_queue_family_index =
        graphics_queue_family_index.ok_or("No graphics queue family found")?;

    Ok((
        present_queue_family_index,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ))
}

fn create_command_pool(
    graphics_queue_family_index: u32,
    device: &ash::Device,
) -> Result<vk::CommandPool, Box<dyn std::error::Error + 'static>> {
    let command_pool = {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        unsafe { device.create_command_pool(&command_pool_info, None)? }
    };
    Ok(command_pool)
}

fn create_allocator(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
) -> Result<gpu_allocator::vulkan::Allocator, Box<dyn std::error::Error + 'static>> {
    let allocator_create_info = gpu_allocator::vulkan::AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings {
            log_memory_information: true,
            ..Default::default()
        },
        buffer_device_address: true,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    };
    let allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_info)?;
    Ok(allocator)
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain(
    instance: &ash::Instance,
    device: &ash::Device,
    surface: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
) -> Result<Swapchain, Box<dyn std::error::Error + 'static>> {
    let format = {
        let formats =
            unsafe { surface.get_physical_device_surface_formats(physical_device, surface_khr)? };
        if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            }
        } else {
            *formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_UNORM
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(&formats[0])
        }
    };

    let present_mode = {
        let present_modes = unsafe {
            surface.get_physical_device_surface_present_modes(physical_device, surface_khr)
        }?;
        if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };

    let capabilities =
        unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr) }?;

    let extent = {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = width.min(max.width).max(min.width);
            let height = height.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    };

    let image_count = capabilities.min_image_count;

    let families_indices = [graphics_queue_family_index, present_queue_family_index];
    let create_info = {
        let mut builder = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        builder = if graphics_queue_family_index != present_queue_family_index {
            builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families_indices)
        } else {
            builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
    };

    let swapchain = swapchain::Device::new(instance, device);
    let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None)? };

    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr)? };
    let image_views = images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe { device.create_image_view(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Swapchain {
        swapchain,
        swapchain_khr,
        extent,
        format,
        images,
        image_views,
    })
}

fn read_shader(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

fn create_and_record_command_buffers(
    renderer: &Renderer,
) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
    let device = &renderer.device;
    let pool = renderer.command_pool;
    let count = renderer.swapchain.images.len();

    let buffers = {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count as _);

        unsafe { device.allocate_command_buffers(&allocate_info)? }
    };

    Ok(buffers)
}

fn recreate_swapchain(
    renderer: &mut Renderer,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe { renderer.device.device_wait_idle() }?;

    cleanup_swapchain(renderer);

    unsafe {
        renderer.device.destroy_pipeline(renderer.pipeline, None);
        renderer
            .device
            .destroy_pipeline_layout(renderer.pipeline_layout, None);
    }

    renderer.current_frame = 0;

    let swapchain = create_swapchain(
        &renderer.instance,
        &renderer.device,
        &renderer.surface,
        renderer.surface_khr,
        renderer.physical_device,
        width,
        height,
        renderer.graphics_queue_family_index,
        renderer.present_queue_family_index,
    )?;

    if let Some(allocator) = &renderer.allocator {
        let (depth_image, depth_image_view) = create_depth_buffer(
            &renderer.device,
            allocator,
            swapchain.extent,
            renderer.depth_format,
        )?;
        renderer.depth_image = Some(depth_image);
        renderer.depth_image_view = Some(depth_image_view);
    }

    let (pipeline_layout, pipeline) =
        create_pipeline(&renderer.device, &swapchain, renderer.depth_format)?;

    renderer.swapchain = swapchain;
    renderer.pipeline = pipeline;
    renderer.pipeline_layout = pipeline_layout;

    let command_buffers = create_and_record_command_buffers(renderer)?;
    renderer.command_buffers = command_buffers;

    if let Some(allocator) = &renderer.allocator {
        let egui_renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            renderer.device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: renderer.swapchain.format.format,
                depth_attachment_format: Some(renderer.depth_format),
            },
            egui_ash_renderer::Options {
                in_flight_frames: renderer.frames_in_flight,
                ..Default::default()
            },
        )?;
        renderer.egui_renderer = Some(egui_renderer);
    }

    Ok(())
}

fn cleanup_swapchain(renderer: &mut Renderer) {
    unsafe {
        if let Some(depth_image_view) = renderer.depth_image_view.take() {
            renderer.device.destroy_image_view(depth_image_view, None);
        }

        if let Some(allocator_arc) = &renderer.allocator {
            if let Ok(mut alloc) = allocator_arc.lock() {
                if let Some(depth_image) = renderer.depth_image.take() {
                    renderer.device.destroy_image(depth_image.image, None);
                    let _ = alloc.free(depth_image.allocation);
                }
            }
        }

        for image_view in renderer.swapchain.image_views.iter() {
            renderer.device.destroy_image_view(*image_view, None);
        }
        renderer.swapchain.image_views.clear();

        renderer
            .swapchain
            .swapchain
            .destroy_swapchain(renderer.swapchain.swapchain_khr, None);
    }
}
