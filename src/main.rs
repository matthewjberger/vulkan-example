use ash::{ext::debug_utils, khr::swapchain, vk};
use std::sync::{Arc, Mutex};

const RING_STAGING_SIZE: u64 = 64 * 1024 * 1024;
const MAX_BINDLESS_BUFFERS: u32 = 1000;
const MAX_BINDLESS_TEXTURES: u32 = 1000;
const FRAMES_IN_FLIGHT: usize = 3;
const MAX_CACHED_COMMAND_BUFFERS: usize = 32;

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
}

impl winit::application::ApplicationHandler for Context {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let mut attributes = winit::window::Window::default_attributes();
        attributes.title = "Rust + Vulkan Example".to_string();
        if let Ok(window) = event_loop.create_window(attributes) {
            match create_renderer(&window, 800, 600) {
                Ok(renderer) => {
                    self.renderer = Some(renderer);
                }
                Err(error) => log::error!("Failed to create renderer: {error}"),
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
        if let Some(window) = self.window_handle.as_mut()
            && let Some(gui_state) = &mut self.egui_state
        {
            let _consumed_event = gui_state.on_window_event(window, &event).consumed;
        }

        if matches!(event, winit::event::WindowEvent::CloseRequested) {
            event_loop.exit();
            return;
        }

        if let winit::event::WindowEvent::Resized(winit::dpi::PhysicalSize { width, height }) =
            event
            && width > 0
            && height > 0
        {
            if let Some(gui_state) = &mut self.egui_state
                && let Some(window) = self.window_handle.as_ref()
            {
                gui_state
                    .egui_ctx()
                    .set_pixels_per_point(window.scale_factor() as _);
            }

            if let Some(renderer) = &mut self.renderer {
                renderer.is_swapchain_dirty = true;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
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
        egui::Window::new("Texture Selection").show(&egui_ctx, |ui| {
            ui.heading("Texture Viewer");

            if !renderer.texture_names.is_empty() {
                ui.horizontal(|ui| {
                    if ui.button("Previous").clicked() {
                        if renderer.current_texture_index > 0 {
                            renderer.current_texture_index -= 1;
                        } else {
                            renderer.current_texture_index = renderer.texture_names.len() - 1;
                        }
                    }

                    if ui.button("Next").clicked() {
                        renderer.current_texture_index =
                            (renderer.current_texture_index + 1) % renderer.texture_names.len();
                    }
                });

                ui.separator();

                egui::ComboBox::from_label("Select Texture")
                    .selected_text(&renderer.texture_names[renderer.current_texture_index])
                    .show_ui(ui, |ui| {
                        for (index, name) in renderer.texture_names.iter().enumerate() {
                            ui.selectable_value(&mut renderer.current_texture_index, index, name);
                        }
                    });

                ui.label(format!(
                    "Texture {}/{}: {}",
                    renderer.current_texture_index + 1,
                    renderer.texture_names.len(),
                    &renderer.texture_names[renderer.current_texture_index]
                ));
            } else {
                ui.label("No textures loaded");
            }
        });
        let output = egui_state.egui_ctx().end_pass();
        egui_state.handle_platform_output(window_handle, output.platform_output.clone());
        let paint_jobs = egui_ctx.tessellate(output.shapes.clone(), output.pixels_per_point);
        let ui_frame_output = Some((output, paint_jobs));

        let should_render =
            window_handle.inner_size().width > 0 && window_handle.inner_size().height > 0;
        if should_render {
            if let Err(error) = render_frame(renderer, ui_frame_output) {
                log::error!("Failed to draw frame: {error}");
            } else {
                renderer.is_swapchain_dirty = false;
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_mut() {
            let _ = unsafe { renderer.device.device_wait_idle() };
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResourceState {
    layout: vk::ImageLayout,
    queue_family: u32,
    access_stage: vk::PipelineStageFlags2,
    access_flags: vk::AccessFlags2,
}

#[derive(Debug)]
enum ResourceType {
    Image { image: vk::Image, mip_levels: u32 },
    Buffer { buffer: vk::Buffer, size: u64 },
}

#[derive(Debug)]
struct TrackedResource {
    resource_type: ResourceType,
    current_state: ResourceState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum QueueType {
    Graphics,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessType {
    Read,
    Write,
}

#[derive(Debug, Clone)]
struct ResourceAccess {
    resource_name: String,
    desired_layout: Option<vk::ImageLayout>,
    stage_flags: vk::PipelineStageFlags2,
    access_flags: vk::AccessFlags2,
    access_type: AccessType,
}

#[derive(Debug, Clone)]
struct FrameGraphPass {
    queue_type: QueueType,
    resource_accesses: Vec<ResourceAccess>,
}

#[derive(Debug)]
struct GeneratedBarrier {
    image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier2<'static>>,
}

struct FrameGraph {
    resources: std::collections::HashMap<String, TrackedResource>,
    graphics_queue_family: u32,
    transfer_queue_family: Option<u32>,
}

impl FrameGraph {
    fn new(graphics_queue_family: u32, transfer_queue_family: Option<u32>) -> Self {
        Self {
            resources: std::collections::HashMap::new(),
            graphics_queue_family,
            transfer_queue_family,
        }
    }

    fn begin_frame(&mut self) {
        self.resources
            .retain(|name, _| !name.starts_with("swapchain") && !name.ends_with("_transient"));
    }

    fn register_image(
        &mut self,
        name: &str,
        image: vk::Image,
        mip_levels: u32,
        initial_layout: vk::ImageLayout,
        initial_queue: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.resources.contains_key(name) {
            self.resources.remove(name);
        }

        self.resources.insert(
            name.to_string(),
            TrackedResource {
                resource_type: ResourceType::Image { image, mip_levels },
                current_state: ResourceState {
                    layout: initial_layout,
                    queue_family: initial_queue,
                    access_stage: vk::PipelineStageFlags2::NONE,
                    access_flags: vk::AccessFlags2::NONE,
                },
            },
        );

        log::debug!("Registered image resource: {}", name);
        Ok(())
    }

    fn register_buffer(
        &mut self,
        name: &str,
        buffer: vk::Buffer,
        size: u64,
        initial_queue: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.resources.contains_key(name) {
            return Err(format!("Resource '{}' already registered", name).into());
        }

        self.resources.insert(
            name.to_string(),
            TrackedResource {
                resource_type: ResourceType::Buffer { buffer, size },
                current_state: ResourceState {
                    layout: vk::ImageLayout::UNDEFINED,
                    queue_family: initial_queue,
                    access_stage: vk::PipelineStageFlags2::NONE,
                    access_flags: vk::AccessFlags2::NONE,
                },
            },
        );

        log::debug!("Registered buffer resource: {}", name);
        Ok(())
    }

    fn generate_barriers_for_pass(
        &self,
        pass: &FrameGraphPass,
    ) -> Result<GeneratedBarrier, Box<dyn std::error::Error>> {
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();

        let dst_queue_family = match pass.queue_type {
            QueueType::Graphics => self.graphics_queue_family,
            QueueType::Transfer => self
                .transfer_queue_family
                .expect("Transfer queue not available"),
        };

        for access in &pass.resource_accesses {
            let resource = self
                .resources
                .get(&access.resource_name)
                .ok_or(format!("Resource not found: {}", access.resource_name))?;

            let current_state = &resource.current_state;

            let layout_changed = access.desired_layout.is_some()
                && access.desired_layout.unwrap() != current_state.layout;
            let queue_changed = current_state.queue_family != dst_queue_family;
            let has_write_hazard = access.access_type == AccessType::Write
                || current_state
                    .access_flags
                    .contains(vk::AccessFlags2::MEMORY_WRITE);

            let needs_barrier = layout_changed || queue_changed || has_write_hazard;

            if needs_barrier {
                match &resource.resource_type {
                    ResourceType::Image {
                        image, mip_levels, ..
                    } => {
                        let old_layout = current_state.layout;
                        let new_layout = access.desired_layout.unwrap_or(current_state.layout);

                        let (src_queue_family_index, dst_queue_family_index) =
                            if current_state.queue_family == dst_queue_family {
                                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
                            } else {
                                (current_state.queue_family, dst_queue_family)
                            };

                        let barrier = vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(
                                if current_state.access_stage == vk::PipelineStageFlags2::NONE {
                                    vk::PipelineStageFlags2::TOP_OF_PIPE
                                } else {
                                    current_state.access_stage
                                },
                            )
                            .src_access_mask(current_state.access_flags)
                            .old_layout(old_layout)
                            .dst_stage_mask(access.stage_flags)
                            .dst_access_mask(access.access_flags)
                            .new_layout(new_layout)
                            .src_queue_family_index(src_queue_family_index)
                            .dst_queue_family_index(dst_queue_family_index)
                            .image(*image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: *mip_levels,
                                base_array_layer: 0,
                                layer_count: 1,
                            });

                        log::debug!(
                            "  Generated barrier for '{}': layout {:?}->{:?}, queue {}->{}",
                            access.resource_name,
                            old_layout,
                            new_layout,
                            current_state.queue_family,
                            dst_queue_family
                        );

                        image_barriers.push(barrier);
                    }
                    ResourceType::Buffer { buffer, size } => {
                        let (src_queue_family_index, dst_queue_family_index) =
                            if current_state.queue_family == dst_queue_family {
                                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
                            } else {
                                (current_state.queue_family, dst_queue_family)
                            };

                        let barrier = vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(
                                if current_state.access_stage == vk::PipelineStageFlags2::NONE {
                                    vk::PipelineStageFlags2::TOP_OF_PIPE
                                } else {
                                    current_state.access_stage
                                },
                            )
                            .src_access_mask(current_state.access_flags)
                            .dst_stage_mask(access.stage_flags)
                            .dst_access_mask(access.access_flags)
                            .src_queue_family_index(src_queue_family_index)
                            .dst_queue_family_index(dst_queue_family_index)
                            .buffer(*buffer)
                            .offset(0)
                            .size(*size);

                        log::debug!(
                            "  Generated barrier for buffer '{}': queue {}->{}",
                            access.resource_name,
                            current_state.queue_family,
                            dst_queue_family
                        );

                        buffer_barriers.push(barrier);
                    }
                }
            }
        }

        Ok(GeneratedBarrier {
            image_barriers,
            buffer_barriers,
        })
    }

    fn update_resource_states(&mut self, pass: &FrameGraphPass) {
        let queue_family = match pass.queue_type {
            QueueType::Graphics => self.graphics_queue_family,
            QueueType::Transfer => self
                .transfer_queue_family
                .expect("Transfer queue not available"),
        };

        for access in &pass.resource_accesses {
            if let Some(resource) = self.resources.get_mut(&access.resource_name) {
                if let Some(layout) = access.desired_layout {
                    resource.current_state.layout = layout;
                }
                resource.current_state.queue_family = queue_family;
                resource.current_state.access_stage = access.stage_flags;
                resource.current_state.access_flags = access.access_flags;
            }
        }
    }

    fn add_pass(&mut self, queue_type: QueueType) -> PassBuilder<'_> {
        PassBuilder {
            graph: self,
            pass: FrameGraphPass {
                queue_type,
                resource_accesses: Vec::new(),
            },
        }
    }
}

struct PassBuilder<'a> {
    graph: &'a mut FrameGraph,
    pass: FrameGraphPass,
}

impl<'a> PassBuilder<'a> {
    fn read_image(
        mut self,
        resource_name: &str,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> Self {
        self.pass.resource_accesses.push(ResourceAccess {
            resource_name: resource_name.to_string(),
            desired_layout: Some(layout),
            stage_flags: stage,
            access_flags: access,
            access_type: AccessType::Read,
        });
        self
    }

    fn write_image(
        mut self,
        resource_name: &str,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> Self {
        self.pass.resource_accesses.push(ResourceAccess {
            resource_name: resource_name.to_string(),
            desired_layout: Some(layout),
            stage_flags: stage,
            access_flags: access,
            access_type: AccessType::Write,
        });
        self
    }

    fn read_buffer(
        mut self,
        resource_name: &str,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> Self {
        self.pass.resource_accesses.push(ResourceAccess {
            resource_name: resource_name.to_string(),
            desired_layout: None,
            stage_flags: stage,
            access_flags: access,
            access_type: AccessType::Read,
        });
        self
    }

    fn write_buffer(
        mut self,
        resource_name: &str,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> Self {
        self.pass.resource_accesses.push(ResourceAccess {
            resource_name: resource_name.to_string(),
            desired_layout: None,
            stage_flags: stage,
            access_flags: access,
            access_type: AccessType::Write,
        });
        self
    }

    fn execute_inline<F>(
        self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        func: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), Box<dyn std::error::Error>>,
    {
        for access in &self.pass.resource_accesses {
            if !self.graph.resources.contains_key(&access.resource_name) {
                return Err(format!("Unknown resource: {}", access.resource_name).into());
            }
        }

        let barriers = self.graph.generate_barriers_for_pass(&self.pass)?;

        if !barriers.image_barriers.is_empty() || !barriers.buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&barriers.image_barriers)
                .buffer_memory_barriers(&barriers.buffer_barriers);
            unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
        }

        func(device, cmd)?;

        self.graph.update_resource_states(&self.pass);

        Ok(())
    }

    fn execute<F>(
        self,
        device: &ash::Device,
        graphics_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        func: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), Box<dyn std::error::Error>>,
    {
        for access in &self.pass.resource_accesses {
            if !self.graph.resources.contains_key(&access.resource_name) {
                return Err(format!("Unknown resource: {}", access.resource_name).into());
            }
        }

        let (pool, queue) = match self.pass.queue_type {
            QueueType::Graphics => (graphics_pool, graphics_queue),
            QueueType::Transfer => panic!("Transfer queue not supported in execute method"),
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin_info)? };

        let barriers = self.graph.generate_barriers_for_pass(&self.pass)?;

        if !barriers.image_barriers.is_empty() || !barriers.buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&barriers.image_barriers)
                .buffer_memory_barriers(&barriers.buffer_barriers);
            unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
        }

        func(device, cmd)?;

        unsafe { device.end_command_buffer(cmd)? };

        let cmd_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit_info =
            vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_buffer_info));

        unsafe {
            device.queue_submit2(queue, std::slice::from_ref(&submit_info), vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
        };

        self.graph.update_resource_states(&self.pass);

        unsafe { device.free_command_buffers(pool, &[cmd]) };

        Ok(())
    }
}

struct CommandBufferPool {
    device: ash::Device,
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    next_index: usize,
}

impl CommandBufferPool {
    fn new(
        device: ash::Device,
        pool: vk::CommandPool,
        initial_count: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(initial_count);

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info)? };

        Ok(Self {
            device: device.clone(),
            pool,
            buffers,
            next_index: 0,
        })
    }

    fn acquire(&mut self) -> Result<vk::CommandBuffer, Box<dyn std::error::Error>> {
        if self.next_index >= self.buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let new_buffer = unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] };
            self.buffers.push(new_buffer);
        }

        let buffer = self.buffers[self.next_index];
        self.next_index += 1;
        Ok(buffer)
    }

    fn reset(&mut self) {
        if self.buffers.len() > MAX_CACHED_COMMAND_BUFFERS {
            let excess = &self.buffers[MAX_CACHED_COMMAND_BUFFERS..];
            unsafe {
                self.device.free_command_buffers(self.pool, excess);
            }
            self.buffers.truncate(MAX_CACHED_COMMAND_BUFFERS);
        }
        self.next_index = 0;
    }
}

struct StagingBuffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    size: u64,
    frames_since_shrink_check: u32,
    max_used_size: u64,
}

impl StagingBuffer {
    fn ensure_capacity(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        required_size: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.max_used_size = self.max_used_size.max(required_size);

        if required_size > self.size {
            if self.buffer != vk::Buffer::null() {
                unsafe { device.destroy_buffer(self.buffer, None) };
            }
            if let Some(allocation) = self.allocation.take() {
                allocator.lock().unwrap().free(allocation)?;
            }

            let new_size = (required_size * 3) / 2;

            let buffer_info = vk::BufferCreateInfo::default()
                .size(new_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            self.buffer = unsafe { device.create_buffer(&buffer_info, None)? };
            let requirements = unsafe { device.get_buffer_memory_requirements(self.buffer) };

            self.allocation = Some(allocator.lock().unwrap().allocate(
                &gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "staging_buffer",
                    requirements,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                },
            )?);

            unsafe {
                device.bind_buffer_memory(
                    self.buffer,
                    self.allocation.as_ref().unwrap().memory(),
                    self.allocation.as_ref().unwrap().offset(),
                )?;
            }

            self.size = new_size;
            self.max_used_size = required_size;
            self.frames_since_shrink_check = 0;
        }
        Ok(())
    }

    fn check_and_shrink(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        const SHRINK_CHECK_INTERVAL: u32 = 300;
        const SHRINK_THRESHOLD: f32 = 0.25;

        self.frames_since_shrink_check += 1;

        if self.frames_since_shrink_check >= SHRINK_CHECK_INTERVAL && self.size > 0 {
            let usage_ratio = self.max_used_size as f32 / self.size as f32;

            if usage_ratio < SHRINK_THRESHOLD {
                let new_size = (self.max_used_size * 3) / 2;

                if self.buffer != vk::Buffer::null() {
                    unsafe { device.destroy_buffer(self.buffer, None) };
                }
                if let Some(allocation) = self.allocation.take() {
                    allocator.lock().unwrap().free(allocation)?;
                }

                let buffer_info = vk::BufferCreateInfo::default()
                    .size(new_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);

                self.buffer = unsafe { device.create_buffer(&buffer_info, None)? };
                let requirements = unsafe { device.get_buffer_memory_requirements(self.buffer) };

                self.allocation = Some(allocator.lock().unwrap().allocate(
                    &gpu_allocator::vulkan::AllocationCreateDesc {
                        name: "staging_buffer",
                        requirements,
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                    },
                )?);

                unsafe {
                    device.bind_buffer_memory(
                        self.buffer,
                        self.allocation.as_ref().unwrap().memory(),
                        self.allocation.as_ref().unwrap().offset(),
                    )?;
                }

                log::info!(
                    "Shrunk staging buffer from {} to {} bytes (usage: {:.1}%)",
                    self.size,
                    new_size,
                    usage_ratio * 100.0
                );

                self.size = new_size;
            }

            self.max_used_size = 0;
            self.frames_since_shrink_check = 0;
        }

        Ok(())
    }

    fn get_mapped_ptr(&self) -> *mut u8 {
        self.allocation
            .as_ref()
            .unwrap()
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8
    }
}

struct RingStagingBuffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    total_size: u64,
    head: u64,
    tail: u64,
    in_flight: std::collections::VecDeque<(u64, u64, u64)>,
}

impl RingStagingBuffer {
    fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        size: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = Some(allocator.lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "ring_staging",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            },
        )?);

        unsafe {
            device.bind_buffer_memory(
                buffer,
                allocation.as_ref().unwrap().memory(),
                allocation.as_ref().unwrap().offset(),
            )?;
        }

        Ok(Self {
            buffer,
            allocation,
            total_size: size,
            head: 0,
            tail: 0,
            in_flight: std::collections::VecDeque::new(),
        })
    }

    fn free_completed(&mut self, current_timeline_value: u64) {
        while let Some((offset, size, completion_value)) = self.in_flight.front() {
            if current_timeline_value >= *completion_value {
                self.tail = (offset + size) % self.total_size;
                self.in_flight.pop_front();
            } else {
                break;
            }
        }
    }

    fn allocate(
        &mut self,
        size: u64,
        completion_value: u64,
    ) -> Result<(u64, *mut u8), Box<dyn std::error::Error>> {
        let aligned_size = (size + 255) & !255;

        let space_to_end = self.total_size - self.head;

        let offset = if space_to_end < aligned_size {
            if space_to_end > 0 {
                self.in_flight
                    .push_back((self.head, space_to_end, completion_value));
            }
            self.head = 0;

            if self.tail > 0 && aligned_size > self.tail {
                return Err("Ring buffer full - not enough space after wrap".into());
            } else if self.tail == 0 {
                return Err("Ring buffer full - GPU not consuming fast enough".into());
            }

            0
        } else {
            if self.head >= self.tail {
                self.head
            } else {
                if self.head + aligned_size > self.tail {
                    return Err("Ring buffer full - would collide with tail".into());
                }
                self.head
            }
        };

        let ptr = unsafe {
            self.allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr()
                .add(offset as usize) as *mut u8
        };

        self.in_flight
            .push_back((offset, aligned_size, completion_value));
        self.head = (offset + aligned_size) % self.total_size;

        Ok((offset, ptr))
    }

    fn get_usage_stats(&self) -> (u64, u64, usize) {
        let used = if self.head >= self.tail {
            self.head - self.tail
        } else {
            self.total_size - (self.tail - self.head)
        };
        let available = self.total_size - used;
        let pending_ops = self.in_flight.len();

        (used, available, pending_ops)
    }

    fn allocate_blocking(
        &mut self,
        size: u64,
        completion_value: u64,
        device: &ash::Device,
        semaphore: vk::Semaphore,
    ) -> Result<(u64, *mut u8), Box<dyn std::error::Error>> {
        const TIMEOUT_NS: u64 = 5_000_000_000;

        loop {
            match self.allocate(size, completion_value) {
                Ok(result) => return Ok(result),
                Err(_) => {
                    if let Some((_, _, oldest_value)) = self.in_flight.front() {
                        let wait_values = [*oldest_value];
                        let wait_semaphores = [semaphore];
                        let wait_info = vk::SemaphoreWaitInfo::default()
                            .semaphores(&wait_semaphores)
                            .values(&wait_values);
                        let wait_result = unsafe {
                            device.wait_semaphores(&wait_info, TIMEOUT_NS)
                        };
                        match wait_result {
                            Ok(()) => {
                                let current = unsafe { device.get_semaphore_counter_value(semaphore)? };
                                self.free_completed(current);
                            }
                            Err(vk::Result::TIMEOUT) => {
                                return Err(format!(
                                    "Timeout waiting for ring buffer space (waited {}s)",
                                    TIMEOUT_NS as f64 / 1_000_000_000.0
                                ).into());
                            }
                            Err(error) => {
                                return Err(format!("Failed to wait for semaphore: {}", error).into());
                            }
                        }
                    } else {
                        return Err("Ring buffer exhausted with no in-flight operations".into());
                    }
                }
            }
        }
    }

    fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

struct Buffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    _binding_array_index: u32,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureState {
    Uploading { completion_value: u64 },
    Ready,
}

struct Texture {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    _binding_array_index: u32,
    width: u32,
    height: u32,
    mip_levels: u32,
    state: TextureState,
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
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub bindless_set_layout: vk::DescriptorSetLayout,
    pub bindless_pool: vk::DescriptorPool,
    pub buffers: Vec<Buffer>,
    pub textures: Vec<Texture>,
    pub transfer_command_pool: Option<vk::CommandPool>,
    pub transfer_timeline_semaphore: vk::Semaphore,
    pub transfer_timeline_counter: u64,
    pub current_texture_index: usize,
    pub texture_names: Vec<String>,
    pub swapchain: Swapchain,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline_cache: vk::PipelineCache,
    pub command_buffers: Vec<vk::CommandBuffer>,

    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
    pub current_frame: usize,
    pub frames_in_flight: usize,

    pub is_swapchain_dirty: bool,

    pub frame_graph: FrameGraph,
    pub command_buffer_pool: CommandBufferPool,
    pub transfer_command_buffer_pool: Option<CommandBufferPool>,
    pub staging_buffer: Option<StagingBuffer>,
    pub ring_staging: Option<RingStagingBuffer>,

    pub egui_renderer: Option<egui_ash_renderer::Renderer>,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                log::error!("Failed to wait for device idle during cleanup: {:?}", e);
            }
            self.egui_renderer = None;

            for i in 0..self.frames_in_flight {
                self.device.destroy_fence(self.in_flight_fences[i], None);
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
            }

            for i in 0..self.render_finished_semaphores.len() {
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
            }

            cleanup_swapchain(self);

            for texture in &mut self.textures {
                self.device.destroy_sampler(texture.sampler, None);
                self.device.destroy_image_view(texture.view, None);
                self.device.destroy_image(texture.image, None);
                if let Some(allocation) = texture.allocation.take() {
                    if let Err(e) = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation)
                    {
                        log::error!("Failed to free texture allocation: {:?}", e);
                    }
                }
            }

            for buffer in &mut self.buffers {
                self.device.destroy_buffer(buffer.buffer, None);
                if let Some(allocation) = buffer.allocation.take() {
                    if let Err(e) = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation)
                    {
                        log::error!("Failed to free buffer allocation: {:?}", e);
                    }
                }
            }

            if let Some(staging) = &mut self.staging_buffer {
                if staging.buffer != vk::Buffer::null() {
                    self.device.destroy_buffer(staging.buffer, None);
                }
                if let Some(allocation) = staging.allocation.take() {
                    if let Err(e) = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation)
                    {
                        log::error!("Failed to free staging buffer allocation: {:?}", e);
                    }
                }
            }

            if let Some(ring) = &mut self.ring_staging {
                if ring.buffer != vk::Buffer::null() {
                    self.device.destroy_buffer(ring.buffer, None);
                }
                if let Some(allocation) = ring.allocation.take() {
                    if let Err(e) = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation)
                    {
                        log::error!("Failed to free ring staging allocation: {:?}", e);
                    }
                }
            }

            self.device
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless_set_layout, None);

            if let Err(error) = save_pipeline_cache(&self.device, self.pipeline_cache) {
                log::warn!("Failed to save pipeline cache: {}", error);
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            self.allocator = None;

            self.device
                .destroy_semaphore(self.transfer_timeline_semaphore, None);

            if let Some(pool) = self.transfer_command_pool {
                self.device.destroy_command_pool(pool, None);
            }

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

fn create_renderer<W>(
    window_handle: W,
    initial_width: u32,
    initial_height: u32,
) -> std::result::Result<Renderer, Box<dyn std::error::Error>>
where
    W: raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle,
{
    log::info!("Creating vulkan render backend");

    let entry = unsafe { ash::Entry::load()? };
    log::info!("Loaded vulkan entry");

    let instance = create_instance(&window_handle, &entry)?;
    log::info!("Loaded vulkan instance");

    let (surface, surface_khr) = create_surface(window_handle, &entry, &instance)?;
    log::info!("Created vulkan surface");

    let (debug_utils, debug_utils_messenger) = create_debug_utils(&entry, &instance)?;
    log::info!("Created vulkan debug utils");

    let (
        physical_device,
        present_queue_family_index,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ) = find_vulkan_physical_device(&instance, &surface, surface_khr)?;
    log::info!("Found a supported vulkan physical device");

    let mut queue_indices = vec![present_queue_family_index, graphics_queue_family_index];
    if let Some(transfer_queue_index) = transfer_queue_family_index {
        queue_indices.push(transfer_queue_index);
    }
    queue_indices.dedup();
    let device = create_device(&instance, physical_device, &queue_indices)?;
    log::info!("Created vulkan logical device");

    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_queue_family_index, 0) };
    let transfer_queue =
        transfer_queue_family_index.map(|index| unsafe { device.get_device_queue(index, 0) });
    log::info!("Got device queues");

    let command_pool = create_command_pool(graphics_queue_family_index, &device)?;
    log::info!("Created command pool");

    let transfer_command_pool = transfer_queue_family_index
        .map(|index| create_command_pool(index, &device))
        .transpose()?;
    if transfer_command_pool.is_some() {
        log::info!("Created transfer command pool");
    }

    let allocator = create_allocator(&instance, physical_device, &device)?;
    let allocator = Arc::new(Mutex::new(allocator));
    log::info!("Created gpu memory allocator");

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
    log::info!("Created swapchain");

    let format_properties = unsafe {
        instance.get_physical_device_format_properties(
            physical_device,
            vk::Format::R8G8B8A8_UNORM,
        )
    };
    if !format_properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err("R8G8B8A8_UNORM does not support linear filtering for mipmaps".into());
    }
    if !format_properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
    {
        return Err("R8G8B8A8_UNORM does not support blit operations for mipmaps".into());
    }
    log::info!("Validated format features for mipmap generation");

    let bindless_set_layout = create_bindless_descriptor_layout(&device)?;
    log::info!("Created bindless descriptor layout");

    let bindless_pool = create_bindless_descriptor_pool(&device)?;
    log::info!("Created bindless descriptor pool");

    let bindless_descriptor_set =
        allocate_bindless_descriptor_set(&device, bindless_pool, bindless_set_layout)?;
    log::info!("Allocated bindless descriptor set");

    let pipeline_cache = create_pipeline_cache(&device)?;

    let (pipeline_layout, pipeline) = create_pipeline(&device, &swapchain, bindless_set_layout, pipeline_cache)?;
    log::info!("Created render pipeline and layout");

    let egui_renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
        allocator.clone(),
        device.clone(),
        egui_ash_renderer::DynamicRendering {
            color_attachment_format: swapchain.format.format,
            depth_attachment_format: None,
        },
        egui_ash_renderer::Options {
            in_flight_frames: FRAMES_IN_FLIGHT,
            ..Default::default()
        },
    )?;

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

    let mut timeline_semaphore_type_info =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let timeline_semaphore_info =
        vk::SemaphoreCreateInfo::default().push_next(&mut timeline_semaphore_type_info);
    let transfer_timeline_semaphore =
        unsafe { device.create_semaphore(&timeline_semaphore_info, None)? };
    log::info!("Created timeline semaphore for async transfers");

    let command_buffer_pool = CommandBufferPool::new(device.clone(), command_pool, 8)?;

    let transfer_command_buffer_pool = transfer_command_pool
        .map(|pool| CommandBufferPool::new(device.clone(), pool, 8))
        .transpose()?;

    let command_buffers = {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain.images.len() as _);
        unsafe { device.allocate_command_buffers(&allocate_info)? }
    };

    let ring_staging = Some(RingStagingBuffer::new(
        &device,
        &allocator,
        RING_STAGING_SIZE,
    )?);

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
        bindless_descriptor_set,
        bindless_set_layout,
        bindless_pool,
        buffers: vec![],
        textures: vec![],
        transfer_command_pool,
        transfer_timeline_semaphore,
        transfer_timeline_counter: 0,
        current_texture_index: 0,
        texture_names: vec![],
        swapchain,
        pipeline,
        pipeline_layout,
        pipeline_cache,
        command_buffers,
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,
        current_frame: 0,
        frames_in_flight: FRAMES_IN_FLIGHT,
        is_swapchain_dirty: false,
        frame_graph: FrameGraph::new(graphics_queue_family_index, transfer_queue_family_index),
        command_buffer_pool,
        transfer_command_buffer_pool,
        staging_buffer: Some(StagingBuffer {
            buffer: vk::Buffer::null(),
            allocation: None,
            size: 0,
            frames_since_shrink_check: 0,
            max_used_size: 0,
        }),
        ring_staging,
        egui_renderer: Some(egui_renderer),
    };

    log::info!("Allocated command buffers");

    let test_textures = generate_test_textures();
    for (name, width, height, data) in test_textures {
        renderer.upload_texture_async(name, width, height, data)?;
    }

    #[repr(C)]
    struct Vertex {
        position: [f32; 3],
        tex_coord: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.5, 0.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.0],
            tex_coord: [1.0, 1.0],
        },
    ];

    let vertex_data_size = std::mem::size_of_val(&vertices) as u64;
    renderer.create_buffer(
        vertex_data_size,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        gpu_allocator::MemoryLocation::CpuToGpu,
    )?;

    let vertex_buffer_index = renderer.buffers.len() - 1;
    renderer.upload_buffer_data(vertex_buffer_index, &vertices)?;

    log::info!("Created test vertex buffer");

    log::debug!("Registering resources in frame graph");
    for (index, texture) in renderer.textures.iter().enumerate() {
        renderer.frame_graph.register_image(
            &format!("texture_{}", index),
            texture.image,
            texture.mip_levels,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            renderer.graphics_queue_family_index,
        )?;
    }

    if let Some(vertex_buffer) = renderer.buffers.first() {
        renderer.frame_graph.register_buffer(
            "vertex_buffer",
            vertex_buffer.buffer,
            vertex_buffer.size,
            renderer.graphics_queue_family_index,
        )?;
    }

    log::info!("Creating test buffer for write->read hazard test");
    let test_buffer_size = 1024u64;
    renderer.create_buffer(
        test_buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;

    let test_buffer_index = renderer.buffers.len() - 1;
    renderer.frame_graph.register_buffer(
        "test_buffer",
        renderer.buffers[test_buffer_index].buffer,
        test_buffer_size,
        renderer.graphics_queue_family_index,
    )?;

    if renderer._transfer_queue.is_some() {
        log::debug!("Testing frame graph graphics queue support");

        renderer
            .frame_graph
            .add_pass(QueueType::Graphics)
            .read_image(
                "texture_0",
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )
            .execute(
                &renderer.device,
                renderer.command_pool,
                renderer.graphics_queue,
                |_device, _cmd| {
                    log::debug!("  Executing on graphics queue!");
                    Ok(())
                },
            )?;
    }

    Ok(renderer)
}

fn create_pipeline_cache(
    device: &ash::Device,
) -> Result<vk::PipelineCache, Box<dyn std::error::Error + 'static>> {
    let cache_path = std::path::Path::new("pipeline_cache.bin");
    let initial_data = if cache_path.exists() {
        match std::fs::read(cache_path) {
            Ok(data) => {
                log::info!("Loaded pipeline cache from disk ({} bytes)", data.len());
                data
            }
            Err(error) => {
                log::warn!("Failed to read pipeline cache: {}", error);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(&initial_data);
    let cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };
    log::info!("Created pipeline cache");
    Ok(cache)
}

fn save_pipeline_cache(
    device: &ash::Device,
    cache: vk::PipelineCache,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let cache_data = unsafe { device.get_pipeline_cache_data(cache)? };
    std::fs::write("pipeline_cache.bin", &cache_data)?;
    log::info!("Saved pipeline cache to disk ({} bytes)", cache_data.len());
    Ok(())
}

fn create_pipeline(
    device: &ash::Device,
    swapchain: &Swapchain,
    bindless_set_layout: vk::DescriptorSetLayout,
    pipeline_cache: vk::PipelineCache,
) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn std::error::Error + 'static>> {
    let descriptor_set_layouts = [bindless_set_layout];
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(4);
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;
    let entry_point_name = c"main";
    let vertex_source = read_shader(&include_bytes!("shaders/quad.vert.spv")[..])?;
    let vertex_create_info = vk::ShaderModuleCreateInfo::default().code(&vertex_source);
    let vertex_module = unsafe { device.create_shader_module(&vertex_create_info, None)? };
    let fragment_source = read_shader(&include_bytes!("shaders/quad.frag.spv")[..])?;
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
    let vertex_binding_descriptions = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(20)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let vertex_attribute_descriptions = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
    ];
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descriptions)
        .vertex_attribute_descriptions(&vertex_attribute_descriptions);
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
        .cull_mode(vk::CullModeFlags::BACK)
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
        .color_attachment_formats(&color_attachment_formats);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .color_blend_state(&color_blending_info)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);
    let pipeline = unsafe {
        device
            .create_graphics_pipelines(
                pipeline_cache,
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
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .descriptor_binding_partially_bound(true)
        .runtime_descriptor_array(true)
        .descriptor_binding_variable_descriptor_count(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .timeline_semaphore(true);
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features13)
        .push_next(&mut features12);
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
    let app_name = c"Rust + Vulkan Example";
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
            log::info!("No discrete GPU is available, using integrated GPU");
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
    let supports_dynamic_rendering = features.dynamic_rendering == vk::TRUE;
    if !supports_dynamic_rendering {
        return Err("Physical device does not support dynamic rendering".into());
    }
    let supports_synchronization = features.synchronization2 == vk::TRUE;
    if !supports_synchronization {
        return Err("Physical device does not support synchronization".into());
    }

    log::info!("Found graphics queue {graphics_queue_family_index}");
    log::info!("Found present queue {present_queue_family_index}");

    if let Some(transfer_queue_index) = transfer_queue_family_index {
        log::info!("Found dedicated transfer queue {transfer_queue_index}");
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
) -> vk::PhysicalDeviceVulkan13Features<'static> {
    let mut vulkan_features_1_3 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_features_1_3);
    unsafe { instance.get_physical_device_features2(*physical_device, &mut features) };
    vulkan_features_1_3
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
    let mut debug_settings = gpu_allocator::AllocatorDebugSettings::default();
    debug_settings.log_memory_information = true;

    let allocator_create_info = gpu_allocator::vulkan::AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings,
        buffer_device_address: true,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    };
    let allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_info)?;
    Ok(allocator)
}

fn create_bindless_descriptor_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error + 'static>> {
    let binding_flags = [
        vk::DescriptorBindingFlags::UPDATE_AFTER_BIND | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
    ];

    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1000)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1000)
            .stage_flags(vk::ShaderStageFlags::ALL),
    ];

    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .push_next(&mut binding_flags_info);

    let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
    Ok(layout)
}

fn create_bindless_descriptor_pool(
    device: &ash::Device,
) -> Result<vk::DescriptorPool, Box<dyn std::error::Error + 'static>> {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1000,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1000,
        },
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

    let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    Ok(pool)
}

fn allocate_bindless_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, Box<dyn std::error::Error + 'static>> {
    let descriptor_counts = [1000u32];
    let mut variable_descriptor_count_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&descriptor_counts);

    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts)
        .push_next(&mut variable_descriptor_count_info);

    let sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };
    Ok(sets[0])
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
    log::info!("Swapchain format: {format:?}");

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
    log::info!("Swapchain present mode: {present_mode:?}");

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
    log::info!("Swapchain extent: {:?}", extent);

    let image_count = capabilities.min_image_count;
    log::info!("Swapchain image count: {:?}", image_count);

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

fn recreate_swapchain(
    renderer: &mut Renderer,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!(
        "Recreating the swapchain with dimensions {}x{}",
        width,
        height
    );

    unsafe { renderer.device.device_wait_idle() }?;

    cleanup_swapchain(renderer);

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
    log::info!("Recreated swapchain");

    let (pipeline_layout, pipeline) =
        create_pipeline(&renderer.device, &swapchain, renderer.bindless_set_layout, renderer.pipeline_cache)?;
    log::info!("Recreated render pipeline and layout");

    renderer.swapchain = swapchain;
    renderer.pipeline = pipeline;
    renderer.pipeline_layout = pipeline_layout;

    let needed_count = renderer.swapchain.images.len();
    let current_count = renderer.command_buffers.len();

    if needed_count > current_count {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(renderer.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count((needed_count - current_count) as _);
        let mut new_buffers = unsafe { renderer.device.allocate_command_buffers(&allocate_info)? };
        renderer.command_buffers.append(&mut new_buffers);
        log::info!("Allocated {} additional command buffers", needed_count - current_count);
    } else if needed_count < current_count {
        let buffers_to_free: Vec<_> = renderer.command_buffers.drain(needed_count..).collect();
        unsafe {
            renderer.device.free_command_buffers(renderer.command_pool, &buffers_to_free);
        }
        log::info!("Freed {} excess command buffers", current_count - needed_count);
    } else {
        log::info!("Reused existing command buffers");
    }

    renderer.images_in_flight = vec![vk::Fence::null(); renderer.swapchain.images.len()];

    Ok(())
}

fn render_frame(
    renderer: &mut Renderer,
    ui_frame_output: Option<(egui::FullOutput, Vec<egui::ClippedPrimitive>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let current_fence = renderer.in_flight_fences[renderer.current_frame];
    unsafe {
        renderer
            .device
            .wait_for_fences(&[current_fence], true, u64::MAX)?
    };

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
            log::info!("Swapchain is out of date");
            return Ok(());
        }
        Err(error) => panic!("Error while acquiring next image: {error}"),
    };

    if renderer.images_in_flight[image_index as usize] != vk::Fence::null() {
        unsafe {
            if let Err(fence_wait_result) = renderer.device.wait_for_fences(
                &[renderer.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            ) {
                log::error!("Error while waiting for fences: {fence_wait_result}");
            }
        }
    }

    renderer.images_in_flight[image_index as usize] = current_fence;
    unsafe { renderer.device.reset_fences(&[current_fence])? };

    renderer.frame_graph.begin_frame();
    renderer.command_buffer_pool.reset();
    if let Some(transfer_pool) = &mut renderer.transfer_command_buffer_pool {
        transfer_pool.reset();
    }

    let (textures_delta, clipped_primitives, pixels_per_point) =
        if let Some((full_output, primitives)) = ui_frame_output {
            if !full_output.textures_delta.set.is_empty()
                && let Some(egui_renderer) = &mut renderer.egui_renderer
            {
                egui_renderer.set_textures(
                    renderer.graphics_queue,
                    renderer.command_pool,
                    full_output.textures_delta.set.as_slice(),
                )?;
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

    renderer.frame_graph.register_image(
        "swapchain",
        renderer.swapchain.images[image_index as usize],
        1,
        vk::ImageLayout::UNDEFINED,
        renderer.graphics_queue_family_index,
    )?;

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    unsafe {
        renderer
            .device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?
    };

    renderer.process_pending_mipmaps(command_buffer)?;

    let device = &renderer.device;
    let swapchain_extent = renderer.swapchain.extent;
    let swapchain_image_view = renderer.swapchain.image_views[image_index as usize];
    let pipeline = renderer.pipeline;
    let pipeline_layout = renderer.pipeline_layout;
    let bindless_descriptor_set = renderer.bindless_descriptor_set;
    let vertex_buffer = renderer.buffers.first().map(|b| b.buffer);
    let texture_index = renderer.current_texture_index;
    let egui_primitives = clipped_primitives;
    let egui_ppp = pixels_per_point;
    let egui_renderer_ptr = renderer.egui_renderer.as_mut().map(|r| r as *mut _);

    renderer
        .frame_graph
        .add_pass(QueueType::Graphics)
        .write_image(
            "swapchain",
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )
        .read_image(
            &format!("texture_{}", texture_index),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )
        .read_buffer(
            "vertex_buffer",
            vk::PipelineStageFlags2::VERTEX_INPUT,
            vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        )
        .execute_inline(device, command_buffer, |device, cmd| {
            let color_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(swapchain_image_view)
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment_info));

            unsafe {
                device.cmd_begin_rendering(cmd, &rendering_info);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[bindless_descriptor_set],
                    &[],
                );

                if let Some(vb) = vertex_buffer {
                    device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                }

                let tex_idx = texture_index as u32;
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &tex_idx.to_ne_bytes(),
                );

                device.cmd_draw(cmd, 6, 1, 0, 0);

                if let Some(primitives) = egui_primitives
                    && let Some(egui_ptr) = egui_renderer_ptr
                {
                    let egui_renderer: &mut egui_ash_renderer::Renderer = &mut *egui_ptr;
                    egui_renderer
                        .cmd_draw(cmd, swapchain_extent, egui_ppp, &primitives)
                        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
                }

                device.cmd_end_rendering(cmd);
            }

            Ok(())
        })?;

    renderer
        .frame_graph
        .add_pass(QueueType::Graphics)
        .write_buffer(
            "test_buffer",
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
        )
        .execute_inline(device, command_buffer, |_device, _cmd| {
            log::debug!("  Writing to test buffer (compute)");
            Ok(())
        })?;

    renderer
        .frame_graph
        .add_pass(QueueType::Graphics)
        .read_buffer(
            "test_buffer",
            vk::PipelineStageFlags2::VERTEX_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )
        .execute_inline(device, command_buffer, |_device, _cmd| {
            log::debug!("  Reading from test buffer (vertex)");
            Ok(())
        })?;

    renderer
        .frame_graph
        .add_pass(QueueType::Graphics)
        .read_buffer(
            "vertex_buffer",
            vk::PipelineStageFlags2::VERTEX_INPUT,
            vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        )
        .execute_inline(device, command_buffer, |_device, _cmd| {
            log::debug!("  Reusing vertex buffer (should have no barrier)");
            Ok(())
        })?;

    renderer
        .frame_graph
        .add_pass(QueueType::Graphics)
        .read_image(
            "swapchain",
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::NONE,
        )
        .execute_inline(device, command_buffer, |_device, _cmd| Ok(()))?;

    unsafe { renderer.device.end_command_buffer(command_buffer)? };

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
        )?
    };

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
        Ok(is_suboptimal) if is_suboptimal => {
            return Ok(());
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            return Ok(());
        }
        Err(error) => panic!("Failed to present queue. Cause: {}", error),
        _ => {}
    }
    renderer.current_frame = (renderer.current_frame + 1) % renderer.frames_in_flight;

    if let Some(staging_buffer) = &mut renderer.staging_buffer {
        if let Some(allocator) = &renderer.allocator {
            let _ = staging_buffer.check_and_shrink(&renderer.device, allocator);
        }
    }

    if let Some(textures_delta) = textures_delta
        && !textures_delta.free.is_empty()
        && let Some(egui_renderer) = &mut renderer.egui_renderer
    {
        egui_renderer.free_textures(&textures_delta.free)?;
    }

    Ok(())
}

fn cleanup_swapchain(renderer: &mut Renderer) {
    unsafe {
        let _ = renderer.device.device_wait_idle();

        renderer
            .device
            .free_command_buffers(renderer.command_pool, &renderer.command_buffers);
        renderer.command_buffers.clear();

        renderer.device.destroy_pipeline(renderer.pipeline, None);
        renderer
            .device
            .destroy_pipeline_layout(renderer.pipeline_layout, None);

        renderer
            .swapchain
            .image_views
            .iter()
            .for_each(|v| renderer.device.destroy_image_view(*v, None));
        renderer.swapchain.image_views.clear();
        renderer
            .swapchain
            .swapchain
            .destroy_swapchain(renderer.swapchain.swapchain_khr, None);
    }
}

fn generate_checkerboard_texture(square_size: u32) -> Vec<u8> {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    const CHANNELS: u32 = 4;

    let mut data = vec![0u8; (WIDTH * HEIGHT * CHANNELS) as usize];

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let checker_x = x / square_size;
            let checker_y = y / square_size;
            let is_white = (checker_x + checker_y).is_multiple_of(2);

            let color = if is_white { 255u8 } else { 0u8 };
            let index = ((y * WIDTH + x) * CHANNELS) as usize;
            data[index] = color;
            data[index + 1] = color;
            data[index + 2] = color;
            data[index + 3] = 255;
        }
    }

    data
}

fn generate_test_textures() -> Vec<(String, u32, u32, Vec<u8>)> {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    const CHANNELS: u32 = 4;
    let size = (WIDTH * HEIGHT * CHANNELS) as usize;

    let mut textures = Vec::new();

    textures.push((
        "checkerboard_8x8".to_string(),
        WIDTH,
        HEIGHT,
        generate_checkerboard_texture(8),
    ));

    textures.push((
        "checkerboard_16x16".to_string(),
        WIDTH,
        HEIGHT,
        generate_checkerboard_texture(16),
    ));

    let mut gradient_horizontal = vec![0u8; size];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = ((x as f32 / WIDTH as f32) * 255.0) as u8;
            let index = ((y * WIDTH + x) * CHANNELS) as usize;
            gradient_horizontal[index] = value;
            gradient_horizontal[index + 1] = value;
            gradient_horizontal[index + 2] = value;
            gradient_horizontal[index + 3] = 255;
        }
    }
    textures.push((
        "gradient_horizontal".to_string(),
        WIDTH,
        HEIGHT,
        gradient_horizontal,
    ));

    let mut gradient_vertical = vec![0u8; size];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = ((y as f32 / HEIGHT as f32) * 255.0) as u8;
            let index = ((y * WIDTH + x) * CHANNELS) as usize;
            gradient_vertical[index] = value;
            gradient_vertical[index + 1] = value;
            gradient_vertical[index + 2] = value;
            gradient_vertical[index + 3] = 255;
        }
    }
    textures.push((
        "gradient_vertical".to_string(),
        WIDTH,
        HEIGHT,
        gradient_vertical,
    ));

    let mut red_border = vec![0u8; size];
    const BORDER: u32 = 16;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let is_border =
                !(BORDER..WIDTH - BORDER).contains(&x) || !(BORDER..HEIGHT - BORDER).contains(&y);
            let index = ((y * WIDTH + x) * CHANNELS) as usize;
            if is_border {
                red_border[index] = 255;
                red_border[index + 1] = 0;
                red_border[index + 2] = 0;
                red_border[index + 3] = 255;
            } else {
                red_border[index] = 255;
                red_border[index + 1] = 255;
                red_border[index + 2] = 255;
                red_border[index + 3] = 255;
            }
        }
    }
    textures.push(("red_border".to_string(), WIDTH, HEIGHT, red_border));

    textures
}

impl Renderer {
    fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                usage
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.as_ref().unwrap().lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };

        let binding_array_index = self.buffers.len() as u32;

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(size);

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor_set)
            .dst_binding(0)
            .dst_array_element(binding_array_index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe { self.device.update_descriptor_sets(&[descriptor_write], &[]) };

        self.buffers.push(Buffer {
            buffer,
            allocation: Some(allocation),
            _binding_array_index: binding_array_index,
            size,
        });

        Ok(())
    }

    fn upload_buffer_data<T>(
        &mut self,
        buffer_index: usize,
        data: &[T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(buffer) = self.buffers.get(buffer_index) {
            if let Some(allocation) = &buffer.allocation {
                unsafe {
                    let mapped_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr, data.len());
                }
                Ok(())
            } else {
                Err("Buffer has no allocation".into())
            }
        } else {
            Err("Buffer index out of bounds".into())
        }
    }

    fn upload_texture_async(
        &mut self,
        name: String,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self._transfer_queue.is_none() || self.transfer_command_buffer_pool.is_none() {
            let textures = vec![(name.clone(), width, height, data)];
            self.batch_upload_textures(textures)?;
            return Ok(self.textures.len() - 1);
        }

        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { self.device.create_image(&image_info, None)? };
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let allocation = self.allocator.as_ref().unwrap().lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "texture_async",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?
        };

        self.transfer_timeline_counter += 1;
        let signal_value = self.transfer_timeline_counter;

        let current_value = unsafe {
            self.device
                .get_semaphore_counter_value(self.transfer_timeline_semaphore)?
        };
        self.ring_staging
            .as_mut()
            .unwrap()
            .free_completed(current_value);

        let data_size = (width * height * 4) as u64;
        let (staging_offset, staging_ptr) = self
            .ring_staging
            .as_mut()
            .unwrap()
            .allocate(data_size, signal_value)?;

        if log::log_enabled!(log::Level::Debug) {
            let (used, avail, pending) = self.ring_staging.as_ref().unwrap().get_usage_stats();
            log::debug!(
                "Ring staging: used={} MB, available={} MB, pending ops={}",
                used / (1024 * 1024),
                avail / (1024 * 1024),
                pending
            );
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), staging_ptr, data.len());
        }

        let staging_buffer = self.ring_staging.as_ref().unwrap().get_buffer();

        let cmd = self
            .transfer_command_buffer_pool
            .as_mut()
            .unwrap()
            .acquire()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info)? };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let dep =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

        let copy = vk::BufferImageCopy::default()
            .buffer_offset(staging_offset)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }

        let release = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(self._transfer_queue_family_index.unwrap())
            .dst_queue_family_index(self.graphics_queue_family_index)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let dep =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
        unsafe { self.device.end_command_buffer(cmd)? };

        let cmd_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let signal_semaphore_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline_semaphore)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_info));

        unsafe {
            self.device
                .queue_submit2(self._transfer_queue.unwrap(), std::slice::from_ref(&submit_info), vk::Fence::null())?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(mip_levels)
                    .layer_count(1),
            );
        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .max_lod(mip_levels as f32);
        let sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };

        let binding_idx = self.textures.len() as u32;
        let desc_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor_set)
            .dst_binding(1)
            .dst_array_element(binding_idx)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&desc_info));
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        let texture_idx = self.textures.len();
        self.textures.push(Texture {
            image,
            view,
            sampler,
            allocation: Some(allocation),
            _binding_array_index: binding_idx,
            width,
            height,
            mip_levels,
            state: TextureState::Uploading {
                completion_value: signal_value,
            },
        });
        self.texture_names.push(name.clone());

        self.frame_graph.register_image(
            &format!("texture_{}", texture_idx),
            image,
            mip_levels,
            vk::ImageLayout::UNDEFINED,
            self._transfer_queue_family_index.unwrap(),
        )?;

        Ok(texture_idx)
    }

    fn process_pending_mipmaps(
        &mut self,
        cmd: vk::CommandBuffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let current_value = unsafe {
            self.device
                .get_semaphore_counter_value(self.transfer_timeline_semaphore)?
        };

        let mut completed_indices = Vec::new();
        for (idx, texture) in self.textures.iter().enumerate() {
            if let TextureState::Uploading { completion_value } = texture.state {
                if current_value >= completion_value {
                    completed_indices.push(idx);
                }
            }
        }

        if completed_indices.is_empty() {
            return Ok(());
        }

        for idx in completed_indices {
            let texture = &self.textures[idx];
            let image = texture.image;
            let mip_levels = texture.mip_levels;
            let width = texture.width;
            let height = texture.height;

            let acquire_mip0 = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(self._transfer_queue_family_index.unwrap())
                .dst_queue_family_index(self.graphics_queue_family_index)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .layer_count(1),
                );

            let barriers = if mip_levels > 1 {
                let init_remaining_mips = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(1)
                            .level_count(mip_levels - 1)
                            .layer_count(1),
                    );
                vec![acquire_mip0, init_remaining_mips]
            } else {
                vec![acquire_mip0]
            };

            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

            let base_mip_transition = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .layer_count(1),
                );

            let dep = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&base_mip_transition));
            unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

            let mut mip_w = width;
            let mut mip_h = height;

            for mip in 1..mip_levels {
                let next_w = if mip_w > 1 { mip_w / 2 } else { 1 };
                let next_h = if mip_h > 1 { mip_h / 2 } else { 1 };

                let blit = vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip - 1)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_w as i32,
                            y: mip_h as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_w as i32,
                            y: next_h as i32,
                            z: 1,
                        },
                    ]);

                unsafe {
                    self.device.cmd_blit_image(
                        cmd,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit],
                        vk::Filter::LINEAR,
                    );
                }

                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(mip - 1)
                            .level_count(1)
                            .layer_count(1),
                    );

                let dep = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));
                unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

                if mip < mip_levels - 1 {
                    let barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(mip)
                                .level_count(1)
                                .layer_count(1),
                        );

                    let dep = vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&barrier));
                    unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
                }

                mip_w = next_w;
                mip_h = next_h;
            }

            let final_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip_levels - 1)
                        .level_count(1)
                        .layer_count(1),
                );

            let dep = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&final_barrier));
            unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

            self.textures[idx].state = TextureState::Ready;

            self.frame_graph.register_image(
                &format!("texture_{}", idx),
                image,
                mip_levels,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                self.graphics_queue_family_index,
            )?;
        }

        Ok(())
    }

    fn batch_upload_textures(
        &mut self,
        textures: Vec<(String, u32, u32, Vec<u8>)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if textures.is_empty() {
            return Ok(());
        }

        let calculate_mip_levels = |width: u32, height: u32| -> u32 {
            (width.max(height) as f32).log2().floor() as u32 + 1
        };

        let properties = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let alignment = properties.limits.optimal_buffer_copy_offset_alignment;

        let mut texture_info: Vec<(String, u32, u32, u32, u64, Vec<u8>)> = vec![];
        let mut total_size = 0u64;

        for (name, width, height, data) in textures {
            let mip_levels = calculate_mip_levels(width, height);
            let base_size = (width * height * 4) as u64;
            let aligned_offset = (total_size + alignment - 1) & !(alignment - 1);
            texture_info.push((name, width, height, mip_levels, aligned_offset, data));
            total_size = aligned_offset + base_size;
        }

        let staging = self.staging_buffer.as_mut().unwrap();
        staging.ensure_capacity(&self.device, &self.allocator.as_ref().unwrap(), total_size)?;

        unsafe {
            let mapped_ptr = staging.get_mapped_ptr();
            for (_, _, _, _, offset, data) in &texture_info {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    mapped_ptr.add(*offset as usize),
                    data.len(),
                );
            }
        }

        let staging_buffer = staging.buffer;

        let has_mipmaps = texture_info
            .iter()
            .any(|(_, _, _, mip_levels, _, _)| *mip_levels > 1);

        let use_transfer_queue = self.transfer_command_pool.is_some()
            && self._transfer_queue.is_some()
            && self._transfer_queue_family_index != Some(self.graphics_queue_family_index)
            && !has_mipmaps;

        let command_pool = if use_transfer_queue {
            self.transfer_command_pool.unwrap()
        } else {
            self.command_pool
        };

        let queue_family_index = if use_transfer_queue {
            self._transfer_queue_family_index.unwrap()
        } else {
            self.graphics_queue_family_index
        };

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer =
            unsafe { self.device.allocate_command_buffers(&command_buffer_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        let mut created_images: Vec<(vk::Image, u32)> = vec![];

        for (name, width, height, mip_levels, staging_offset, _) in &texture_info {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: *width,
                    height: *height,
                    depth: 1,
                })
                .mip_levels(*mip_levels)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_UNORM)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1);

            let image = unsafe { self.device.create_image(&image_info, None)? };
            created_images.push((image, *mip_levels));
            let requirements = unsafe { self.device.get_image_memory_requirements(image) };

            let allocation = self.allocator.as_ref().unwrap().lock().unwrap().allocate(
                &gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "texture_image",
                    requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                },
            )?;

            unsafe {
                self.device
                    .bind_image_memory(image, allocation.memory(), allocation.offset())?
            };

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(*mip_levels)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

            unsafe {
                self.device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info)
            };

            let buffer_image_copy = vk::BufferImageCopy::default()
                .buffer_offset(*staging_offset)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: *width,
                    height: *height,
                    depth: 1,
                });

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_image_copy],
                )
            };

            let mut mip_width = *width;
            let mut mip_height = *height;

            for mip_level in 1..*mip_levels {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(mip_level - 1)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                unsafe {
                    self.device
                        .cmd_pipeline_barrier2(command_buffer, &dependency_info)
                };

                let next_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
                let next_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

                let blit = vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip_level - 1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip_level)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_mip_width as i32,
                            y: next_mip_height as i32,
                            z: 1,
                        },
                    ]);

                unsafe {
                    self.device.cmd_blit_image(
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit],
                        vk::Filter::LINEAR,
                    )
                };

                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(mip_level - 1)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                unsafe {
                    self.device
                        .cmd_pipeline_barrier2(command_buffer, &dependency_info)
                };

                mip_width = next_mip_width;
                mip_height = next_mip_height;
            }

            let last_mip_barrier = if use_transfer_queue {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(queue_family_index)
                    .dst_queue_family_index(self.graphics_queue_family_index)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(*mip_levels)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
            } else {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(*mip_levels - 1)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
            };

            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&last_mip_barrier));

            unsafe {
                self.device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info)
            };

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(*mip_levels)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let view = unsafe { self.device.create_image_view(&view_info, None)? };

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(false)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(*mip_levels as f32);

            let sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };

            let binding_array_index = self.textures.len() as u32;

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(self.bindless_descriptor_set)
                .dst_binding(1)
                .dst_array_element(binding_array_index)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            unsafe { self.device.update_descriptor_sets(&[descriptor_write], &[]) };

            self.textures.push(Texture {
                image,
                view,
                sampler,
                allocation: Some(allocation),
                _binding_array_index: binding_array_index,
                width: *width,
                height: *height,
                mip_levels: *mip_levels,
                state: TextureState::Ready,
            });

            self.texture_names.push(name.clone());
        }

        unsafe { self.device.end_command_buffer(command_buffer)? };

        self.transfer_timeline_counter += 1;
        let signal_value = self.transfer_timeline_counter;

        let cmd_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
        let signal_semaphore_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline_semaphore)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_info));

        let queue = if use_transfer_queue {
            self._transfer_queue.unwrap()
        } else {
            self.graphics_queue
        };

        unsafe {
            self.device
                .queue_submit2(queue, std::slice::from_ref(&submit_info), vk::Fence::null())?
        };

        log::info!(
            "Batch uploaded {} textures with mipmaps using timeline semaphore (signal: {})",
            texture_info.len(),
            signal_value
        );

        let wait_semaphores = [self.transfer_timeline_semaphore];
        let wait_values = [signal_value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&wait_semaphores)
            .values(&wait_values);

        unsafe {
            self.device.wait_semaphores(&wait_info, u64::MAX)?;
        }

        if use_transfer_queue {
            log::info!(
                "Performing queue family ownership transfer for {} textures from transfer queue to graphics queue",
                created_images.len()
            );

            let graphics_cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let graphics_cmd_buffer =
                unsafe { self.device.allocate_command_buffers(&graphics_cmd_info)?[0] };

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.device
                    .begin_command_buffer(graphics_cmd_buffer, &begin_info)?
            };

            for (image, mip_levels) in &created_images {
                let acquire_barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self._transfer_queue_family_index.unwrap())
                    .dst_queue_family_index(self.graphics_queue_family_index)
                    .image(*image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(*mip_levels)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&acquire_barrier));

                unsafe {
                    self.device
                        .cmd_pipeline_barrier2(graphics_cmd_buffer, &dependency_info)
                };
            }

            unsafe { self.device.end_command_buffer(graphics_cmd_buffer)? };

            let cmd_buffer_infos =
                [vk::CommandBufferSubmitInfo::default().command_buffer(graphics_cmd_buffer)];
            let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&cmd_buffer_infos);

            unsafe {
                self.device.queue_submit2(
                    self.graphics_queue,
                    &[submit_info],
                    vk::Fence::null(),
                )?;
                self.device.queue_wait_idle(self.graphics_queue)?;
                self.device
                    .free_command_buffers(self.command_pool, &[graphics_cmd_buffer]);
            }
        }

        Ok(())
    }
}
