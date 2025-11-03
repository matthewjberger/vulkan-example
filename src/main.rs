use ash::{ext::debug_utils, khr::swapchain, vk};
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
        egui::Window::new("Hello World").show(&egui_ctx, |ui| {
            ui.heading("Hello World");
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

struct Buffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    _binding_array_index: u32,
}

struct Texture {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    _binding_array_index: u32,
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
    pub swapchain: Swapchain,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub command_buffers: Vec<vk::CommandBuffer>,

    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
    pub current_frame: usize,
    pub frames_in_flight: usize,

    pub is_swapchain_dirty: bool,

    pub egui_renderer: Option<egui_ash_renderer::Renderer>,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
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
                    let _ = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation);
                }
            }

            for buffer in &mut self.buffers {
                self.device.destroy_buffer(buffer.buffer, None);
                if let Some(allocation) = buffer.allocation.take() {
                    let _ = self
                        .allocator
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .free(allocation);
                }
            }

            self.device
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless_set_layout, None);

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

    let bindless_set_layout = create_bindless_descriptor_layout(&device)?;
    log::info!("Created bindless descriptor layout");

    let bindless_pool = create_bindless_descriptor_pool(&device)?;
    log::info!("Created bindless descriptor pool");

    let bindless_descriptor_set =
        allocate_bindless_descriptor_set(&device, bindless_pool, bindless_set_layout)?;
    log::info!("Allocated bindless descriptor set");

    let (pipeline_layout, pipeline) = create_pipeline(&device, &swapchain, bindless_set_layout)?;
    log::info!("Created render pipeline and layout");

    const FRAMES_IN_FLIGHT: usize = 3;

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
        swapchain,
        pipeline,
        pipeline_layout,
        command_buffers: vec![],
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,
        current_frame: 0,
        frames_in_flight: FRAMES_IN_FLIGHT,
        is_swapchain_dirty: false,
        egui_renderer: Some(egui_renderer),
    };
    renderer.command_buffers = create_and_record_command_buffers(&renderer)?;
    log::info!("Created and recorded command buffers");

    let checkerboard_data = generate_checkerboard_texture(8);
    renderer.create_texture(256, 256, &checkerboard_data)?;
    log::info!("Created test checkerboard texture");

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

    Ok(renderer)
}

fn create_pipeline(
    device: &ash::Device,
    swapchain: &Swapchain,
    bindless_set_layout: vk::DescriptorSetLayout,
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
        .descriptor_binding_sampled_image_update_after_bind(true);
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

fn create_and_record_command_buffers(
    renderer: &Renderer,
) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
    let device = &renderer.device;
    let pool = renderer.command_pool;
    let count = renderer.swapchain.images.len();

    log::info!("Creating and recording command buffers");
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
        create_pipeline(&renderer.device, &swapchain, renderer.bindless_set_layout)?;
    log::info!("Recreated render pipeline and layout");

    renderer.swapchain = swapchain;
    renderer.pipeline = pipeline;
    renderer.pipeline_layout = pipeline_layout;

    let command_buffers = create_and_record_command_buffers(renderer)?;
    renderer.command_buffers = command_buffers;
    log::info!("Recreated and recorded command buffers");

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

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    unsafe {
        renderer
            .device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?
    };

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
            .cmd_pipeline_barrier2(command_buffer, &dependency_info)
    };

    let color_attachment_info = vk::RenderingAttachmentInfo::default()
        .image_view(renderer.swapchain.image_views[image_index as usize])
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
            extent: renderer.swapchain.extent,
        })
        .layer_count(1)
        .color_attachments(std::slice::from_ref(&color_attachment_info));

    unsafe {
        renderer
            .device
            .cmd_begin_rendering(command_buffer, &rendering_info)
    };

    unsafe {
        renderer.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            renderer.pipeline,
        );

        renderer.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            renderer.pipeline_layout,
            0,
            &[renderer.bindless_descriptor_set],
            &[],
        );

        if let Some(vertex_buffer) = renderer.buffers.first() {
            renderer.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[vertex_buffer.buffer],
                &[0],
            );
        }

        let texture_index: u32 = 0;
        renderer.device.cmd_push_constants(
            command_buffer,
            renderer.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            &texture_index.to_ne_bytes(),
        );

        renderer.device.cmd_draw(command_buffer, 6, 1, 0, 0);
    };

    if let Some(primitives) = clipped_primitives
        && let Some(egui_renderer) = &mut renderer.egui_renderer
    {
        egui_renderer.cmd_draw(
            command_buffer,
            renderer.swapchain.extent,
            pixels_per_point,
            &primitives,
        )?;
    }

    unsafe { renderer.device.cmd_end_rendering(command_buffer) };

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
            .cmd_pipeline_barrier2(command_buffer, &dependency_info)
    };

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
        });

        Ok(())
    }

    fn create_texture(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { self.device.create_image(&image_info, None)? };
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let allocation = self.allocator.as_ref().unwrap().lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "texture",
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

        let staging_buffer_info = vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { self.device.create_buffer(&staging_buffer_info, None)? };
        let staging_requirements =
            unsafe { self.device.get_buffer_memory_requirements(staging_buffer) };

        let staging_allocation = self.allocator.as_ref().unwrap().lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "staging_buffer",
                requirements: staging_requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        unsafe {
            self.device.bind_buffer_memory(
                staging_buffer,
                staging_allocation.memory(),
                staging_allocation.offset(),
            )?
        };

        unsafe {
            let mapped_ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr, data.len());
        }

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
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

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            self.device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            self.device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };

        unsafe { self.device.end_command_buffer(command_buffer)? };

        let command_buffer_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_infos);

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        unsafe {
            self.device
                .queue_submit2(self.graphics_queue, &[submit_info], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
        };

        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
            self.device.destroy_buffer(staging_buffer, None);
        }

        self.allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free(staging_allocation)?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

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
}
