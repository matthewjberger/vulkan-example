use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum RenderGraphError {
    #[error("Invalid resource {id:?}: {reason}")]
    InvalidResourceId { id: ResourceId, reason: String },

    #[error("Stale resource {id:?}: expected generation {expected_gen} but got {got_gen}")]
    StaleResourceId {
        id: ResourceId,
        expected_gen: u32,
        got_gen: u32,
    },

    #[error("Pass {pass_index} requires {queue_type:?} queue but none available")]
    MissingQueue {
        queue_type: QueueType,
        pass_index: usize,
    },

    #[error("Memory budget exceeded: {current}MB + {requested}MB > {budget}MB")]
    BudgetExceeded {
        current: u64,
        requested: u64,
        budget: u64,
    },

    #[error("Allocation failed for {size} bytes: {reason}")]
    AllocationFailed { size: u64, reason: String },

    #[error("Dependency cycle detected: {message}")]
    DependencyCycle {
        pass_indices: Vec<usize>,
        message: String,
    },

    #[error("Empty render graph")]
    EmptyGraph,

    #[error("{0}")]
    Other(String),

    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),

    #[error("GPU allocator error: {0}")]
    GpuAllocatorError(#[from] gpu_allocator::AllocationError),
}

impl From<&str> for RenderGraphError {
    fn from(s: &str) -> Self {
        RenderGraphError::Other(s.to_string())
    }
}

impl From<String> for RenderGraphError {
    fn from(s: String) -> Self {
        RenderGraphError::Other(s)
    }
}

impl From<Box<dyn std::error::Error>> for RenderGraphError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        RenderGraphError::Other(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RenderGraphError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId {
    index: u32,
    generation: u32,
}

impl ResourceId {
    fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(u32);

impl PassId {
    fn new(id: u32) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
    Transfer,
    Compute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct ImportedImageDesc {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub initial_layout: vk::ImageLayout,
    pub initial_queue_family: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDesc {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub auto_generate_mipmaps: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: gpu_allocator::MemoryLocation,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ResourceType {
    Imported {
        handle: ResourceHandle,
        initial_layout: vk::ImageLayout,
        initial_queue_family: u32,
    },
    TransientImage {
        desc: ImageDesc,
    },
    TransientBuffer {
        desc: BufferDesc,
    },
    TemporalImage {
        desc: ImageDesc,
        history_length: usize,
    },
    TemporalBuffer {
        desc: BufferDesc,
        history_length: usize,
    },
}

#[derive(Debug, Clone)]
pub enum ResourceHandle {
    Image {
        image: vk::Image,
        view: vk::ImageView,
        extent: vk::Extent3D,
        format: vk::Format,
        mip_levels: u32,
        array_layers: u32,
    },
    Buffer {
        buffer: vk::Buffer,
        size: u64,
    },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ResourceInfo {
    resource_type: ResourceType,
    generation: u32,
    temporal_parent: Option<ResourceId>,
    temporal_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,
    pub access_mask: vk::AccessFlags2,
    pub stage_mask: vk::PipelineStageFlags2,
    pub queue_family: u32,
}

impl Default for ResourceState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access_mask: vk::AccessFlags2::NONE,
            stage_mask: vk::PipelineStageFlags2::NONE,
            queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceAccess {
    pub resource_id: ResourceId,
    pub access_type: AccessType,
    pub desired_layout: vk::ImageLayout,
    pub access_mask: vk::AccessFlags2,
    pub stage_mask: vk::PipelineStageFlags2,
}

pub struct PassContext<'a> {
    pub cmd: vk::CommandBuffer,
    pub device: &'a ash::Device,
    resources: &'a HashMap<ResourceId, ResourceHandle>,
}

impl<'a> PassContext<'a> {
    pub fn get_image(&self, id: ResourceId) -> Result<(vk::Image, vk::ImageView)> {
        match self.resources.get(&id) {
            Some(ResourceHandle::Image { image, view, .. }) => Ok((*image, *view)),
            Some(ResourceHandle::Buffer { .. }) => Err(RenderGraphError::InvalidResourceId {
                id,
                reason: "Resource is a buffer, not an image".to_string(),
            }),
            None => Err(RenderGraphError::InvalidResourceId {
                id,
                reason: "Resource not found in pass context".to_string(),
            }),
        }
    }

    pub fn get_buffer(&self, id: ResourceId) -> Result<vk::Buffer> {
        match self.resources.get(&id) {
            Some(ResourceHandle::Buffer { buffer, .. }) => Ok(*buffer),
            Some(ResourceHandle::Image { .. }) => Err(RenderGraphError::InvalidResourceId {
                id,
                reason: "Resource is an image, not a buffer".to_string(),
            }),
            None => Err(RenderGraphError::InvalidResourceId {
                id,
                reason: "Resource not found in pass context".to_string(),
            }),
        }
    }
}

pub type PassExecuteFn = Box<dyn FnOnce(PassContext) -> Result<()>>;

pub struct PassDesc {
    pub queue_type: QueueType,
    pub accesses: Vec<ResourceAccess>,
    pub execute: Option<PassExecuteFn>,
    pub enabled: bool,
}

struct PooledImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: gpu_allocator::vulkan::Allocation,
    in_use: bool,
    last_used_frame: u64,
}

struct PooledBuffer {
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    in_use: bool,
    last_used_frame: u64,
}

struct SizeClassKey {
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mip_levels: u32,
    array_layers: u32,
}

impl std::hash::Hash for SizeClassKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extent.width.hash(state);
        self.extent.height.hash(state);
        self.extent.depth.hash(state);
        (self.format.as_raw()).hash(state);
        self.usage.as_raw().hash(state);
        self.mip_levels.hash(state);
        self.array_layers.hash(state);
    }
}

impl PartialEq for SizeClassKey {
    fn eq(&self, other: &Self) -> bool {
        self.extent == other.extent
            && self.format == other.format
            && self.usage == other.usage
            && self.mip_levels == other.mip_levels
            && self.array_layers == other.array_layers
    }
}

impl Eq for SizeClassKey {}

fn aspect_mask_from_format(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT | vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D32_SFLOAT_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D16_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn sanitize_stage_mask_for_queue(
    stage_mask: vk::PipelineStageFlags2,
    queue_family: u32,
    graphics_queue_family: u32,
) -> vk::PipelineStageFlags2 {
    if queue_family == graphics_queue_family {
        return stage_mask;
    }

    const GRAPHICS_ONLY_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::DRAW_INDIRECT.as_raw()
            | vk::PipelineStageFlags2::VERTEX_INPUT.as_raw()
            | vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
            | vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER.as_raw()
            | vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER.as_raw()
            | vk::PipelineStageFlags2::GEOMETRY_SHADER.as_raw()
            | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
            | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
    );

    if stage_mask.intersects(GRAPHICS_ONLY_STAGES) {
        vk::PipelineStageFlags2::ALL_COMMANDS
    } else {
        stage_mask
    }
}

#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    pub total_images: usize,
    pub images_in_use: usize,
    pub total_memory_bytes: u64,
    pub memory_in_use_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

struct BufferSizeClassKey {
    size: u64,
    usage: vk::BufferUsageFlags,
    memory_location: gpu_allocator::MemoryLocation,
}

impl std::hash::Hash for BufferSizeClassKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.size.hash(state);
        self.usage.as_raw().hash(state);
        (self.memory_location as u32).hash(state);
    }
}

impl PartialEq for BufferSizeClassKey {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.usage == other.usage
            && self.memory_location == other.memory_location
    }
}

impl Eq for BufferSizeClassKey {}

pub struct TransientResourcePool {
    images_by_class: HashMap<SizeClassKey, Vec<PooledImage>>,
    buffers_by_class: HashMap<BufferSizeClassKey, Vec<PooledBuffer>>,
    allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    device: ash::Device,
    current_frame: u64,
    frames_in_flight: u64,
    total_allocated_bytes: u64,
    budget_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl TransientResourcePool {
    pub fn new(
        device: ash::Device,
        allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        frames_in_flight: u64,
        budget_bytes: u64,
    ) -> Self {
        Self {
            images_by_class: HashMap::new(),
            buffers_by_class: HashMap::new(),
            allocator,
            device,
            current_frame: 0,
            frames_in_flight,
            total_allocated_bytes: 0,
            budget_bytes,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn acquire_image(&mut self, desc: &ImageDesc) -> Result<(vk::Image, vk::ImageView)> {
        let safe_frame = self.current_frame.saturating_sub(self.frames_in_flight);
        let size_class = get_size_class(desc.extent);

        let mut pooled_desc = *desc;
        pooled_desc.extent = size_class;

        let key = SizeClassKey {
            extent: size_class,
            format: desc.format,
            usage: desc.usage,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
        };

        if let Some(pool) = self.images_by_class.get_mut(&key) {
            for pooled in pool.iter_mut() {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    pooled.in_use = true;
                    pooled.last_used_frame = self.current_frame;
                    self.cache_hits += 1;
                    return Ok((pooled.image, pooled.view));
                }
            }
        }

        self.cache_misses += 1;

        let estimated_size = estimate_image_size(&pooled_desc);

        if self.total_allocated_bytes + estimated_size > self.budget_bytes {
            self.cleanup_unused();

            if self.total_allocated_bytes + estimated_size > self.budget_bytes {
                return Err(RenderGraphError::BudgetExceeded {
                    current: self.total_allocated_bytes / 1_000_000,
                    requested: estimated_size / 1_000_000,
                    budget: self.budget_bytes / 1_000_000,
                });
            }
        }

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(size_class)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .format(desc.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { self.device.create_image(&image_create_info, None)? };

        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let allocation = self.allocator.lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "transient_image",
                requirements: mem_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(image),
            },
        )?;

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        self.total_allocated_bytes += allocation.size();

        let aspect_mask = aspect_mask_from_format(desc.format);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(desc.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(desc.mip_levels)
                    .base_array_layer(0)
                    .layer_count(desc.array_layers),
            );

        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        let pooled = PooledImage {
            image,
            view,
            allocation,
            in_use: true,
            last_used_frame: self.current_frame,
        };

        self.images_by_class.entry(key).or_default().push(pooled);

        Ok((image, view))
    }

    pub fn acquire_buffer(&mut self, desc: &BufferDesc) -> Result<vk::Buffer> {
        let safe_frame = self.current_frame.saturating_sub(self.frames_in_flight);
        let size_class = get_buffer_size_class(desc.size);

        let key = BufferSizeClassKey {
            size: size_class,
            usage: desc.usage,
            memory_location: desc.memory_location,
        };

        if let Some(pool) = self.buffers_by_class.get_mut(&key) {
            for pooled in pool.iter_mut() {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    pooled.in_use = true;
                    pooled.last_used_frame = self.current_frame;
                    self.cache_hits += 1;
                    return Ok(pooled.buffer);
                }
            }
        }

        self.cache_misses += 1;

        let estimated_size = size_class;

        if self.total_allocated_bytes + estimated_size > self.budget_bytes {
            self.cleanup_unused();

            if self.total_allocated_bytes + estimated_size > self.budget_bytes {
                return Err(RenderGraphError::BudgetExceeded {
                    current: self.total_allocated_bytes / 1_000_000,
                    requested: estimated_size / 1_000_000,
                    budget: self.budget_bytes / 1_000_000,
                });
            }
        }

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size_class)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None)? };

        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.lock().unwrap().allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "transient_buffer",
                requirements: mem_requirements,
                location: desc.memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            },
        )?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        self.total_allocated_bytes += allocation.size();

        let pooled = PooledBuffer {
            buffer,
            allocation,
            in_use: true,
            last_used_frame: self.current_frame,
        };

        self.buffers_by_class.entry(key).or_default().push(pooled);

        Ok(buffer)
    }

    fn cleanup_unused(&mut self) {
        let safe_frame = self.current_frame.saturating_sub(self.frames_in_flight);

        for pool in self.images_by_class.values_mut() {
            let mut indices_to_remove = Vec::new();

            for (index, pooled) in pool.iter().enumerate() {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    indices_to_remove.push(index);
                }
            }

            for index in indices_to_remove.iter().rev() {
                let pooled = pool.remove(*index);
                unsafe {
                    self.device.destroy_image_view(pooled.view, None);
                    self.device.destroy_image(pooled.image, None);
                }
                self.total_allocated_bytes = self
                    .total_allocated_bytes
                    .saturating_sub(pooled.allocation.size());
                self.allocator.lock().unwrap().free(pooled.allocation).ok();
            }
        }

        for pool in self.buffers_by_class.values_mut() {
            let mut indices_to_remove = Vec::new();

            for (index, pooled) in pool.iter().enumerate() {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    indices_to_remove.push(index);
                }
            }

            for index in indices_to_remove.iter().rev() {
                let pooled = pool.remove(*index);
                unsafe {
                    self.device.destroy_buffer(pooled.buffer, None);
                }
                self.total_allocated_bytes = self
                    .total_allocated_bytes
                    .saturating_sub(pooled.allocation.size());
                self.allocator.lock().unwrap().free(pooled.allocation).ok();
            }
        }

        self.images_by_class.retain(|_, pool| !pool.is_empty());
        self.buffers_by_class.retain(|_, pool| !pool.is_empty());
    }

    pub fn release_all(&mut self) {
        for pool in self.images_by_class.values_mut() {
            for pooled in pool.iter_mut() {
                pooled.in_use = false;
            }
        }
        for pool in self.buffers_by_class.values_mut() {
            for pooled in pool.iter_mut() {
                pooled.in_use = false;
            }
        }
        self.current_frame += 1;
    }

    pub fn get_statistics(&self) -> PoolStatistics {
        let mut stats = PoolStatistics::default();

        for pool in self.images_by_class.values() {
            stats.total_images += pool.len();
            for pooled in pool {
                if pooled.in_use {
                    stats.images_in_use += 1;
                    stats.memory_in_use_bytes += pooled.allocation.size();
                }
            }
        }

        stats.total_memory_bytes = self.total_allocated_bytes;
        stats.cache_hits = self.cache_hits;
        stats.cache_misses = self.cache_misses;
        stats
    }

    pub fn defragment(&mut self, max_unused_frames: u64) {
        let cutoff = self.current_frame.saturating_sub(max_unused_frames);

        for pool in self.images_by_class.values_mut() {
            let mut indices_to_remove = Vec::new();

            for (index, pooled) in pool.iter().enumerate() {
                if !pooled.in_use && pooled.last_used_frame < cutoff {
                    indices_to_remove.push(index);
                }
            }

            for index in indices_to_remove.iter().rev() {
                let pooled = pool.remove(*index);
                unsafe {
                    self.device.destroy_image_view(pooled.view, None);
                    self.device.destroy_image(pooled.image, None);
                }
                let size = pooled.allocation.size();
                self.allocator.lock().unwrap().free(pooled.allocation).ok();
                self.total_allocated_bytes = self.total_allocated_bytes.saturating_sub(size);
            }
        }

        self.images_by_class.retain(|_, pool| !pool.is_empty());
    }

    pub fn cleanup(&mut self) {
        for pool in self.images_by_class.values_mut() {
            for pooled in pool.drain(..) {
                unsafe {
                    self.device.destroy_image_view(pooled.view, None);
                    self.device.destroy_image(pooled.image, None);
                }
                self.allocator.lock().unwrap().free(pooled.allocation).ok();
            }
        }
        for pool in self.buffers_by_class.values_mut() {
            for pooled in pool.drain(..) {
                unsafe {
                    self.device.destroy_buffer(pooled.buffer, None);
                }
                self.allocator.lock().unwrap().free(pooled.allocation).ok();
            }
        }
    }

    pub fn destroy(self) {
        drop(self);
    }
}

impl Drop for TransientResourcePool {
    fn drop(&mut self) {
        self.cleanup();
    }
}

pub struct RenderGraph {
    resources: Vec<ResourceInfo>,
    passes: Vec<PassDesc>,
    resource_map: HashMap<String, ResourceId>,
    graphics_queue_family: u32,
    transfer_queue_family: u32,
    compute_queue_family: u32,
    next_generation: u32,
}

impl RenderGraph {
    pub fn new(
        graphics_queue_family: u32,
        transfer_queue_family: u32,
        compute_queue_family: u32,
    ) -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            resource_map: HashMap::new(),
            graphics_queue_family,
            transfer_queue_family,
            compute_queue_family,
            next_generation: 0,
        }
    }

    pub fn import_image(&mut self, name: &str, desc: ImportedImageDesc) -> ResourceId {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        let handle = ResourceHandle::Image {
            image: desc.image,
            view: desc.view,
            extent: desc.extent,
            format: desc.format,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
        };

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::Imported {
                handle,
                initial_layout: desc.initial_layout,
                initial_queue_family: desc.initial_queue_family,
            },
            generation,
            temporal_parent: None,
            temporal_index: 0,
        });

        id
    }

    pub fn create_image(&mut self, name: &str, desc: ImageDesc) -> ResourceId {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::TransientImage { desc },
            generation,
            temporal_parent: None,
            temporal_index: 0,
        });

        id
    }

    pub fn import_buffer(&mut self, name: &str, buffer: vk::Buffer, size: u64) -> ResourceId {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        let handle = ResourceHandle::Buffer { buffer, size };

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::Imported {
                handle,
                initial_layout: vk::ImageLayout::UNDEFINED,
                initial_queue_family: self.graphics_queue_family,
            },
            generation,
            temporal_parent: None,
            temporal_index: 0,
        });

        id
    }

    pub fn create_buffer(&mut self, name: &str, desc: BufferDesc) -> ResourceId {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::TransientBuffer { desc },
            generation,
            temporal_parent: None,
            temporal_index: 0,
        });

        id
    }

    #[allow(dead_code)]
    pub fn create_temporal_image(
        &mut self,
        name: &str,
        desc: ImageDesc,
        history_length: usize,
    ) -> Vec<ResourceId> {
        let mut ids = Vec::new();

        for frame_index in 0..history_length {
            let generation = self.next_generation;
            self.next_generation = self.next_generation.wrapping_add(1);

            let id = ResourceId::new(self.resources.len() as u32, generation);

            if frame_index == 0 {
                self.resource_map.insert(name.to_string(), id);
            } else {
                self.resource_map
                    .insert(format!("{}_history_{}", name, frame_index - 1), id);
            }

            self.resources.push(ResourceInfo {
                resource_type: ResourceType::TemporalImage {
                    desc,
                    history_length,
                },
                generation,
                temporal_parent: if frame_index == 0 { None } else { Some(ids[0]) },
                temporal_index: frame_index,
            });

            ids.push(id);
        }

        ids
    }

    #[allow(dead_code)]
    pub fn create_temporal_buffer(
        &mut self,
        name: &str,
        desc: BufferDesc,
        history_length: usize,
    ) -> Vec<ResourceId> {
        let mut ids = Vec::new();

        for frame_index in 0..history_length {
            let generation = self.next_generation;
            self.next_generation = self.next_generation.wrapping_add(1);

            let id = ResourceId::new(self.resources.len() as u32, generation);

            if frame_index == 0 {
                self.resource_map.insert(name.to_string(), id);
            } else {
                self.resource_map
                    .insert(format!("{}_history_{}", name, frame_index - 1), id);
            }

            self.resources.push(ResourceInfo {
                resource_type: ResourceType::TemporalBuffer {
                    desc,
                    history_length,
                },
                generation,
                temporal_parent: if frame_index == 0 { None } else { Some(ids[0]) },
                temporal_index: frame_index,
            });

            ids.push(id);
        }

        ids
    }

    #[allow(dead_code)]
    pub fn get_previous_frame(&self, current: ResourceId, frames_ago: usize) -> Option<ResourceId> {
        let resource_index = current.index as usize;
        if resource_index >= self.resources.len() {
            return None;
        }

        let resource = &self.resources[resource_index];
        let parent_id = resource.temporal_parent.or(Some(current))?;
        let current_index = resource.temporal_index;

        let parent_index = parent_id.index as usize;
        let parent_resource = &self.resources[parent_index];

        let history_length = match &parent_resource.resource_type {
            ResourceType::TemporalImage { history_length, .. } => *history_length,
            ResourceType::TemporalBuffer { history_length, .. } => *history_length,
            _ => return None,
        };

        let target_index = (current_index + frames_ago) % history_length;

        for (idx, res) in self.resources.iter().enumerate() {
            if res.temporal_parent == Some(parent_id) && res.temporal_index == target_index {
                return Some(ResourceId::new(idx as u32, res.generation));
            }
            if idx == parent_index && target_index == 0 {
                return Some(parent_id);
            }
        }

        None
    }

    pub fn get_resource(&self, name: &str) -> Option<ResourceId> {
        self.resource_map.get(name).copied()
    }

    fn validate_resource(&self, id: ResourceId) -> Result<()> {
        if id.index as usize >= self.resources.len() {
            return Err(RenderGraphError::InvalidResourceId {
                id,
                reason: format!("index {} out of bounds", id.index),
            });
        }

        let resource = &self.resources[id.index as usize];
        if resource.generation != id.generation {
            return Err(RenderGraphError::StaleResourceId {
                id,
                expected_gen: resource.generation,
                got_gen: id.generation,
            });
        }

        Ok(())
    }

    pub fn add_pass(&mut self, desc: PassDesc) -> Result<PassId> {
        for access in &desc.accesses {
            self.validate_resource(access.resource_id)?;
        }

        let id = PassId::new(self.passes.len() as u32);
        self.passes.push(desc);
        Ok(id)
    }

    #[allow(dead_code)]
    pub fn enable_pass(&mut self, pass_id: PassId) {
        let pass_index = pass_id.0 as usize;
        if pass_index < self.passes.len() {
            self.passes[pass_index].enabled = true;
        }
    }

    pub fn disable_pass(&mut self, pass_id: PassId) {
        let pass_index = pass_id.0 as usize;
        if pass_index < self.passes.len() {
            self.passes[pass_index].enabled = false;
        }
    }

    #[allow(dead_code)]
    pub fn is_pass_enabled(&self, pass_id: PassId) -> bool {
        let pass_index = pass_id.0 as usize;
        if pass_index < self.passes.len() {
            self.passes[pass_index].enabled
        } else {
            false
        }
    }

    pub fn export_graphviz(&self) -> String {
        let mut dot = String::from("digraph RenderGraph {\n");
        dot.push_str("  rankdir=LR;\n");

        for (idx, _pass) in self.passes.iter().enumerate() {
            dot.push_str(&format!(
                "  p{} [label=\"Pass {}\", shape=box];\n",
                idx, idx
            ));
        }

        for (idx, _resource) in self.resources.iter().enumerate() {
            dot.push_str(&format!(
                "  r{} [label=\"Resource {}\", shape=ellipse];\n",
                idx, idx
            ));
        }

        for (idx, pass) in self.passes.iter().enumerate() {
            for access in &pass.accesses {
                let (from, to, style) = match access.access_type {
                    AccessType::Read => (
                        format!("r{}", access.resource_id.index),
                        format!("p{}", idx),
                        "solid",
                    ),
                    AccessType::Write => (
                        format!("p{}", idx),
                        format!("r{}", access.resource_id.index),
                        "bold",
                    ),
                };
                dot.push_str(&format!("  {} -> {} [style={}];\n", from, to, style));
            }
        }

        dot.push_str("}\n");
        dot
    }

    pub fn compile(self, options: CompileOptions) -> Result<CompiledRenderGraph> {
        for (index, pass) in self.passes.iter().enumerate() {
            match pass.queue_type {
                QueueType::Transfer if !options.has_transfer_queue => {
                    return Err(RenderGraphError::MissingQueue {
                        queue_type: QueueType::Transfer,
                        pass_index: index,
                    });
                }
                QueueType::Compute if !options.has_compute_queue => {
                    return Err(RenderGraphError::MissingQueue {
                        queue_type: QueueType::Compute,
                        pass_index: index,
                    });
                }
                _ => {}
            }
        }

        validate_no_cycles(&self.passes, &self.resource_map)?;

        let aliasing_groups = compute_aliasing_groups(&self.resources, &self.passes);

        for group in &aliasing_groups {
            if group.resources.is_empty() {
                return Err(RenderGraphError::EmptyGraph);
            }
            if group.desc.extent.width == 0 || group.desc.extent.height == 0 {
                return Err(RenderGraphError::AllocationFailed {
                    size: 0,
                    reason: "Invalid aliasing group descriptor with zero dimensions".to_string(),
                });
            }
        }

        let resource_to_alias_group = build_alias_map(&aliasing_groups, self.resources.len());

        let compiled = compile_passes(CompilePassesParams {
            passes: self.passes,
            resources: self.resources,
            graphics_queue_family: self.graphics_queue_family,
            transfer_queue_family: self.transfer_queue_family,
            compute_queue_family: self.compute_queue_family,
            graphics_timeline_base: options.graphics_timeline_base,
            transfer_timeline_base: options.transfer_timeline_base,
            compute_timeline_base: options.compute_timeline_base,
            resource_to_alias_group,
        })?;

        if options.print_aliasing_report {
            compiled.print_aliasing_report();
        }

        Ok(compiled)
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    pub graphics_timeline_base: u64,
    pub transfer_timeline_base: u64,
    pub compute_timeline_base: u64,
    pub has_transfer_queue: bool,
    pub has_compute_queue: bool,
    pub print_aliasing_report: bool,
}

#[derive(Debug, Clone)]
struct ResourceLifetime {
    resource_id: ResourceId,
    desc: ImageDesc,
    first_pass: usize,
    last_pass: usize,
}

impl ResourceLifetime {
    fn overlaps(&self, other: &ResourceLifetime) -> bool {
        self.last_pass >= other.first_pass && self.first_pass <= other.last_pass
    }
}

#[derive(Debug, Clone)]
struct AliasingGroup {
    desc: ImageDesc,
    resources: Vec<ResourceId>,
}

struct CompiledResource {
    info: ResourceInfo,
    current_state: ResourceState,
    last_written_pass: Option<PassId>,
    first_use_pass: Option<PassId>,
}

struct CompiledPass {
    desc: PassDesc,
    same_queue_dependencies: Vec<PassId>,
    cross_queue_dependencies: Vec<PassId>,
    barriers_before: Vec<Barrier>,
    barriers_after: Vec<Barrier>,
    queue_family: u32,
    timeline_value: u64,
}

#[derive(Debug, Clone)]
enum Barrier {
    Image {
        image: vk::Image,
        src_stage: vk::PipelineStageFlags2,
        dst_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family: u32,
        dst_queue_family: u32,
        subresource_range: vk::ImageSubresourceRange,
    },
    Buffer {
        buffer: vk::Buffer,
        src_stage: vk::PipelineStageFlags2,
        dst_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
        src_queue_family: u32,
        dst_queue_family: u32,
        offset: u64,
        size: u64,
    },
}

impl PartialEq for Barrier {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Barrier::Image {
                    image: img1,
                    src_stage: src_stage1,
                    dst_stage: dst_stage1,
                    src_access: src_access1,
                    dst_access: dst_access1,
                    old_layout: old_layout1,
                    new_layout: new_layout1,
                    src_queue_family: src_qf1,
                    dst_queue_family: dst_qf1,
                    subresource_range: range1,
                },
                Barrier::Image {
                    image: img2,
                    src_stage: src_stage2,
                    dst_stage: dst_stage2,
                    src_access: src_access2,
                    dst_access: dst_access2,
                    old_layout: old_layout2,
                    new_layout: new_layout2,
                    src_queue_family: src_qf2,
                    dst_queue_family: dst_qf2,
                    subresource_range: range2,
                },
            ) => {
                img1 == img2
                    && src_stage1 == src_stage2
                    && dst_stage1 == dst_stage2
                    && src_access1 == src_access2
                    && dst_access1 == dst_access2
                    && old_layout1 == old_layout2
                    && new_layout1 == new_layout2
                    && src_qf1 == src_qf2
                    && dst_qf1 == dst_qf2
                    && range1.aspect_mask == range2.aspect_mask
                    && range1.base_mip_level == range2.base_mip_level
                    && range1.level_count == range2.level_count
                    && range1.base_array_layer == range2.base_array_layer
                    && range1.layer_count == range2.layer_count
            }
            (
                Barrier::Buffer {
                    buffer: buf1,
                    src_stage: src_stage1,
                    dst_stage: dst_stage1,
                    src_access: src_access1,
                    dst_access: dst_access1,
                    src_queue_family: src_qf1,
                    dst_queue_family: dst_qf1,
                    offset: offset1,
                    size: size1,
                },
                Barrier::Buffer {
                    buffer: buf2,
                    src_stage: src_stage2,
                    dst_stage: dst_stage2,
                    src_access: src_access2,
                    dst_access: dst_access2,
                    src_queue_family: src_qf2,
                    dst_queue_family: dst_qf2,
                    offset: offset2,
                    size: size2,
                },
            ) => {
                buf1 == buf2
                    && src_stage1 == src_stage2
                    && dst_stage1 == dst_stage2
                    && src_access1 == src_access2
                    && dst_access1 == dst_access2
                    && src_qf1 == src_qf2
                    && dst_qf1 == dst_qf2
                    && offset1 == offset2
                    && size1 == size2
            }
            _ => false,
        }
    }
}

struct CompilePassesParams {
    passes: Vec<PassDesc>,
    resources: Vec<ResourceInfo>,
    graphics_queue_family: u32,
    transfer_queue_family: u32,
    compute_queue_family: u32,
    graphics_timeline_base: u64,
    transfer_timeline_base: u64,
    compute_timeline_base: u64,
    resource_to_alias_group: HashMap<ResourceId, usize>,
}

fn compile_passes(params: CompilePassesParams) -> Result<CompiledRenderGraph> {
    let CompilePassesParams {
        mut passes,
        resources,
        graphics_queue_family,
        transfer_queue_family,
        compute_queue_family,
        graphics_timeline_base,
        transfer_timeline_base,
        compute_timeline_base,
        resource_to_alias_group,
    } = params;

    passes.retain(|pass| pass.enabled);

    let used_resources = compute_used_resources(&passes, &resources);
    let used_passes = compute_used_passes(&passes, &used_resources);

    passes = passes
        .into_iter()
        .enumerate()
        .filter(|(index, _)| used_passes.contains(index))
        .map(|(_, pass)| pass)
        .collect();

    let mut compiled_resources: Vec<CompiledResource> = resources
        .into_iter()
        .map(|info| {
            let initial_state = match &info.resource_type {
                ResourceType::Imported {
                    initial_layout,
                    initial_queue_family,
                    ..
                } => ResourceState {
                    layout: *initial_layout,
                    access_mask: vk::AccessFlags2::NONE,
                    stage_mask: vk::PipelineStageFlags2::NONE,
                    queue_family: *initial_queue_family,
                },
                ResourceType::TransientImage { .. }
                | ResourceType::TransientBuffer { .. }
                | ResourceType::TemporalImage { .. }
                | ResourceType::TemporalBuffer { .. } => ResourceState {
                    layout: vk::ImageLayout::UNDEFINED,
                    access_mask: vk::AccessFlags2::NONE,
                    stage_mask: vk::PipelineStageFlags2::NONE,
                    queue_family: vk::QUEUE_FAMILY_IGNORED,
                },
            };

            CompiledResource {
                info,
                current_state: initial_state,
                last_written_pass: None,
                first_use_pass: None,
            }
        })
        .collect();

    let mut compiled_passes: Vec<CompiledPass> = Vec::new();
    let mut pass_order: Vec<PassId> = Vec::new();

    let mut graphics_timeline_counter = graphics_timeline_base;
    let mut transfer_timeline_counter = transfer_timeline_base;
    let mut compute_timeline_counter = compute_timeline_base;

    for (pass_index, pass_desc) in passes.into_iter().enumerate() {
        let pass_id = PassId::new(pass_index as u32);
        let mut same_queue_dependencies = Vec::new();
        let mut cross_queue_dependencies = Vec::new();
        let mut barriers_before = Vec::new();

        let queue_family = match pass_desc.queue_type {
            QueueType::Graphics => graphics_queue_family,
            QueueType::Transfer => transfer_queue_family,
            QueueType::Compute => compute_queue_family,
        };

        let mut barrier_keys: std::collections::HashSet<(
            vk::Image,
            vk::ImageLayout,
            vk::ImageLayout,
            u32,
            u32,
        )> = std::collections::HashSet::new();

        for access in &pass_desc.accesses {
            let resource = &mut compiled_resources[access.resource_id.index as usize];

            if resource.first_use_pass.is_none() {
                resource.first_use_pass = Some(pass_id);
            }

            match &resource.info.resource_type {
                ResourceType::TransientBuffer { .. }
                | ResourceType::Imported {
                    handle: ResourceHandle::Buffer { .. },
                    ..
                } => {
                    let buffer = match &resource.info.resource_type {
                        ResourceType::Imported {
                            handle: ResourceHandle::Buffer { buffer, .. },
                            ..
                        } => *buffer,
                        ResourceType::TransientBuffer { .. } => vk::Buffer::null(),
                        _ => unreachable!(),
                    };

                    let buffer_size = match &resource.info.resource_type {
                        ResourceType::Imported {
                            handle: ResourceHandle::Buffer { size, .. },
                            ..
                        } => *size,
                        ResourceType::TransientBuffer { desc } => desc.size,
                        _ => unreachable!(),
                    };

                    let effective_current_queue =
                        if resource.current_state.queue_family == vk::QUEUE_FAMILY_IGNORED {
                            queue_family
                        } else {
                            resource.current_state.queue_family
                        };

                    let needs_ownership_transfer = effective_current_queue != queue_family;

                    let needs_barrier = needs_ownership_transfer
                        || access.access_type == AccessType::Write
                        || resource
                            .current_state
                            .access_mask
                            .contains(vk::AccessFlags2::MEMORY_WRITE);

                    if needs_ownership_transfer {
                        let prev_pass_id = resource.last_written_pass.or(resource.first_use_pass);
                        if let Some(prev_id) = prev_pass_id {
                            let prev_pass_index = prev_id.0 as usize;
                            if prev_pass_index < compiled_passes.len() {
                                if !cross_queue_dependencies.contains(&prev_id) {
                                    cross_queue_dependencies.push(prev_id);
                                }

                                let release_barrier = Barrier::Buffer {
                                    buffer,
                                    src_stage: resource.current_state.stage_mask,
                                    dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                    src_access: resource.current_state.access_mask,
                                    dst_access: vk::AccessFlags2::NONE,
                                    src_queue_family: effective_current_queue,
                                    dst_queue_family: queue_family,
                                    offset: 0,
                                    size: buffer_size,
                                };
                                compiled_passes[prev_pass_index]
                                    .barriers_after
                                    .push(release_barrier);
                            }
                        }

                        let acquire_barrier = Barrier::Buffer {
                            buffer,
                            src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                            dst_stage: access.stage_mask,
                            src_access: vk::AccessFlags2::NONE,
                            dst_access: access.access_mask,
                            src_queue_family: effective_current_queue,
                            dst_queue_family: queue_family,
                            offset: 0,
                            size: buffer_size,
                        };
                        barriers_before.push(acquire_barrier);
                    } else if needs_barrier {
                        let src_stage =
                            if resource.current_state.stage_mask == vk::PipelineStageFlags2::NONE {
                                vk::PipelineStageFlags2::TOP_OF_PIPE
                            } else {
                                sanitize_stage_mask_for_queue(
                                    resource.current_state.stage_mask,
                                    queue_family,
                                    graphics_queue_family,
                                )
                            };

                        let barrier = Barrier::Buffer {
                            buffer,
                            src_stage,
                            dst_stage: access.stage_mask,
                            src_access: resource.current_state.access_mask,
                            dst_access: access.access_mask,
                            src_queue_family: queue_family,
                            dst_queue_family: queue_family,
                            offset: 0,
                            size: buffer_size,
                        };
                        barriers_before.push(barrier);
                    }

                    if let Some(last_writer) = resource.last_written_pass {
                        let last_writer_queue =
                            compiled_passes[last_writer.0 as usize].queue_family;
                        if last_writer_queue == queue_family {
                            if !same_queue_dependencies.contains(&last_writer) {
                                same_queue_dependencies.push(last_writer);
                            }
                        } else if !cross_queue_dependencies.contains(&last_writer) {
                            cross_queue_dependencies.push(last_writer);
                        }
                    }

                    if access.access_type == AccessType::Write {
                        resource.last_written_pass = Some(pass_id);
                    }

                    let new_queue_family = match &resource.info.resource_type {
                        ResourceType::Imported {
                            initial_queue_family,
                            ..
                        } if *initial_queue_family == vk::QUEUE_FAMILY_IGNORED => {
                            vk::QUEUE_FAMILY_IGNORED
                        }
                        _ => queue_family,
                    };

                    resource.current_state = ResourceState {
                        layout: vk::ImageLayout::UNDEFINED,
                        access_mask: access.access_mask,
                        stage_mask: access.stage_mask,
                        queue_family: new_queue_family,
                    };

                    continue;
                }
                _ => {}
            }

            let (image, _extent, format, mip_levels, array_layers) =
                match &resource.info.resource_type {
                    ResourceType::Imported {
                        handle:
                            ResourceHandle::Image {
                                image,
                                extent,
                                format,
                                mip_levels,
                                array_layers,
                                ..
                            },
                        ..
                    } => (*image, *extent, *format, *mip_levels, *array_layers),
                    ResourceType::TransientImage { desc }
                    | ResourceType::TemporalImage { desc, .. } => (
                        vk::Image::null(),
                        desc.extent,
                        desc.format,
                        desc.mip_levels,
                        desc.array_layers,
                    ),
                    _ => unreachable!(),
                };

            let effective_current_queue =
                if resource.current_state.queue_family == vk::QUEUE_FAMILY_IGNORED {
                    queue_family
                } else {
                    resource.current_state.queue_family
                };

            let needs_ownership_transfer = effective_current_queue != queue_family;

            let needs_layout_transition = resource.current_state.layout != access.desired_layout;

            let needs_barrier = needs_layout_transition
                || needs_ownership_transfer
                || access.access_type == AccessType::Write
                || resource
                    .current_state
                    .access_mask
                    .contains(vk::AccessFlags2::MEMORY_WRITE);

            let barrier_key = (
                image,
                resource.current_state.layout,
                access.desired_layout,
                resource.current_state.queue_family,
                queue_family,
            );
            if barrier_keys.contains(&barrier_key) {
                if access.access_type == AccessType::Write {
                    resource.last_written_pass = Some(pass_id);
                }
                resource.current_state = ResourceState {
                    layout: access.desired_layout,
                    access_mask: access.access_mask,
                    stage_mask: access.stage_mask,
                    queue_family,
                };
                continue;
            }
            barrier_keys.insert(barrier_key);

            if needs_ownership_transfer {
                let prev_pass_id = resource.last_written_pass.or(resource.first_use_pass);
                if let Some(prev_id) = prev_pass_id {
                    let prev_pass_index = prev_id.0 as usize;
                    if prev_pass_index < compiled_passes.len() {
                        if !cross_queue_dependencies.contains(&prev_id) {
                            cross_queue_dependencies.push(prev_id);
                        }

                        let aspect_mask = aspect_mask_from_format(format);

                        let release_barrier = Barrier::Image {
                            image,
                            src_stage: resource.current_state.stage_mask,
                            dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                            src_access: resource.current_state.access_mask,
                            dst_access: vk::AccessFlags2::NONE,
                            old_layout: resource.current_state.layout,
                            new_layout: resource.current_state.layout,
                            src_queue_family: effective_current_queue,
                            dst_queue_family: queue_family,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask,
                                base_mip_level: 0,
                                level_count: mip_levels,
                                base_array_layer: 0,
                                layer_count: array_layers,
                            },
                        };
                        compiled_passes[prev_pass_index]
                            .barriers_after
                            .push(release_barrier);
                    }
                }

                let aspect_mask = aspect_mask_from_format(format);

                let acquire_barrier = Barrier::Image {
                    image,
                    src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                    dst_stage: if needs_layout_transition {
                        vk::PipelineStageFlags2::TOP_OF_PIPE
                    } else {
                        access.stage_mask
                    },
                    src_access: vk::AccessFlags2::NONE,
                    dst_access: if needs_layout_transition {
                        vk::AccessFlags2::NONE
                    } else {
                        access.access_mask
                    },
                    old_layout: resource.current_state.layout,
                    new_layout: resource.current_state.layout,
                    src_queue_family: effective_current_queue,
                    dst_queue_family: queue_family,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: mip_levels,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                };
                barriers_before.push(acquire_barrier);

                if needs_layout_transition {
                    let layout_barrier = Barrier::Image {
                        image,
                        src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                        dst_stage: access.stage_mask,
                        src_access: vk::AccessFlags2::NONE,
                        dst_access: access.access_mask,
                        old_layout: resource.current_state.layout,
                        new_layout: access.desired_layout,
                        src_queue_family: queue_family,
                        dst_queue_family: queue_family,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask,
                            base_mip_level: 0,
                            level_count: mip_levels,
                            base_array_layer: 0,
                            layer_count: array_layers,
                        },
                    };
                    barriers_before.push(layout_barrier);
                }
            } else if needs_barrier {
                let aspect_mask = aspect_mask_from_format(format);

                let src_stage =
                    if resource.current_state.stage_mask == vk::PipelineStageFlags2::NONE {
                        vk::PipelineStageFlags2::TOP_OF_PIPE
                    } else {
                        sanitize_stage_mask_for_queue(
                            resource.current_state.stage_mask,
                            queue_family,
                            graphics_queue_family,
                        )
                    };

                let barrier = Barrier::Image {
                    image,
                    src_stage,
                    dst_stage: access.stage_mask,
                    src_access: resource.current_state.access_mask,
                    dst_access: access.access_mask,
                    old_layout: resource.current_state.layout,
                    new_layout: access.desired_layout,
                    src_queue_family: queue_family,
                    dst_queue_family: queue_family,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: mip_levels,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    },
                };
                barriers_before.push(barrier);
            }

            if let Some(last_writer) = resource.last_written_pass {
                let last_writer_queue = compiled_passes[last_writer.0 as usize].queue_family;
                if last_writer_queue == queue_family {
                    if !same_queue_dependencies.contains(&last_writer) {
                        same_queue_dependencies.push(last_writer);
                    }
                } else if !cross_queue_dependencies.contains(&last_writer) {
                    cross_queue_dependencies.push(last_writer);
                }
            }

            if access.access_type == AccessType::Write {
                resource.last_written_pass = Some(pass_id);
            }

            let new_queue_family = match &resource.info.resource_type {
                ResourceType::Imported {
                    initial_queue_family,
                    ..
                } if *initial_queue_family == vk::QUEUE_FAMILY_IGNORED => vk::QUEUE_FAMILY_IGNORED,
                _ => queue_family,
            };

            resource.current_state = ResourceState {
                layout: access.desired_layout,
                access_mask: access.access_mask,
                stage_mask: access.stage_mask,
                queue_family: new_queue_family,
            };
        }

        let timeline_value = match pass_desc.queue_type {
            QueueType::Graphics => {
                graphics_timeline_counter += 1;
                graphics_timeline_counter
            }
            QueueType::Transfer => {
                transfer_timeline_counter += 1;
                transfer_timeline_counter
            }
            QueueType::Compute => {
                compute_timeline_counter += 1;
                compute_timeline_counter
            }
        };

        compiled_passes.push(CompiledPass {
            desc: pass_desc,
            same_queue_dependencies,
            cross_queue_dependencies,
            barriers_before,
            barriers_after: Vec::new(),
            queue_family,
            timeline_value,
        });
    }

    for pass_id in 0..compiled_passes.len() {
        pass_order.push(PassId::new(pass_id as u32));
    }

    merge_barriers_across_passes(&mut compiled_passes);

    add_final_release_barriers(&mut compiled_passes, &mut compiled_resources);

    Ok(CompiledRenderGraph {
        passes: compiled_passes,
        pass_order,
        resources: compiled_resources,
        max_graphics_timeline: graphics_timeline_counter,
        max_transfer_timeline: transfer_timeline_counter,
        max_compute_timeline: compute_timeline_counter,
        resource_to_alias_group,
    })
}

pub struct CompiledRenderGraph {
    passes: Vec<CompiledPass>,
    pass_order: Vec<PassId>,
    resources: Vec<CompiledResource>,
    max_graphics_timeline: u64,
    max_transfer_timeline: u64,
    max_compute_timeline: u64,
    resource_to_alias_group: HashMap<ResourceId, usize>,
}

pub struct ExecutionContext<'a> {
    pub device: &'a ash::Device,
    pub graphics_cmd_pool: &'a mut crate::CommandBufferPool,
    pub graphics_queue: vk::Queue,
    pub transfer_cmd_pool: Option<&'a mut crate::CommandBufferPool>,
    pub transfer_queue: Option<vk::Queue>,
    pub compute_cmd_pool: Option<&'a mut crate::CommandBufferPool>,
    pub compute_queue: Option<vk::Queue>,
    pub fence: vk::Fence,
    pub wait_semaphores: &'a [(vk::Semaphore, vk::PipelineStageFlags2)],
    pub signal_semaphores: &'a [(vk::Semaphore, vk::PipelineStageFlags2)],
    pub graphics_timeline_semaphore: vk::Semaphore,
    pub transfer_timeline_semaphore: Option<vk::Semaphore>,
    pub compute_timeline_semaphore: Option<vk::Semaphore>,
    pub transient_pool: &'a mut TransientResourcePool,
}

impl CompiledRenderGraph {
    pub fn get_max_timeline_values(&self) -> (u64, u64, u64) {
        (
            self.max_graphics_timeline,
            self.max_transfer_timeline,
            self.max_compute_timeline,
        )
    }

    pub fn print_aliasing_report(&self) {
        println!("=== Resource Aliasing Report ===");

        let mut group_map: HashMap<usize, Vec<ResourceId>> = HashMap::new();
        for (resource_id, &group_id) in &self.resource_to_alias_group {
            group_map.entry(group_id).or_default().push(*resource_id);
        }

        for (group_id, resources) in &group_map {
            println!("Physical Resource {}:", group_id);
            for resource_id in resources {
                println!("  - {:?}", resource_id);
            }
        }

        let logical_count = self.resources.len();
        let physical_count = group_map.len();
        let savings = if logical_count > 0 {
            (1.0 - physical_count as f32 / logical_count as f32) * 100.0
        } else {
            0.0
        };
        println!(
            "Total: {} logical → {} physical ({:.1}% savings)",
            logical_count, physical_count, savings
        );
    }

    fn generate_mipmaps_for_image(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        extent: vk::Extent3D,
        mip_levels: u32,
        format: vk::Format,
    ) {
        let aspect_mask = aspect_mask_from_format(format);

        let mut mip_width = extent.width as i32;
        let mut mip_height = extent.height as i32;

        for mip_level in 1..mip_levels {
            let src_subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect_mask)
                .mip_level(mip_level - 1)
                .base_array_layer(0)
                .layer_count(1);

            let dst_subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect_mask)
                .mip_level(mip_level)
                .base_array_layer(0)
                .layer_count(1);

            let next_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let next_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            let barrier = vk::ImageMemoryBarrier2::default()
                .image(image)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(aspect_mask)
                        .base_mip_level(mip_level - 1)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            unsafe {
                device.cmd_pipeline_barrier2(cmd, &dependency_info);
            }

            let blit = vk::ImageBlit::default()
                .src_subresource(src_subresource)
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .dst_subresource(dst_subresource)
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_mip_width,
                        y: next_mip_height,
                        z: 1,
                    },
                ]);

            unsafe {
                device.cmd_blit_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&blit),
                    vk::Filter::LINEAR,
                );
            }

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        let final_barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&final_barrier));
        unsafe {
            device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
    }

    pub fn execute(mut self, ctx: ExecutionContext) -> Result<()> {
        let ExecutionContext {
            device,
            graphics_cmd_pool,
            graphics_queue,
            mut transfer_cmd_pool,
            transfer_queue,
            mut compute_cmd_pool,
            compute_queue,
            fence,
            wait_semaphores,
            signal_semaphores,
            graphics_timeline_semaphore,
            transfer_timeline_semaphore,
            compute_timeline_semaphore,
            transient_pool,
        } = ctx;

        let mut resource_handles: HashMap<ResourceId, ResourceHandle> = HashMap::new();
        let mut alias_group_images: HashMap<usize, (vk::Image, vk::ImageView)> = HashMap::new();

        for (idx, resource) in self.resources.iter().enumerate() {
            let resource_id = ResourceId::new(idx as u32, resource.info.generation);

            match &resource.info.resource_type {
                ResourceType::Imported { handle, .. } => {
                    resource_handles.insert(resource_id, handle.clone());
                }
                ResourceType::TransientImage { desc }
                | ResourceType::TemporalImage { desc, .. } => {
                    let (image, view) = if let Some(&alias_group_id) =
                        self.resource_to_alias_group.get(&resource_id)
                    {
                        *alias_group_images
                            .entry(alias_group_id)
                            .or_insert_with(|| transient_pool.acquire_image(desc).unwrap())
                    } else {
                        transient_pool.acquire_image(desc)?
                    };

                    let handle = ResourceHandle::Image {
                        image,
                        view,
                        extent: desc.extent,
                        format: desc.format,
                        mip_levels: desc.mip_levels,
                        array_layers: desc.array_layers,
                    };

                    resource_handles.insert(resource_id, handle);
                }
                ResourceType::TransientBuffer { desc }
                | ResourceType::TemporalBuffer { desc, .. } => {
                    let buffer = transient_pool.acquire_buffer(desc)?;

                    let handle = ResourceHandle::Buffer {
                        buffer,
                        size: desc.size,
                    };

                    resource_handles.insert(resource_id, handle);
                }
            }
        }

        use std::collections::HashSet;

        struct Batch {
            queue_type: QueueType,
            passes: Vec<PassId>,
        }

        let mut submitted_passes: HashSet<PassId> = HashSet::new();
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let total_pass_count = self.passes.len();
        let mut batches: Vec<Batch> = Vec::new();

        while submitted_passes.len() < total_pass_count {
            let mut graphics_batch = Vec::new();
            let mut transfer_batch = Vec::new();
            let mut compute_batch = Vec::new();

            for pass_id in &self.pass_order {
                if submitted_passes.contains(pass_id) {
                    continue;
                }

                let pass_index = pass_id.0 as usize;
                let pass = &self.passes[pass_index];

                let dependencies_met = pass
                    .same_queue_dependencies
                    .iter()
                    .all(|dep| submitted_passes.contains(dep))
                    && pass
                        .cross_queue_dependencies
                        .iter()
                        .all(|dep| submitted_passes.contains(dep));

                if !dependencies_met {
                    continue;
                }

                let has_cross_queue_deps = !pass.cross_queue_dependencies.is_empty();
                let batch_is_empty = match pass.desc.queue_type {
                    QueueType::Graphics => graphics_batch.is_empty(),
                    QueueType::Transfer => transfer_batch.is_empty(),
                    QueueType::Compute => compute_batch.is_empty(),
                };

                if has_cross_queue_deps && !batch_is_empty {
                    break;
                }

                match pass.desc.queue_type {
                    QueueType::Graphics => graphics_batch.push(*pass_id),
                    QueueType::Transfer => transfer_batch.push(*pass_id),
                    QueueType::Compute => compute_batch.push(*pass_id),
                }

                submitted_passes.insert(*pass_id);

                if has_cross_queue_deps {
                    break;
                }
            }

            if graphics_batch.is_empty() && transfer_batch.is_empty() && compute_batch.is_empty() {
                return Err("Deadlock detected: no passes can be submitted".into());
            }

            if !graphics_batch.is_empty() {
                batches.push(Batch {
                    queue_type: QueueType::Graphics,
                    passes: graphics_batch,
                });
            }
            if !transfer_batch.is_empty() {
                batches.push(Batch {
                    queue_type: QueueType::Transfer,
                    passes: transfer_batch,
                });
            }
            if !compute_batch.is_empty() {
                batches.push(Batch {
                    queue_type: QueueType::Compute,
                    passes: compute_batch,
                });
            }
        }

        let total_batch_count = batches.len();

        for (batch_index, batch) in batches.into_iter().enumerate() {
            let is_last_batch_overall = batch_index == total_batch_count - 1;

            let mut cmd_buffers = Vec::new();
            let mut first_pass_cross_queue_deps = Vec::new();
            let mut last_pass_timeline_value = 0;

            let (cmd_pool, queue, timeline_semaphore): (
                &mut crate::CommandBufferPool,
                vk::Queue,
                vk::Semaphore,
            ) = match batch.queue_type {
                QueueType::Graphics => (
                    graphics_cmd_pool,
                    graphics_queue,
                    graphics_timeline_semaphore,
                ),
                QueueType::Transfer => {
                    let pool = transfer_cmd_pool
                        .as_mut()
                        .ok_or("Transfer pass scheduled but no transfer pool available")?;
                    let queue_handle = transfer_queue
                        .ok_or("Transfer pass scheduled but no transfer queue available")?;
                    let timeline = transfer_timeline_semaphore.ok_or(
                        "Transfer pass scheduled but no transfer timeline semaphore available",
                    )?;
                    (pool, queue_handle, timeline)
                }
                QueueType::Compute => {
                    let pool = compute_cmd_pool
                        .as_mut()
                        .ok_or("Compute pass scheduled but no compute pool available")?;
                    let queue_handle = compute_queue
                        .ok_or("Compute pass scheduled but no compute queue available")?;
                    let timeline = compute_timeline_semaphore.ok_or(
                        "Compute pass scheduled but no compute timeline semaphore available",
                    )?;
                    (pool, queue_handle, timeline)
                }
            };

            for (pass_index_in_batch, pass_id) in batch.passes.iter().enumerate() {
                let pass_index = pass_id.0 as usize;
                let pass = &mut self.passes[pass_index];

                let cmd = cmd_pool.acquire()?;

                unsafe { device.begin_command_buffer(cmd, &begin_info)? };

                let updated_barriers_before =
                    update_barriers(&pass.barriers_before, &resource_handles);
                if !updated_barriers_before.is_empty() {
                    Self::record_barriers_static(device, cmd, &updated_barriers_before);
                }

                if let Some(execute) = pass.desc.execute.take() {
                    let pass_ctx = PassContext {
                        cmd,
                        device,
                        resources: &resource_handles,
                    };
                    execute(pass_ctx)?;
                }

                for access in &pass.desc.accesses {
                    if access.access_type == AccessType::Write {
                        let resource_index = access.resource_id.index as usize;
                        if let ResourceType::TransientImage { desc } =
                            &self.resources[resource_index].info.resource_type
                            && desc.auto_generate_mipmaps
                            && desc.mip_levels > 1
                            && let Some(ResourceHandle::Image {
                                image,
                                extent,
                                format,
                                mip_levels,
                                ..
                            }) = resource_handles.get(&access.resource_id)
                        {
                            Self::generate_mipmaps_for_image(
                                device,
                                cmd,
                                *image,
                                *extent,
                                *mip_levels,
                                *format,
                            );
                        }
                    }
                }

                let updated_barriers_after =
                    update_barriers(&pass.barriers_after, &resource_handles);
                if !updated_barriers_after.is_empty() {
                    Self::record_barriers_static(device, cmd, &updated_barriers_after);
                }

                unsafe { device.end_command_buffer(cmd)? };

                cmd_buffers.push(cmd);

                if pass_index_in_batch == 0 {
                    first_pass_cross_queue_deps = pass.cross_queue_dependencies.clone();
                }

                last_pass_timeline_value = pass.timeline_value;
            }

            let cmd_infos: Vec<_> = cmd_buffers
                .iter()
                .map(|cmd| vk::CommandBufferSubmitInfo::default().command_buffer(*cmd))
                .collect();

            let mut wait_infos = Vec::new();

            let is_first_batch_overall = batch_index == 0;
            if is_first_batch_overall {
                for (sem, stage) in wait_semaphores {
                    wait_infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*sem)
                            .stage_mask(*stage),
                    );
                }
            }

            for dep_id in &first_pass_cross_queue_deps {
                let dep_pass = &self.passes[dep_id.0 as usize];
                let dep_timeline_semaphore = match dep_pass.desc.queue_type {
                    QueueType::Graphics => graphics_timeline_semaphore,
                    QueueType::Transfer => transfer_timeline_semaphore.ok_or(
                        "Cross-queue dependency on transfer but no transfer timeline semaphore",
                    )?,
                    QueueType::Compute => compute_timeline_semaphore.ok_or(
                        "Cross-queue dependency on compute but no compute timeline semaphore",
                    )?,
                };

                let wait_stage = match batch.queue_type {
                    QueueType::Graphics => vk::PipelineStageFlags2::ALL_COMMANDS,
                    QueueType::Transfer => vk::PipelineStageFlags2::TRANSFER,
                    QueueType::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
                };

                wait_infos.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(dep_timeline_semaphore)
                        .stage_mask(wait_stage)
                        .value(dep_pass.timeline_value),
                );
            }

            let mut signal_infos = vec![
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(timeline_semaphore)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .value(last_pass_timeline_value),
            ];

            let submit_fence = if is_last_batch_overall {
                for (sem, stage) in signal_semaphores {
                    signal_infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*sem)
                            .stage_mask(*stage),
                    );
                }
                fence
            } else {
                vk::Fence::null()
            };

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(&cmd_infos)
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos);

            unsafe {
                device.queue_submit2(queue, std::slice::from_ref(&submit_info), submit_fence)?;
            }
        }

        let stats = transient_pool.get_statistics();
        if stats.total_images > 0 {
            log::debug!(
                "Pool stats: {}/{} images in use, {:.2}MB/{:.2}MB memory, {} hits/{} misses",
                stats.images_in_use,
                stats.total_images,
                stats.memory_in_use_bytes as f64 / 1_000_000.0,
                stats.total_memory_bytes as f64 / 1_000_000.0,
                stats.cache_hits,
                stats.cache_misses
            );
        }

        if transient_pool.current_frame % 300 == 0 {
            transient_pool.defragment(60);
        }

        transient_pool.release_all();

        Ok(())
    }

    fn record_barriers_static(device: &ash::Device, cmd: vk::CommandBuffer, barriers: &[Barrier]) {
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();

        for barrier in barriers {
            match barrier {
                Barrier::Image {
                    image,
                    src_stage,
                    dst_stage,
                    src_access,
                    dst_access,
                    old_layout,
                    new_layout,
                    src_queue_family,
                    dst_queue_family,
                    subresource_range,
                } => {
                    if *image == vk::Image::null() {
                        continue;
                    }

                    let (src_qf, dst_qf) = if *src_queue_family == *dst_queue_family {
                        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
                    } else {
                        (*src_queue_family, *dst_queue_family)
                    };

                    image_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .image(*image)
                            .src_stage_mask(*src_stage)
                            .dst_stage_mask(*dst_stage)
                            .src_access_mask(*src_access)
                            .dst_access_mask(*dst_access)
                            .old_layout(*old_layout)
                            .new_layout(*new_layout)
                            .src_queue_family_index(src_qf)
                            .dst_queue_family_index(dst_qf)
                            .subresource_range(*subresource_range),
                    );
                }
                Barrier::Buffer {
                    buffer,
                    src_stage,
                    dst_stage,
                    src_access,
                    dst_access,
                    src_queue_family,
                    dst_queue_family,
                    offset,
                    size,
                } => {
                    if *buffer == vk::Buffer::null() {
                        continue;
                    }

                    let (src_qf, dst_qf) = if *src_queue_family == *dst_queue_family {
                        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
                    } else {
                        (*src_queue_family, *dst_queue_family)
                    };

                    buffer_barriers.push(
                        vk::BufferMemoryBarrier2::default()
                            .buffer(*buffer)
                            .src_stage_mask(*src_stage)
                            .dst_stage_mask(*dst_stage)
                            .src_access_mask(*src_access)
                            .dst_access_mask(*dst_access)
                            .src_queue_family_index(src_qf)
                            .dst_queue_family_index(dst_qf)
                            .offset(*offset)
                            .size(*size),
                    );
                }
            }
        }

        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&image_barriers)
                .buffer_memory_barriers(&buffer_barriers);

            unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
        }
    }
}

fn update_barriers(
    barriers: &[Barrier],
    _resource_handles: &HashMap<ResourceId, ResourceHandle>,
) -> Vec<Barrier> {
    barriers
        .iter()
        .filter_map(|barrier| match barrier {
            Barrier::Image { image, .. } => {
                if *image != vk::Image::null() {
                    return Some(barrier.clone());
                }

                None
            }
            Barrier::Buffer { buffer, .. } => {
                if *buffer != vk::Buffer::null() {
                    return Some(barrier.clone());
                }

                None
            }
        })
        .collect()
}

fn add_final_release_barriers(
    compiled_passes: &mut [CompiledPass],
    compiled_resources: &mut [CompiledResource],
) {
    for resource in compiled_resources.iter_mut() {
        if let ResourceType::Imported {
            initial_queue_family,
            ..
        } = &resource.info.resource_type
        {
            if *initial_queue_family == vk::QUEUE_FAMILY_IGNORED {
                continue;
            }

            if resource.current_state.queue_family != *initial_queue_family
                && resource.current_state.queue_family != vk::QUEUE_FAMILY_IGNORED
            {
                let barrier = match &resource.info.resource_type {
                    ResourceType::Imported {
                        handle: ResourceHandle::Image { image, .. },
                        ..
                    } => {
                        let format = match &resource.info.resource_type {
                            ResourceType::Imported {
                                handle: ResourceHandle::Image { format, .. },
                                ..
                            } => *format,
                            _ => unreachable!(),
                        };
                        let aspect_mask = aspect_mask_from_format(format);
                        let subresource_range = vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect_mask)
                            .base_mip_level(0)
                            .level_count(vk::REMAINING_MIP_LEVELS)
                            .base_array_layer(0)
                            .layer_count(vk::REMAINING_ARRAY_LAYERS);

                        Barrier::Image {
                            image: *image,
                            src_stage: resource.current_state.stage_mask,
                            dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                            src_access: resource.current_state.access_mask,
                            dst_access: vk::AccessFlags2::NONE,
                            old_layout: resource.current_state.layout,
                            new_layout: resource.current_state.layout,
                            src_queue_family: resource.current_state.queue_family,
                            dst_queue_family: *initial_queue_family,
                            subresource_range,
                        }
                    }
                    ResourceType::Imported {
                        handle: ResourceHandle::Buffer { buffer, size },
                        ..
                    } => Barrier::Buffer {
                        buffer: *buffer,
                        src_stage: resource.current_state.stage_mask,
                        dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                        src_access: resource.current_state.access_mask,
                        dst_access: vk::AccessFlags2::NONE,
                        src_queue_family: resource.current_state.queue_family,
                        dst_queue_family: *initial_queue_family,
                        offset: 0,
                        size: *size,
                    },
                    _ => continue,
                };

                for pass in compiled_passes.iter_mut().rev() {
                    if pass.queue_family == resource.current_state.queue_family {
                        pass.barriers_after.push(barrier);
                        break;
                    }
                }

                resource.current_state.queue_family = *initial_queue_family;
            }
        }
    }
}

fn merge_barriers_across_passes(compiled_passes: &mut [CompiledPass]) {
    for pass_index in 0..compiled_passes.len() {
        let queue_family = compiled_passes[pass_index].queue_family;
        let mut merged_barriers = Vec::new();
        let mut barrier_map: std::collections::HashMap<BarrierKey, BarrierMergeInfo> =
            std::collections::HashMap::new();

        for barrier in &compiled_passes[pass_index].barriers_before {
            let key = barrier_key(barrier);

            barrier_map
                .entry(key)
                .and_modify(|info| {
                    info.dst_stage |= barrier_stage_dst(barrier);
                    info.dst_access |= barrier_access_dst(barrier);
                    if barrier_layout_new(barrier) != vk::ImageLayout::UNDEFINED {
                        info.new_layout = barrier_layout_new(barrier);
                    }
                })
                .or_insert_with(|| BarrierMergeInfo {
                    barrier: barrier.clone(),
                    dst_stage: barrier_stage_dst(barrier),
                    dst_access: barrier_access_dst(barrier),
                    new_layout: barrier_layout_new(barrier),
                });
        }

        for (_, info) in barrier_map {
            let mut barrier = info.barrier;
            match &mut barrier {
                Barrier::Image {
                    dst_stage,
                    dst_access,
                    new_layout,
                    ..
                } => {
                    *dst_stage = info.dst_stage;
                    *dst_access = info.dst_access;
                    *new_layout = info.new_layout;
                }
                Barrier::Buffer {
                    dst_stage,
                    dst_access,
                    ..
                } => {
                    *dst_stage = info.dst_stage;
                    *dst_access = info.dst_access;
                }
            }
            merged_barriers.push(barrier);
        }

        compiled_passes[pass_index].barriers_before = merged_barriers;

        if !compiled_passes[pass_index]
            .same_queue_dependencies
            .is_empty()
        {
            let last_dep = compiled_passes[pass_index]
                .same_queue_dependencies
                .last()
                .unwrap();
            let last_dep_index = last_dep.0 as usize;

            if last_dep_index < pass_index
                && compiled_passes[last_dep_index].queue_family == queue_family
            {
                let mut barriers_to_move = Vec::new();
                let barriers_before = &compiled_passes[pass_index].barriers_before;

                for barrier in barriers_before {
                    if is_same_queue_barrier(barrier, queue_family) {
                        barriers_to_move.push(barrier.clone());
                    }
                }

                if !barriers_to_move.is_empty() {
                    compiled_passes[pass_index]
                        .barriers_before
                        .retain(|b| !barriers_to_move.contains(b));
                    compiled_passes[last_dep_index]
                        .barriers_after
                        .extend(barriers_to_move);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BarrierKey {
    Image(vk::Image, vk::ImageLayout),
    Buffer(vk::Buffer),
}

struct BarrierMergeInfo {
    barrier: Barrier,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    new_layout: vk::ImageLayout,
}

fn barrier_key(barrier: &Barrier) -> BarrierKey {
    match barrier {
        Barrier::Image {
            image, old_layout, ..
        } => BarrierKey::Image(*image, *old_layout),
        Barrier::Buffer { buffer, .. } => BarrierKey::Buffer(*buffer),
    }
}

fn barrier_stage_dst(barrier: &Barrier) -> vk::PipelineStageFlags2 {
    match barrier {
        Barrier::Image { dst_stage, .. } => *dst_stage,
        Barrier::Buffer { dst_stage, .. } => *dst_stage,
    }
}

fn barrier_access_dst(barrier: &Barrier) -> vk::AccessFlags2 {
    match barrier {
        Barrier::Image { dst_access, .. } => *dst_access,
        Barrier::Buffer { dst_access, .. } => *dst_access,
    }
}

fn barrier_layout_new(barrier: &Barrier) -> vk::ImageLayout {
    match barrier {
        Barrier::Image { new_layout, .. } => *new_layout,
        Barrier::Buffer { .. } => vk::ImageLayout::UNDEFINED,
    }
}

fn is_same_queue_barrier(barrier: &Barrier, queue_family: u32) -> bool {
    match barrier {
        Barrier::Image {
            src_queue_family,
            dst_queue_family,
            ..
        } => *src_queue_family == queue_family && *dst_queue_family == queue_family,
        Barrier::Buffer {
            src_queue_family,
            dst_queue_family,
            ..
        } => *src_queue_family == queue_family && *dst_queue_family == queue_family,
    }
}

fn validate_no_cycles(
    passes: &[PassDesc],
    resource_map: &HashMap<String, ResourceId>,
) -> Result<()> {
    use std::collections::HashSet;

    let id_to_name: HashMap<ResourceId, String> = resource_map
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); passes.len()];
    let mut dependency_resources: Vec<HashMap<usize, Vec<ResourceId>>> =
        vec![HashMap::new(); passes.len()];

    for (pass_index, pass) in passes.iter().enumerate() {
        for access in &pass.accesses {
            if access.access_type == AccessType::Read {
                for (other_index, other_pass) in passes.iter().enumerate() {
                    if other_index < pass_index {
                        for other_access in &other_pass.accesses {
                            if other_access.resource_id == access.resource_id
                                && other_access.access_type == AccessType::Write
                            {
                                dependencies[pass_index].insert(other_index);
                                dependency_resources[pass_index]
                                    .entry(other_index)
                                    .or_default()
                                    .push(access.resource_id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn find_cycle_path(
        node: usize,
        dependencies: &[HashSet<usize>],
        visited: &mut HashSet<usize>,
        path: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        if path.contains(&node) {
            let cycle_start = path.iter().position(|&n| n == node).unwrap();
            return Some(path[cycle_start..].to_vec());
        }

        if visited.contains(&node) {
            return None;
        }

        visited.insert(node);
        path.push(node);

        for &dep in &dependencies[node] {
            if let Some(cycle) = find_cycle_path(dep, dependencies, visited, path) {
                return Some(cycle);
            }
        }

        path.pop();
        None
    }

    let mut visited = HashSet::new();

    for pass_index in 0..passes.len() {
        let mut path = Vec::new();
        if let Some(cycle_path) =
            find_cycle_path(pass_index, &dependencies, &mut visited, &mut path)
        {
            let mut message = "Cycle detected in passes: ".to_string();
            for (index, &pass_idx) in cycle_path.iter().enumerate() {
                if index > 0 {
                    message.push_str(" → ");

                    if let Some(resources) =
                        dependency_resources[pass_idx].get(&cycle_path[index - 1])
                    {
                        let resource_names: Vec<String> = resources
                            .iter()
                            .map(|rid| {
                                id_to_name
                                    .get(rid)
                                    .map(|s| format!("'{}'", s))
                                    .unwrap_or_else(|| format!("resource_{}", rid.index))
                            })
                            .collect();
                        message.push_str(&format!("[depends on {}] → ", resource_names.join(", ")));
                    }
                }
                message.push_str(&format!("Pass {}", pass_idx));
            }

            if let Some(&first_pass) = cycle_path.first()
                && let Some(resources) =
                    dependency_resources[first_pass].get(cycle_path.last().unwrap())
            {
                let resource_names: Vec<String> = resources
                    .iter()
                    .map(|rid| {
                        id_to_name
                            .get(rid)
                            .map(|s| format!("'{}'", s))
                            .unwrap_or_else(|| format!("resource_{}", rid.index))
                    })
                    .collect();
                message.push_str(&format!(
                    " → [depends on {}] → Pass {} (cycle)",
                    resource_names.join(", "),
                    first_pass
                ));
            }

            return Err(RenderGraphError::DependencyCycle {
                pass_indices: cycle_path,
                message,
            });
        }
    }

    Ok(())
}

fn build_alias_map(
    aliasing_groups: &[AliasingGroup],
    _resource_count: usize,
) -> HashMap<ResourceId, usize> {
    let mut map = HashMap::new();

    for (group_index, group) in aliasing_groups.iter().enumerate() {
        for &resource_id in &group.resources {
            map.insert(resource_id, group_index);
        }
    }

    map
}

fn compute_used_resources(
    passes: &[PassDesc],
    resources: &[ResourceInfo],
) -> std::collections::HashSet<ResourceId> {
    use std::collections::HashSet;

    let mut used = HashSet::new();
    let mut to_visit = Vec::new();

    for (resource_index, resource) in resources.iter().enumerate() {
        if let ResourceType::Imported { .. } = resource.resource_type {
            let resource_id = ResourceId::new(resource_index as u32, resource.generation);
            to_visit.push(resource_id);
        }
    }

    while let Some(resource_id) = to_visit.pop() {
        if !used.insert(resource_id) {
            continue;
        }

        for pass in passes {
            let mut pass_reads_resource = false;
            for access in &pass.accesses {
                if access.resource_id == resource_id && access.access_type == AccessType::Read {
                    pass_reads_resource = true;
                    break;
                }
            }

            if pass_reads_resource {
                for access in &pass.accesses {
                    if access.access_type == AccessType::Write
                        && !used.contains(&access.resource_id)
                    {
                        to_visit.push(access.resource_id);
                    }
                }
            }
        }
    }

    used
}

fn compute_used_passes(
    passes: &[PassDesc],
    used_resources: &std::collections::HashSet<ResourceId>,
) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;

    let mut used_passes = HashSet::new();

    for (pass_index, pass) in passes.iter().enumerate() {
        for access in &pass.accesses {
            if used_resources.contains(&access.resource_id) {
                used_passes.insert(pass_index);
                break;
            }
        }
    }

    used_passes
}

fn analyze_lifetimes(resources: &[ResourceInfo], passes: &[PassDesc]) -> Vec<ResourceLifetime> {
    let mut lifetimes = Vec::new();

    for (resource_index, resource) in resources.iter().enumerate() {
        if let ResourceType::TransientImage { desc } = &resource.resource_type {
            let resource_id = ResourceId::new(resource_index as u32, resource.generation);
            let mut first_pass = None;
            let mut last_pass = None;

            for (pass_index, pass_desc) in passes.iter().enumerate() {
                for access in &pass_desc.accesses {
                    if access.resource_id == resource_id {
                        if first_pass.is_none() {
                            first_pass = Some(pass_index);
                        }
                        last_pass = Some(pass_index);
                    }
                }
            }

            if let (Some(first), Some(last)) = (first_pass, last_pass) {
                let lifetime = ResourceLifetime {
                    resource_id,
                    desc: *desc,
                    first_pass: first,
                    last_pass: last,
                };

                for existing in &lifetimes {
                    if lifetime.overlaps(existing)
                        && !descs_compatible(&lifetime.desc, &existing.desc)
                    {}
                }

                lifetimes.push(lifetime);
            }
        }
    }

    lifetimes
}

fn estimate_image_size(desc: &ImageDesc) -> u64 {
    let bytes_per_pixel = match desc.format {
        vk::Format::R8G8B8A8_UNORM => 4,
        vk::Format::R8G8B8A8_SRGB => 4,
        vk::Format::B8G8R8A8_UNORM => 4,
        vk::Format::B8G8R8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        vk::Format::D32_SFLOAT => 4,
        vk::Format::D24_UNORM_S8_UINT => 4,
        vk::Format::D32_SFLOAT_S8_UINT => 8,
        _ => 4,
    };

    desc.extent.width as u64
        * desc.extent.height as u64
        * desc.extent.depth as u64
        * bytes_per_pixel
        * desc.mip_levels as u64
}

fn get_size_class(extent: vk::Extent3D) -> vk::Extent3D {
    let width = extent.width.next_power_of_two();
    let height = extent.height.next_power_of_two();
    vk::Extent3D {
        width,
        height,
        depth: 1,
    }
}

fn get_buffer_size_class(size: u64) -> u64 {
    const MIN_SIZE: u64 = 256;
    const MAX_SIZE: u64 = 256 * 1024 * 1024;

    if size <= MIN_SIZE {
        return MIN_SIZE;
    }

    if size >= MAX_SIZE {
        return size.next_power_of_two();
    }

    size.next_power_of_two()
}

fn descs_compatible(a: &ImageDesc, b: &ImageDesc) -> bool {
    get_size_class(a.extent) == get_size_class(b.extent)
        && a.format == b.format
        && a.usage == b.usage
        && a.mip_levels == b.mip_levels
        && a.array_layers == b.array_layers
        && a.auto_generate_mipmaps == b.auto_generate_mipmaps
}

fn group_by_compatibility(lifetimes: Vec<ResourceLifetime>) -> Vec<Vec<ResourceId>> {
    let mut groups: Vec<Vec<ResourceLifetime>> = Vec::new();

    for lifetime in lifetimes {
        let mut placed = false;
        for group in &mut groups {
            if descs_compatible(&group[0].desc, &lifetime.desc) {
                group.push(lifetime.clone());
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![lifetime]);
        }
    }

    groups
        .into_iter()
        .map(|group| group.into_iter().map(|lt| lt.resource_id).collect())
        .collect()
}

fn compute_aliasing_groups(resources: &[ResourceInfo], passes: &[PassDesc]) -> Vec<AliasingGroup> {
    let lifetimes = analyze_lifetimes(resources, passes);
    let compatibility_groups = group_by_compatibility(lifetimes.clone());

    let mut aliasing_groups = Vec::new();

    for group in compatibility_groups {
        if group.is_empty() {
            continue;
        }

        let first_resource_id = group[0];
        let desc = lifetimes
            .iter()
            .find(|lt| lt.resource_id == first_resource_id)
            .map(|lt| lt.desc)
            .unwrap();

        let mut intervals: Vec<(usize, usize, ResourceId)> = Vec::new();
        for resource_id in &group {
            if let Some(lt) = lifetimes.iter().find(|lt| lt.resource_id == *resource_id) {
                intervals.push((lt.first_pass, lt.last_pass, *resource_id));
            }
        }

        intervals.sort_by_key(|(start, _end, _id)| *start);

        let mut physical_resources: Vec<Vec<ResourceId>> = Vec::new();

        for (start, end, resource_id) in intervals {
            let mut placed = false;
            for physical_resource in &mut physical_resources {
                let last_interval_in_physical = physical_resource.last();
                if let Some(last_id) = last_interval_in_physical
                    && let Some(last_lifetime) =
                        lifetimes.iter().find(|lt| lt.resource_id == *last_id)
                    && last_lifetime.last_pass < start
                    && start <= end
                {
                    physical_resource.push(resource_id);
                    placed = true;
                    break;
                }
            }

            if !placed {
                physical_resources.push(vec![resource_id]);
            }
        }

        for physical_resource in physical_resources {
            if !physical_resource.is_empty() {
                aliasing_groups.push(AliasingGroup {
                    desc,
                    resources: physical_resource,
                });
            }
        }
    }

    aliasing_groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transient_image_creation() {
        let mut graph = RenderGraph::new(0, 1, 2);

        let desc = ImageDesc {
            extent: vk::Extent3D {
                width: 1920,
                height: 1080,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            mip_levels: 1,
            array_layers: 1,
            auto_generate_mipmaps: false,
        };

        let resource_id = graph.create_image("test_image", desc);

        assert!(graph.get_resource("test_image").is_some());
        assert_eq!(graph.get_resource("test_image").unwrap(), resource_id);
    }

    #[test]
    fn test_pool_statistics() {
        let stats = PoolStatistics {
            total_images: 10,
            images_in_use: 5,
            total_memory_bytes: 1024000,
            memory_in_use_bytes: 512000,
            cache_hits: 100,
            cache_misses: 20,
        };

        assert_eq!(stats.total_images, 10);
        assert_eq!(stats.cache_hits, 100);
    }

    #[test]
    fn test_graphviz_export() {
        let graph = RenderGraph::new(0, 1, 2);
        let dot = graph.export_graphviz();
        assert!(dot.contains("digraph RenderGraph"));
        assert!(dot.contains("rankdir=LR"));
    }

    #[test]
    fn test_aspect_mask_helper() {
        let color_aspect = aspect_mask_from_format(vk::Format::R8G8B8A8_UNORM);
        assert_eq!(color_aspect, vk::ImageAspectFlags::COLOR);

        let depth_aspect = aspect_mask_from_format(vk::Format::D32_SFLOAT);
        assert_eq!(depth_aspect, vk::ImageAspectFlags::DEPTH);

        let depth_stencil_aspect = aspect_mask_from_format(vk::Format::D24_UNORM_S8_UINT);
        assert_eq!(
            depth_stencil_aspect,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn test_resource_lifetime_overlap() {
        let lifetime1 = ResourceLifetime {
            resource_id: ResourceId::new(0, 0),
            desc: ImageDesc {
                extent: vk::Extent3D {
                    width: 1920,
                    height: 1080,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                mip_levels: 1,
                array_layers: 1,
                auto_generate_mipmaps: false,
            },
            first_pass: 0,
            last_pass: 2,
        };

        let lifetime2 = ResourceLifetime {
            resource_id: ResourceId::new(1, 0),
            desc: ImageDesc {
                extent: vk::Extent3D {
                    width: 1920,
                    height: 1080,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                mip_levels: 1,
                array_layers: 1,
                auto_generate_mipmaps: false,
            },
            first_pass: 1,
            last_pass: 3,
        };

        assert!(lifetime1.overlaps(&lifetime2));
        assert!(lifetime2.overlaps(&lifetime1));
    }
}
