use ash::vk;
use std::collections::{HashMap, HashSet};
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

    #[error("Memory budget exceeded: {current}MB + {requested}MB > {budget}MB")]
    BudgetExceeded {
        current: u64,
        requested: u64,
        budget: u64,
    },

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
pub enum AccessType {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct ImportedImageDesc {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub initial_layout: vk::ImageLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDesc {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub mip_levels: u32,
    pub array_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: gpu_allocator::MemoryLocation,
}

#[derive(Debug, Clone)]
enum ResourceType {
    Imported {
        handle: ResourceHandle,
        initial_layout: vk::ImageLayout,
    },
    TransientImage {
        desc: ImageDesc,
    },
    TransientBuffer {
        desc: BufferDesc,
    },
}

#[derive(Debug, Clone)]
pub enum ResourceHandle {
    Image {
        image: vk::Image,
        view: vk::ImageView,
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
struct ResourceInfo {
    resource_type: ResourceType,
    generation: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,
    pub access_mask: vk::AccessFlags2,
    pub stage_mask: vk::PipelineStageFlags2,
}

impl Default for ResourceState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access_mask: vk::AccessFlags2::NONE,
            stage_mask: vk::PipelineStageFlags2::NONE,
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
    pub accesses: Vec<ResourceAccess>,
    pub execute: Option<PassExecuteFn>,
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
        let device = &self.device;
        let total_bytes = &mut self.total_allocated_bytes;

        for pool in self.images_by_class.values_mut() {
            pool.retain_mut(|pooled| {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    *total_bytes = total_bytes.saturating_sub(pooled.allocation.size());
                    unsafe {
                        device.destroy_image_view(pooled.view, None);
                        device.destroy_image(pooled.image, None);
                    }
                    false
                } else {
                    true
                }
            });
        }

        for pool in self.buffers_by_class.values_mut() {
            pool.retain_mut(|pooled| {
                if !pooled.in_use && pooled.last_used_frame < safe_frame {
                    *total_bytes = total_bytes.saturating_sub(pooled.allocation.size());
                    unsafe {
                        device.destroy_buffer(pooled.buffer, None);
                    }
                    false
                } else {
                    true
                }
            });
        }
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
    }

    pub fn advance_frame(&mut self) {
        self.current_frame += 1;

        if self.current_frame.is_multiple_of(300) {
            self.defragment(100);
        }
    }

    pub fn defragment(&mut self, max_unused_frames: u64) {
        let threshold_frame = self.current_frame.saturating_sub(max_unused_frames);
        let device = &self.device;
        let total_bytes = &mut self.total_allocated_bytes;

        for pool in self.images_by_class.values_mut() {
            pool.retain_mut(|pooled| {
                if pooled.last_used_frame >= threshold_frame {
                    true
                } else {
                    *total_bytes = total_bytes.saturating_sub(pooled.allocation.size());
                    unsafe {
                        device.destroy_image_view(pooled.view, None);
                        device.destroy_image(pooled.image, None);
                    }
                    false
                }
            });
        }

        for pool in self.buffers_by_class.values_mut() {
            pool.retain_mut(|pooled| {
                if pooled.last_used_frame >= threshold_frame {
                    true
                } else {
                    *total_bytes = total_bytes.saturating_sub(pooled.allocation.size());
                    unsafe {
                        device.destroy_buffer(pooled.buffer, None);
                    }
                    false
                }
            });
        }

        self.images_by_class.retain(|_, pool| !pool.is_empty());
        self.buffers_by_class.retain(|_, pool| !pool.is_empty());
    }

    pub fn cleanup(&mut self) {
        for pool in self.images_by_class.values() {
            for pooled in pool.iter() {
                unsafe {
                    self.device.destroy_image_view(pooled.view, None);
                    self.device.destroy_image(pooled.image, None);
                }
            }
        }

        for pool in self.buffers_by_class.values() {
            for pooled in pool.iter() {
                unsafe {
                    self.device.destroy_buffer(pooled.buffer, None);
                }
            }
        }

        self.images_by_class.clear();
        self.buffers_by_class.clear();
        self.total_allocated_bytes = 0;
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
    next_generation: u32,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            resource_map: HashMap::new(),
            next_generation: 0,
        }
    }

    pub fn import_image(&mut self, name: &str, desc: ImportedImageDesc) -> ResourceId {
        assert!(
            desc.image != vk::Image::null(),
            "Imported image handle cannot be null"
        );
        assert!(
            desc.view != vk::ImageView::null(),
            "Imported image view handle cannot be null"
        );
        assert!(desc.mip_levels > 0, "Image mip_levels must be > 0");
        assert!(desc.array_layers > 0, "Image array_layers must be > 0");

        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        let handle = ResourceHandle::Image {
            image: desc.image,
            view: desc.view,
            format: desc.format,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
        };

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::Imported {
                handle,
                initial_layout: desc.initial_layout,
            },
            generation,
        });

        id
    }

    pub fn create_image(&mut self, name: &str, desc: ImageDesc) -> ResourceId {
        assert!(desc.extent.width > 0, "Image width must be > 0");
        assert!(desc.extent.height > 0, "Image height must be > 0");
        assert!(desc.extent.depth > 0, "Image depth must be > 0");
        assert!(desc.mip_levels > 0, "Image mip_levels must be > 0");
        assert!(desc.array_layers > 0, "Image array_layers must be > 0");

        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::TransientImage { desc },
            generation,
        });

        id
    }

    pub fn import_buffer(&mut self, name: &str, buffer: vk::Buffer, size: u64) -> ResourceId {
        assert!(
            buffer != vk::Buffer::null(),
            "Imported buffer handle cannot be null"
        );
        assert!(size > 0, "Buffer size must be > 0");

        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        let handle = ResourceHandle::Buffer { buffer, size };

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::Imported {
                handle,
                initial_layout: vk::ImageLayout::UNDEFINED,
            },
            generation,
        });

        id
    }

    pub fn create_buffer(&mut self, name: &str, desc: BufferDesc) -> ResourceId {
        assert!(desc.size > 0, "Buffer size must be > 0");

        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let id = ResourceId::new(self.resources.len() as u32, generation);
        self.resource_map.insert(name.to_string(), id);

        self.resources.push(ResourceInfo {
            resource_type: ResourceType::TransientBuffer { desc },
            generation,
        });

        id
    }

    pub fn get_resource(&self, name: &str) -> Option<ResourceId> {
        self.resource_map.get(name).copied()
    }

    fn validate_resource(&self, id: ResourceId) -> Result<()> {
        if (id.index as usize) >= self.resources.len() {
            return Err(RenderGraphError::InvalidResourceId {
                id,
                reason: "Index out of bounds".to_string(),
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

        let mut seen_resources = HashSet::new();
        for access in &desc.accesses {
            if !seen_resources.insert(access.resource_id) {
                return Err(RenderGraphError::Other(format!(
                    "Pass accesses resource {:?} multiple times",
                    access.resource_id
                )));
            }
        }

        self.passes.push(desc);
        Ok(PassId::new((self.passes.len() - 1) as u32))
    }

    pub fn compile(self) -> Result<CompiledRenderGraph> {
        if self.passes.is_empty() {
            return Err(RenderGraphError::EmptyGraph);
        }

        validate_no_cycles(&self.passes, &self.resource_map)?;

        compile_passes(self.passes, self.resources)
    }
}

struct CompiledResource {
    info: ResourceInfo,
    current_state: ResourceState,
}

struct CompiledPass {
    desc: PassDesc,
    barriers_before: Vec<Barrier>,
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
        subresource_range: vk::ImageSubresourceRange,
    },
    Buffer {
        buffer: vk::Buffer,
        src_stage: vk::PipelineStageFlags2,
        dst_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
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
                    src_stage: ss1,
                    dst_stage: ds1,
                    src_access: sa1,
                    dst_access: da1,
                    old_layout: ol1,
                    new_layout: nl1,
                    subresource_range: sr1,
                },
                Barrier::Image {
                    image: img2,
                    src_stage: ss2,
                    dst_stage: ds2,
                    src_access: sa2,
                    dst_access: da2,
                    old_layout: ol2,
                    new_layout: nl2,
                    subresource_range: sr2,
                },
            ) => {
                img1 == img2
                    && ss1 == ss2
                    && ds1 == ds2
                    && sa1 == sa2
                    && da1 == da2
                    && ol1 == ol2
                    && nl1 == nl2
                    && sr1.aspect_mask == sr2.aspect_mask
                    && sr1.base_mip_level == sr2.base_mip_level
                    && sr1.level_count == sr2.level_count
                    && sr1.base_array_layer == sr2.base_array_layer
                    && sr1.layer_count == sr2.layer_count
            }
            (
                Barrier::Buffer {
                    buffer: b1,
                    src_stage: ss1,
                    dst_stage: ds1,
                    src_access: sa1,
                    dst_access: da1,
                    offset: o1,
                    size: sz1,
                },
                Barrier::Buffer {
                    buffer: b2,
                    src_stage: ss2,
                    dst_stage: ds2,
                    src_access: sa2,
                    dst_access: da2,
                    offset: o2,
                    size: sz2,
                },
            ) => {
                b1 == b2
                    && ss1 == ss2
                    && ds1 == ds2
                    && sa1 == sa2
                    && da1 == da2
                    && o1 == o2
                    && sz1 == sz2
            }
            _ => false,
        }
    }
}

fn compile_passes(
    passes: Vec<PassDesc>,
    resources: Vec<ResourceInfo>,
) -> Result<CompiledRenderGraph> {
    let mut compiled_resources: Vec<CompiledResource> = resources
        .into_iter()
        .map(|info| {
            let initial_state = match &info.resource_type {
                ResourceType::Imported { initial_layout, .. } => ResourceState {
                    layout: *initial_layout,
                    access_mask: vk::AccessFlags2::NONE,
                    stage_mask: vk::PipelineStageFlags2::NONE,
                },
                ResourceType::TransientImage { .. } | ResourceType::TransientBuffer { .. } => {
                    ResourceState::default()
                }
            };

            CompiledResource {
                info,
                current_state: initial_state,
            }
        })
        .collect();

    let mut compiled_passes: Vec<CompiledPass> = Vec::new();

    for pass_desc in passes {
        let mut barriers_before = Vec::new();

        for access in &pass_desc.accesses {
            let resource = &mut compiled_resources[access.resource_id.index as usize];

            match &resource.info.resource_type {
                ResourceType::TransientBuffer { .. }
                | ResourceType::Imported {
                    handle: ResourceHandle::Buffer { .. },
                    ..
                } => {
                    match &resource.info.resource_type {
                        ResourceType::Imported {
                            handle: ResourceHandle::Buffer { buffer, size, .. },
                            ..
                        } => {
                            let needs_barrier = access.access_type == AccessType::Write
                                || resource
                                    .current_state
                                    .access_mask
                                    .contains(vk::AccessFlags2::MEMORY_WRITE);

                            if needs_barrier {
                                let src_stage = if resource.current_state.stage_mask
                                    == vk::PipelineStageFlags2::NONE
                                {
                                    vk::PipelineStageFlags2::TOP_OF_PIPE
                                } else {
                                    resource.current_state.stage_mask
                                };

                                let barrier = Barrier::Buffer {
                                    buffer: *buffer,
                                    src_stage,
                                    dst_stage: access.stage_mask,
                                    src_access: resource.current_state.access_mask,
                                    dst_access: access.access_mask,
                                    offset: 0,
                                    size: *size,
                                };
                                barriers_before.push(barrier);
                            }
                        }
                        ResourceType::TransientBuffer { .. } => {}
                        _ => unreachable!(),
                    }

                    resource.current_state = ResourceState {
                        layout: vk::ImageLayout::UNDEFINED,
                        access_mask: access.access_mask,
                        stage_mask: access.stage_mask,
                    };

                    continue;
                }
                _ => {}
            }

            match &resource.info.resource_type {
                ResourceType::Imported {
                    handle:
                        ResourceHandle::Image {
                            image,
                            format,
                            mip_levels,
                            array_layers,
                            ..
                        },
                    ..
                } => {
                    let needs_layout_transition =
                        resource.current_state.layout != access.desired_layout;

                    let needs_barrier = needs_layout_transition
                        || access.access_type == AccessType::Write
                        || resource
                            .current_state
                            .access_mask
                            .contains(vk::AccessFlags2::MEMORY_WRITE);

                    if needs_barrier {
                        let aspect_mask = aspect_mask_from_format(*format);

                        let src_stage =
                            if resource.current_state.stage_mask == vk::PipelineStageFlags2::NONE {
                                vk::PipelineStageFlags2::TOP_OF_PIPE
                            } else {
                                resource.current_state.stage_mask
                            };

                        let barrier = Barrier::Image {
                            image: *image,
                            src_stage,
                            dst_stage: access.stage_mask,
                            src_access: resource.current_state.access_mask,
                            dst_access: access.access_mask,
                            old_layout: resource.current_state.layout,
                            new_layout: access.desired_layout,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask,
                                base_mip_level: 0,
                                level_count: *mip_levels,
                                base_array_layer: 0,
                                layer_count: *array_layers,
                            },
                        };
                        barriers_before.push(barrier);
                    }
                }
                ResourceType::TransientImage { .. } => {}
                _ => unreachable!(),
            }

            resource.current_state = ResourceState {
                layout: access.desired_layout,
                access_mask: access.access_mask,
                stage_mask: access.stage_mask,
            };
        }

        compiled_passes.push(CompiledPass {
            desc: pass_desc,
            barriers_before,
        });
    }

    Ok(CompiledRenderGraph {
        passes: compiled_passes,
        resources: compiled_resources,
    })
}

pub struct CompiledRenderGraph {
    passes: Vec<CompiledPass>,
    resources: Vec<CompiledResource>,
}

pub struct ExecutionContext<'a> {
    pub device: &'a ash::Device,
    pub graphics_cmd_pool: vk::CommandPool,
    pub graphics_queue: vk::Queue,
    pub fence: vk::Fence,
    pub wait_semaphores: &'a [(vk::Semaphore, vk::PipelineStageFlags2)],
    pub signal_semaphores: &'a [vk::Semaphore],
    pub transient_pool: &'a mut TransientResourcePool,
}

impl CompiledRenderGraph {
    pub fn execute(mut self, ctx: ExecutionContext) -> Result<()> {
        let ExecutionContext {
            device,
            graphics_cmd_pool,
            graphics_queue,
            fence,
            wait_semaphores,
            signal_semaphores,
            transient_pool,
        } = ctx;

        let mut resource_handles: HashMap<ResourceId, ResourceHandle> = HashMap::new();

        for (idx, resource) in self.resources.iter().enumerate() {
            let resource_id = ResourceId::new(idx as u32, resource.info.generation);

            match &resource.info.resource_type {
                ResourceType::Imported { handle, .. } => {
                    resource_handles.insert(resource_id, handle.clone());
                }
                ResourceType::TransientImage { desc } => {
                    let (image, view) = transient_pool.acquire_image(desc)?;

                    let handle = ResourceHandle::Image {
                        image,
                        view,
                        format: desc.format,
                        mip_levels: desc.mip_levels,
                        array_layers: desc.array_layers,
                    };

                    resource_handles.insert(resource_id, handle);
                }
                ResourceType::TransientBuffer { desc } => {
                    let buffer = transient_pool.acquire_buffer(desc)?;

                    let handle = ResourceHandle::Buffer {
                        buffer,
                        size: desc.size,
                    };

                    resource_handles.insert(resource_id, handle);
                }
            }
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let mut cmd_buffers = Vec::new();

        for pass in &mut self.passes {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(graphics_cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

            unsafe { device.begin_command_buffer(cmd, &begin_info)? };

            if !pass.barriers_before.is_empty() {
                Self::record_barriers_static(device, cmd, &pass.barriers_before);
            }

            if let Some(execute_fn) = pass.desc.execute.take() {
                let pass_ctx = PassContext {
                    cmd,
                    device,
                    resources: &resource_handles,
                };
                execute_fn(pass_ctx)?;
            }

            unsafe { device.end_command_buffer(cmd)? };

            cmd_buffers.push(cmd);
        }

        let cmd_infos: Vec<_> = cmd_buffers
            .iter()
            .map(|cmd| vk::CommandBufferSubmitInfo::default().command_buffer(*cmd))
            .collect();

        let mut wait_infos = Vec::new();
        for (sem, stage) in wait_semaphores {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .stage_mask(*stage),
            );
        }

        let mut signal_infos = Vec::new();
        for sem in signal_semaphores {
            signal_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_infos)
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(&signal_infos);

        unsafe {
            device.queue_submit2(graphics_queue, &[submit_info], fence)?;
        }

        transient_pool.release_all();
        transient_pool.advance_frame();

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
                    subresource_range,
                } => {
                    image_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(*src_stage)
                            .dst_stage_mask(*dst_stage)
                            .src_access_mask(*src_access)
                            .dst_access_mask(*dst_access)
                            .old_layout(*old_layout)
                            .new_layout(*new_layout)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(*image)
                            .subresource_range(*subresource_range),
                    );
                }
                Barrier::Buffer {
                    buffer,
                    src_stage,
                    dst_stage,
                    src_access,
                    dst_access,
                    offset,
                    size,
                } => {
                    buffer_barriers.push(
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(*src_stage)
                            .dst_stage_mask(*dst_stage)
                            .src_access_mask(*src_access)
                            .dst_access_mask(*dst_access)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(*buffer)
                            .offset(*offset)
                            .size(*size),
                    );
                }
            }
        }

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers);

        unsafe {
            device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
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
            for (other_index, other_pass) in passes.iter().enumerate() {
                if other_index < pass_index {
                    for other_access in &other_pass.accesses {
                        if other_access.resource_id == access.resource_id {
                            let has_dependency =
                                match (other_access.access_type, access.access_type) {
                                    (AccessType::Write, AccessType::Read) => true,
                                    (AccessType::Write, AccessType::Write) => true,
                                    (AccessType::Read, AccessType::Write) => true,
                                    (AccessType::Read, AccessType::Read) => false,
                                };

                            if has_dependency {
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

fn estimate_image_size(desc: &ImageDesc) -> u64 {
    let bytes_per_pixel = match desc.format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        vk::Format::D32_SFLOAT => 4,
        vk::Format::D24_UNORM_S8_UINT => 4,
        _ => 4,
    };

    let mut total_size = 0u64;
    let mut width = desc.extent.width as u64;
    let mut height = desc.extent.height as u64;

    for _ in 0..desc.mip_levels {
        let mip_size = width * height * bytes_per_pixel * desc.array_layers as u64;
        total_size += mip_size;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }

    total_size
}

fn get_size_class(extent: vk::Extent3D) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width.next_power_of_two(),
        height: extent.height.next_power_of_two(),
        depth: extent.depth.next_power_of_two(),
    }
}

fn get_buffer_size_class(size: u64) -> u64 {
    if size == 0 {
        return 0;
    }

    size.next_power_of_two()
}
