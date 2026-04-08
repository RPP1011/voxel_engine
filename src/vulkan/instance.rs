use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

/// Tracks validation errors via debug messenger callback.
struct ValidationState {
    error_count: AtomicU32,
}

/// Holds a Vulkan queue handle and its family index.
#[derive(Clone, Copy)]
pub struct QueueHandle {
    pub queue: vk::Queue,
    pub family_index: u32,
}

/// Owns the core Vulkan objects: instance, device, queues, debug messenger.
pub struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    enabled_device_extensions: Vec<String>,
    device_properties: vk::PhysicalDeviceProperties,
    graphics_queue: Option<QueueHandle>,
    compute_queue: Option<QueueHandle>,
    transfer_queue: Option<QueueHandle>,
    validation_state: Arc<ValidationState>,
    has_validation: bool,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let state = unsafe { &*(p_user_data as *const ValidationState) };
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        state.error_count.fetch_add(1, Ordering::SeqCst);
    }
    if !p_callback_data.is_null() {
        let msg = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
        eprintln!(
            "[Vulkan {:?}] {}",
            message_severity,
            msg.to_string_lossy()
        );
    }
    vk::FALSE
}

impl VulkanContext {
    pub fn new() -> Result<Self> {
        Self::new_inner(&[], false)
    }

    #[cfg(feature = "app-harness")]
    pub fn new_with_surface_extensions(window: &winit::window::Window) -> Result<Self> {
        use raw_window_handle::HasDisplayHandle;
        let display_handle = window.display_handle().unwrap();
        let extra_exts = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .context("Failed to get required surface extensions")?;
        Self::new_inner(extra_exts, true)
    }

    fn new_inner(extra_instance_extensions: &[*const i8], enable_swapchain: bool) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("Failed to load Vulkan entry")?;

        let validation_state = Arc::new(ValidationState {
            error_count: AtomicU32::new(0),
        });

        // Check if validation layer is available
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let has_validation = available_layers.iter().any(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            name == c"VK_LAYER_KHRONOS_validation"
        });

        // Check if debug utils extension is available
        let available_instance_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let has_debug_utils = available_instance_exts.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name.to_bytes() == ash::ext::debug_utils::NAME.to_bytes()
        });

        let use_validation = has_validation && has_debug_utils;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"VoxelPhysics")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"VoxelEngine")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let layer_names_raw = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        let mut all_extensions: Vec<*const i8> = extra_instance_extensions.to_vec();
        if use_validation {
            all_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let mut instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&all_extensions);

        if use_validation {
            instance_ci = instance_ci.enabled_layer_names(&layer_names_raw);
        }

        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .context("Failed to create Vulkan instance")?;

        // Debug messenger (only if validation available)
        let mut debug_utils_loader = None;
        let mut debug_messenger = vk::DebugUtilsMessengerEXT::null();

        if use_validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let state_ptr = Arc::as_ptr(&validation_state) as *mut std::ffi::c_void;
            let debug_messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback))
                .user_data(state_ptr);

            debug_messenger =
                unsafe { loader.create_debug_utils_messenger(&debug_messenger_ci, None) }
                    .context("Failed to create debug messenger")?;
            debug_utils_loader = Some(loader);
        }

        if !use_validation {
            eprintln!("WARNING: Vulkan validation layers not available — running without validation");
        }

        // Physical device selection — prefer discrete GPU with required extensions
        let mut required_device_extensions: Vec<&CStr> = vec![
            ash::khr::buffer_device_address::NAME,
            ash::khr::synchronization2::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
            ash::khr::acceleration_structure::NAME,
            // Dependencies of ray tracing
            ash::khr::deferred_host_operations::NAME,
        ];
        if enable_swapchain {
            required_device_extensions.push(ash::khr::swapchain::NAME);
        }

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("Failed to enumerate physical devices")?;

        let (physical_device, _available_ext_names) = physical_devices
            .iter()
            .filter_map(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let available_exts =
                    unsafe { instance.enumerate_device_extension_properties(pd) }.ok()?;
                let available_ext_names: Vec<String> = available_exts
                    .iter()
                    .map(|e| {
                        unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect();

                // Check all required extensions are present
                let has_all = required_device_extensions.iter().all(|req| {
                    let req_str = req.to_str().unwrap_or("");
                    available_ext_names.iter().any(|a| a == req_str)
                });
                if !has_all {
                    return None;
                }

                // Score: prefer discrete
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                    _ => 1,
                };
                Some((pd, available_ext_names, score))
            })
            .max_by_key(|&(_, _, score)| score)
            .map(|(pd, exts, _)| (pd, exts))
            .context("No suitable physical device found with required extensions")?;

        let device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        // Queue family selection
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics_family = None;
        let mut compute_family = None;
        let mut transfer_family = None;

        for (i, qf) in queue_families.iter().enumerate() {
            let idx = i as u32;
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_family.is_none() {
                graphics_family = Some(idx);
            } else if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute_family.is_none()
            {
                compute_family = Some(idx);
            } else if qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && transfer_family.is_none()
            {
                transfer_family = Some(idx);
            }
        }

        // Fall back: if dedicated compute/transfer not found, use graphics family
        let graphics_idx = graphics_family.context("No graphics queue family")?;
        let compute_idx = compute_family.unwrap_or(graphics_idx);
        let transfer_idx = transfer_family.unwrap_or(graphics_idx);

        // Collect unique queue families for device creation
        let mut unique_families = vec![graphics_idx];
        if compute_idx != graphics_idx {
            unique_families.push(compute_idx);
        }
        if transfer_idx != graphics_idx && transfer_idx != compute_idx {
            unique_families.push(transfer_idx);
        }

        let queue_priorities = [1.0f32];
        let queue_cis: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        // Enable required features
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true);

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true);

        let mut rt_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                .ray_tracing_pipeline(true);

        let mut as_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let ext_name_ptrs: Vec<*const i8> = required_device_extensions
            .iter()
            .map(|n| n.as_ptr())
            .collect();

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&ext_name_ptrs)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features)
            .push_next(&mut rt_features)
            .push_next(&mut as_features);

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .context("Failed to create logical device")?;

        let graphics_queue = Some(QueueHandle {
            queue: unsafe { device.get_device_queue(graphics_idx, 0) },
            family_index: graphics_idx,
        });
        let compute_queue = Some(QueueHandle {
            queue: unsafe { device.get_device_queue(compute_idx, 0) },
            family_index: compute_idx,
        });
        let transfer_queue = Some(QueueHandle {
            queue: unsafe { device.get_device_queue(transfer_idx, 0) },
            family_index: transfer_idx,
        });

        Ok(Self {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            physical_device,
            device,
            enabled_device_extensions: ext_name_ptrs
                .iter()
                .map(|&p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
                .collect(),
            device_properties,
            graphics_queue,
            compute_queue,
            transfer_queue,
            validation_state,
            has_validation: use_validation,
        })
    }

    pub fn validation_error_count(&self) -> u32 {
        self.validation_state.error_count.load(Ordering::SeqCst)
    }

    pub fn has_validation_layers(&self) -> bool {
        self.has_validation
    }

    pub fn is_discrete_gpu(&self) -> bool {
        self.device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }

    pub fn has_device_extension(&self, name: &str) -> bool {
        self.enabled_device_extensions.iter().any(|e| e == name)
    }

    pub fn graphics_queue(&self) -> Option<QueueHandle> {
        self.graphics_queue
    }

    pub fn compute_queue(&self) -> Option<QueueHandle> {
        self.compute_queue
    }

    pub fn transfer_queue(&self) -> Option<QueueHandle> {
        self.transfer_queue
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            if let Some(ref loader) = self.debug_utils_loader {
                loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
