use anyhow::{Context, Result};
use ash::vk;

use super::instance::VulkanContext;

pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub push_constant_stages: vk::ShaderStageFlags,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
}

impl GraphicsPipeline {
    pub fn destroy(&self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            if let Some(dsl) = self.descriptor_set_layout {
                device.destroy_descriptor_set_layout(dsl, None);
            }
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }
    }
}

/// Descriptor binding specification.
pub struct DescriptorBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
}

pub struct GraphicsPipelineBuilder<'a> {
    ctx: &'a VulkanContext,
    vert_spv: Option<&'a [u8]>,
    frag_spv: Option<&'a [u8]>,
    render_pass: vk::RenderPass,
    push_constant_size: u32,
    push_constant_stages: vk::ShaderStageFlags,
    descriptors: Vec<DescriptorBinding>,
    depth_write: bool,
    depth_test: bool,
    color_attachment_count: u32,
    no_vertex_input: bool,
    cull_mode: vk::CullModeFlags,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new(ctx: &'a VulkanContext) -> Self {
        Self {
            ctx,
            vert_spv: None,
            frag_spv: None,
            render_pass: vk::RenderPass::null(),
            push_constant_size: 64,
            push_constant_stages: vk::ShaderStageFlags::VERTEX,
            descriptors: Vec::new(),
            depth_write: true,
            depth_test: true,
            color_attachment_count: 1,
            no_vertex_input: false,
            cull_mode: vk::CullModeFlags::BACK,
        }
    }

    pub fn vertex_shader(mut self, spv: &'a [u8]) -> Self {
        self.vert_spv = Some(spv);
        self
    }

    pub fn fragment_shader(mut self, spv: &'a [u8]) -> Self {
        self.frag_spv = Some(spv);
        self
    }

    pub fn render_pass(mut self, rp: vk::RenderPass) -> Self {
        self.render_pass = rp;
        self
    }

    pub fn push_constant_size(mut self, size: u32, stages: vk::ShaderStageFlags) -> Self {
        self.push_constant_size = size;
        self.push_constant_stages = stages;
        self
    }

    pub fn descriptor(mut self, binding: u32, ty: vk::DescriptorType, stages: vk::ShaderStageFlags) -> Self {
        self.descriptors.push(DescriptorBinding {
            binding,
            descriptor_type: ty,
            stage_flags: stages,
        });
        self
    }

    pub fn depth_write(mut self, enable: bool) -> Self {
        self.depth_write = enable;
        self
    }

    pub fn color_attachment_count(mut self, count: u32) -> Self {
        self.color_attachment_count = count;
        self
    }

    pub fn no_vertex_input(mut self) -> Self {
        self.no_vertex_input = true;
        self
    }

    pub fn cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    pub fn no_depth_test(mut self) -> Self {
        self.depth_test = false;
        self.depth_write = false;
        self
    }

    pub fn build(self) -> Result<GraphicsPipeline> {
        let device = self.ctx.device();

        let vert_module = create_shader_module(device, self.vert_spv.context("No vertex shader")?)?;
        let frag_module = create_shader_module(device, self.frag_spv.context("No fragment shader")?)?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main");
        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main");
        let stages = [vert_stage, frag_stage];

        let binding = vk::VertexInputBindingDescription {
            binding: 0, stride: 12, input_rate: vk::VertexInputRate::VERTEX,
        };
        let attribute = vk::VertexInputAttributeDescription {
            location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0,
        };
        let bindings = [binding];
        let attributes = [attribute];
        let vertex_input = if self.no_vertex_input {
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&bindings)
                .vertex_attribute_descriptions(&attributes)
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1).scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(self.cull_mode)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(vk::CompareOp::LESS);
        let blend_atts: Vec<vk::PipelineColorBlendAttachmentState> = (0..self.color_attachment_count)
            .map(|_| vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA))
            .collect();
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_atts);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        // Descriptor set layout
        let dsl_bindings: Vec<vk::DescriptorSetLayoutBinding> = self.descriptors.iter().map(|d| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(d.binding)
                .descriptor_type(d.descriptor_type)
                .descriptor_count(1)
                .stage_flags(d.stage_flags)
        }).collect();

        let descriptor_set_layout = if !dsl_bindings.is_empty() {
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings);
            Some(unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }?)
        } else {
            None
        };

        let set_layouts: Vec<vk::DescriptorSetLayout> = descriptor_set_layout.into_iter().collect();

        let push_range = vk::PushConstantRange {
            stage_flags: self.push_constant_stages,
            offset: 0,
            size: self.push_constant_size,
        };
        let push_ranges = [push_range];
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }?;

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(self.render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .map_err(|(_, e)| e)
        .context("Failed to create graphics pipeline")?[0];

        Ok(GraphicsPipeline {
            pipeline, layout, descriptor_set_layout,
            push_constant_stages: self.push_constant_stages,
            vert_module, frag_module,
        })
    }
}

fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ci = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&ci, None) }?)
}
