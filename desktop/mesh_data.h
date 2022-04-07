#pragma once

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>
#include <taichi/inc/constants.h>
#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>
constexpr int N_VERTS = 616;
constexpr int N_CELLS = 1770;
constexpr int N_FACES = 1138;
constexpr int N_EDGES = 2954;
float dt = 7.5e-3;

#include "data.h"

using namespace taichi::lang;

void set_ctx_arg_devalloc(taichi::lang::RuntimeContext &host_ctx, int arg_id, taichi::lang::DeviceAllocation& alloc, int x, int y, int z) {
  host_ctx.set_arg(arg_id, &alloc);
  host_ctx.set_device_allocation(arg_id, true);
  host_ctx.extra_args[arg_id][0] = x;
  host_ctx.extra_args[arg_id][1] = y;
  host_ctx.extra_args[arg_id][2] = z;
}

// TODO: provide a proper API from taichi
void set_ctx_arg_float(taichi::lang::RuntimeContext &host_ctx, int arg_id, float x) {
  host_ctx.set_arg(arg_id, x);
  host_ctx.set_device_allocation(arg_id, false);
}

float *map(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc) {
    float *device_arr_ptr = reinterpret_cast<float *>(vulkan_runtime.get_ti_device()->map(alloc));
    return device_arr_ptr;
}

void unmap(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc) {
    vulkan_runtime.get_ti_device()->unmap(alloc);
}

void print_debug(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc, int it, bool use_int = false) {
    vulkan_runtime.synchronize();
    auto ptr = map(vulkan_runtime, alloc);
    if (!use_int) printf("%d %.10f %.10f %.10f\n", it, ptr[0], ptr[1], ptr[2]);
    else
    {
        auto p = reinterpret_cast<int*>(ptr);
        printf("%d %d %d %d\n", it, p[0], p[1], p[2]);
    }
    unmap(vulkan_runtime, alloc);
}

void load_data(taichi::lang::vulkan::VkRuntime *vulkan_runtime, taichi::lang::DeviceAllocation& alloc, void *data, size_t size) {
  char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  std::memcpy(device_arr_ptr, data, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}

struct ImplicitFemKernels {
  taichi::lang::aot::Kernel* init_kernel;
  taichi::lang::aot::Kernel* get_vertices_kernel;
  taichi::lang::aot::Kernel* get_indices_kernel;
  taichi::lang::aot::Kernel* get_force_kernel;
  taichi::lang::aot::Kernel* advect_kernel;
  taichi::lang::aot::Kernel* floor_bound_kernel;
  taichi::lang::aot::Kernel* get_b_kernel;
  taichi::lang::aot::Kernel* matmul_cell_kernel;
  taichi::lang::aot::Kernel* ndarray_to_ndarray_kernel;
  taichi::lang::aot::Kernel* fill_ndarray_kernel;
  taichi::lang::aot::Kernel* add_ndarray_kernel;
  taichi::lang::aot::Kernel* dot_kernel;
  taichi::lang::aot::Kernel* add_kernel;
  taichi::lang::aot::Kernel* update_alpha_kernel;
  taichi::lang::aot::Kernel* update_beta_r_2_kernel;
  taichi::lang::aot::Kernel* add_hack_kernel;
  taichi::lang::aot::Kernel* dot2scalar_kernel;
  taichi::lang::aot::Kernel* init_r_2_kernel;
  taichi::lang::aot::Kernel* get_matrix_kernel;
  taichi::lang::aot::Kernel* clear_field_kernel;
  taichi::lang::aot::Kernel* matmul_edge_kernel;
};

// TODO: cleanup these global variables
ImplicitFemKernels loaded_kernels;

std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
std::unique_ptr<taichi::lang::aot::Module> module;
taichi::lang::RuntimeContext host_ctx;
std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator> embedded_device;
std::unique_ptr<taichi::lang::Surface> surface;

taichi::lang::vulkan::VulkanDevice *device;

int width, height;


taichi::lang::DeviceAllocation devalloc_x;
taichi::lang::DeviceAllocation devalloc_v;
taichi::lang::DeviceAllocation devalloc_f;
taichi::lang::DeviceAllocation devalloc_mul_ans;
taichi::lang::DeviceAllocation devalloc_c2e;
taichi::lang::DeviceAllocation devalloc_b;
taichi::lang::DeviceAllocation devalloc_r0;
taichi::lang::DeviceAllocation devalloc_p0;
taichi::lang::DeviceAllocation devalloc_indices;
taichi::lang::DeviceAllocation devalloc_vertices;
taichi::lang::DeviceAllocation devalloc_edges;
taichi::lang::DeviceAllocation devalloc_ox;
taichi::lang::DeviceAllocation devalloc_alpha_scalar;
taichi::lang::DeviceAllocation devalloc_beta_scalar;

std::unique_ptr<Pipeline> render_point_pipeline;
std::unique_ptr<Pipeline> render_surface_pipeline;

DeviceAllocation depth_allocation;
DeviceAllocation render_constants;

struct RenderConstants {
    glm::mat4 proj;
    glm::mat4 view;
};

void run_init(int _width, int _height, std::string path_prefix, taichi::ui::TaichiWindow* window) {
    width = _width;
    height = _height;

#ifdef ANDROID
    std::vector<std::string> extensions;

    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#else
    // Create a Vulkan Device
    std::vector<std::string> extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    uint32_t glfw_ext_count = 0;
    const char **glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    for (int i = 0; i < glfw_ext_count; ++i) {
        extensions.push_back(glfw_extensions[i]);
    }
#endif
    taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params;
    evd_params.api_version = VK_API_VERSION_1_2;
    evd_params.additional_instance_extensions = extensions;
    evd_params.additional_device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    evd_params.is_for_ui = false;
    evd_params.surface_creator = nullptr;
    // [&](VkInstance instance) -> VkSurfaceKHR {
    //   VkSurfaceKHR surface = VK_NULL_HANDLE;
    //   TI_TRACE("before glfwCreateWindowSurface {} {}", (void *)window,
    //            (void *)instance);
    //   int status = VK_SUCCESS;
    //   if ((status = glfwCreateWindowSurface(instance, window, nullptr,
    //                                         &surface)) != VK_SUCCESS) {
    //     TI_ERROR("Failed to create window surface! err: {}", status);
    //     throw std::runtime_error("failed to create window surface!");
    //   }
    //   return surface;
    // };

    embedded_device = std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

    device = static_cast<taichi::lang::vulkan::VulkanDevice *>(embedded_device->device());

    {
        taichi::lang::SurfaceConfig config;
        config.vsync = true;
        config.window_handle = window;
        config.width = width;
        config.height = height;
        surface = device->create_surface(config);
    }

    {
        taichi::lang::ImageParams params;
        params.dimension = ImageDimension::d2D;
        params.format = BufferFormat::depth32f;
        params.initial_layout = ImageLayout::undefined;
        params.x = width;
        params.y = height;
        params.export_sharing = false;

        depth_allocation = device->create_image(params);
    }

    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    memory_pool = std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = embedded_device->device();
    vulkan_runtime = std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

    std::string shader_source = path_prefix + "/implicit_mesh_fem";
    taichi::lang::vulkan::AotModuleParams aot_params{shader_source, vulkan_runtime.get()};
    module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
    auto root_size = module->get_root_size();
    // printf("root buffer size=%ld\n", root_size);
    vulkan_runtime->add_root_buffer(root_size);


    loaded_kernels.get_force_kernel = module->get_kernel("get_force");
    loaded_kernels.init_kernel = module->get_kernel("init");
    loaded_kernels.floor_bound_kernel = module->get_kernel("floor_bound");
    loaded_kernels.get_matrix_kernel = module->get_kernel("get_matrix");
    loaded_kernels.matmul_edge_kernel = module->get_kernel("matmul_edge");
    loaded_kernels.add_kernel = module->get_kernel("add");
    loaded_kernels.add_hack_kernel = module->get_kernel("add_hack");
    loaded_kernels.dot2scalar_kernel = module->get_kernel("dot2scalar");
    loaded_kernels.get_b_kernel = module->get_kernel("get_b");
    loaded_kernels.ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
    loaded_kernels.fill_ndarray_kernel = module->get_kernel("fill_ndarray");
    loaded_kernels.clear_field_kernel = module->get_kernel("clear_field");
    loaded_kernels.init_r_2_kernel = module->get_kernel("init_r_2");
    loaded_kernels.update_alpha_kernel = module->get_kernel("update_alpha");
    loaded_kernels.update_beta_r_2_kernel = module->get_kernel("update_beta_r_2");


    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.host_write = true;
    // x
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    alloc_params.usage = taichi::lang::AllocUsage::Vertex | taichi::lang::AllocUsage::Storage;
    devalloc_x = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;
    // v
    devalloc_v = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // f
    devalloc_f = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // mul_ans
    devalloc_mul_ans = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // c2e
    alloc_params.size = N_CELLS * 6 * sizeof(int);
    devalloc_c2e = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // b
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    devalloc_b = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // r0
    devalloc_r0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // p0
    devalloc_p0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // indices
    alloc_params.size = N_FACES * 3 * sizeof(int);
    alloc_params.usage = taichi::lang::AllocUsage::Index;
    devalloc_indices = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;
    // vertices
    alloc_params.size = N_CELLS * 4 * sizeof(int);
    devalloc_vertices = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // edges
    alloc_params.size = N_EDGES * 2 * sizeof(int);
    devalloc_edges = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    // ox
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    devalloc_ox = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

    alloc_params.size = sizeof(float);
    devalloc_alpha_scalar = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
    devalloc_beta_scalar = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

    load_data(vulkan_runtime.get(), devalloc_indices, indices_data, sizeof(indices_data));
    load_data(vulkan_runtime.get(), devalloc_c2e, c2e_data, sizeof(c2e_data));
    load_data(vulkan_runtime.get(), devalloc_vertices, vertices_data, sizeof(vertices_data));
    load_data(vulkan_runtime.get(), devalloc_ox, ox_data, sizeof(ox_data));
    load_data(vulkan_runtime.get(), devalloc_edges, edges_data, sizeof(edges_data));

    memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
    host_ctx.result_buffer = result_buffer;
    loaded_kernels.clear_field_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 3, devalloc_ox, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 4, devalloc_vertices, N_CELLS, 4, 1);
    // init(x, v, f, ox, vertices)
    loaded_kernels.init_kernel->launch(&host_ctx);
    // get_matrix(c2e, vertices)
    set_ctx_arg_devalloc(host_ctx, 0, devalloc_c2e, N_CELLS, 6, 1);
    set_ctx_arg_devalloc(host_ctx, 1, devalloc_vertices, N_CELLS, 4, 1);
    loaded_kernels.get_matrix_kernel->launch(&host_ctx);
    vulkan_runtime->synchronize();

    {
        // auto vert_code = taichi::ui::read_file("shaders/render_point.vert.spv");
        // auto frag_code = taichi::ui::read_file("shaders/render_point.frag.spv");

        // std::vector<PipelineSourceDesc> source(2);
        // source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
        //             frag_code.size(), PipelineStageType::fragment};
        // source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
        //             vert_code.size(), PipelineStageType::vertex};

        // RasterParams raster_params;
        // raster_params.prim_topology = TopologyType::Points;
        // raster_params.depth_test = true;
        // raster_params.depth_write = true;

        // std::vector<VertexInputBinding> vertex_inputs = {
        //     {/*binding=*/0, /*stride=*/3 * sizeof(float), /*instance=*/false}};
        // std::vector<VertexInputAttribute> vertex_attribs;
        // vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
        //                         /*format=*/BufferFormat::rgb32f,
        //                         /*offset=*/0});

        // render_point_pipeline = device->create_raster_pipeline(
        //     source, raster_params, vertex_inputs, vertex_attribs);
    }

        {
        auto vert_code = taichi::ui::read_file(path_prefix + "/rhi_shaders/surface.vert.spv");
        auto frag_code = taichi::ui::read_file(path_prefix + "/rhi_shaders/surface.frag.spv");

        std::vector<PipelineSourceDesc> source(2);
        source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                    frag_code.size(), PipelineStageType::fragment};
        source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                    vert_code.size(), PipelineStageType::vertex};

        RasterParams raster_params;
        raster_params.prim_topology = TopologyType::Triangles;
        raster_params.depth_test = true;
        raster_params.depth_write = true;

        std::vector<VertexInputBinding> vertex_inputs = {
            {/*binding=*/0, /*stride=*/3 * sizeof(float), /*instance=*/false}};
        std::vector<VertexInputAttribute> vertex_attribs;
        vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                                /*format=*/BufferFormat::rgb32f,
                                /*offset=*/0});

        render_surface_pipeline = device->create_raster_pipeline(
            source, raster_params, vertex_inputs, vertex_attribs);
    }


    render_constants = device->allocate_memory({
        sizeof(RenderConstants),
        true, false, false, AllocUsage::Uniform
    });
}


void run_render_loop(float a_x = 0, float a_y = -9.8, float a_z = 0) {
        for (int i = 0; i < 4; i++) {
            // get_force(x, f, vertices)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_f, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_vertices, N_CELLS, 4, 1);
            set_ctx_arg_float(host_ctx, 3, a_x);
            set_ctx_arg_float(host_ctx, 4, a_y);
            set_ctx_arg_float(host_ctx, 5, a_z);
            loaded_kernels.get_force_kernel->launch(&host_ctx);
            // get_b(v, b, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_v, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_b, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
            loaded_kernels.get_b_kernel->launch(&host_ctx);

            // matmul_edge(mul_ans, v, edges)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_mul_ans, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_edges, N_EDGES, 2, 1);
            loaded_kernels.matmul_edge_kernel->launch(&host_ctx);
            // add(r0, b, -1, mul_ans)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_b, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 2, -1.0f);
            set_ctx_arg_devalloc(host_ctx, 3, devalloc_mul_ans, N_VERTS, 3, 1);
            loaded_kernels.add_kernel->launch(&host_ctx);
            // ndarray_to_ndarray(p0, r0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
            loaded_kernels.ndarray_to_ndarray_kernel->launch(&host_ctx);
            // dot2scalar(r0, r0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
            loaded_kernels.dot2scalar_kernel->launch(&host_ctx);
            // init_r_2()
            loaded_kernels.init_r_2_kernel->launch(&host_ctx);

            int n_iter = 2;

            for (int i = 0; i < n_iter; i++) {
                // matmul_edge(mul_ans, p0, edges);
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_mul_ans, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 2, devalloc_edges, N_EDGES, 2, 1);
                loaded_kernels.matmul_edge_kernel->launch(&host_ctx);
                // dot2scalar(p0, mul_ans)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_mul_ans, N_VERTS, 3, 1);
                loaded_kernels.dot2scalar_kernel->launch(&host_ctx);
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_alpha_scalar, 1, 1, 1);
                loaded_kernels.update_alpha_kernel->launch(&host_ctx);
          		// add(v, v, alpha, p0)
          		set_ctx_arg_devalloc(host_ctx, 0, devalloc_v, N_VERTS, 3, 1);
          		set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
          		set_ctx_arg_float(host_ctx, 2, 1.0f);
                set_ctx_arg_devalloc(host_ctx, 3, devalloc_alpha_scalar, 1, 1, 1);
          		set_ctx_arg_devalloc(host_ctx, 4, devalloc_p0, N_VERTS, 3, 1);
          		loaded_kernels.add_hack_kernel->launch(&host_ctx);
			    // add(r0, r0, -alpha, mul_ans)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_float(host_ctx, 2, -1.0f);
                set_ctx_arg_devalloc(host_ctx, 3, devalloc_alpha_scalar, 1, 1, 1);
                set_ctx_arg_devalloc(host_ctx, 4, devalloc_mul_ans, N_VERTS, 3, 1);
                loaded_kernels.add_hack_kernel->launch(&host_ctx);

                // r_2_new = dot(r0, r0)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                loaded_kernels.dot2scalar_kernel->launch(&host_ctx);

                set_ctx_arg_devalloc(host_ctx, 0, devalloc_beta_scalar, 1, 1, 1);
                loaded_kernels.update_beta_r_2_kernel->launch(&host_ctx);


                // add(p0, r0, beta, p0)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_float(host_ctx, 2, 1.0f);
                set_ctx_arg_devalloc(host_ctx, 3, devalloc_beta_scalar, 1, 1, 1);
                set_ctx_arg_devalloc(host_ctx, 4, devalloc_p0, N_VERTS, 3, 1);
                loaded_kernels.add_hack_kernel->launch(&host_ctx);
            }

            // fill_ndarray(f, 0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_f, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 1, 0);
            loaded_kernels.fill_ndarray_kernel->launch(&host_ctx);

            // add(x, x, dt, v)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 2, dt);
            set_ctx_arg_devalloc(host_ctx, 3, devalloc_v, N_VERTS, 3, 1);
            loaded_kernels.add_kernel->launch(&host_ctx);
        }
        // floor_bound(x, v)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
        loaded_kernels.floor_bound_kernel->launch(&host_ctx);
        vulkan_runtime->synchronize();
#ifdef ONLY_INIT
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 3, devalloc_ox, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 4, devalloc_vertices, N_CELLS, 4, 1);
        loaded_kernels.init_kernel->launch(&host_ctx);
        print_debug(vulkan_runtime, devalloc_x, 0);
#endif

    // Render elements
    auto stream = device->get_graphics_stream();
    auto cmd_list = stream->new_command_list();
    bool color_clear = true;
    std::vector<float> clear_colors = {0.2, 0.5, 0.8, 1};
    auto image = surface->get_target_image();
    cmd_list->begin_renderpass(
        /*xmin=*/0, /*ymin=*/0, /*xmax=*/width,
        /*ymax=*/height, /*num_color_attachments=*/1, &image,
        &color_clear, &clear_colors, &depth_allocation,
        /*depth_clear=*/true);

    RenderConstants *constants = (RenderConstants *)device->map(render_constants);
    constants->proj = glm::perspective(glm::radians(55.0f), float(width) / float(height), 0.1f, 10.0f);
    constants->proj[1][1] *= -1.0f;
    constants->view = glm::lookAt(glm::vec3(0.0, 1.5, 2.95), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    device->unmap(render_constants);

    // Draw mesh
    {
        auto resource_binder = render_surface_pipeline->resource_binder();
        resource_binder->buffer(0, 0, render_constants.get_ptr(0));
        resource_binder->vertex_buffer(devalloc_x.get_ptr(0));
        resource_binder->index_buffer(devalloc_indices.get_ptr(0), 32);

        cmd_list->bind_pipeline(render_surface_pipeline.get());
        cmd_list->bind_resources(resource_binder);
        cmd_list->draw_indexed(N_FACES * 3);
    }

    // Draw points
    // {
    //     auto resource_binder = render_point_pipeline->resource_binder();
    //     resource_binder->buffer(0, 0, render_constants.get_ptr(0));
    //     resource_binder->vertex_buffer(devalloc_x.get_ptr(0));

    //     cmd_list->bind_pipeline(render_point_pipeline.get());
    //     cmd_list->bind_resources(resource_binder);
    //     cmd_list->draw(N_VERTS);
    // }

    cmd_list->end_renderpass();
    stream->submit_synced(cmd_list.get());

    surface->present_image();
}

void cleanup() {
    device->dealloc_memory(devalloc_x);
    device->dealloc_memory(devalloc_v);
    device->dealloc_memory(devalloc_f);
    device->dealloc_memory(devalloc_mul_ans);
    device->dealloc_memory(devalloc_c2e);
    device->dealloc_memory(devalloc_b);
    device->dealloc_memory(devalloc_r0);
    device->dealloc_memory(devalloc_p0);
    device->dealloc_memory(devalloc_indices);
    device->dealloc_memory(devalloc_vertices);
    device->dealloc_memory(devalloc_edges);
    device->dealloc_memory(devalloc_ox);
    device->dealloc_memory(devalloc_alpha_scalar);
    device->dealloc_memory(devalloc_beta_scalar);

    device->dealloc_memory(render_constants);

    device->destroy_image(depth_allocation);

    vulkan_runtime = nullptr;
    embedded_device = nullptr;
}
