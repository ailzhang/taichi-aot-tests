#include <signal.h>
#include <iostream>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>
#include <taichi/inc/constants.h>
#include <inttypes.h>

#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>

#define NR_PARTICLES 512
// Uncomment this line to show explicit fem deomo
//#define USE_EXPLICIT

std::vector<std::string> get_required_instance_extensions() {
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  return extensions;
}

void set_ctx_arg_devalloc(taichi::lang::RuntimeContext &host_ctx, int arg_id, taichi::lang::DeviceAllocation& alloc) {
  host_ctx.set_arg(arg_id, &alloc);
  host_ctx.set_device_allocation(arg_id, true);
  // This is hack since our ndarrays happen to have exactly the same size in implicit_fem demo.
  host_ctx.extra_args[arg_id][0] = 512;
  host_ctx.extra_args[arg_id][1] = 3;
  host_ctx.extra_args[arg_id][2] = 1;
}

// TODO: provide a proper API from taichi
void set_ctx_arg_float(taichi::lang::RuntimeContext &host_ctx, int arg_id, float x) {
  host_ctx.set_arg(arg_id, x);
  host_ctx.set_device_allocation(arg_id, false);
}

#include <unistd.h>
int main() {
    // Init gl window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    float dt = 1e-2;

    // Create a GGUI configuration
    taichi::ui::AppConfig app_config;
    app_config.name         = "FEM";
    app_config.width        = 512;
    app_config.height       = 512;
    app_config.vsync        = true;
    app_config.show_window  = false;
    app_config.package_path = "../"; // make it flexible later
    app_config.ti_arch      = taichi::Arch::vulkan;

    // // Create GUI & renderer
    auto renderer  = std::make_unique<taichi::ui::vulkan::Renderer>();
    renderer->init(nullptr, window, app_config);

    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = &(renderer->app_context().device());
    auto vulkan_runtime = taichi::lang::vulkan::VkRuntime(std::move(params));

    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
#ifdef USE_EXPLICIT
    std::string shader_source = "../explicit_fem";
#else
    std::string shader_source = "../implicit_fem";
#endif
    taichi::lang::vulkan::AotModuleParams aot_params{shader_source, &vulkan_runtime};
    auto module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
    auto root_size = module->get_root_size();
    printf("root buffer size=%ld\n", root_size);
    vulkan_runtime.add_root_buffer(root_size);

    auto init_kernel = module->get_kernel("init");
    auto get_vertices_kernel = module->get_kernel("get_vertices");
    auto get_indices_kernel = module->get_kernel("get_indices");
    auto get_force_kernel = module->get_kernel("get_force");
    auto advect_kernel = module->get_kernel("advect");
    auto floor_bound_kernel = module->get_kernel("floor_bound");
    auto get_b_kernel = module->get_kernel("get_b");
    auto matmul_cell_kernel = module->get_kernel("matmul_cell");
    auto add_kernel = module->get_kernel("add");
    auto ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
    auto dot_kernel = module->get_kernel("dot");
    //auto add_ndarray_kernel = module->get_kernel("add_ndarray");
    //auto add2_kernel = module->get_kernel("add2");
    auto fill_ndarray_kernel = module->get_kernel("fill_ndarray");

    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    constexpr int kVecSize = 3;
    alloc_params.size = NR_PARTICLES * kVecSize * sizeof(float);
    // x
    taichi::lang::DeviceAllocation devalloc_x = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // v
    taichi::lang::DeviceAllocation devalloc_v = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // b
    taichi::lang::DeviceAllocation devalloc_b = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // mul_ans
    taichi::lang::DeviceAllocation devalloc_mul_ans = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // r0
    taichi::lang::DeviceAllocation devalloc_r0 = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // p0
    taichi::lang::DeviceAllocation devalloc_p0 = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // f
    taichi::lang::DeviceAllocation devalloc_f = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);

    taichi::lang::RuntimeContext host_ctx;
    memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
    host_ctx.result_buffer = result_buffer;

    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);

    set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
    set_ctx_arg_devalloc(host_ctx, 1, devalloc_v);
    set_ctx_arg_devalloc(host_ctx, 2, devalloc_f);
    // get_vertices()
    get_vertices_kernel->launch(&host_ctx);
    // init(x, v, f)
    init_kernel->launch(&host_ctx);
    // get_indices(x)
    get_indices_kernel->launch(&host_ctx);
    vulkan_runtime.synchronize();

    // Describe information to render the circle with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Scalar;
    f_info.matrix_rows  = 1;
    f_info.matrix_cols  = 1;
    f_info.shape        = {NR_PARTICLES};
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = devalloc_x;
    taichi::ui::CirclesInfo circles;
    circles.renderable_info.has_per_vertex_color = false;
    circles.renderable_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
    circles.renderable_info.vbo                  = f_info;
    circles.color                                = {0.8, 0.4, 0.1};
    circles.radius                               = 0.005f; // 0.0015f looks unclear on desktop

    renderer->set_background_color({0.6, 0.6, 0.6});

    while (!glfwWindowShouldClose(window)) {
        // Run 'substep' 40 times
        // Explicit FEM
#ifdef USE_EXPLICIT
        for (int i = 0; i < 40; i++) {
            // get_force(x, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_f);
            get_force_kernel->launch(&host_ctx);
            // advect(x, v, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_v);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_f);
            advect_kernel->launch(&host_ctx);
        }
#else
        // get_force(x, f)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_f);
        get_force_kernel->launch(&host_ctx);
        // get_b(v, b, f)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_v);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_b);
        set_ctx_arg_devalloc(host_ctx, 2, devalloc_f);
        get_b_kernel->launch(&host_ctx);

        // matmul_cell(v, mul_ans)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_v);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_mul_ans);
        matmul_cell_kernel->launch(&host_ctx);

        // add(r0, b, -1, mul_ans)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_b);
        set_ctx_arg_float(host_ctx, 2, -1);
        set_ctx_arg_devalloc(host_ctx, 3, devalloc_mul_ans);
        add_kernel->launch(&host_ctx);

        // ndarray_to_ndarray(p0, r0)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0);
        ndarray_to_ndarray_kernel->launch(&host_ctx);

        // r_2 = dot(r0, r0)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0);
        dot_kernel->launch(&host_ctx);
        float r_2 = host_ctx.get_ret<float>(0);

        int n_iter = 8;
        float epsilon = 1e-6;
        float r_2_init = r_2;
        float r_2_new = r_2;

        for (int i = 0; i < n_iter; ++i) {
          // matmul_cell(p0, mul_ans)
          set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0);
          set_ctx_arg_devalloc(host_ctx, 1, devalloc_mul_ans);
          matmul_cell_kernel->launch(&host_ctx);

          // alpha = r_2_new / dot(p0, mul_ans)
          dot_kernel->launch(&host_ctx);
          vulkan_runtime.synchronize();
          float alpha = r_2_new / host_ctx.get_ret<float>(0);

          // add(v, v, alpha, p0)
          set_ctx_arg_devalloc(host_ctx, 0, devalloc_v);
          set_ctx_arg_devalloc(host_ctx, 1, devalloc_v);
          set_ctx_arg_float(host_ctx, 2, alpha);
          set_ctx_arg_devalloc(host_ctx, 3, devalloc_p0);
          add_kernel->launch(&host_ctx);

          // add(r0, r0, -alpha, mul_ans)
          set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0);
          set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0);
          set_ctx_arg_float(host_ctx, 2, -alpha);
          set_ctx_arg_devalloc(host_ctx, 3, devalloc_mul_ans);
          add_kernel->launch(&host_ctx);

          r_2 = r_2_new;
          // r_2_new = dot(r0, r0);
          set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0);
          set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0);
          dot_kernel->launch(&host_ctx);
          vulkan_runtime.synchronize();
          r_2_new = host_ctx.get_ret<float>(0);

          // if r_2_new <= r_2_init * epsilon ** 2: break
          if (r_2_new <= r_2_init * epsilon * epsilon) {break;}

          float beta = r_2_new / r_2;

          // add(p0, r0, beta, p0)
          set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0);
          set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0);
          set_ctx_arg_float(host_ctx, 2, beta);
          set_ctx_arg_devalloc(host_ctx, 3, devalloc_p0);
          add_kernel->launch(&host_ctx);
        }
        // fill_ndarray(f, 0)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_f);
        set_ctx_arg_float(host_ctx, 1, 0);
        fill_ndarray_kernel->launch(&host_ctx);

        // add(x, x, dt, v)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_x);
        set_ctx_arg_float(host_ctx, 2, dt);
        set_ctx_arg_devalloc(host_ctx, 3, devalloc_v);
        add_kernel->launch(&host_ctx);
#endif
        // floor_bound(x, v)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_v);
        floor_bound_kernel->launch(&host_ctx);
        vulkan_runtime.synchronize();

        // Render elements
        renderer->circles(circles);
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}
