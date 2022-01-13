#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_window_jni.h>
#include <android/sensor.h>
#include <jni.h>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>

#include <taichi/ui/backends/vulkan/renderer.h>

#include <assert.h>
#include <map>
#include <stdint.h>
#include <vector>

#define ALOGI(fmt, ...)                                                  \
  ((void)__android_log_print(ANDROID_LOG_INFO, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))
#define ALOGE(fmt, ...)                                                   \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))

std::vector<std::string> get_required_instance_extensions() {
  std::vector<std::string> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  return extensions;
}

#define NR_PARTICLES 8192

std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
taichi::lang::vulkan::VkRuntime::KernelHandle init_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle substep_kernel_handle;
taichi::ui::vulkan::Renderer *renderer;
taichi::ui::vulkan::Gui *gui;
taichi::lang::DeviceAllocation dalloc_circles;

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env,
                                                        jclass,
                                                        jobject assets,
                                                        jobject surface) {
  ANativeWindow *native_window = ANativeWindow_fromSurface(env, surface);
  taichi::lang::vulkan::AotModuleLoaderImpl aot_loader("/data/local/tmp/mpm88");

  // Initialize our Vulkan Program pipeline
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;

  // Create a memory pool to allocate GPU memory
  memory_pool = std::make_unique<taichi::lang::MemoryPool>(
      taichi::lang::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "MPM88";
  app_config.width = ANativeWindow_getWidth(native_window);
  app_config.height = ANativeWindow_getHeight(native_window);
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.package_path = "/data/local/tmp/";  // Use CacheDir()
  app_config.ti_arch = taichi::lang::Arch::vulkan;
  renderer = new taichi::ui::vulkan::Renderer();
  renderer->init(native_window, app_config);

  // Create a GUI even though it's not used in our case (required to
  // render the renderer)
  gui = new taichi::ui::vulkan::Gui(&renderer->app_context(),
                                    &renderer->swap_chain(), native_window);

  // Create the Vk Runtime
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = &renderer->app_context().device();
  vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // Retrieve kernels/fields/etc from AOT module so we can initialize our
  // runtime
  taichi::lang::vulkan::VkRuntime::RegisterParams init_kernel, substep_kernel;
  bool ret = aot_loader.get_kernel("init", init_kernel);
  if (!ret) {
    ALOGE("Cannot find 'init' kernel\n");
    return;
  }
  ret = aot_loader.get_kernel("substep", substep_kernel);
  if (!ret) {
    ALOGE("Cannot find 'substep' kernel\n");
    return;
  }
  auto root_size = aot_loader.get_root_size();
  ALOGI("root buffer size=%d\n", root_size);

  vulkan_runtime->add_root_buffer(root_size);
  init_kernel_handle = vulkan_runtime->register_taichi_kernel(init_kernel);
  substep_kernel_handle =
      vulkan_runtime->register_taichi_kernel(substep_kernel);

  // Allocate memory for Circles position
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.size = NR_PARTICLES * sizeof(taichi::ui::Vertex);
  alloc_params.host_write = true;
  alloc_params.host_read = true;
  dalloc_circles =
      renderer->app_context().device().allocate_memory(alloc_params);

  //
  // Run MPM88 from AOT module similar to Python code
  //
  vulkan_runtime->launch_kernel(init_kernel_handle, &host_ctx);

#if 0
  // Sanity check to make sure the shaders are running properly, we should have
  // the same float values as the python scripts aot->get_field("x");
  float x[10];
  vulkan_runtime->synchronize();
  vulkan_runtime->read_memory((uint8_t *)x, 0, 5 * 2 * sizeof(taichi::float32));

  for (int i = 0; i < 10; i += 2) {
    ALOGI("[%f, %f]\n", x[i], x[i + 1]);
  }
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_destroy(JNIEnv *env,
                                                           jclass,
                                                           jobject surface) {
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_pause(JNIEnv *env,
                                                         jclass,
                                                         jobject surface) {
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resume(JNIEnv *env,
                                                          jclass,
                                                          jobject surface) {
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resize(JNIEnv *,
                                                          jclass,
                                                          jobject,
                                                          jint width,
                                                          jint height) {
  ALOGI("Resize requested for %dx%d", width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_render(JNIEnv *env,
                                                          jclass,
                                                          jobject surface) {
  taichi::lang::RuntimeContext host_ctx;
  float x[NR_PARTICLES * 2];

  // Clear the background
  renderer->set_background_color({0.6, 0.6, 0.6});

  // timer starts before launch kernel
  auto start = std::chrono  ::steady_clock::now();

  // Run 'substep' 50 times
  for (int i = 0; i < 50; i++) {
    vulkan_runtime->launch_kernel(substep_kernel_handle, &host_ctx);
  }

  // Make sure to sync the GPU memory so we can read the latest update from CPU
  // And read the 'x' calculated on GPU to our local variable
  // @TODO: Skip this with support of NdArray as we will be able to specify 'dalloc_circles'
  // in the host_ctx before running the kernel, allowing the data to be updated automatically
  vulkan_runtime->synchronize();

  // timer ends after synchronization
  auto end = std::chrono::steady_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ALOGI("Execution time is %" PRId64 "ns\n", cpu_time);

  vulkan_runtime->read_memory((uint8_t *)x, 0,
                              NR_PARTICLES * 2 * sizeof(taichi::float32));

  // Copy the results from the kernel to our DeviceAlloc so it can be used to
  // render circle
  taichi::ui::Vertex *vs_buffer =
      (taichi::ui::Vertex *)renderer->app_context().device().map(
          dalloc_circles);
  for (int i = 0, j = 0; i < NR_PARTICLES; i++, j += 2) {
    vs_buffer[i].pos = {x[j], x[j + 1], 0.0};
  }
  renderer->app_context().device().unmap(dalloc_circles);

  // Describe information to render the circle with Vulkan
  taichi::ui::FieldInfo f_info;
  f_info.valid = true;
  f_info.field_type = taichi::ui::FieldType::Scalar;
  f_info.matrix_rows = 1;
  f_info.matrix_cols = 1;
  f_info.shape = {NR_PARTICLES};
  f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
  f_info.dtype = taichi::lang::PrimitiveType::f32;
  f_info.snode = nullptr;
  f_info.dev_alloc = dalloc_circles;

  taichi::ui::CirclesInfo circles;
  circles.renderable_info.has_per_vertex_color = false;
  circles.renderable_info.vbo = f_info;
  circles.color = {0.6, 0, 1};
  circles.radius = 0.0015f;

  // Render the UI
  renderer->circles(circles);
  renderer->draw_frame(gui);
  renderer->swap_chain().surface().present_image();
  renderer->prepare_for_next_frame();
}
