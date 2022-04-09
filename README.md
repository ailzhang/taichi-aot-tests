# Taichi AOT Demo: implicit FEM

This repo includes Taichi AOT demo for implicit FEM.

```
python/  # Taichi python script for implicit_fem
desktop/  # desktop demo
android/  # android demo
shaders/
  aot/  # generated by `python implicit_fem.py --aot` in python/
  render/  # shaders used in rendering
```

## Desktop Demo
```
export TAICHI_REPO_DIR=/path/github/taichi/
cd desktop
mkdir build
cd build && cmake .. && make
```

Taichi built with

```
python setup.py clean && TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=OFF -DTI_EXPORT_CORE:BOOL=ON -DTI_WITH_LLVM:BOOL=OFF" python3 setup.py build_ext
```

## Android Demo
If you are building Taichi with custom changes, make sure to override the prebuilt binaries in: `app/src/main/jniLibs/arm64-v8a/`
```
export TAICHI_REPO_DIR=/path/github/taichi/
cd android
./gradlew assembleDebug
adb install ./app/build/outputs/apk/debug/app-debug.apk
adb push ../shaders /data/local/tmp/
adb shell chmod -R 777 /data/local/tmp/
```

Taichi built with
```
python setup.py clean && TAICHI_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_SDK_ROOT}/ndk/22.1.7171670/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=29 -DANDROID_ABI=arm64-v8a -DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=OFF -DTI_EXPORT_CORE:BOOL=ON" python3 setup.py build_ext
```

## Troubleshooting

- Note that if you built for the desktop demo first and then decide to cross-compile to android, the manual cleanup (`python setup.py clean`) is required. CMake cannot automatically adjust build directory from one build type to another. More details can be found at [this stackoverflow question](https://stackoverflow.com/questions/40528254/how-do-i-detect-that-i-am-cross-compiling-in-cmakelists-txt).

- Monitor Android logs

```
adb logcat -s "TaichiTest" "*:F" "AndroidRuntime" vulkan taichi VALIDATION
```
