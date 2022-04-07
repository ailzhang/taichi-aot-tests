%.frag.spv : %.frag
	glslc --target-env=vulkan1.1 $< -o $@

%.vert.spv : %.vert
	glslc --target-env=vulkan1.1 $< -o $@

all: render_point.frag.spv render_point.vert.spv surface.frag.spv surface.vert.spv
