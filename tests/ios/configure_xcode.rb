#!/usr/bin/env ruby
require 'xcodeproj'

project_root = ENV['PROJECT_ROOT']
tests_root = ENV['TESTS_ROOT']
cactus_root = ENV['CACTUS_ROOT']
apple_root = ENV['APPLE_ROOT']
project_path = ENV['XCODEPROJ_PATH']

project = Xcodeproj::Project.open(project_path)
target = project.targets.first

# Create Tests group pointing to actual test directory
tests_group = project.main_group.find_subpath('Tests', true)
tests_group.set_path(tests_root)
tests_group.set_source_tree('<absolute>')

# Map of test files to their renamed main functions
test_files = {
  'test_kernel.cpp' => 'test_kernel_main',
  'test_graph.cpp' => 'test_graph_main',
  'test_kv_cache.cpp' => 'test_kv_cache_main',
  'test_engine.cpp' => 'test_engine_main',
  'test_performance.cpp' => 'test_performance_main',
  'test_utils.cpp' => nil
}

# Add test files directly from their original location
test_files.each do |filename, renamed_main|
  file_path = File.join(tests_root, filename)

  existing_file = tests_group.files.find { |f| f.path == filename || f.real_path.to_s == file_path }

  if existing_file
    file_ref = existing_file
  else
    file_ref = tests_group.new_reference(file_path)
    file_ref.set_source_tree('<absolute>')
  end

  build_file = target.source_build_phase.files.find { |bf| bf.file_ref == file_ref }
  build_file = target.source_build_phase.add_file_reference(file_ref) unless build_file

  if renamed_main
    build_file.settings = { 'COMPILER_FLAGS' => "-Dmain=#{renamed_main}" }
  end
end

# Add test_utils.h
test_utils_h = File.join(tests_root, 'test_utils.h')
unless tests_group.files.any? { |f| f.path == 'test_utils.h' }
  file_ref = tests_group.new_reference(test_utils_h)
  file_ref.set_source_tree('<absolute>')
end

# Add Cactus source files group
cactus_sources_group = project.main_group.find_subpath('CactusSources', true)
cactus_sources_group.set_source_tree('<group>')

# Add all cactus source files
['engine', 'graph', 'kernel', 'ffi', 'models'].each do |subdir|
  subdir_path = File.join(cactus_root, subdir)
  next unless Dir.exist?(subdir_path)

  Dir.glob(File.join(subdir_path, '*.cpp')).each do |source_file|
    filename = File.basename(source_file)

    existing_file = cactus_sources_group.files.find { |f| f.real_path.to_s == source_file }

    unless existing_file
      file_ref = cactus_sources_group.new_reference(source_file)
      file_ref.set_source_tree('<absolute>')

      build_file = target.source_build_phase.files.find { |bf| bf.file_ref == file_ref }
      target.source_build_phase.add_file_reference(file_ref) unless build_file
    end
  end
end

# Configure build settings
target.build_configurations.each do |config|
  config.build_settings['HEADER_SEARCH_PATHS'] ||= ['$(inherited)']
  config.build_settings['HEADER_SEARCH_PATHS'] << tests_root unless config.build_settings['HEADER_SEARCH_PATHS'].include?(tests_root)
  config.build_settings['HEADER_SEARCH_PATHS'] << cactus_root unless config.build_settings['HEADER_SEARCH_PATHS'].include?(cactus_root)

  # Add subdirectory paths for headers
  ['graph', 'engine', 'kernel', 'ffi', 'models'].each do |subdir|
    subdir_path = File.join(cactus_root, subdir)
    config.build_settings['HEADER_SEARCH_PATHS'] << subdir_path unless config.build_settings['HEADER_SEARCH_PATHS'].include?(subdir_path)
  end

  config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++20'
  config.build_settings['CLANG_CXX_LIBRARY'] = 'libc++'

  # Set the deployment target
  config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'

  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'

  # Add compiler flags
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] ||= ['$(inherited)']
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-DPLATFORM_CPU_ONLY=1'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-D__ARM_NEON=1'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-D__ARM_FEATURE_DOTPROD=1'
  # Enable ARM features at compiler level (match CMake: armv8.2-a+fp16+simd+dotprod)
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-march=armv8.2-a+fp16+simd+dotprod'

  # Optimization flags (match CMake)
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-O3'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-DNDEBUG'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-fomit-frame-pointer'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-funroll-loops'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-ftree-vectorize'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-fvisibility=hidden'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-fvisibility-inlines-hidden'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-ffunction-sections'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-fdata-sections'

  # Linker flags (match CMake)
  config.build_settings['OTHER_LDFLAGS'] ||= ['$(inherited)']
  config.build_settings['OTHER_LDFLAGS'] << '-Wl,-dead_strip'

  # Enable LTO (Link Time Optimization)
  config.build_settings['LLVM_LTO'] = 'YES'
end

project.save
puts "Xcode project configured"
