#!/usr/bin/env ruby
require 'xcodeproj'

def fail_with(message)
  STDERR.puts "Error: #{message}"
  exit 1
end

project_root   = ENV['PROJECT_ROOT']   or fail_with("PROJECT_ROOT not set")
xcodeproj_path = ENV['XCODEPROJ_PATH'] or fail_with("XCODEPROJ_PATH not set")
bundle_id      = ENV['BUNDLE_ID']      or fail_with("BUNDLE_ID not set")
device_type    = ENV['DEVICE_TYPE']    or fail_with("DEVICE_TYPE not set")
apple_dir      = ENV['APPLE_DIR']      or fail_with("APPLE_DIR not set")
cactus_dir     = ENV['CACTUS_DIR']     or fail_with("CACTUS_DIR not set")
static_lib     = ENV['STATIC_LIB']     or fail_with("STATIC_LIB not set")
team_id        = ENV['DEVELOPMENT_TEAM']

fail_with("Xcode project not found at: #{xcodeproj_path}") unless File.exist?(xcodeproj_path)
fail_with("Static library not found at: #{static_lib}") unless File.exist?(static_lib)

project = Xcodeproj::Project.open(xcodeproj_path) rescue fail_with("Failed to open Xcode project")
target  = project.targets.first or fail_with("No targets found")

# Remove any previously added CactusLib group to keep this idempotent
existing = project.main_group.groups.find { |g| g.name == 'CactusLib' }
if existing
  existing.files.each do |f|
    bf = target.source_build_phase.files.find { |b| b.file_ref == f }
    target.source_build_phase.files.delete(bf) if bf
  end
  existing.remove_from_project
end

# Add Cactus.swift from apple/ as a source file reference
cactus_swift_path = File.join(apple_dir, 'Cactus.swift')
fail_with("Cactus.swift not found at: #{cactus_swift_path}") unless File.exist?(cactus_swift_path)

lib_group = project.main_group.new_group('CactusLib', apple_dir, '<absolute>')
ref = lib_group.new_reference(cactus_swift_path)
ref.set_source_tree('<absolute>')
target.source_build_phase.add_file_reference(ref)
puts "Added Cactus.swift from #{cactus_swift_path}"

curl_root = ENV['CACTUS_CURL_ROOT']
vendored_curl_lib = nil
if curl_root && !curl_root.empty?
  vendored_curl_lib = device_type == 'simulator' ?
    File.join(curl_root, 'ios', 'simulator', 'libcurl.a') :
    File.join(curl_root, 'ios', 'device', 'libcurl.a')
  vendored_curl_lib = nil unless File.exist?(vendored_curl_lib.to_s)
end

target.build_configurations.each do |config|
  # Swift needs the module map from apple/ to resolve `import cactus`
  config.build_settings['SWIFT_INCLUDE_PATHS'] = ['$(inherited)', apple_dir]

  # C headers for the module map's umbrella header (cactus_ffi.h)
  config.build_settings['HEADER_SEARCH_PATHS'] = [
    '$(inherited)',
    cactus_dir,
    File.join(cactus_dir, 'ffi')
  ]

  config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
  config.build_settings['SWIFT_VERSION'] = '5.0'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = bundle_id
  config.build_settings['DEVELOPMENT_TEAM'] = team_id if team_id

  # Link static library and required Apple frameworks
  ldflags = ['$(inherited)', static_lib,
    '-framework CoreML', '-framework Foundation',
    '-framework Accelerate', '-framework Security',
    '-framework SystemConfiguration', '-framework CFNetwork'
  ]
  ldflags << vendored_curl_lib if vendored_curl_lib
  config.build_settings['OTHER_LDFLAGS'] = ldflags
end

project.save rescue fail_with("Failed to save Xcode project")
puts "Xcode project configured successfully"
