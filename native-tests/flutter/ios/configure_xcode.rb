#!/usr/bin/env ruby
require 'xcodeproj'

def fail_with(message)
  STDERR.puts "Error: #{message}"
  exit 1
end

xcodeproj_path = ENV['XCODEPROJ_PATH'] or fail_with("XCODEPROJ_PATH not set")
static_lib     = ENV['STATIC_LIB']     or fail_with("STATIC_LIB not set")
curl_root      = ENV['CACTUS_CURL_ROOT']

fail_with("Xcode project not found at: #{xcodeproj_path}") unless File.exist?(xcodeproj_path)
fail_with("Static library not found at: #{static_lib}") unless File.exist?(static_lib)

project = Xcodeproj::Project.open(xcodeproj_path) rescue fail_with("Failed to open Xcode project")
target  = project.targets.find { |t| t.name == 'Runner' } or fail_with("Runner target not found")

vendored_curl_lib = nil
if curl_root && !curl_root.empty?
  device_type = ENV['DEVICE_TYPE'] || 'simulator'
  curl_candidate = device_type == 'simulator' ?
    File.join(curl_root, 'ios', 'simulator', 'libcurl.a') :
    File.join(curl_root, 'ios', 'device', 'libcurl.a')
  vendored_curl_lib = curl_candidate if File.exist?(curl_candidate)
end

target.build_configurations.each do |config|
  config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '14.0'

  ldflags = ['$(inherited)', '-force_load', static_lib,
    '-framework CoreML', '-framework Foundation',
    '-framework Accelerate', '-framework Security',
    '-framework SystemConfiguration', '-framework CFNetwork'
  ]
  ldflags << vendored_curl_lib if vendored_curl_lib
  config.build_settings['OTHER_LDFLAGS'] = ldflags
  config.build_settings['EXCLUDED_ARCHS[sdk=iphonesimulator*]'] = 'i386 x86_64'
end

project.save rescue fail_with("Failed to save Xcode project")
puts "Flutter iOS project configured successfully"
