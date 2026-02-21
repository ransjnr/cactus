#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACTUS_CURL_ROOT="${CACTUS_CURL_ROOT:-$PROJECT_ROOT/libs/curl}"
export CACTUS_CURL_ROOT

MODEL_NAME="${1:-LiquidAI/LFM2-VL-450M}"
TRANSCRIBE_MODEL_NAME="${2:-openai/whisper-small}"

echo "Running Cactus Swift wrapper tests on iOS..."
echo "============================================"

if [ ! -d "/Applications/Xcode.app" ]; then
    echo "Xcode not installed"
    exit 1
fi

if ! command -v xcodebuild &>/dev/null; then
    echo "xcodebuild not found"
    exit 1
fi

echo ""
echo "Step 1: Selecting iOS device..."

simulators=$(xcrun simctl list devices available | grep -E "^\s+(iPhone|iPad)" | grep -v "unavailable" | sed 's/^[[:space:]]*//' | while read line; do
    uuid=$(echo "$line" | grep -oE '\([A-F0-9-]{36}\)' | head -1 | tr -d '()')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | sed -E 's/ \([^)]*\)//g' | xargs)
        echo "${name}|simulator|${uuid}"
    fi
done)

xctrace_output=$(xcrun xctrace list devices 2>&1)
physical_devices=$(echo "$xctrace_output" | awk '
    /== Devices ==/ { in_online=1; in_offline=0; next }
    /== Devices Offline ==/ { in_online=0; in_offline=1; next }
    /== Simulators ==/ { exit }
    /00008[A-F0-9]{3}-[A-F0-9]{16}/ {
        if (in_online || in_offline) {
            status = in_offline ? "offline" : ""
            print $0 "|" status
        }
    }
' | while read line; do
    uuid=$(echo "$line" | grep -oE '00008[A-F0-9]{3}-[A-F0-9]{16}')
    status=$(echo "$line" | awk -F'|' '{print $2}')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | awk -F'|' '{print $1}' | sed -E 's/ \([0-9]+\.[0-9]+.*$//' | xargs)
        ios_version=$(echo "$line" | awk -F'|' '{print $1}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        echo "${name} (iOS ${ios_version})|device|${uuid}|${status}"
    fi
done)

all_devices=$(printf "%s\n%s\n" "$physical_devices" "$simulators" | grep -v '^$')

if [ -z "$all_devices" ]; then
    echo "No devices or simulators found"
    exit 1
fi

physical_count=$(echo "$all_devices" | grep -c '|device|' || true)
device_num=0

if [ "$physical_count" -gt 0 ]; then
    echo "Devices:"
    while IFS='|' read -r name type uuid status; do
        if [ "$type" = "device" ]; then
            device_num=$((device_num + 1))
            printf "  %2d. %s\n" "$device_num" "$name"
        fi
    done <<< "$all_devices"
    echo ""
fi

echo "Simulators:"
while IFS='|' read -r name type uuid status; do
    if [ "$type" = "simulator" ]; then
        device_num=$((device_num + 1))
        printf "  %2d. %s\n" "$device_num" "$name"
    fi
done <<< "$all_devices"

echo ""
read -p "Select device number (1-$device_num): " device_number

if ! [[ "$device_number" =~ ^[0-9]+$ ]] || [ "$device_number" -lt 1 ] || [ "$device_number" -gt "$device_num" ]; then
    echo "Invalid selection"
    exit 1
fi

selected_line=$(echo "$all_devices" | sed -n "${device_number}p")
device_name=$(echo "$selected_line" | cut -d'|' -f1)
device_type=$(echo "$selected_line" | cut -d'|' -f2)
device_uuid=$(echo "$selected_line" | cut -d'|' -f3)

echo ""
echo "Selected: $device_name ($device_type)"

if [ "$device_type" = "device" ]; then
    if ! security find-identity -v -p codesigning | grep -q "Apple Development"; then
        echo "No development certificates found. Add your Apple ID in Xcode > Settings > Accounts."
        exit 1
    fi
fi

echo ""
echo "Step 2: Building Cactus static library..."

if ! BUILD_STATIC=true BUILD_XCFRAMEWORK=false "$PROJECT_ROOT/apple/build.sh"; then
    echo "Failed to build Cactus library"
    exit 1
fi

echo ""
echo "Step 3: Configuring Xcode project..."

xcodeproj_path="$SCRIPT_DIR/CactusSwiftTest/CactusSwiftTest.xcodeproj"
bundle_id="com.cactus.nativetest.${USER}"

if [ "$device_type" = "device" ]; then
    static_lib="$PROJECT_ROOT/apple/libcactus-device.a"
    development_team=$(security find-certificate -a -c "Apple Development" -p | openssl x509 -noout -subject | grep -oE 'OU=[A-Z0-9]{10}' | head -1 | cut -d= -f2)
else
    static_lib="$PROJECT_ROOT/apple/libcactus-simulator.a"
    development_team=""
fi

if ! gem list xcodeproj -i &>/dev/null; then
    echo "Installing xcodeproj gem..."
    gem install --user-install xcodeproj || { echo "Failed - try: brew install rbenv && rbenv install && gem install xcodeproj"; exit 1; }
fi

export PROJECT_ROOT XCODEPROJ_PATH="$xcodeproj_path" BUNDLE_ID="$bundle_id" \
       DEVICE_TYPE="$device_type" APPLE_DIR="$PROJECT_ROOT/apple" \
       CACTUS_DIR="$PROJECT_ROOT/cactus" STATIC_LIB="$static_lib" \
       DEVELOPMENT_TEAM="$development_team" CACTUS_CURL_ROOT="$CACTUS_CURL_ROOT"

if ! ruby "$SCRIPT_DIR/configure_xcode.rb"; then
    echo "Failed to configure Xcode project"
    exit 1
fi

echo ""
echo "Step 4: Building app..."

if [ "$device_type" = "simulator" ]; then
    sdk_path=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if ! xcodebuild -project "$xcodeproj_path" \
         -scheme CactusSwiftTest \
         -configuration Release \
         -destination "platform=iOS Simulator,id=$device_uuid" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=14.0 \
         SDKROOT="$sdk_path" \
         PRODUCT_BUNDLE_IDENTIFIER="$bundle_id" \
         build 2>&1 | grep -E "error:|warning:|BUILD FAILED|BUILD SUCCEEDED" | grep -v "^$" | tail -40; then
        echo "Build failed"
        exit 1
    fi
    app_path="$SCRIPT_DIR/build/Build/Products/Release-iphonesimulator/CactusSwiftTest.app"
else
    sdk_path=$(xcrun --sdk iphoneos --show-sdk-path)
    if ! xcodebuild -project "$xcodeproj_path" \
         -scheme CactusSwiftTest \
         -configuration Release \
         -destination "platform=iOS,id=$device_uuid" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         -allowProvisioningUpdates \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=14.0 \
         SDKROOT="$sdk_path" \
         PRODUCT_BUNDLE_IDENTIFIER="$bundle_id" \
         CODE_SIGN_STYLE=Automatic \
         build 2>&1 | grep -E "error:|warning:|BUILD FAILED|BUILD SUCCEEDED" | grep -v "^$" | tail -40; then
        echo "Build failed"
        exit 1
    fi
    app_path="$SCRIPT_DIR/build/Build/Products/Release-iphoneos/CactusSwiftTest.app"
fi

echo ""
echo "Step 5: Copying models and assets to app bundle..."

model_dir=""
if [ -n "$MODEL_NAME" ]; then
    model_dir=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    model_src="$PROJECT_ROOT/weights/$model_dir"
    if [ -d "$model_src" ]; then
        cp -R "$model_src" "$app_path/"
        echo "Copied model: $model_dir"
    else
        echo "Warning: model not found at $model_src"
        model_dir=""
    fi
else
    echo "No LLM model specified (pass as first arg)"
fi

transcribe_dir=""
if [ -n "$TRANSCRIBE_MODEL_NAME" ]; then
    transcribe_dir=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    transcribe_src="$PROJECT_ROOT/weights/$transcribe_dir"
    if [ -d "$transcribe_src" ]; then
        cp -R "$transcribe_src" "$app_path/"
        echo "Copied transcribe model: $transcribe_dir"
        # Inject VAD into transcribe model bundle if needed
        if [[ "$transcribe_dir" == *"whisper"* ]] || [[ "$transcribe_dir" == *"moonshine"* ]]; then
            vad_src="$PROJECT_ROOT/weights/silero-vad"
            if [ -d "$vad_src" ] && [ ! -d "$app_path/$transcribe_dir/vad" ]; then
                mkdir -p "$app_path/$transcribe_dir/vad"
                rsync -a "$vad_src/" "$app_path/$transcribe_dir/vad/"
                echo "Injected VAD into transcribe model bundle"
            fi
        fi
    else
        echo "Warning: transcribe model not found at $transcribe_src"
        transcribe_dir=""
    fi
fi

assets_dir=""
assets_src="$PROJECT_ROOT/tests/assets"
if [ -d "$assets_src" ]; then
    cp -R "$assets_src" "$app_path/"
    assets_dir="assets"
    echo "Copied test assets"
fi

echo ""
echo "Step 6: Running tests..."
echo "------------------------"

if [ "$device_type" = "simulator" ]; then
    xcrun simctl boot "$device_uuid" 2>/dev/null || true
    xcrun simctl install "$device_uuid" "$app_path"

    sim_env=(
        "SIMCTL_CHILD_CACTUS_TEST_MODEL=$model_dir"
        "SIMCTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL=$transcribe_dir"
        "SIMCTL_CHILD_CACTUS_TEST_ASSETS=$assets_dir"
    )
    env "${sim_env[@]}" xcrun simctl launch --console-pty "$device_uuid" "$bundle_id"
else
    xcrun devicectl device install app --device "$device_uuid" "$app_path"

    launch_output=$(env \
        "DEVICECTL_CHILD_CACTUS_TEST_MODEL=$model_dir" \
        "DEVICECTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL=$transcribe_dir" \
        "DEVICECTL_CHILD_CACTUS_TEST_ASSETS=$assets_dir" \
        xcrun devicectl device process launch --device "$device_uuid" "$bundle_id" 2>&1) || true
    echo "$launch_output"

    max_wait=120
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if xcrun devicectl device info processes --device "$device_uuid" | grep -q "CactusSwiftTest"; then
            sleep 2; elapsed=$((elapsed + 2))
        else
            break
        fi
    done

    temp_log=$(mktemp)
    if xcrun devicectl device copy from \
        --device "$device_uuid" \
        --source "Documents/cactus_test.log" \
        --destination "$temp_log" \
        --domain-type appDataContainer \
        --domain-identifier "$bundle_id" 2>/dev/null; then
        cat "$temp_log"
    fi
    rm -f "$temp_log"
fi
