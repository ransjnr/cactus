#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACTUS_CURL_ROOT="${CACTUS_CURL_ROOT:-$PROJECT_ROOT/libs/curl}"

MODEL_NAME="${1:-LiquidAI/LFM2-VL-450M}"
TRANSCRIBE_MODEL_NAME="${2:-openai/whisper-small}"
PLATFORM="${3:-}"  # optional: android or ios

echo "Running Cactus Flutter wrapper tests..."
echo "======================================="

if ! command -v flutter &>/dev/null; then
    echo "flutter not found - install with: brew install flutter"
    exit 1
fi

model_dir=""
if [ -n "$MODEL_NAME" ]; then
    model_dir=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    model_src="$PROJECT_ROOT/weights/$model_dir"
fi

transcribe_dir=""
if [ -n "$TRANSCRIBE_MODEL_NAME" ]; then
    transcribe_dir=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    transcribe_src="$PROJECT_ROOT/weights/$transcribe_dir"
fi

assets_src="$PROJECT_ROOT/tests/assets"

# ---------------------------------------------------------------
# Android
# ---------------------------------------------------------------

if [ "$PLATFORM" = "android" ]; then
    echo ""
    echo "Step 1: Building Android native library..."
    if ! "$PROJECT_ROOT/android/build.sh"; then
        echo "Failed to build Android library"
        exit 1
    fi
    mkdir -p "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a"
    cp "$PROJECT_ROOT/android/libcactus.so" "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a/"
    echo "Copied libcactus.so"

    echo ""
    echo "Step 2: Copying cactus.dart wrapper..."
    ruby "$SCRIPT_DIR/patch_cactus_dart.rb" "$PROJECT_ROOT/flutter/cactus.dart" "$SCRIPT_DIR/lib/cactus.dart"
    echo "Patched cactus.dart (missing Utf8 + non-const malloc)"

    echo ""
    echo "Step 3: Getting Flutter packages..."
    cd "$SCRIPT_DIR" && flutter pub get && cd - >/dev/null

    echo ""
    echo "Step 4: Pushing models and assets to device..."
    DEVICE_DATA_DIR="/sdcard/Android/data/com.cactus.cactus_flutter_test/files/cactus_test"
    android_model_path=""
    android_transcribe_path=""
    android_assets_path=""

    if [ -n "$model_dir" ] && [ -d "$model_src" ]; then
        adb shell mkdir -p "$DEVICE_DATA_DIR/$model_dir"
        adb push "$model_src/." "$DEVICE_DATA_DIR/$model_dir/"
        echo "Pushed model: $model_dir"
        android_model_path="$DEVICE_DATA_DIR/$model_dir"
    else
        [ -n "$MODEL_NAME" ] && echo "Warning: model not found at $model_src"
    fi

    if [ -n "$transcribe_dir" ] && [ -d "$transcribe_src" ]; then
        adb shell mkdir -p "$DEVICE_DATA_DIR/$transcribe_dir"
        adb push "$transcribe_src/." "$DEVICE_DATA_DIR/$transcribe_dir/"
        echo "Pushed transcribe model: $transcribe_dir"
        if [[ "$transcribe_dir" == *"whisper"* ]] || [[ "$transcribe_dir" == *"moonshine"* ]]; then
            vad_src="$PROJECT_ROOT/weights/silero-vad"
            if [ -d "$vad_src" ]; then
                adb shell mkdir -p "$DEVICE_DATA_DIR/$transcribe_dir/vad"
                adb push "$vad_src/." "$DEVICE_DATA_DIR/$transcribe_dir/vad/"
                echo "Pushed VAD into transcribe model"
            fi
        fi
        android_transcribe_path="$DEVICE_DATA_DIR/$transcribe_dir"
    else
        [ -n "$TRANSCRIBE_MODEL_NAME" ] && echo "Warning: transcribe model not found at $transcribe_src"
    fi

    if [ -d "$assets_src" ]; then
        adb shell mkdir -p "$DEVICE_DATA_DIR/assets"
        adb push "$assets_src/." "$DEVICE_DATA_DIR/assets/"
        echo "Pushed test assets"
        android_assets_path="$DEVICE_DATA_DIR/assets"
    fi

    echo ""
    echo "Step 5: Running on Android..."
    flutter run \
        "--dart-define=CACTUS_TEST_MODEL=$android_model_path" \
        "--dart-define=CACTUS_TEST_TRANSCRIBE_MODEL=$android_transcribe_path" \
        "--dart-define=CACTUS_TEST_ASSETS=$android_assets_path" \
        -d android

# ---------------------------------------------------------------
# iOS
# ---------------------------------------------------------------

elif [ "$PLATFORM" = "ios" ]; then
    if [ ! -d "/Applications/Xcode.app" ]; then
        echo "Xcode not installed"
        exit 1
    fi

    echo ""
    echo "Step 1: Building Cactus static library..."
    if ! BUILD_STATIC=true BUILD_XCFRAMEWORK=false "$PROJECT_ROOT/apple/build.sh"; then
        echo "Failed to build Cactus library"
        exit 1
    fi

    echo ""
    echo "Step 2: Selecting simulator..."
    sim_uuid=$(xcrun simctl list devices available | grep -E "^\s+iPhone" | grep -v "unavailable" | \
        grep -oE '\([A-F0-9-]{36}\)' | head -1 | tr -d '()')
    if [ -z "$sim_uuid" ]; then
        echo "No available iPhone simulator found"
        exit 1
    fi
    sim_name=$(xcrun simctl list devices available | grep "$sim_uuid" | sed -E 's/ \([^)]*\)//g' | xargs)
    echo "Using simulator: $sim_name ($sim_uuid)"

    echo ""
    echo "Step 3: Configuring Xcode project..."
    xcodeproj_path="$SCRIPT_DIR/ios/Runner.xcodeproj"
    if ! gem list xcodeproj -i &>/dev/null; then
        gem install --user-install xcodeproj || { echo "Failed - try: gem install xcodeproj"; exit 1; }
    fi
    export XCODEPROJ_PATH="$xcodeproj_path" \
           STATIC_LIB="$PROJECT_ROOT/apple/libcactus-simulator.a" \
           DEVICE_TYPE="simulator" \
           CACTUS_CURL_ROOT="$CACTUS_CURL_ROOT"
    if ! ruby "$SCRIPT_DIR/ios/configure_xcode.rb"; then
        echo "Failed to configure Xcode project"
        exit 1
    fi

    echo ""
    echo "Step 4: Copying cactus.dart wrapper..."
    ruby "$SCRIPT_DIR/patch_cactus_dart.rb" "$PROJECT_ROOT/flutter/cactus.dart" "$SCRIPT_DIR/lib/cactus.dart"
    echo "Patched cactus.dart (missing Utf8 + non-const malloc)"

    echo ""
    echo "Step 5: Getting Flutter packages..."
    cd "$SCRIPT_DIR" && flutter pub get && cd - >/dev/null

    echo ""
    echo "Step 6: Staging models to /tmp..."
    # iOS simulator runs as a macOS process and can access macOS /tmp paths directly.
    # This lets us pass paths via --dart-define without data container UUID churn.
    tmp_dir="/tmp/cactus_flutter_test"
    rm -rf "$tmp_dir" && mkdir -p "$tmp_dir"

    ios_model_path=""
    ios_transcribe_path=""
    ios_assets_path=""

    if [ -n "$model_dir" ] && [ -d "$model_src" ]; then
        cp -R "$model_src" "$tmp_dir/$model_dir"
        ios_model_path="$tmp_dir/$model_dir"
        echo "Staged model: $model_dir"
    else
        [ -n "$MODEL_NAME" ] && echo "Warning: model not found at $model_src"
    fi

    if [ -n "$transcribe_dir" ] && [ -d "$transcribe_src" ]; then
        cp -R "$transcribe_src" "$tmp_dir/$transcribe_dir"
        if [[ "$transcribe_dir" == *"whisper"* ]] || [[ "$transcribe_dir" == *"moonshine"* ]]; then
            vad_src="$PROJECT_ROOT/weights/silero-vad"
            if [ -d "$vad_src" ]; then
                mkdir -p "$tmp_dir/$transcribe_dir/vad"
                rsync -a "$vad_src/" "$tmp_dir/$transcribe_dir/vad/"
                echo "Staged VAD"
            fi
        fi
        ios_transcribe_path="$tmp_dir/$transcribe_dir"
        echo "Staged transcribe model: $transcribe_dir"
    else
        [ -n "$TRANSCRIBE_MODEL_NAME" ] && echo "Warning: transcribe model not found at $transcribe_src"
    fi

    if [ -d "$assets_src" ]; then
        cp -R "$assets_src" "$tmp_dir/assets"
        ios_assets_path="$tmp_dir/assets"
        echo "Staged assets"
    fi

    echo ""
    echo "Step 7: Running tests on simulator..."
    echo "--------------------------------------"
    xcrun simctl boot "$sim_uuid" 2>/dev/null || true
    cd "$SCRIPT_DIR"
    flutter run \
        --dart-define="CACTUS_TEST_MODEL=$ios_model_path" \
        --dart-define="CACTUS_TEST_TRANSCRIBE_MODEL=$ios_transcribe_path" \
        --dart-define="CACTUS_TEST_ASSETS=$ios_assets_path" \
        -d "$sim_uuid"
    cd - >/dev/null

else
    echo "Usage: $0 [model] [transcribe_model] android|ios"
    exit 1
fi
