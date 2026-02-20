#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_NAME="${1:-LiquidAI/LFM2-VL-450M}"
TRANSCRIBE_MODEL_NAME="${2:-openai/whisper-small}"
PLATFORM="${3:-}"  # optional: android or ios

echo "Running Cactus Flutter wrapper tests..."
echo "======================================="

if ! command -v flutter &>/dev/null; then
    echo "flutter not found - install with: brew install flutter"
    exit 1
fi

echo ""
echo "Step 1: Building Cactus native library..."

if [ "$PLATFORM" = "android" ]; then
    if ! "$PROJECT_ROOT/android/build.sh"; then
        echo "Failed to build Android library"
        exit 1
    fi
    mkdir -p "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a"
    cp "$PROJECT_ROOT/android/libcactus.so" "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a/"
    echo "Copied libcactus.so"

elif [ "$PLATFORM" = "ios" ]; then
    if ! BUILD_XCFRAMEWORK=true "$PROJECT_ROOT/apple/build.sh"; then
        echo "Failed to build iOS library"
        exit 1
    fi
    rm -rf "$SCRIPT_DIR/ios/Runner/cactus-ios.xcframework"
    cp -R "$PROJECT_ROOT/apple/cactus-ios.xcframework" "$SCRIPT_DIR/ios/Runner/"
    echo "Copied cactus-ios.xcframework"
    echo ""
    echo "NOTE: For iOS you must also add the xcframework to Xcode:"
    echo "  1. Open $SCRIPT_DIR/ios/Runner.xcworkspace in Xcode"
    echo "  2. Drag cactus-ios.xcframework into the Runner target"
    echo "  3. Set to 'Embed & Sign' in Frameworks section"

else
    echo "No platform specified, building both..."
    if "$PROJECT_ROOT/android/build.sh"; then
        mkdir -p "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a"
        cp "$PROJECT_ROOT/android/libcactus.so" "$SCRIPT_DIR/android/app/src/main/jniLibs/arm64-v8a/"
        echo "Copied libcactus.so (Android)"
    fi
    if BUILD_XCFRAMEWORK=true "$PROJECT_ROOT/apple/build.sh"; then
        rm -rf "$SCRIPT_DIR/ios/Runner/cactus-ios.xcframework"
        cp -R "$PROJECT_ROOT/apple/cactus-ios.xcframework" "$SCRIPT_DIR/ios/Runner/"
        echo "Copied cactus-ios.xcframework (iOS)"
    fi
fi

echo ""
echo "Step 2: Copying cactus.dart wrapper..."
cp "$PROJECT_ROOT/flutter/cactus.dart" "$SCRIPT_DIR/lib/cactus.dart"
echo "Copied cactus.dart"

echo ""
echo "Step 3: Getting Flutter packages..."
cd "$SCRIPT_DIR"
flutter pub get

echo ""
echo "Step 4: Pushing models and assets to device..."

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

assets_dir=""
assets_src="$PROJECT_ROOT/tests/assets"

if [ "$PLATFORM" = "android" ] || [ -z "$PLATFORM" ]; then
    DEVICE_DATA_DIR="/sdcard/cactus_test"

    if [ -n "$model_dir" ] && [ -d "$model_src" ]; then
        adb shell mkdir -p "$DEVICE_DATA_DIR/$model_dir"
        adb push "$model_src/." "$DEVICE_DATA_DIR/$model_dir/"
        echo "Pushed model: $model_dir"
        model_dir="$DEVICE_DATA_DIR/$model_dir"
    else
        [ -n "$MODEL_NAME" ] && echo "Warning: model not found at $model_src"
        model_dir=""
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
        transcribe_dir="$DEVICE_DATA_DIR/$transcribe_dir"
    else
        [ -n "$TRANSCRIBE_MODEL_NAME" ] && echo "Warning: transcribe model not found at $transcribe_src"
        transcribe_dir=""
    fi

    if [ -d "$assets_src" ]; then
        adb shell mkdir -p "$DEVICE_DATA_DIR/assets"
        adb push "$assets_src/." "$DEVICE_DATA_DIR/assets/"
        echo "Pushed test assets"
        assets_dir="$DEVICE_DATA_DIR/assets"
    fi
fi

echo ""
echo "Step 5: Running on device..."

dart_defines=(
    "--dart-define=CACTUS_TEST_MODEL=$model_dir"
    "--dart-define=CACTUS_TEST_TRANSCRIBE_MODEL=$transcribe_dir"
    "--dart-define=CACTUS_TEST_ASSETS=$assets_dir"
)

if [ "$PLATFORM" = "android" ]; then
    flutter run "${dart_defines[@]}" -d android
elif [ "$PLATFORM" = "ios" ]; then
    flutter run "${dart_defines[@]}" -d iPhone
else
    flutter run "${dart_defines[@]}"
fi
