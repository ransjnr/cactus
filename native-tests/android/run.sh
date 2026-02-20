#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_NAME="${1:-LiquidAI/LFM2-VL-450M}"
TRANSCRIBE_MODEL_NAME="${2:-openai/whisper-small}"

echo "Running Cactus Kotlin wrapper tests..."
echo "======================================="

# ---------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------

if ! command -v adb &>/dev/null; then
    echo "adb not found. Run: sdkmanager \"platform-tools\""
    exit 1
fi

if [ -z "$ANDROID_HOME" ]; then
    if [ -d "/opt/homebrew/share/android-commandlinetools" ]; then
        export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
    elif [ -d "$HOME/Library/Android/sdk" ]; then
        export ANDROID_HOME="$HOME/Library/Android/sdk"
    else
        echo "ANDROID_HOME not set and SDK not found at default locations"
        exit 1
    fi
fi

if ! adb devices | grep -q "device$"; then
    echo "No Android device/emulator connected"
    exit 1
fi

# ---------------------------------------------------------------
# Step 1: Gradle wrapper
# ---------------------------------------------------------------

echo ""
echo "Step 1: Setting up Gradle wrapper..."

if [ ! -f "$SCRIPT_DIR/gradlew" ]; then
    if ! command -v gradle &>/dev/null; then
        echo "gradle not found - installing via brew..."
        brew install gradle || { echo "Failed to install gradle"; exit 1; }
    fi
    cd "$SCRIPT_DIR"
    gradle wrapper --gradle-version 8.7
    cd - >/dev/null
fi

# ---------------------------------------------------------------
# Step 2: Build native library
# ---------------------------------------------------------------

echo ""
echo "Step 2: Building Cactus native library..."

if ! "$PROJECT_ROOT/android/build.sh"; then
    echo "Failed to build Android native library"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a"
cp "$PROJECT_ROOT/android/libcactus.so" "$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a/"
echo "Copied libcactus.so"

# ---------------------------------------------------------------
# Step 3: Copy Cactus.kt wrapper
# ---------------------------------------------------------------

echo ""
echo "Step 3: Copying Cactus.kt wrapper..."

mkdir -p "$SCRIPT_DIR/app/src/main/java/com/cactus"
cp "$PROJECT_ROOT/android/Cactus.kt" "$SCRIPT_DIR/app/src/main/java/com/cactus/Cactus.kt"
echo "Copied Cactus.kt"

# ---------------------------------------------------------------
# Step 4: Build APK
# ---------------------------------------------------------------

echo ""
echo "Step 4: Building APK..."

cd "$SCRIPT_DIR"
if ! ./gradlew assembleDebug 2>&1 | tail -20; then
    echo "Build failed"
    exit 1
fi
cd - >/dev/null

APK_PATH="$SCRIPT_DIR/app/build/outputs/apk/debug/app-debug.apk"

# ---------------------------------------------------------------
# Step 5: Install APK
# ---------------------------------------------------------------

echo ""
echo "Step 5: Installing APK..."
adb install -r "$APK_PATH"

# ---------------------------------------------------------------
# Step 6: Push models and assets
# ---------------------------------------------------------------

echo ""
echo "Step 6: Pushing models and assets to device..."

APP_DATA_DIR="/sdcard/Android/data/com.cactus.nativetest/files/cactus_test"
adb shell mkdir -p "$APP_DATA_DIR"

model_path=""
if [ -n "$MODEL_NAME" ]; then
    model_dir=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    model_src="$PROJECT_ROOT/weights/$model_dir"
    if [ -d "$model_src" ]; then
        adb shell mkdir -p "$APP_DATA_DIR/$model_dir"
        adb push "$model_src/." "$APP_DATA_DIR/$model_dir/"
        echo "Pushed model: $model_dir"
        model_path="$APP_DATA_DIR/$model_dir"
    else
        echo "Warning: model not found at $model_src"
    fi
fi

transcribe_path=""
if [ -n "$TRANSCRIBE_MODEL_NAME" ]; then
    transcribe_dir=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    transcribe_src="$PROJECT_ROOT/weights/$transcribe_dir"
    if [ -d "$transcribe_src" ]; then
        adb shell mkdir -p "$APP_DATA_DIR/$transcribe_dir"
        adb push "$transcribe_src/." "$APP_DATA_DIR/$transcribe_dir/"
        echo "Pushed transcribe model: $transcribe_dir"
        if [[ "$transcribe_dir" == *"whisper"* ]] || [[ "$transcribe_dir" == *"moonshine"* ]]; then
            vad_src="$PROJECT_ROOT/weights/silero-vad"
            if [ -d "$vad_src" ]; then
                adb shell mkdir -p "$APP_DATA_DIR/$transcribe_dir/vad"
                adb push "$vad_src/." "$APP_DATA_DIR/$transcribe_dir/vad/"
                echo "Pushed VAD into transcribe model"
            fi
        fi
        transcribe_path="$APP_DATA_DIR/$transcribe_dir"
    else
        echo "Warning: transcribe model not found at $transcribe_src"
    fi
fi

assets_path=""
assets_src="$PROJECT_ROOT/tests/assets"
if [ -d "$assets_src" ]; then
    adb shell mkdir -p "$APP_DATA_DIR/assets"
    adb push "$assets_src/." "$APP_DATA_DIR/assets/"
    echo "Pushed test assets"
    assets_path="$APP_DATA_DIR/assets"
fi

# ---------------------------------------------------------------
# Step 7: Launch and stream logs
# ---------------------------------------------------------------

echo ""
echo "Step 7: Launching tests..."
echo "--------------------------"

adb shell am start -n "com.cactus.nativetest/.MainActivity" \
    --es "MODEL_PATH" "$model_path" \
    --es "TRANSCRIBE_PATH" "$transcribe_path" \
    --es "ASSETS_PATH" "$assets_path"

echo ""
echo "Streaming logcat (Ctrl+C to stop)..."
adb logcat -s System.out:I | grep -v "^$"
