#!/bin/bash

echo "=========================================="
echo "  Cactus iOS Test Suite"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_MODEL="LiquidAI/LFM2-VL-450M"
DEFAULT_TRANSCRIBE_MODEL="openai/whisper-small"

MODEL_NAME="$DEFAULT_MODEL"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --transcribe_model)
            TRANSCRIBE_MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model to use for tests (default: $DEFAULT_MODEL)"
            echo "  --transcribe_model <name> Transcribe model to use (default: $DEFAULT_TRANSCRIBE_MODEL)"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Transcribe Model: $TRANSCRIBE_MODEL_NAME"

echo ""
echo "=========================================="
echo "  Device Selection"
echo "=========================================="
echo ""

# Collect simulators - extract UUID first, then clean the name
SIMULATORS=$(xcrun simctl list devices available 2>/dev/null | grep -E "^\s+(iPhone|iPad)" | grep -v "unavailable" | sed 's/^[[:space:]]*//' | while read line; do
    uuid=$(echo "$line" | grep -oE '\([A-F0-9-]{36}\)' | head -1 | tr -d '()')
    if [ -n "$uuid" ]; then
        # Remove all parenthetical info to get clean name
        name=$(echo "$line" | sed -E 's/ \([^)]*\)//g' | xargs)
        echo "${name}|simulator|${uuid}"
    fi
done)

# Collect physical devices - only those with physical device UUID pattern starting with 00008
# Physical device UUIDs have format: 00008XXX-XXXXXXXXXXXXXXXX (8 chars, dash, 16 chars)
# Check device availability by looking at xctrace sections
XCTRACE_OUTPUT=$(xcrun xctrace list devices 2>&1)

PHYSICAL_DEVICES=$(echo "$XCTRACE_OUTPUT" | awk '
    /== Devices ==/ { in_online=1; in_offline=0; next }
    /== Devices Offline ==/ { in_online=0; in_offline=1; next }
    /== Simulators ==/ { exit }
    /00008[A-F0-9]{3}-[A-F0-9]{16}/ {
        if (in_online || in_offline) {
            status = in_online ? "online" : "offline"
            print $0 "|" status
        }
    }
' | while read line; do
    uuid=$(echo "$line" | grep -oE '00008[A-F0-9]{3}-[A-F0-9]{16}')
    status=$(echo "$line" | awk -F'|' '{print $2}')
    if [ -n "$uuid" ]; then
        # Extract device name and iOS version (remove the status from the name)
        name=$(echo "$line" | awk -F'|' '{print $1}' | sed -E 's/ \([0-9]+\.[0-9]+.*$//' | xargs)
        ios_version=$(echo "$line" | awk -F'|' '{print $1}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        echo "${name} (iOS ${ios_version})|device|${uuid}|${status}"
    fi
done)

# Combine all devices - physical devices first, then simulators
ALL_DEVICES=$(printf "%s\n%s\n" "$PHYSICAL_DEVICES" "$SIMULATORS" | grep -v '^$')

if [ -z "$ALL_DEVICES" ]; then
    echo "Error: No devices found"
    echo ""
    echo "Make sure:"
    echo "  - Xcode is installed"
    echo "  - At least one simulator is available"
    echo "  - Physical devices are connected and trusted (if using physical devices)"
    exit 1
fi

# Display physical devices first if any
PHYSICAL_COUNT=$(echo "$ALL_DEVICES" | grep -c '|device|')
DEVICE_NUM=0

if [ "$PHYSICAL_COUNT" -gt 0 ]; then
    echo "Physical Devices:"
    while IFS='|' read -r name type uuid status; do
        if [ "$type" = "device" ]; then
            DEVICE_NUM=$((DEVICE_NUM + 1))
            if [ "$status" = "online" ]; then
                printf "  %2d) %s\n" "$DEVICE_NUM" "$name"
            else
                printf "  %2d) %s [offline]\n" "$DEVICE_NUM" "$name"
            fi
        fi
    done <<< "$ALL_DEVICES"
    echo ""
fi

# Display simulators
echo "Simulators:"
while IFS='|' read -r name type uuid status; do
    if [ "$type" = "simulator" ]; then
        DEVICE_NUM=$((DEVICE_NUM + 1))
        printf "  %2d) %s\n" "$DEVICE_NUM" "$name"
    fi
done <<< "$ALL_DEVICES"

echo ""
read -p "Select device number (1-$DEVICE_NUM): " DEVICE_NUMBER

# Validate input
if ! [[ "$DEVICE_NUMBER" =~ ^[0-9]+$ ]] || [ "$DEVICE_NUMBER" -lt 1 ] || [ "$DEVICE_NUMBER" -gt "$DEVICE_NUM" ]; then
    echo ""
    echo "Invalid selection. Please enter a number between 1 and $DEVICE_NUM"
    exit 1
fi

# Get selected device info
SELECTED_LINE=$(echo "$ALL_DEVICES" | sed -n "${DEVICE_NUMBER}p")
DEVICE_NAME=$(echo "$SELECTED_LINE" | cut -d'|' -f1)
DEVICE_TYPE=$(echo "$SELECTED_LINE" | cut -d'|' -f2)
DEVICE_UUID=$(echo "$SELECTED_LINE" | cut -d'|' -f3)
DEVICE_STATUS=$(echo "$SELECTED_LINE" | cut -d'|' -f4)

if [ -z "$DEVICE_UUID" ]; then
    echo ""
    echo "Error: Could not parse device information"
    exit 1
fi

echo ""
if [ "$DEVICE_TYPE" = "simulator" ]; then
    echo "Selected: $DEVICE_NAME (Simulator)"
else
    if [ "$DEVICE_STATUS" = "online" ]; then
        echo "Selected: $DEVICE_NAME (Physical Device - Online)"
    else
        echo "Selected: $DEVICE_NAME (Physical Device - Offline)"
        echo ""
        echo "Warning: This device is currently offline"
        echo "   Please ensure the device is:"
        echo "   - Connected via USB or network"
        echo "   - Unlocked and trusted"
        echo "   - Has developer mode enabled"
        echo ""
        read -p "Continue anyway? (y/N): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# Handle code signing for physical devices
if [ "$DEVICE_TYPE" = "device" ]; then
    echo "=========================================="
    echo "  Code Signing Configuration"
    echo "=========================================="
    echo ""
    echo "Checking for development certificates..."

    # Check if certificates exist
    if ! security find-identity -v -p codesigning 2>/dev/null | grep -q "Apple Development"; then
        echo ""
        echo "Error: No development certificates found"
        echo ""
        echo "To fix this:"
        echo "  1. Open Xcode"
        echo "  2. Go to Settings > Accounts"
        echo "  3. Add your Apple ID"
        echo "  4. Download development certificates"
        exit 1
    fi

    # Get certificate subject containing Team ID (OU field) and developer name (CN field)
    CERT_SUBJECT=$(security find-certificate -a -c "Apple Development" -p | openssl x509 -subject -noout)

    # Extract Team ID from OU field
    TEAM_ID=$(echo "$CERT_SUBJECT" | grep -oE 'OU=[A-Z0-9]{10}' | cut -d= -f2)

    # Extract developer name from CN field (remove certificate ID suffix)
    TEAM_NAME=$(echo "$CERT_SUBJECT" | sed -E 's/.*CN=Apple Development: ([^,]+).*/\1/' | sed -E 's/ \([A-Z0-9]{10}\)$//')

    echo "Found development certificate:"
    echo "  Developer: $TEAM_NAME"
    echo "  Team ID: $TEAM_ID"
fi

echo ""
echo "=========================================="
echo "  Build Process"
echo "=========================================="
echo ""
echo "[1/4] Downloading model weights..."
if ! "$PROJECT_ROOT/cli/cactus" download "$MODEL_NAME"; then
    echo ""
    echo "Failed to download model weights"
    exit 1
fi

if ! "$PROJECT_ROOT/cli/cactus" download "$TRANSCRIBE_MODEL_NAME"; then
    echo ""
    echo "Failed to download transcribe model weights"
    exit 1
fi
echo "      Model weights downloaded"

echo ""
echo "[2/4] Building Cactus library..."
cd "$PROJECT_ROOT"
if ! cactus/build.sh > /dev/null 2>&1; then
    echo ""
    echo "Failed to build cactus library"
    exit 1
fi
echo "      Cactus library built"

echo ""
echo "[3/4] Configuring Xcode project..."

XCODEPROJ_PATH="$SCRIPT_DIR/CactusTest/CactusTest.xcodeproj"
TESTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACTUS_ROOT="$PROJECT_ROOT/cactus"
APPLE_ROOT="$PROJECT_ROOT/apple"

if ! command -v ruby &> /dev/null; then
    echo ""
    echo "Error: Ruby not found. Please install Ruby."
    exit 1
fi

if ! gem list xcodeproj -i &> /dev/null; then
    echo "   Installing xcodeproj gem..."
    if ! gem install xcodeproj > /dev/null 2>&1; then
        echo ""
        echo "Failed to install xcodeproj gem"
        exit 1
    fi
fi

export PROJECT_ROOT TESTS_ROOT CACTUS_ROOT APPLE_ROOT XCODEPROJ_PATH
if ! ruby "$SCRIPT_DIR/configure_xcode.rb" > /dev/null 2>&1; then
    echo ""
    echo "Failed to configure Xcode project"
    exit 1
fi
echo "      Xcode project configured"

echo ""
echo "[4/4] Building test application..."
if [ "$DEVICE_TYPE" = "simulator" ]; then
    echo "      Target: iOS Simulator"
else
    echo "      Target: Physical Device (Code Signing with Team ID: $TEAM_ID)"
fi

if [ "$DEVICE_TYPE" = "simulator" ]; then
    IOS_SIM_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if [ -z "$IOS_SIM_SDK_PATH" ] || [ ! -d "$IOS_SIM_SDK_PATH" ]; then
        echo ""
        echo "Error: iOS Simulator SDK not found. Make sure Xcode is installed."
        exit 1
    fi

    if ! xcodebuild -project "$XCODEPROJ_PATH" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS Simulator,id=$DEVICE_UUID" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$IOS_SIM_SDK_PATH" \
         build > /dev/null 2>&1; then
        echo ""
        echo "Build failed. Run without '> /dev/null 2>&1' to see detailed errors."
        exit 1
    fi

    APP_PATH="$SCRIPT_DIR/build/Build/Products/Release-iphonesimulator/CactusTest.app"
else
    IOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
    if [ -z "$IOS_SDK_PATH" ] || [ ! -d "$IOS_SDK_PATH" ]; then
        echo ""
        echo "Error: iOS SDK not found. Make sure Xcode is installed."
        exit 1
    fi

    if ! xcodebuild -project "$XCODEPROJ_PATH" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS,id=$DEVICE_UUID" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         -allowProvisioningUpdates \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$IOS_SDK_PATH" \
         DEVELOPMENT_TEAM="$TEAM_ID" \
         CODE_SIGN_IDENTITY="Apple Development" \
         CODE_SIGN_STYLE="Automatic" \
         build > /dev/null 2>&1; then
        echo ""
        echo "Build failed. Run without '> /dev/null 2>&1' to see detailed errors."
        exit 1
    fi

    APP_PATH="$SCRIPT_DIR/build/Build/Products/Release-iphoneos/CactusTest.app"
fi
echo "      Build completed successfully"

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

MODEL_SRC="$PROJECT_ROOT/weights/$MODEL_DIR"
TRANSCRIBE_MODEL_SRC="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"

echo ""
echo "Copying model weights to app bundle..."
if ! cp -R "$MODEL_SRC" "$APP_PATH/" 2>/dev/null; then
    echo "Warning: Could not copy model weights"
fi
if ! cp -R "$TRANSCRIBE_MODEL_SRC" "$APP_PATH/" 2>/dev/null; then
    echo "Warning: Could not copy transcribe model weights"
fi

echo ""
echo "=========================================="
echo "  Test Execution"
echo "=========================================="
echo ""

BUNDLE_ID="cactus.CactusTest"

if [ "$DEVICE_TYPE" = "simulator" ]; then
    echo "Installing on simulator: $DEVICE_NAME"
    echo ""

    # Boot simulator if needed
    xcrun simctl boot "$DEVICE_UUID" 2>/dev/null || true

    # Uninstall previous version
    xcrun simctl uninstall "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null || true

    # Install app
    if ! xcrun simctl install "$DEVICE_UUID" "$APP_PATH" 2>/dev/null; then
        echo "Failed to install app on simulator"
        exit 1
    fi
    echo "App installed successfully"

    echo ""
    echo "Launching tests..."
    echo "=========================================="
    echo ""

    SIMCTL_CHILD_CACTUS_TEST_MODEL="$MODEL_DIR" \
    SIMCTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL="$TRANSCRIBE_MODEL_DIR" \
    xcrun simctl launch --console-pty "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null

    echo ""
    echo "=========================================="
    echo "Tests completed on: $DEVICE_NAME (Simulator)"
    echo "=========================================="
else
    echo "Installing on physical device: $DEVICE_NAME"
    echo ""

    # Uninstall previous version if exists
    echo "Removing previous installation..."
    xcrun devicectl device uninstall app --device "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null || true

    # Install the app
    echo "Installing app..."
    if ! xcrun devicectl device install app --device "$DEVICE_UUID" "$APP_PATH" 2>/dev/null; then
        echo ""
        echo "Failed to install app on device"
        echo ""
        echo "Possible issues:"
        echo "  - Device is not trusted"
        echo "  - Code signing failed"
        echo "  - Device is locked"
        exit 1
    fi
    echo "App installed successfully"

    echo ""
    echo "Launching tests..."
    echo "=========================================="
    echo ""
    echo "Note: Model paths for physical devices:"
    echo "  - Model: $MODEL_DIR"
    echo "  - Transcribe model: $TRANSCRIBE_MODEL_DIR"
    echo ""
    echo "Note: Logs will be saved to device and fetched after completion"
    echo ""

    # Launch the app with environment variables
    DEVICECTL_CHILD_CACTUS_TEST_MODEL="$MODEL_DIR" \
    DEVICECTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL="$TRANSCRIBE_MODEL_DIR" \
    xcrun devicectl device process launch --device "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null || true

    echo "App launched"
    echo "Waiting for tests to complete..."

    # Wait for the process to finish by checking if CactusTest is still running
    # Check every 2 seconds for up to 5 minutes
    MAX_WAIT=300
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if xcrun devicectl device info processes --device "$DEVICE_UUID" 2>/dev/null | grep -q "CactusTest.app/CactusTest"; then
            sleep 2
            ELAPSED=$((ELAPSED + 2))
        else
            echo "Tests completed (process ended after ${ELAPSED}s)"
            break
        fi
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "Warning: Test execution timeout reached (${MAX_WAIT}s)"
    fi

    sleep 1

    echo ""
    echo "Fetching logs from device..."

    # Create temporary directory for the log file
    TEMP_LOG_DIR=$(mktemp -d)
    TEMP_LOG_FILE="$TEMP_LOG_DIR/cactus_test.log"

    # Fetch the log file from the device
    if xcrun devicectl device copy from \
        --device "$DEVICE_UUID" \
        --source "Documents/cactus_test.log" \
        --destination "$TEMP_LOG_FILE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" 2>/dev/null; then

        if [ -f "$TEMP_LOG_FILE" ]; then
            echo "=========================================="
            echo "Test Output:"
            echo "=========================================="
            echo ""
            cat "$TEMP_LOG_FILE"
            echo ""
        else
            echo "Warning: Could not find downloaded log file"
        fi

        # Clean up temporary directory
        rm -rf "$TEMP_LOG_DIR"
    else
        echo "Warning: Could not fetch log file from device"
        echo ""
        echo "To manually view logs, run:"
        echo "  xcrun devicectl device copy from --device $DEVICE_UUID --source Documents/cactus_test.log --destination ./cactus_test.log --domain-type appDataContainer --domain-identifier $BUNDLE_ID"
    fi

    echo ""
    echo "=========================================="
    echo "Tests completed on: $DEVICE_NAME (Physical Device)"
    echo "=========================================="
fi
