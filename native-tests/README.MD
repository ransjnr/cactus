### SETUP
Have xcode.
Install following libs:
- iOS: `gem install xcodeproj`
- Android:
  1. `brew install --cask android-commandlinetools temurin@21`
  2. The SDK installs to `/opt/homebrew/share/android-commandlinetools` (not `android-sdk`). Add to your shell profile:
     ```
     export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
     export PATH="$ANDROID_HOME/platform-tools:$PATH"
     ```
  3. Install required SDK components:
     ```
     sdkmanager "platform-tools" "platforms;android-34" "platforms;android-36" "build-tools;34.0.0" "build-tools;28.0.3" "ndk;26.3.11579264" "emulator" "system-images;android-34;google_apis;arm64-v8a"
     ```
  4. Point Flutter to the SDK: `flutter config --android-sdk $ANDROID_HOME`
  5. Accept licenses: `sdkmanager --licenses`
  6. Create an emulator: `avdmanager create avd --name "Pixel_6_API_34" --package "system-images;android-34;google_apis;arm64-v8a" --device "pixel_6"`
- Flutter: `brew install flutter`

### COMMANDS
iOS (defaults to LFM2-VL-450M + whisper-small)
./ios/run.sh [model] [transcribe_model]

Android (defaults to LFM2-VL-450M + whisper-small)
./android/run.sh [model] [transcribe_model]

Flutter (defaults to LFM2-VL-450M + whisper-small)
./flutter/run.sh [model] [transcribe_model] [android|ios]
./flutter/run.sh LFM2-VL-450M whisper-small ios
./flutter/run.sh LFM2-VL-450M whisper-small android


### USAGE
The idea: have a native test folder in a branch that we never merge to main, we keep it separate from the actual build. Reason: would clog the actual lib.
Essentially a standalone addon to main cactus repo, we can just keep merging main to it and retest these native platform builds. We can do this because we are no longer supporting sdks and keep the api wrappers in main cactus repo.