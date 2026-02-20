### SETUP
Have xcode.
Install following libs:
- iOS (if no xcode): `gem install xcodeproj`
- Android: `brew install --cask android-commandlinetools temurin`, then `sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;26.3.11579264"`
- Flutter: `brew install flutter`

### COMMANDS
iOS (defaults to LFM2-VL-450M + whisper-small)
./ios/run.sh [model] [transcribe_model]

Android (defaults to LFM2-VL-450M + whisper-small)
./android/run.sh [model] [transcribe_model]

Flutter (defaults to LFM2-VL-450M + whisper-small)
./flutter/run.sh [model] [transcribe_model] [android|ios]


### USAGE
The idea: have a native test folder in a branch that we never merge to main, we keep it separate from the actual build. Reason: would clog the actual lib.
Essentially a standalone addon to main cactus repo, we can just keep merging main to it and retest these native platform builds. We can do this because we are no longer supporting sdks and keep the api wrappers in main cactus repo.