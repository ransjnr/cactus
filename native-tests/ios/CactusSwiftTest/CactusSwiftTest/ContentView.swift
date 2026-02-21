//
//  ContentView.swift
//  CactusSwiftTest
//
//  Created by Justin L on 2/20/26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var runner = TestRunner()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Cactus Swift Test")
                    .font(.headline)
                Spacer()
                if !runner.finished {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            .padding()
            .background(Color(.systemGroupedBackground))

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(runner.logs.enumerated()), id: \.offset) { index, log in
                            Text(log)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(color(for: log))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(index)
                        }
                    }
                    .padding()
                }
                .onChange(of: runner.logs.count) { _ in
                    if let last = runner.logs.indices.last {
                        proxy.scrollTo(last)
                    }
                }
            }
        }
        .onAppear {
            let bundle = Bundle.main.bundlePath
            let env = ProcessInfo.processInfo.environment

            let modelName = env["CACTUS_TEST_MODEL"] ?? ""
            let transcribeName = env["CACTUS_TEST_TRANSCRIBE_MODEL"] ?? ""
            let assetsName = env["CACTUS_TEST_ASSETS"] ?? ""

            runner.run(
                modelPath: modelName.isEmpty ? "" : bundle + "/" + modelName,
                transcribeModelPath: transcribeName.isEmpty ? "" : bundle + "/" + transcribeName,
                assetsPath: assetsName.isEmpty ? "" : bundle + "/" + assetsName
            )
        }
    }

    private func color(for log: String) -> Color {
        if log.hasPrefix("[PASS]") { return .green }
        if log.hasPrefix("[FAIL]") { return .red }
        if log.hasPrefix("===") { return .primary }
        return .secondary
    }
}
