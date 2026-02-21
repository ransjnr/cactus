package com.cactus.nativetest

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView
    private val builder = SpannableStringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logView = findViewById(R.id.logView)
        scrollView = findViewById(R.id.scrollView)

        val modelPath = intent.getStringExtra("MODEL_PATH") ?: ""
        val transcribePath = intent.getStringExtra("TRANSCRIBE_PATH") ?: ""
        val assetsPath = intent.getStringExtra("ASSETS_PATH") ?: ""

        TestRunner(::appendLog) {}.run(modelPath, transcribePath, assetsPath)
    }

    private fun appendLog(message: String, color: Int) {
        val start = builder.length
        builder.append(message).append("\n")
        builder.setSpan(ForegroundColorSpan(color), start, builder.length, SpannableStringBuilder.SPAN_EXCLUSIVE_EXCLUSIVE)
        logView.text = builder
        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
    }
}
