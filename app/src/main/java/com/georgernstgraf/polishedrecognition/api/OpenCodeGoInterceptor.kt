package com.georgernstgraf.polishedrecognition.api

import okhttp3.Interceptor
import okhttp3.Response
import java.util.UUID

/** OpenCode Go's headers and model-list route, restricted to its documented HTTPS endpoints. */
class OpenCodeGoInterceptor(private val userAgent: String) : Interceptor {

    /** Prepares one Go conversation without modifying requests to other providers. */
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val url = request.url
        val isModels = request.method == "GET" && url.encodedPath == "/inference/go/openai/v1/models"
        val isChat = request.method == "POST" && url.encodedPath == "/inference/go/openai/v1/chat/completions"
        if (url.scheme != "https" || url.host != "opencode.ai" || url.port != 443 || (!isModels && !isChat)) {
            return chain.proceed(request)
        }

        // One dictation makes exactly one LLM call, so each call is a new conversation.
        // An application interceptor runs once across OkHttp's transport retries.
        val builder = request.newBuilder()
            .header("User-Agent", userAgent)
            .header("x-opencode-session", UUID.randomUUID().toString())

        // Go publishes models outside its OpenAI adapter; retain any query parameters.
        if (isModels) builder.url(url.newBuilder().encodedPath("/inference/go/v1/models").build())
        return chain.proceed(builder.build())
    }
}
