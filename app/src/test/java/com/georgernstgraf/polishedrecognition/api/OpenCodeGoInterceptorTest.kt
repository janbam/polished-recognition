package com.georgernstgraf.polishedrecognition.api

import com.google.common.truth.Truth.assertThat
import io.mockk.every
import io.mockk.mockk
import okhttp3.Interceptor
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.junit.Test
import java.util.UUID

/** Protects Go's owned routing and headers without pretending to emulate the provider. */
class OpenCodeGoInterceptorTest {

    private val interceptor = OpenCodeGoInterceptor("PolishedRecognition/test")

    /** Captures the request forwarded to the next interceptor, without a network dependency. */
    private fun forwarded(request: Request): Request {
        val chain = mockk<Interceptor.Chain>()
        every { chain.request() } returns request
        every { chain.proceed(any()) } answers {
            Response.Builder().request(firstArg()).protocol(Protocol.HTTP_1_1).code(200).message("OK").build()
        }
        return interceptor.intercept(chain).request
    }

    @Test
    fun `chat retains body and authorization and replaces required headers for each conversation`() {
        val request = Request.Builder()
            .url("https://opencode.ai/inference/go/openai/v1/chat/completions")
            .post("{\"model\":\"gpt-6-luna\"}".toRequestBody())
            .header("Authorization", "Bearer test-token")
            .addHeader("User-Agent", "generic-sdk")
            .addHeader("x-opencode-session", "old-session")
            .build()

        val first = forwarded(request)
        val second = forwarded(request)

        assertThat(first.url).isEqualTo(request.url)
        assertThat(first.body).isSameInstanceAs(request.body)
        assertThat(first.header("Authorization")).isEqualTo("Bearer test-token")
        assertThat(first.headers.values("User-Agent")).containsExactly("PolishedRecognition/test")
        assertThat(first.headers.values("x-opencode-session")).hasSize(1)
        assertThat(UUID.fromString(first.header("x-opencode-session")).toString()).isEqualTo(first.header("x-opencode-session"))
        assertThat(second.header("x-opencode-session")).isNotEqualTo(first.header("x-opencode-session"))
    }

    @Test
    fun `models use the Go catalog route and retain query and credentials`() {
        val request = Request.Builder()
            .url("https://opencode.ai/inference/go/openai/v1/models?limit=50")
            .header("Authorization", "Bearer test-token").build()

        val result = forwarded(request)

        assertThat(result.url.toString()).isEqualTo("https://opencode.ai/inference/go/v1/models?limit=50")
        assertThat(result.method).isEqualTo("GET")
        assertThat(result.header("Authorization")).isEqualTo("Bearer test-token")
        assertThat(result.header("User-Agent")).isEqualTo("PolishedRecognition/test")
        assertThat(result.header("x-opencode-session")).isNotEmpty()
    }

    @Test
    fun `unrelated hosts ports schemes API paths and methods are untouched`() {
        val urls = listOf(
            "https://api.openai.com/v1/models",
            "https://opencode.ai.example/inference/go/openai/v1/models",
            "http://opencode.ai/inference/go/openai/v1/models",
            "https://opencode.ai:8443/inference/go/openai/v1/models",
            "https://opencode.ai/inference/openai/v1/models",
            "https://opencode.ai/inference/go/openai/v1/audio/transcriptions",
            "https://opencode.ai/inference/go/openai/v1/chat/completions"
        )
        for (url in urls) {
            val request = Request.Builder().url(url).build()
            assertThat(forwarded(request)).isSameInstanceAs(request)
        }
        val postModels = Request.Builder().url("https://opencode.ai/inference/go/openai/v1/models")
            .post("".toRequestBody()).build()
        assertThat(forwarded(postModels)).isSameInstanceAs(postModels)
    }
}
