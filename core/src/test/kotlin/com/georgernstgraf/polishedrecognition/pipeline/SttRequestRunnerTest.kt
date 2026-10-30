package com.georgernstgraf.polishedrecognition.pipeline

import com.georgernstgraf.polishedrecognition.api.OpenAiSttApiService
import com.georgernstgraf.polishedrecognition.api.dto.SttResponse
import com.georgernstgraf.polishedrecognition.audio.WavWriter
import com.georgernstgraf.polishedrecognition.config.SttProviderConfig
import com.google.common.truth.Truth.assertThat
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import okhttp3.ResponseBody
import okhttp3.RequestBody
import okhttp3.ResponseBody.Companion.toResponseBody
import com.google.gson.Gson
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import retrofit2.Call
import retrofit2.Response
import java.io.File
import java.io.IOException

/**
 * #116: single-request STT execution with retry ×3 (transient-only) and
 * per-attempt completion records in `stt-latency.json`.
 */
class SttRequestRunnerTest {

    private val sttApi = mockk<OpenAiSttApiService>(relaxed = true)

    @get:Rule
    val tmp = TemporaryFolder()

    private lateinit var logDir: File
    private val delays = mutableListOf<Long>()

    private val config = SttProviderConfig(
        displayName = "LAN",
        baseUrl = "http://10.8.0.16:11437/v1/",
        apiToken = "token",
        model = "large-v3"
    )

    @Before
    fun setUp() {
        logDir = tmp.newFolder("logs")
        delays.clear()
    }

    private fun runner(): SttRequestRunner =
        SttRequestRunner(
            getSttApi = { sttApi },
            logger = RotatingJsonLogger(logDir),
            backoffMs = listOf(1L, 2L),
            sleep = { delays.add(it) }
        )

    private fun mockStt(response: Response<SttResponse>) {
        val call = mockk<Call<SttResponse>>()
        every { call.execute() } returns response
        every { sttApi.transcribeAudioSync(any(), any(), any(), any(), any()) } returns call
    }

    private fun mockSttMany(vararg responses: Response<SttResponse>) {
        val call = mockk<Call<SttResponse>>()
        every { call.execute() } returnsMany responses.toList()
        every { sttApi.transcribeAudioSync(any(), any(), any(), any(), any()) } returns call
    }

    private fun success(text: String = "hallo welt") =
        Response.success(SttResponse(text = text, language = "de"))

    private fun httpError(code: Int) =
        @Suppress("DEPRECATION")
        Response.error<SttResponse>(code, ResponseBody.create(null, "error"))

    /** Captures actual multipart format values while supplying observed API responses. */
    private fun formatsFor(vararg responses: Response<SttResponse>): List<RequestBody> {
        val formats = mutableListOf<RequestBody>()
        val call = mockk<Call<SttResponse>>()
        every { call.execute() } returnsMany responses.toList()
        every { sttApi.transcribeAudioSync(any(), any(), any(), capture(formats), any()) } returns call
        return formats
    }

    /** Error wording captured from OpenAI's rejection of gpt-transcribe verbose JSON. */
    private fun formatRejection(): Response<SttResponse> = Response.error(
        400,
        """{"error":{"message":"response_format 'verbose_json' is not compatible with model 'gpt-transcribe-api-ev3'. Use 'json' or 'text' instead.","type":"invalid_request_error","param":"response_format"}}""".toResponseBody()
    )

    /** Decodes captured text multipart parts, not their object identities. */
    private fun List<RequestBody>.texts(): List<String> = map { body ->
        val buffer = okio.Buffer()
        body.writeTo(buffer)
        buffer.readUtf8()
    }

    /** 16-bit mono @ 16 kHz: 16000 data bytes = 500 ms (WavWriter layout). */
    private fun wavFile(name: String = "fragment.wav"): File =
        tmp.newFile(name).apply { writeBytes(WavWriter.write(ByteArray(16_000), sampleRate = 16_000)) }

    private fun readLog(baseName: String): Map<*, *> =
        com.google.gson.Gson().fromJson(File(logDir, "$baseName.json").readText(), Map::class.java)

    @Test
    fun `success on first attempt logs upload and completion with duration`() = kotlinx.coroutines.runBlocking {
        mockStt(success())
        val wav = wavFile()

        val result = runner().run(wav, config, chunk = 1, chunkCount = 1)

        assertThat(result.isSuccess).isTrue()
        assertThat(result.getOrThrow().text).isEqualTo("hallo welt")

        val upload = readLog("stt-upload")
        assertThat(upload["chunk"]).isEqualTo(1.0)
        assertThat(upload["file"]).isEqualTo("fragment.wav")
        assertThat(upload["mediaType"]).isEqualTo("audio/wav")
        assertThat(upload["model"]).isEqualTo("large-v3")
        assertThat(upload["baseUrl"]).isEqualTo("http://10.8.0.16:11437/v1/")
        // legacy upload entries carry no completion fields
        assertThat(upload.containsKey("durationMs")).isFalse()
        assertThat(upload.containsKey("httpCode")).isFalse()

        val completion = readLog("stt-latency")
        assertThat(completion["event"]).isEqualTo("complete")
        assertThat((completion["durationMs"] as Double).toLong()).isEqualTo(500L)
        assertThat((completion["httpCode"] as Double).toInt()).isEqualTo(200)
        assertThat((completion["attempt"] as Double).toInt()).isEqualTo(1)
        assertThat(completion["elapsedMs"] as Double).isAtLeast(0.0)
        assertThat(completion["error"]).isNull()
    }

    @Test
    fun `ogg duration is parsed from the last page granule`() = kotlinx.coroutines.runBlocking {
        mockStt(success())
        val ogg = tmp.newFile("fragment.ogg")
        ogg.writeBytes(singlePageOgg(granulePosition = 336_000)) // 7 s

        val result = runner().run(ogg, config)

        assertThat(result.isSuccess).isTrue()
        assertThat((readLog("stt-latency")["durationMs"] as Double).toLong()).isEqualTo(7_000L)
    }

    @Test
    fun `transient HTTP 500 retries and succeeds`() = kotlinx.coroutines.runBlocking {
        mockSttMany(httpError(500), success("second try"))

        val result = runner().run(wavFile(), config)

        assertThat(result.isSuccess).isTrue()
        assertThat(result.getOrThrow().text).isEqualTo("second try")
        assertThat(delays).containsExactly(1L).inOrder()
        // attempt 1 (500) rotated into _1, attempt 2 (200) is current
        assertThat((readLog("stt-latency")["attempt"] as Double).toInt()).isEqualTo(2)
        assertThat((readLog("stt-latency")["httpCode"] as Double).toInt()).isEqualTo(200)
        assertThat((readLog("stt-latency_1")["httpCode"] as Double).toInt()).isEqualTo(500)
    }

    @Test
    fun `HTTP 429 is transient and retries`() = kotlinx.coroutines.runBlocking {
        mockSttMany(httpError(429), success())

        val result = runner().run(wavFile(), config)

        assertThat(result.isSuccess).isTrue()
        assertThat(delays).containsExactly(1L).inOrder()
    }

    @Test
    fun `HTTP 400 fails immediately without retry`() = kotlinx.coroutines.runBlocking {
        mockStt(httpError(400))

        val result = runner().run(wavFile(), config)

        assertThat(result.isFailure).isTrue()
        assertThat(result.exceptionOrNull()!!.message).isEqualTo("HTTP 400")
        assertThat((result.exceptionOrNull() as SttRequestRunner.SttRequestException).httpCode).isEqualTo(400)
        assertThat(delays).isEmpty()
        assertThat((readLog("stt-latency")["attempt"] as Double).toInt()).isEqualTo(1)
    }

    @Test
    fun `format rejection retries JSON immediately and remembers capability for later fragments`() = kotlinx.coroutines.runBlocking {
        val formats = formatsFor(formatRejection(), success(), success())
        val runner = runner()
        val wav = wavFile()

        assertThat(runner.run(wav, config).isSuccess).isTrue()
        assertThat(formats.texts()).containsExactly("verbose_json", "json").inOrder()
        assertThat(readLog("stt-latency")["attempt"]).isEqualTo(2.0)
        assertThat(readLog("stt-latency_1")["httpCode"]).isEqualTo(400.0)
        assertThat(runner.run(wav, config).isSuccess).isTrue()
        assertThat(formats.texts()).containsExactly("verbose_json", "json", "json").inOrder()
        assertThat(delays).isEmpty()
    }

    @Test
    fun `JSON capability is isolated by endpoint and model but not token`() = kotlinx.coroutines.runBlocking {
        val formats = formatsFor(formatRejection(), success(), success(), success(), success())
        val runner = runner()
        val wav = wavFile()

        assertThat(runner.run(wav, config).isSuccess).isTrue()
        assertThat(runner.run(wav, config.copy(apiToken = "new-token")).isSuccess).isTrue()
        assertThat(runner.run(wav, config.copy(model = "another-model")).isSuccess).isTrue()
        assertThat(runner.run(wav, config.copy(baseUrl = "https://api.openai.com/v1/")).isSuccess).isTrue()
        assertThat(formats.texts()).containsExactly("verbose_json", "json", "json", "verbose_json", "verbose_json").inOrder()
    }

    @Test
    fun `format negotiation preserves the full transient retry budget and JSON choice`() = kotlinx.coroutines.runBlocking {
        val formats = formatsFor(httpError(500), formatRejection(), httpError(429), success())

        assertThat(runner().run(wavFile(), config).isSuccess).isTrue()
        assertThat(formats.texts()).containsExactly("verbose_json", "verbose_json", "json", "json").inOrder()
        assertThat(delays).containsExactly(1L, 2L).inOrder()
        assertThat(readLog("stt-latency")["attempt"]).isEqualTo(4.0)
    }

    @Test
    fun `JSON rejection fails without another format retry`() = kotlinx.coroutines.runBlocking {
        val formats = formatsFor(formatRejection(), formatRejection())

        val result = runner().run(wavFile(), config)

        assertThat((result.exceptionOrNull() as SttRequestRunner.SttRequestException).httpCode).isEqualTo(400)
        assertThat(formats.texts()).containsExactly("verbose_json", "json").inOrder()
        assertThat(delays).isEmpty()
    }

    @Test
    fun `unrelated validation errors and echoed format text do not trigger negotiation or cache changes`() = kotlinx.coroutines.runBlocking {
        val unrelated = Response.error<SttResponse>(400,
            """{"error":{"message":"Invalid prompt","param":"prompt"},"input":"response_format"}""".toResponseBody())
        val formats = formatsFor(unrelated, success())
        val runner = runner()
        val wav = wavFile()

        assertThat(runner.run(wav, config, prompt = "context").isFailure).isTrue()
        assertThat(runner.run(wav, config).isSuccess).isTrue()
        assertThat(formats.texts()).containsExactly("verbose_json", "verbose_json").inOrder()
        assertThat(delays).isEmpty()
    }

    @Test
    fun `OpenAI JSON language codes normalize to the language contract consumed by both pipelines`() = kotlinx.coroutines.runBlocking {
        // Shape observed from a successful gpt-transcribe JSON response.
        val body = Gson().fromJson("""{"text":"Hallo Welt","languages":[{"code":"de"}],"usage":{"type":"duration","seconds":1}}""", SttResponse::class.java)
        mockStt(Response.success(body))

        val response = runner().run(wavFile(), config).getOrThrow()

        assertThat(response.text).isEqualTo("Hallo Welt")
        assertThat(response.language).isEqualTo("de")
        assertThat(response.languageProbability).isNull()
    }

    @Test
    fun `singular language and probability remain authoritative when both language forms appear`() = kotlinx.coroutines.runBlocking {
        val body = Gson().fromJson("""{"text":"Hallo","language":"German","language_probability":0.9,"languages":[{"code":"en"}]}""", SttResponse::class.java)
        mockStt(Response.success(body))

        val response = runner().run(wavFile(), config).getOrThrow()

        assertThat(response.language).isEqualTo("German")
        assertThat(response.languageProbability).isEqualTo(0.9f)
    }

    @Test
    fun `persistent HTTP 500 fails after three attempts with backoff`() = kotlinx.coroutines.runBlocking {
        mockStt(httpError(500))

        val result = runner().run(wavFile(), config)

        assertThat(result.isFailure).isTrue()
        assertThat(result.exceptionOrNull()!!.message).isEqualTo("HTTP 500")
        assertThat(delays).containsExactly(1L, 2L).inOrder()
    }

    @Test
    fun `IOException is transient and fails after three attempts`() = kotlinx.coroutines.runBlocking {
        val call = mockk<Call<SttResponse>>()
        every { call.execute() } throws IOException("read timeout")
        every { sttApi.transcribeAudioSync(any(), any(), any(), any(), any()) } returns call

        val result = runner().run(wavFile(), config)

        assertThat(result.isFailure).isTrue()
        assertThat(result.exceptionOrNull()!!.message).isEqualTo("read timeout")
        assertThat(delays).containsExactly(1L, 2L).inOrder()
        val completion = readLog("stt-latency")
        assertThat(completion["httpCode"]).isNull()
        assertThat(completion["error"]).isEqualTo("IOException")
    }

    /** Minimal structurally valid Ogg page (see AudioDurationTest). */
    private fun singlePageOgg(granulePosition: Long): ByteArray {
        val page = ByteArray(28)
        page[0] = 'O'.code.toByte()
        page[1] = 'g'.code.toByte()
        page[2] = 'g'.code.toByte()
        page[3] = 'S'.code.toByte()
        page[4] = 0
        for (i in 0 until 8) page[6 + i] = ((granulePosition shr (8 * i)) and 0xFF).toByte()
        page[26] = 1
        page[27] = 0
        return page
    }

    // ---- #117: prompt carry-over + evidence suppression --------------------

    private fun stubWithPromptSlot(): io.mockk.CapturingSlot<okhttp3.RequestBody?> {
        val promptSlot = slot<okhttp3.RequestBody?>()
        val call = mockk<Call<SttResponse>>()
        every { call.execute() } returns success()
        every {
            sttApi.transcribeAudioSync(any(), any(), any(), any(), captureNullable(promptSlot))
        } returns call
        return promptSlot
    }

    @Test
    fun `prompt is forwarded as the prompt multipart part`() = kotlinx.coroutines.runBlocking {
        val promptSlot = stubWithPromptSlot()

        val result = runner().run(wavFile(), config, chunk = 1, prompt = "context tail")

        assertThat(result.isSuccess).isTrue()
        val buffer = okio.Buffer()
        promptSlot.captured!!.writeTo(buffer)
        assertThat(buffer.readUtf8()).isEqualTo("context tail")
        // prompt marker in the completion record (#117) keeps the Phase 2
        // t(S) fit unpolluted by prompt-induced latency
        assertThat((readLog("stt-latency")["promptChars"] as Double).toInt()).isEqualTo(12)
    }

    @Test
    fun `no prompt means the prompt part is omitted`() = kotlinx.coroutines.runBlocking {
        val promptSlot = stubWithPromptSlot()

        val result = runner().run(wavFile(), config, chunk = 1)

        assertThat(result.isSuccess).isTrue()
        assertThat(promptSlot.captured).isNull()
        assertThat(readLog("stt-latency").containsKey("promptChars")).isFalse()
    }
}
