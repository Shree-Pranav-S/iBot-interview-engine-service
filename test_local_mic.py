import asyncio
import os
import sys

# Add the parent directory to sys.path so we can import src.config.settings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyaudio
except ImportError:
    print("Please install pyaudio to run this test:")
    print("pip install pyaudio")
    sys.exit(1)

from src.core.services.stt_service import STTService

# Audio recording parameters
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16


async def microphone_stream(stt: STTService):
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("\n🎤 Microphone is LIVE! Start speaking...")
    print("   (Press Ctrl+C to stop)\n")

    try:
        loop = asyncio.get_event_loop()
        while True:
            # Read audio data from the microphone without blocking the event loop
            data = await loop.run_in_executor(None, stream.read, CHUNK, False)

            # Send the raw PCM bytes to Deepgram
            await stt.send_audio(data)

            # Yield control back to the event loop so Deepgram can process
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        pass
    finally:
        print("\nStopping microphone...")
        stream.stop_stream()
        stream.close()
        p.terminate()


async def read_transcripts(stt: STTService):
    try:
        async for event in stt.transcript_events():
            if event.event_type == "transcript":
                if event.is_final:
                    print(
                        f"✅ Final Transcript: {event.text} (Confidence: {event.confidence:.2f})"
                    )
                else:
                    # Print interim results on the same line
                    sys.stdout.write(f"\r⏳ Interim: {event.text}\033[K")
                    sys.stdout.flush()
            elif event.event_type == "speech_started":
                print("\n[Speech Detected]")
            elif event.event_type == "eager_end_of_turn":
                print(f"[Eager End of Turn] {event.text}")

    except asyncio.CancelledError:
        pass


async def main():
    async with STTService(encoding="linear16", sample_rate=16000) as stt:
        mic_task = asyncio.create_task(microphone_stream(stt))
        transcript_task = asyncio.create_task(read_transcripts(stt))

        try:
            # Run forever until the user presses Ctrl+C
            await asyncio.gather(mic_task, transcript_task)
        except KeyboardInterrupt:
            print("\nShutting down...")
            mic_task.cancel()
            transcript_task.cancel()
            await asyncio.gather(mic_task, transcript_task, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
