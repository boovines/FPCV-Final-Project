"""
Audio Service Module

Handles all audio-related functionality including:
- Speech-to-Text (STT) using ElevenLabs
- Text-to-Speech (TTS) using ElevenLabs
- Audio recording and playback
"""

import base64
import io
import os
import requests
from typing import Optional

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import stream
except ImportError:
    ElevenLabs = None
    stream = None


class AudioService:
    """Manages all audio input/output operations."""
    
    def __init__(self):
        """Initialize audio service with ElevenLabs and speech recognition."""
        # Validate speech recognition availability
        if sr is None:
            raise ImportError(
                "speech_recognition package is required for voice input. "
                "Install it with 'pip install SpeechRecognition' and ensure PyAudio dependencies are available."
            )
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.non_speaking_duration = 0.4
        self.recognizer.dynamic_energy_threshold = True
        self.audio_sample_rate = 16000
        self.listen_timeout = float(os.getenv("ELEVENLABS_LISTEN_TIMEOUT", 12))
        self.max_listen_duration = float(os.getenv("ELEVENLABS_MAX_LISTEN_SECONDS", 25))
        
        # Get ElevenLabs API key
        self.eleven_api_key = (
            os.getenv("ELEVENLABS_API_KEY")
            or os.getenv("ELEVEN_API_KEY")
        )
        if not self.eleven_api_key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY or ELEVEN_API_KEY in your environment."
            )
        
        # Get TTS voice ID
        self.eleven_tts_voice_id = (
            os.getenv("ELEVENLABS_TTS_VOICE_ID")
            or os.getenv("ELEVENLABS_VOICE_ID")
            or os.getenv("ELEVEN_VOICE_ID")
        )
        if not self.eleven_tts_voice_id:
            print(
                "Warning: ElevenLabs voice ID not configured. Set ELEVENLABS_TTS_VOICE_ID to enable TTS playback."
            )
        
        # TTS configuration
        self.eleven_tts_model_id = os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_turbo_v2")
        self.eleven_tts_output_format = os.getenv("ELEVENLABS_TTS_OUTPUT_FORMAT", "pcm_16000")
        self.eleven_tts_sample_rate = self._infer_sample_rate(self.eleven_tts_output_format)
        self.use_local_tts_playback = self.eleven_tts_output_format.startswith("pcm_")
        
        # STT configuration
        self.eleven_stt_model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")
        self.eleven_stt_language = os.getenv("ELEVENLABS_STT_LANGUAGE", "en")
        self.eleven_stt_endpoint = os.getenv(
            "ELEVENLABS_STT_ENDPOINT",
            "https://api.elevenlabs.io/v1/speech-to-text",
        )
        
        # Initialize ElevenLabs client
        self.eleven_client = None
        self.tts_ready = False
        
        if ElevenLabs is None or stream is None:
            print(
                "Warning: ElevenLabs SDK streaming utilities not available; AI responses will be text-only."
            )
        else:
            self._initialize_elevenlabs_client()
        
        # Check PyAudio for local playback
        if self.use_local_tts_playback:
            try:
                import pyaudio  # noqa: F401
            except Exception:
                print(
                    "Note: ELEVENLABS_TTS_OUTPUT_FORMAT is set to a PCM format but PyAudio is not available. "
                    "Either install PyAudio or set ELEVENLABS_TTS_OUTPUT_FORMAT=mp3_44100_128 to use SDK streaming playback."
                )
    
    def _initialize_elevenlabs_client(self):
        """Initialize ElevenLabs SDK client and auto-select voice if needed."""
        try:
            self.eleven_client = ElevenLabs(api_key=self.eleven_api_key)
            self.tts_ready = bool(self.eleven_tts_voice_id)
            
            if not self.tts_ready:
                # Try to auto-select a voice
                try:
                    selected_voice = None
                    voices = getattr(self.eleven_client, "voices", None)
                    list_method = getattr(voices, "list", None) if voices else None
                    
                    if callable(list_method):
                        voice_list = list_method()
                        candidate_voices = []
                        if isinstance(voice_list, dict) and "voices" in voice_list:
                            candidate_voices = voice_list.get("voices") or []
                        elif hasattr(voice_list, "voices"):
                            candidate_voices = getattr(voice_list, "voices") or []
                        else:
                            candidate_voices = voice_list if isinstance(voice_list, (list, tuple)) else []
                        
                        for v in candidate_voices:
                            vid = v.get("voice_id") if isinstance(v, dict) else (getattr(v, "voice_id", None) or getattr(v, "id", None))
                            if vid:
                                selected_voice = vid
                                break
                    
                    if selected_voice is None:
                        rest = requests.get(
                            "https://api.elevenlabs.io/v1/voices",
                            headers={"xi-api-key": self.eleven_api_key},
                            timeout=20,
                        )
                        rest.raise_for_status()
                        data = rest.json() or {}
                        for v in data.get("voices", []) or []:
                            vid = v.get("voice_id") or v.get("id")
                            if vid:
                                selected_voice = vid
                                break
                    
                    if selected_voice:
                        self.eleven_tts_voice_id = selected_voice
                        self.tts_ready = True
                        print(f"Note: No ELEVENLABS_TTS_VOICE_ID set; using default voice: {selected_voice}")
                    else:
                        print("Warning: No voices available in ElevenLabs account. Create a voice and set ELEVENLABS_TTS_VOICE_ID.")
                except Exception as voice_err:
                    print(f"Warning: Could not auto-select a voice ({voice_err}). Set ELEVENLABS_TTS_VOICE_ID.")
        except Exception as sdk_error:
            print(f"Warning: ElevenLabs SDK initialization failed ({sdk_error}). Falling back to REST API calls.")
    
    def capture_spoken_prompt(self) -> Optional[str]:
        """Record the user's spoken question and transcribe it via ElevenLabs."""
        try:
            with sr.Microphone(sample_rate=self.audio_sample_rate) as source:
                print("\nListening... (speak your question)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.max_listen_duration,
                    )
                except sr.WaitTimeoutError:
                    print("No speech detected within the listening window.")
                    return None
            
            print("Processing speech...")
            audio_bytes = audio.get_wav_data()
            transcript = self.transcribe_audio_with_elevenlabs(audio_bytes)
            
            if transcript:
                print(f"You said: {transcript}")
                return transcript
            
            print("Transcription failed or returned empty text.")
            return None
        
        except sr.UnknownValueError:
            print("Speech was unintelligible. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error capturing audio: {e}")
            return None
    
    def transcribe_audio_with_elevenlabs(self, audio_bytes: bytes) -> Optional[str]:
        """Send recorded audio to ElevenLabs Speech-to-Text."""
        if not audio_bytes:
            return None
        
        # Try SDK method first
        if self.eleven_client is not None:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "question.wav"
            try:
                result = self.eleven_client.speech_to_text.convert(
                    file=audio_buffer,
                    model_id=self.eleven_stt_model_id,
                    language_code=self.eleven_stt_language,
                )
                
                transcript = self._extract_transcript_from_result(result)
                if transcript:
                    return transcript
            except Exception as sdk_error:
                print(f"ElevenLabs SDK transcription error: {sdk_error}. Falling back to REST API.")
        
        # Fall back to REST API
        try:
            response = requests.post(
                self.eleven_stt_endpoint,
                headers={
                    "xi-api-key": self.eleven_api_key,
                },
                data={
                    "model_id": self.eleven_stt_model_id,
                    "language_code": self.eleven_stt_language,
                },
                files={
                    "file": ("question.wav", audio_bytes, "audio/wav"),
                },
                timeout=90,
            )
            response.raise_for_status()
            
            result = response.json()
            transcript = self._extract_transcript_from_result(result)
            if transcript:
                return transcript
            
            print("ElevenLabs STT response did not include a transcript.")
            return None
        
        except requests.RequestException as rest_error:
            print(f"ElevenLabs REST transcription error: {rest_error}")
            return None
    
    def output_ai_response(self, ai_response: Optional[str]):
        """Stream the AI response via ElevenLabs TTS."""
        if not ai_response:
            return
        
        if self.tts_ready and self.eleven_client is not None and stream is not None:
            try:
                tts_api = self.eleven_client.text_to_speech
                
                call_kwargs = {
                    "voice_id": self.eleven_tts_voice_id,
                    "model_id": self.eleven_tts_model_id,
                    "text": ai_response,
                }
                
                if self.eleven_tts_output_format:
                    call_kwargs["output_format"] = self.eleven_tts_output_format
                
                audio_chunks = tuple()
                
                stream_method = getattr(tts_api, "stream", None)
                if callable(stream_method):
                    try:
                        raw_audio = stream_method(**call_kwargs)
                    except TypeError:
                        call_kwargs.pop("output_format", None)
                        raw_audio = stream_method(**call_kwargs)
                    audio_chunks = self._materialize_audio_chunks(raw_audio)
                else:
                    convert_method = getattr(tts_api, "convert", None)
                    if not callable(convert_method):
                        raise RuntimeError("ElevenLabs SDK does not expose a stream or convert method.")
                    
                    convert_kwargs = call_kwargs.copy()
                    try:
                        tts_result = convert_method(**convert_kwargs)
                    except TypeError:
                        convert_kwargs.pop("output_format", None)
                        tts_result = convert_method(**convert_kwargs)
                    
                    if callable(getattr(tts_result, "stream", None)):
                        audio_chunks = self._materialize_audio_chunks(tts_result.stream())
                    else:
                        audio_attr = getattr(tts_result, "audio", None)
                        audio_chunks = self._materialize_audio_chunks(audio_attr)
                        
                        if not audio_chunks and isinstance(tts_result, dict):
                            audio_chunks = self._materialize_audio_chunks(tts_result.get("audio"))
                        
                        if not audio_chunks:
                            audio_chunks = self._materialize_audio_chunks(tts_result)
                
                if not audio_chunks:
                    raise RuntimeError("ElevenLabs TTS returned no audio chunks to stream.")
                
                playback_success = False
                
                if self.use_local_tts_playback:
                    audio_bytes = self._collect_audio_bytes(audio_chunks)
                    if audio_bytes:
                        playback_success = self._play_audio_bytes(audio_bytes, self.eleven_tts_sample_rate)
                
                if not playback_success:
                    stream(chunk for chunk in audio_chunks)
            except Exception as tts_error:
                print(f"[TTS stream error] {tts_error}")
                print(f"\nResponse:\n{ai_response}\n")
                return
        
        print(f"\nResponse:\n{ai_response}\n")
    
    @staticmethod
    def _extract_transcript_from_result(result: Optional[object]) -> Optional[str]:
        """Extract transcript text from various result formats."""
        if result is None:
            return None
        
        for attr in ("text", "transcription", "transcript"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        
        if isinstance(result, dict):
            for key in ("text", "transcription", "transcript"):
                if key in result and isinstance(result[key], str) and result[key].strip():
                    return result[key].strip()
            
            segments = result.get("segments") if isinstance(result.get("segments"), list) else None
            if segments:
                combined = " ".join(
                    segment.get("text", "").strip()
                    for segment in segments
                    if isinstance(segment, dict)
                ).strip()
                if combined:
                    return combined
        
        return None
    
    @staticmethod
    def _infer_sample_rate(output_format: Optional[str]) -> int:
        """Infer sample rate from output format string."""
        default_rate = 16000
        
        if not output_format or "pcm" not in output_format:
            return default_rate
        
        try:
            suffix = output_format.split("pcm_", 1)[1]
        except IndexError:
            return default_rate
        
        digits = []
        for char in suffix:
            if char.isdigit():
                digits.append(char)
            else:
                break
        
        if not digits:
            return default_rate
        
        try:
            inferred = int("".join(digits))
            return inferred if inferred > 0 else default_rate
        except ValueError:
            return default_rate
    
    @staticmethod
    def _materialize_audio_chunks(audio_candidate: Optional[object]) -> tuple:
        """Convert audio candidate to tuple of chunks."""
        if audio_candidate is None:
            return tuple()
        
        if isinstance(audio_candidate, (bytes, bytearray, str, dict)):
            return (audio_candidate,)
        
        iterator = getattr(audio_candidate, "__iter__", None)
        if callable(iterator):
            try:
                return tuple(audio_candidate)
            except TypeError:
                return tuple()
        
        return tuple()
    
    @staticmethod
    def _extract_audio_payload(chunk: Optional[object]) -> Optional[object]:
        """Extract audio data from chunk."""
        if chunk is None:
            return None
        
        if isinstance(chunk, dict):
            for key in ("audio", "data", "chunk", "value"):
                if key in chunk and chunk[key] is not None:
                    return chunk[key]
            return None
        
        return chunk
    
    @staticmethod
    def _collect_audio_bytes(audio_chunks: tuple) -> bytes:
        """Collect audio bytes from chunks."""
        if not audio_chunks:
            return b""
        
        audio_buffer = bytearray()
        
        for raw_chunk in audio_chunks:
            payload = AudioService._extract_audio_payload(raw_chunk)
            if payload is None:
                continue
            
            if isinstance(payload, str):
                try:
                    decoded = base64.b64decode(payload, validate=True)
                except (ValueError, TypeError):
                    continue
                audio_buffer.extend(decoded)
            elif isinstance(payload, (bytes, bytearray, memoryview)):
                audio_buffer.extend(bytes(payload))
        
        return bytes(audio_buffer)
    
    @staticmethod
    def _play_audio_bytes(audio_bytes: bytes, sample_rate: int) -> bool:
        """Play audio bytes using PyAudio."""
        if not audio_bytes:
            return False
        
        try:
            import pyaudio
        except ImportError:
            print("PyAudio not available for TTS playback; skipping audio output.")
            return False
        
        pyaudio_instance = None
        stream_handle = None
        
        try:
            pyaudio_instance = pyaudio.PyAudio()
            stream_handle = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            
            chunk_size = 4096
            for index in range(0, len(audio_bytes), chunk_size):
                stream_handle.write(audio_bytes[index:index + chunk_size])
            
            return True
        except Exception as playback_error:
            print(f"Local TTS playback failed: {playback_error}")
            return False
        finally:
            if stream_handle is not None:
                try:
                    stream_handle.stop_stream()
                    stream_handle.close()
                except Exception:
                    pass
            
            if pyaudio_instance is not None:
                try:
                    pyaudio_instance.terminate()
                except Exception:
                    pass
