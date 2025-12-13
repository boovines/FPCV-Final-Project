import base64
import io
import cv2
import numpy as np
import os
import time
import requests
from datetime import datetime
from typing import Optional

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient
from mode_switch_detector import ModeSwitchDetector
from dotenv import load_dotenv

try:
    from google_drive_uploader import upload_snapshot_to_drive
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    upload_snapshot_to_drive = None

try:
    from infobip_whatsapp_sender import send_snapshot_via_whatsapp
    INFOBIP_AVAILABLE = True
except ImportError:
    INFOBIP_AVAILABLE = False
    send_snapshot_via_whatsapp = None

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

class VisionPromptGlasses:
    def __init__(self):
        """Initialize the Vision-Prompt Glasses prototype."""
        # Initialize components
        self.hand_detector = HandDetector()
        self.frame_detector = FrameDetector()
        self.mode_detector = ModeSwitchDetector(hold_duration=1.0)
        self.crop_utils = CropUtils()
        self.openai_client = OpenAIClient()
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Create snapshots directory
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # State tracking
        self.last_capture_time = 0
        self.capture_cooldown = 3.0  # seconds between captures

        # Load environment variables (useful when running outside CLI context)
        load_dotenv()

        # Verify speech recognition dependency
        if sr is None:
            raise ImportError(
                "speech_recognition package is required for voice input. "
                "Install it with 'pip install SpeechRecognition' and ensure PyAudio dependencies are available."
            )

        # Speech recognition configuration
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0  # Stop after ~1s silence
        self.recognizer.non_speaking_duration = 0.4
        self.recognizer.dynamic_energy_threshold = True
        self.audio_sample_rate = 16000
        self.listen_timeout = float(os.getenv("ELEVENLABS_LISTEN_TIMEOUT", 12))
        self.max_listen_duration = float(os.getenv("ELEVENLABS_MAX_LISTEN_SECONDS", 25))

        # ElevenLabs configuration
        self.eleven_api_key = (
            os.getenv("ELEVENLABS_API_KEY")
            or os.getenv("ELEVEN_API_KEY")
        )
        if not self.eleven_api_key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY or ELEVEN_API_KEY in your environment."
            )

        self.eleven_tts_voice_id = (
            os.getenv("ELEVENLABS_TTS_VOICE_ID")
            or os.getenv("ELEVENLABS_VOICE_ID")
            or os.getenv("ELEVEN_VOICE_ID")
        )
        if not self.eleven_tts_voice_id:
            print(
                "Warning: ElevenLabs voice ID not configured. Set ELEVENLABS_TTS_VOICE_ID to enable TTS playback."
            )

        self.eleven_tts_model_id = os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_turbo_v2")
        self.eleven_tts_output_format = os.getenv("ELEVENLABS_TTS_OUTPUT_FORMAT", "pcm_16000")
        self.eleven_tts_sample_rate = self._infer_sample_rate(self.eleven_tts_output_format)
        self.use_local_tts_playback = self.eleven_tts_output_format.startswith("pcm_")

        self.eleven_stt_model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")
        self.eleven_stt_language = os.getenv("ELEVENLABS_STT_LANGUAGE", "en")
        self.eleven_stt_endpoint = os.getenv(
            "ELEVENLABS_STT_ENDPOINT",
            "https://api.elevenlabs.io/v1/speech-to-text",
        )

        # Initialize ElevenLabs SDK if available (REST fallback otherwise)
        self.eleven_client = None
        self.tts_ready = False

        if ElevenLabs is None or stream is None:
            print(
                "Warning: ElevenLabs SDK streaming utilities not available; AI responses will be text-only."
            )
        else:
            try:
                self.eleven_client = ElevenLabs(api_key=self.eleven_api_key)
                self.tts_ready = bool(self.eleven_tts_voice_id)
                # If no voice configured, attempt to pick the first available voice
                if not self.tts_ready:
                    try:
                        selected_voice = None
                        # Try SDK listing first
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
                        # Fallback to REST if SDK listing is unavailable
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

        # Provide actionable guidance if local PCM playback is configured but PyAudio is missing
        if self.use_local_tts_playback:
            try:
                import pyaudio  # noqa: F401
            except Exception:
                print(
                    "Note: ELEVENLABS_TTS_OUTPUT_FORMAT is set to a PCM format but PyAudio is not available. "
                    "Either install PyAudio or set ELEVENLABS_TTS_OUTPUT_FORMAT=mp3_44100_128 to use SDK streaming playback."
                )
        
    def initialize_camera(self) -> bool:
        """Initialize webcam capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_cached_snapshots(self) -> list:
        """Get list of cached snapshot files."""
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    snapshot_files.append(os.path.join(self.snapshots_dir, filename))
            return sorted(snapshot_files, key=os.path.getmtime, reverse=True)
        except Exception as e:
            print(f"Error reading snapshots directory: {e}")
            return []
    
    def select_cached_snapshot(self) -> Optional[str]:
        """Allow user to select a cached snapshot for analysis."""
        snapshots = self.get_cached_snapshots()
        
        if not snapshots:
            print("No cached snapshots found.")
            return None
        
        print("\nAvailable cached snapshots:")
        for i, snapshot in enumerate(snapshots[:10]):  # Show last 10
            filename = os.path.basename(snapshot)
            print(f"{i + 1}. {filename}")
        
        try:
            choice = input("\nSelect snapshot number (or press Enter to cancel): ").strip()
            if not choice:
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(snapshots):
                return snapshots[index]
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None
    
    def process_cached_snapshot(self, snapshot_path: str):
        """Process a cached snapshot with OpenAI."""
        try:
            # Load image
            image = cv2.imread(snapshot_path)
            if image is None:
                print(f"Could not load image: {snapshot_path}")
                return
            
            # Encode to base64
            image_base64 = self.crop_utils.encode_image_to_base64(image)
            if not image_base64:
                print("Failed to encode image")
                return
            
            # Get user prompt
            prompt = input(f"\nEnter your question about the image '{os.path.basename(snapshot_path)}': ")
            if not prompt.strip():
                print("No prompt provided.")
                return
            
            # Send to OpenAI
            print("Analyzing image...")
            response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
            
            if response:
                self.output_ai_response(response)
            else:
                print("Failed to get response from AI.")
                
        except Exception as e:
            print(f"Error processing cached snapshot: {e}")
    
    def capture_and_analyze(self, frame: np.ndarray, corners: list, mode: int = 0):
        """
        Capture the framed region and route to appropriate mode handler.
        
        Args:
            frame: Video frame containing the gesture
            corners: Corner points of the detected rectangle gesture
            mode: Current active mode (0-4)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_capture_time < self.capture_cooldown:
            return
        
        try:
            # Crop the framed region
            cropped_image = self.crop_utils.crop_frame_region(frame, corners)
            if cropped_image is None:
                print("Failed to crop image")
                return
            
            # Validate crop quality
            if not self.crop_utils.validate_crop_quality(cropped_image):
                print("Cropped image quality is too poor")
                return
            
            # Rotate image 90 degrees to the left (counterclockwise) - universal rotation for all snapshots
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(self.snapshots_dir, f"snapshot_{timestamp}.jpg")
            
            if not self.crop_utils.save_cropped_image(cropped_image, snapshot_path):
                print("Failed to save snapshot")
                return
            
            print(f"\nSnapshot saved: {snapshot_path}")
            
            # Encode image for API
            image_base64 = self.crop_utils.encode_image_to_base64(cropped_image)
            if not image_base64:
                print("Failed to encode image")
                return
            
            # Route snapshot to appropriate mode handler
            self.handle_snapshot(
                cropped_image=cropped_image,
                snapshot_path=snapshot_path,
                image_base64=image_base64,
                mode=mode
            )
            
            # Update cooldown
            self.last_capture_time = current_time
            
        except Exception as e:
            print(f"Error in capture and analyze: {e}")

    def handle_snapshot(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str, mode: int):
        """
        Central router for snapshot handling based on active mode.
        
        This function dispatches snapshot processing to the appropriate mode handler.
        Each mode has a clearly defined responsibility and implementation pathway.
        
        Args:
            cropped_image: Cropped numpy array image
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
            mode: Current active mode (0-4)
        """
        if mode == 0:
            self.handle_mode_0(cropped_image, snapshot_path, image_base64)
        elif mode == 1:
            self.handle_mode_1(cropped_image, snapshot_path, image_base64)
        elif mode == 2:
            self.handle_mode_2(cropped_image, snapshot_path, image_base64)
        elif mode == 3:
            self.handle_mode_3(cropped_image, snapshot_path, image_base64)
        elif mode == 4:
            self.handle_mode_4(cropped_image, snapshot_path, image_base64)
        else:
            print(f"Unknown mode: {mode}. Defaulting to mode 0.")
            self.handle_mode_0(cropped_image, snapshot_path, image_base64)

    def handle_mode_0(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """
        Mode 0: AI Processing with STT/TTS
        
        Existing behavior - processes snapshot through:
        1. Speech-to-text (capture spoken prompt)
        2. OpenAI vision analysis
        3. Text-to-speech (announce AI response)
        
        Args:
            cropped_image: Cropped numpy array image
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
        """
        # Get spoken user prompt via ElevenLabs speech-to-text
        prompt = self.capture_spoken_prompt()
        if not prompt:
            print("No spoken prompt captured.")
            return
        
        # Send to OpenAI
        print("Analyzing image...")
        response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
        
        if response:
            self.output_ai_response(response)
        else:
            print("Failed to get response from AI.")

    def handle_mode_1(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """
        Mode 1: Google Drive Upload
        
        Uploads snapshot to Google Drive and announces completion via TTS.
        
        Reference: https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple
        
        Args:
            cropped_image: Cropped numpy array image (already rotated 90° left)
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
        """
        if not GOOGLE_DRIVE_AVAILABLE or upload_snapshot_to_drive is None:
            print("[Mode 1] Google Drive upload not available. Install required packages:")
            print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return
        
        print(f"[Mode 1] Uploading snapshot to Google Drive: {snapshot_path}")
        
        # Upload snapshot to Google Drive
        upload_success = upload_snapshot_to_drive(snapshot_path)
        
        if upload_success:
            # Announce completion via TTS
            self.output_ai_response("Upload completed.")
        else:
            print("[Mode 1] Upload failed. Check logs for details.")

    def handle_mode_2(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """
        Mode 2: WhatsApp Send via InfoBip
        
        Sends snapshot as WhatsApp image message using InfoBip's HTTPS API and announces completion via TTS.
        
        Reference: https://www.infobip.com/docs/api/channels/whatsapp/whatsapp-outbound-messages/whatsapp-text-and-media-messages/send-whatsapp-image-message
        
        Args:
            cropped_image: Cropped numpy array image (already rotated 90° left)
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
        """
        if not INFOBIP_AVAILABLE or send_snapshot_via_whatsapp is None:
            print("[Mode 2] InfoBip WhatsApp sender not available. Install required packages:")
            print("  pip install requests")
            return
        
        print(f"[Mode 2] Sending snapshot via WhatsApp: {snapshot_path}")
        
        # Send snapshot via WhatsApp using InfoBip
        send_success = send_snapshot_via_whatsapp(snapshot_path)
        
        if send_success:
            # Announce completion via TTS
            self.output_ai_response("Upload completed.")
        else:
            print("[Mode 2] WhatsApp send failed. Check logs for details.")

    def handle_mode_3(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """
        Mode 3: Location Detection
        
        Intended behavior (NOT IMPLEMENTED):
        1. Determine the physical location where the photo was taken
        2. Announce location via TTS
        
        Reference: https://developers.google.com/maps/documentation/places/web-service/overview
        
        Args:
            cropped_image: Cropped numpy array image
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
        """
        # TODO: Implement location detection
        # TODO: - Extract EXIF GPS data if available
        # TODO: - Use Google Places API or similar to identify location
        # TODO: - Convert coordinates to human-readable address/place name
        # TODO: - Announce location via TTS (e.g., "Photo taken at Central Park, New York")
        print(f"[Mode 3] Location detection not yet implemented for: {snapshot_path}")

    def handle_mode_4(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """
        Mode 4: Visual Product Search
        
        Intended behavior (NOT IMPLEMENTED):
        1. Find visually similar products using Google Cloud Vision Product Search
        2. Print user-friendly results to terminal
        
        Reference: https://docs.cloud.google.com/vision/product-search/docs/create-product-set-search-products
        
        Args:
            cropped_image: Cropped numpy array image
            snapshot_path: File path where snapshot was saved
            image_base64: Base64-encoded image string for API calls
        """
        # TODO: Implement visual product search
        # TODO: - Authenticate with Google Cloud Vision API
        # TODO: - Create/use product set and catalog
        # TODO: - Perform product search on image
        # TODO: - Format and print results to terminal (product names, prices, links, etc.)
        print(f"[Mode 4] Visual product search not yet implemented for: {snapshot_path}")

    def capture_spoken_prompt(self) -> Optional[str]:
        """Record the user's spoken question and transcribe it via ElevenLabs."""
        try:
            with sr.Microphone(sample_rate=self.audio_sample_rate) as source:
                print("\nListening... (speak your question)")
                # Calibrate to ambient noise for more reliable detection
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
        """Send recorded audio to ElevenLabs Speech-to-Text and return the transcript."""
        if not audio_bytes:
            return None

        # Attempt SDK transcription first
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

        # REST API fallback using requests
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

    @staticmethod
    def _extract_transcript_from_result(result: Optional[object]) -> Optional[str]:
        """Normalize various ElevenLabs STT response styles to a plain string."""
        if result is None:
            return None

        # Handle SDK response objects (may expose `.text` or `.transcription`)
        for attr in ("text", "transcription", "transcript"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Handle mapping/dict-style responses
        if isinstance(result, dict):
            for key in ("text", "transcription", "transcript"):
                if key in result and isinstance(result[key], str) and result[key].strip():
                    return result[key].strip()

            # Some responses may include segment arrays
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
        """Extract the PCM sample rate from ElevenLabs output format strings."""
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
        """Materialize potentially streaming audio payloads into a tuple for reuse."""
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
        """Pull the actual audio field out of various ElevenLabs chunk shapes."""
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
        """Decode base64/audio chunks into raw bytes."""
        if not audio_chunks:
            return b""

        audio_buffer = bytearray()

        for raw_chunk in audio_chunks:
            payload = VisionPromptGlasses._extract_audio_payload(raw_chunk)
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
        """Play raw PCM audio bytes using PyAudio."""
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
    
    def output_ai_response(self, ai_response: Optional[str]):
        """Stream the AI response via ElevenLabs TTS and print it as a fallback."""
        if not ai_response:
            return

        if self.tts_ready and self.eleven_client is not None and stream is not None:
            try:
                # --- begin TTS via ElevenLabs streaming ---
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
                    # Use ElevenLabs SDK streaming playback as a fallback
                    stream(chunk for chunk in audio_chunks)
                # --- end TTS via ElevenLabs streaming ---
            except Exception as tts_error:
                print(f"[TTS stream error] {tts_error}")
                print(f"\nAI Response:\n{ai_response}\n")
                return

        print(f"\nAI Response:\n{ai_response}\n")

    

    def run(self):
        """Main application loop."""
        print("Vision-Prompt Glasses Prototype")
        print("==============================")
        print("Instructions:")
        print("- Form a rectangle with both hands (thumb + index extended, others curled)")
        print("- Hold the gesture steady for 2 seconds")
        print("- Press 'q' to quit, 'r' to reset, 's' to select cached snapshot")
        print("- Press 't' to test OpenAI connection")
        print()
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Test OpenAI connection
        print("Testing OpenAI connection...")
        if not self.openai_client.test_connection():
            print("Warning: OpenAI connection test failed. Check your API key.")
        else:
            print("OpenAI connection successful!")
        
        print("\nStarting camera feed...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame with MediaPipe
                annotated_frame, hand_data = self.hand_detector.process_frame(frame)
                
                # Get finger tips
                finger_tips = self.hand_detector.get_finger_tips(hand_data)
                
                # Detect frame gesture
                gesture_detected, corners, progress = self.frame_detector.detect_frame_gesture(
                    hand_data, finger_tips
                )
                
                # Detect mode switch gesture (right hand only)
                mode_state = self.mode_detector.update(hand_data)
                
                # Draw overlay
                overlay_frame = self.crop_utils.draw_frame_overlay(annotated_frame, corners, progress)
                
                # Draw mode switch progress bar if a mode switch is in progress
                overlay_frame = self.crop_utils.draw_mode_switch_progress(
                    overlay_frame, mode_state.progress, mode_state.pending_mode
                )
                
                # Add instructions overlay
                cv2.putText(overlay_frame, "Form rectangle with both hands (thumb+index)", 
                           (10, overlay_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay_frame, "Hold steady for 2 seconds", 
                           (10, overlay_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display current mode in bottom-right corner
                mode_text = f"Mode: {mode_state.current_mode}"
                text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = overlay_frame.shape[1] - text_size[0] - 20
                text_y = overlay_frame.shape[0] - 20
                cv2.putText(overlay_frame, mode_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Capture if gesture detected
                if gesture_detected and corners:
                    self.capture_and_analyze(frame, corners, mode_state.current_mode)
                    # Removed blocking waitKey call - continue with the frame feed
                    self.frame_detector.reset()
                    # Directly continue with the next frames
                
                # Display frame
                cv2.imshow('Vision-Prompt Glasses', overlay_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.frame_detector.reset()
                    print("Gesture detection reset")
                elif key == ord('s'):
                    snapshot_path = self.select_cached_snapshot()
                    if snapshot_path:
                        self.process_cached_snapshot(snapshot_path)
                elif key == ord('t'):
                    if self.openai_client.test_connection():
                        print("OpenAI connection test: SUCCESS")
                    else:
                        print("OpenAI connection test: FAILED")
                
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.hand_detector.cleanup()
        
        print("Cleanup completed")

def main():
    """Main entry point."""
    try:
        app = VisionPromptGlasses()
        app.run()
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    main()