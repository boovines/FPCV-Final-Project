#!/usr/bin/env python3

import json
import time
import statistics
import psutil
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient
from mode_switch_detector import ModeSwitchDetector

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

import cv2
import base64
import io
from PIL import Image
import requests
from dotenv import load_dotenv


class SystemEvaluator:
    def __init__(self, output_file: str = "system_eval_results.json"):
        self.output_file = output_file
        load_dotenv()
        
        self.hand_detector = HandDetector()
        self.frame_detector = FrameDetector()
        self.mode_detector = ModeSwitchDetector(hold_duration=1.0)
        self.crop_utils = CropUtils()
        self.openai_client = OpenAIClient()
        
        self.metrics = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "frame_processing": {
                "fps_samples": [],
                "per_frame_times": [],
                "mediapipe_times": []
            },
            "gesture_reliability": {
                "stabilization_times": [],
                "false_reset_count": 0,
                "total_gesture_attempts": 0
            },
            "crop_performance": {
                "total_crop_times": [],
                "successful_crops": 0,
                "rejected_crops": 0
            },
            "end_to_end_latency": {
                "total_times": [],
                "api_inference_times": [],
                "image_encoding_times": []
            },
            "resource_footprint": {
                "cpu_samples": [],
                "memory_samples": []
            }
        }
        
        self.frame_count = 0
        self.last_frame_time = None
        self.last_capture_time = 0
        self.capture_cooldown = 3.0
        self.capture_times = []
        self.false_reset_count = 0
        self.gesture_attempts = 0
        
        self.eleven_api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
        self.eleven_tts_voice_id = (
            os.getenv("ELEVENLABS_TTS_VOICE_ID") or 
            os.getenv("ELEVENLABS_VOICE_ID") or 
            os.getenv("ELEVEN_VOICE_ID")
        )
        
        if sr:
            self.recognizer = sr.Recognizer()
            self.recognizer.pause_threshold = 1.0
            self.recognizer.non_speaking_duration = 0.4
            self.recognizer.dynamic_energy_threshold = True
            self.audio_sample_rate = 16000
            self.listen_timeout = float(os.getenv("ELEVENLABS_LISTEN_TIMEOUT", 12))
            self.max_listen_duration = float(os.getenv("ELEVENLABS_MAX_LISTEN_SECONDS", 25))
        
        if self.eleven_api_key and ElevenLabs:
            try:
                self.eleven_client = ElevenLabs(api_key=self.eleven_api_key)
            except:
                self.eleven_client = None
        else:
            self.eleven_client = None
        
        self.process = psutil.Process(os.getpid())
        
    def evaluate_frame_processing(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        frame_start = time.perf_counter()
        
        mp_start = time.perf_counter()
        annotated_frame, hand_data = self.hand_detector.process_frame(frame)
        mp_time = (time.perf_counter() - mp_start) * 1000
        
        finger_tips = self.hand_detector.get_finger_tips(hand_data)
        gesture_detected, corners, progress = self.frame_detector.detect_frame_gesture(
            hand_data, finger_tips
        )
        
        overlay_frame = self.crop_utils.draw_frame_overlay(annotated_frame, corners, progress)
        
        total_frame_time = (time.perf_counter() - frame_start) * 1000
        
        self.metrics["frame_processing"]["per_frame_times"].append(total_frame_time)
        self.metrics["frame_processing"]["mediapipe_times"].append(mp_time)
        
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
            if frame_interval > 0:
                fps = 1.0 / frame_interval
                self.metrics["frame_processing"]["fps_samples"].append(fps)
        self.last_frame_time = current_time
        
        return overlay_frame, {
            "hand_data": hand_data,
            "finger_tips": finger_tips,
            "gesture_detected": gesture_detected,
            "corners": corners,
            "progress": progress
        }
    
    def evaluate_gesture_latency(self, corners: Optional[List], gesture_detected: bool):
        if corners is None:
            if self.frame_detector.gesture_start_time is not None:
                self.false_reset_count += 1
            return
        
        self.gesture_attempts += 1
        
        if self.frame_detector.gesture_start_time is not None:
            if gesture_detected:
                stabilization_time = time.time() - self.frame_detector.gesture_start_time
                self.metrics["gesture_reliability"]["stabilization_times"].append(stabilization_time)
    
    def evaluate_crop_performance(self, frame: np.ndarray, corners: List) -> Optional[np.ndarray]:
        if corners is None or len(corners) != 4:
            return None
        
        crop_start = time.perf_counter()
        cropped = self.crop_utils.crop_frame_region(frame, corners)
        is_valid = self.crop_utils.validate_crop_quality(cropped) if cropped is not None else False
        total_crop_time = (time.perf_counter() - crop_start) * 1000
        
        self.metrics["crop_performance"]["total_crop_times"].append(total_crop_time)
        
        if is_valid:
            self.metrics["crop_performance"]["successful_crops"] += 1
        else:
            self.metrics["crop_performance"]["rejected_crops"] += 1
        
        return cropped
    
    def evaluate_speech_pipeline(self) -> Optional[str]:
        if not sr or not self.eleven_api_key:
            return None
        
        try:
            with sr.Microphone(sample_rate=self.audio_sample_rate) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.max_listen_duration,
                    )
                except sr.WaitTimeoutError:
                    return None
                
                audio_bytes = audio.get_wav_data()
                transcript = self._transcribe_audio(audio_bytes)
                return transcript
        except Exception as e:
            print(f"Speech pipeline error: {e}")
            return None
    
    def _transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        if not self.eleven_client:
            return None
        
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "question.wav"
            result = self.eleven_client.speech_to_text.convert(
                file=audio_buffer,
                model_id=os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1"),
                language_code=os.getenv("ELEVENLABS_STT_LANGUAGE", "en"),
            )
            
            if hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict):
                return result.get('text') or result.get('transcription')
        except Exception as e:
            print(f"STT error: {e}")
            return None
    
    def evaluate_llm_response(self, cropped_image: np.ndarray, prompt: str) -> Optional[str]:
        end_to_end_start = time.perf_counter()
        
        encode_start = time.perf_counter()
        image_base64 = self.crop_utils.encode_image_to_base64(cropped_image)
        encode_time = (time.perf_counter() - encode_start) * 1000
        self.metrics["end_to_end_latency"]["image_encoding_times"].append(encode_time)
        
        if not image_base64:
            return None
        
        api_start = time.perf_counter()
        response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
        api_inference_time = (time.perf_counter() - api_start) * 1000
        self.metrics["end_to_end_latency"]["api_inference_times"].append(api_inference_time)
        
        end_to_end_time = (time.perf_counter() - end_to_end_start) * 1000
        self.metrics["end_to_end_latency"]["total_times"].append(end_to_end_time)
        
        return response
    
    def evaluate_tts(self, text: str):
        if not self.eleven_client or not self.eleven_tts_voice_id:
            return
        
        try:
            tts_api = self.eleven_client.text_to_speech
            call_kwargs = {
                "voice_id": self.eleven_tts_voice_id,
                "model_id": os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_turbo_v2"),
                "text": text,
            }
            output_format = os.getenv("ELEVENLABS_TTS_OUTPUT_FORMAT", "pcm_16000")
            if output_format:
                call_kwargs["output_format"] = output_format
            
            try:
                raw_audio = tts_api.convert(**call_kwargs)
                if hasattr(raw_audio, 'audio'):
                    _ = raw_audio.audio
            except:
                pass
        except Exception as e:
            print(f"TTS error: {e}")
    
    
    def sample_resource_utilization(self):
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.metrics["resource_footprint"]["cpu_samples"].append(cpu_percent)
            self.metrics["resource_footprint"]["memory_samples"].append(memory_mb)
        except Exception as e:
            print(f"Resource sampling error: {e}")
    
    def run_evaluation(self, duration_seconds: int = 300, min_captures: int = 5):
        print("Starting System Evaluation")
        print("=" * 50)
        print(f"Duration: {duration_seconds} seconds or {min_captures} captures")
        print("Press 'q' to quit early, 'c' to force capture")
        print()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        captures_collected = 0
        
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > duration_seconds and captures_collected >= min_captures:
                    break
                
                if self.frame_count % 30 == 0:
                    self.sample_resource_utilization()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                overlay_frame, frame_data = self.evaluate_frame_processing(frame)
                
                self.evaluate_gesture_latency(
                    frame_data["corners"],
                    frame_data["gesture_detected"]
                )
                
                self.metrics["gesture_reliability"]["total_gesture_attempts"] = self.gesture_attempts
                self.metrics["gesture_reliability"]["false_reset_count"] = self.false_reset_count
                
                if frame_data["gesture_detected"] and frame_data["corners"]:
                    current_time = time.time()
                    if current_time - self.last_capture_time >= self.capture_cooldown:
                        print(f"\n[Capture #{captures_collected + 1}] Processing...")
                        
                        cropped = self.evaluate_crop_performance(frame, frame_data["corners"])
                        
                        if cropped is not None and self.crop_utils.validate_crop_quality(cropped):
                            prompt = self.evaluate_speech_pipeline()
                            if not prompt:
                                prompt = "What do you see in this image?"
                            
                            response = self.evaluate_llm_response(cropped, prompt)
                            
                            if response:
                                self.evaluate_tts(response)
                            
                            captures_collected += 1
                            print(f"  Capture completed. Total: {captures_collected}")
                            
                            self.frame_detector.reset()
                        else:
                            print("  Crop validation failed")
                
                cv2.imshow('System Evaluation', overlay_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if frame_data["corners"]:
                        cropped = self.evaluate_crop_performance(frame, frame_data["corners"])
                        if cropped:
                            prompt = "What do you see in this image?"
                            response = self.evaluate_llm_response(cropped, prompt)
                            if response:
                                self.evaluate_tts(response)
                            captures_collected += 1
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.cleanup()
    
    def compute_statistics(self) -> Dict:
        stats = {}
        
        fps_samples = self.metrics["frame_processing"]["fps_samples"]
        per_frame_times = self.metrics["frame_processing"]["per_frame_times"]
        mp_times = self.metrics["frame_processing"]["mediapipe_times"]
        
        avg_fps = statistics.mean(fps_samples) if fps_samples else 0
        avg_mp_time = statistics.mean(mp_times) if mp_times else 0
        avg_frame_time = statistics.mean(per_frame_times) if per_frame_times else 0
        mp_percentage = (avg_mp_time / avg_frame_time * 100) if avg_frame_time > 0 else 0
        
        stats["throughput_bottleneck"] = {
            "fps": round(avg_fps, 2),
            "mediapipe_time_ms_per_frame": round(avg_mp_time, 2),
            "mediapipe_percentage_of_frame_time": round(mp_percentage, 1)
        }
        
        stab_times = self.metrics["gesture_reliability"]["stabilization_times"]
        false_resets = self.metrics["gesture_reliability"]["false_reset_count"]
        total_attempts = self.metrics["gesture_reliability"]["total_gesture_attempts"]
        
        avg_stab_time = statistics.mean(stab_times) if stab_times else 0
        std_stab_time = statistics.stdev(stab_times) if len(stab_times) > 1 else 0
        false_reset_rate = (false_resets / max(total_attempts, 1)) * 100
        
        stats["gesture_reliability"] = {
            "stabilization_time_seconds": {
                "mean": round(avg_stab_time, 2),
                "std": round(std_stab_time, 2)
            },
            "false_reset_rate_percent": round(false_reset_rate, 1)
        }
        
        crop_times = self.metrics["crop_performance"]["total_crop_times"]
        successful = self.metrics["crop_performance"]["successful_crops"]
        rejected = self.metrics["crop_performance"]["rejected_crops"]
        total_crops = successful + rejected
        
        avg_crop_time = statistics.mean(crop_times) if crop_times else 0
        success_rate = (successful / max(total_crops, 1)) * 100
        
        stats["crop_performance"] = {
            "total_crop_time_ms": round(avg_crop_time, 2),
            "success_rate_percent": round(success_rate, 1)
        }
        
        total_times = self.metrics["end_to_end_latency"]["total_times"]
        api_times = self.metrics["end_to_end_latency"]["api_inference_times"]
        encode_times = self.metrics["end_to_end_latency"]["image_encoding_times"]
        
        avg_total = statistics.mean(total_times) if total_times else 0
        avg_api = statistics.mean(api_times) if api_times else 0
        avg_encode = statistics.mean(encode_times) if encode_times else 0
        
        stats["end_to_end_latency"] = {
            "total_time_ms": round(avg_total, 2),
            "api_inference_time_ms": round(avg_api, 2),
            "image_encoding_time_ms": round(avg_encode, 2)
        }
        
        cpu_samples = self.metrics["resource_footprint"]["cpu_samples"]
        memory_samples = self.metrics["resource_footprint"]["memory_samples"]
        
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
        avg_memory = statistics.mean(memory_samples) if memory_samples else 0
        
        stats["resource_footprint"] = {
            "cpu_usage_percent_mean": round(avg_cpu, 1),
            "memory_usage_mb_mean": round(avg_memory, 1)
        }
        
        return stats
    
    def save_results(self):
        print("\nComputing statistics...")
        stats = self.compute_statistics()
        
        results = {
            "evaluation_metadata": {
                "timestamp": self.metrics["evaluation_timestamp"],
                "total_frames_processed": self.frame_count,
                "total_captures": len(self.metrics["end_to_end_latency"]["total_times"])
            },
            "summary_statistics": stats
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.output_file}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total captures: {len(self.metrics['end_to_end_latency']['total_times'])}")
        
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"FPS: {stats['throughput_bottleneck']['fps']:.2f}")
        print(f"MediaPipe: {stats['throughput_bottleneck']['mediapipe_time_ms_per_frame']:.2f} ms/frame ({stats['throughput_bottleneck']['mediapipe_percentage_of_frame_time']:.1f}%)")
        print(f"Stabilization time: {stats['gesture_reliability']['stabilization_time_seconds']['mean']:.2f} s (σ = {stats['gesture_reliability']['stabilization_time_seconds']['std']:.2f} s)")
        print(f"False reset rate: {stats['gesture_reliability']['false_reset_rate_percent']:.1f}%")
        print(f"Crop time: {stats['crop_performance']['total_crop_time_ms']:.2f} ms")
        print(f"Crop success rate: {stats['crop_performance']['success_rate_percent']:.1f}%")
        print(f"End-to-end latency: {stats['end_to_end_latency']['total_time_ms']:.2f} ms")
        print(f"  - API+inference: {stats['end_to_end_latency']['api_inference_time_ms']:.2f} ms")
        print(f"  - Image encoding: {stats['end_to_end_latency']['image_encoding_time_ms']:.2f} ms")
        print(f"CPU usage: {stats['resource_footprint']['cpu_usage_percent_mean']:.1f}%")
        print(f"Memory usage: {stats['resource_footprint']['memory_usage_mb_mean']:.1f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="System Evaluation for Vision-Prompt Glasses")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Evaluation duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--min-captures",
        type=int,
        default=5,
        help="Minimum number of captures to collect (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="system_eval_results.json",
        help="Output JSON file path (default: system_eval_results.json)"
    )
    
    args = parser.parse_args()
    
    evaluator = SystemEvaluator(output_file=args.output)
    evaluator.run_evaluation(
        duration_seconds=args.duration,
        min_captures=args.min_captures
    )
    evaluator.save_results()


if __name__ == "__main__":
    main()

