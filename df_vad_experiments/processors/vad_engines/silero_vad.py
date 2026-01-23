import os
import torch
import numpy as np
from scipy import signal
from .base_vad import BaseVAD
import time
import logging

# 모델 기본 경로
MODELS_BASE_DIR = os.path.expanduser("~/.aeirobot_models")

# Silero VAD 모델 정보
SILERO_VAD_REPO_ID = "snakers4/silero-vad"
SILERO_VAD_MODEL_NAME_IN_REPO = "silero_vad"
SILERO_VAD_ONNX = True

class SileroVAD(BaseVAD):
    def __init__(self, 
                 sample_rate=48000, 
                 frame_duration=30, 
                 buffer_size=2, 
                 threshold=0.5, 
                 min_speech_prob=0.01,
                 sensitive_level=3
                 ):
        """
        Silero VAD 구현 클래스 (원본 구조 기반)

        Args:
            node (Node): ROS2 노드
            sample_rate (int): 입력 샘플링 레이트 (Hz)
            frame_duration (int): 프레임 길이 (ms)
            buffer_size (int): VAD 버퍼 크기
            threshold (float): VAD 감지 임계값 (0.0-1.0)
            min_speech_prob (float): 최소 음성 확률
        """
        super().__init__(sample_rate, frame_duration)
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000
        self.required_samples = 512  # 16kHz용
        self.threshold = threshold
        self.min_speech_prob = min_speech_prob
        self.sensitive_level = sensitive_level
                      
        # 프레임 버퍼 초기화
        self.frame_buffer = np.array([], dtype=np.float32)
        
        # 리샘플링을 위해 필요한 입력 샘플 수 계산
        self.resample_ratio = self.target_sample_rate / self.sample_rate
        self.input_samples_needed = int(self.required_samples / self.resample_ratio)

        # 노이즈 에너지 필터링을 위한 설정
        self.noise_threshold = 0.02  # 노이즈 에너지 임계값
        self.enable_energy_check = True  # 에너지 체크 활성화
        
        # VAD 점수 히스토리 (스무딩용)
        self.history_size = buffer_size
        self.score_history = []

        # 성능 모니터링
        self.process_times = []
        self.max_times = 50
        self.total_frames = 0
        self.processed_frames = 0
        self.last_log_time = time.time()
        self.log_interval = 5.0

        # torch.hub 디렉토리 설정
        self.torch_hub_dir = MODELS_BASE_DIR
        torch.hub.set_dir(self.torch_hub_dir)
        self.logger.info(f"Set torch.hub directory to: {self.torch_hub_dir}")

        self._load_model()
        
        self.fast_resampling = False  # 빠른 리샘플링 사용 여부
        # 리샘플러 초기화
        self.resampler = signal.resample_poly

    def _load_model(self):
        """Silero VAD 모델을 로드합니다. download_models.py에 의해 모델이 준비되었다고 가정합니다."""
        print("Loading Silero VAD model...")
        max_retries = 3
        retry_delay = 2

        # 이 파일에 직접 정의된 상수 사용
        repo_or_dir = SILERO_VAD_REPO_ID
        model_name = SILERO_VAD_MODEL_NAME_IN_REPO
        use_onnx = SILERO_VAD_ONNX

        # torch.hub.set_dir()는 __init__에서 이미 호출됨
        for attempt in range(max_retries):
            try:
                self.model, utils = torch.hub.load(
                    repo_or_dir=repo_or_dir,
                    model=model_name,
                    force_reload=False, # 로컬 캐시 우선 사용
                    onnx=use_onnx
                )

                if hasattr(self.model, 'reset_states'):
                    self.model.reset_states()
                elif hasattr(self.model, 'eval'):
                    self.model.eval()

                self.logger.info(f"Silero VAD initialized using {repo_or_dir} from torch.hub (ONNX: {use_onnx})")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warn(f"모델 로드 실패 ({attempt + 1}/{max_retries}): {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"모델 로드 최종 실패: {str(e)}")
                    self.logger.error(
                        f"Please ensure that the model '{repo_or_dir}' ({model_name}) is downloadable "
                        f"or run the download_models.py script in the heroehs_asr package."
                    )
                    raise
    
    def _log_performance(self, process_time):
        """VAD 성능 모니터링 및 로깅"""
        self.process_times.append(process_time)
        if len(self.process_times) > self.max_times:
            self.process_times.pop(0)

        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            avg_time = np.mean(self.process_times) * 1000
            max_time = np.max(self.process_times) * 1000
            process_ratio = self.processed_frames / max(1, self.total_frames)

            self.logger.info(
                f'VAD Performance - Avg: {avg_time:.1f}ms, Max: {max_time:.1f}ms, '
                f'Processed: {process_ratio*100:.1f}% ({self.processed_frames}/{self.total_frames})'
            )

            # 카운터 리셋
            self.total_frames = 0
            self.processed_frames = 0
            self.last_log_time = current_time
                        
    def prepare_input(self, audio_chunk):
        """
        입력 오디오를 검증하고 모델에 맞게 전처리 <- 원본 OnnxWrapper의 _validate_input 참고함
        """
        
        # Float32로 변환
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # 정규화 (-1 ~ 1 범위)
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            audio_chunk = audio_chunk / max_val

        # 
        if self.fast_resampling:
            # 리샘플링 (scipy.signal.resample 사용)
            if self.sample_rate != self.target_sample_rate:
                # 정확한 크기로 리샘플링
                resampled = signal.resample(audio_chunk, self.required_samples)
            else:
                resampled = audio_chunk

            # 정확히 required_samples 크기 맞추기
            if len(resampled) > self.required_samples:
                resampled = resampled[:self.required_samples]
            elif len(resampled) < self.required_samples:
                resampled = np.pad(resampled, (0, self.required_samples - len(resampled)), mode='constant')
            
        else:
            # 유현쌤 기존 코드 방식
            """ 
            더 정확한 리샘플링 (polyphase 필터 사용)
            경계 효과 감소 (패딩 사용) -> 경계 효과는 모델의 context 관리로 해결됨
            정수 비율에 최적화 (48kHz → 16kHz는 3:1)
            빠른 샘플링보다 2~3배 느림 
            고품질 오디오 처리가 필요한 경우
            속도보다 품질이 중요한 경우            
            기존 코드는 패딩 제거 후 크기 계산이 문제 
            """
            # 리샘플링 전에 프레임 패딩 (경계 효과 감소)
            pad_size = int(0.1 * len(audio_chunk))
            padded_frame = np.pad(audio_chunk, (pad_size, pad_size), mode='edge')

            # 리샘플링
            resampled = self.resampler(
                padded_frame,
                self.target_sample_rate,
                self.sample_rate
            )

            # ***중요*** 리샘플링 후 패딩 크기 계산
            pad_size_resampled = int(pad_size * self.resample_ratio)

            # 패딩 제거하고 정확한 크기로 조정
            resampled = resampled[pad_size_resampled:-pad_size_resampled]
            if len(resampled) > self.required_samples:
                resampled = resampled[:self.required_samples]
            elif len(resampled) < self.required_samples:
                resampled = np.pad(
                    resampled,
                    (0, self.required_samples - len(resampled))
                )

        return resampled.astype(np.float32)    
        
        
    def process_frame(self, new_frame):
        """
        오디오 프레임 처리 (원본 OnnxWrapper.__call__ 구조 참고)
        
        Returns:
            tuple: (is_speech, smoothed_prob, raw_prob) or None
        """
        try:
            self.total_frames += 1
            start_time = time.time()

            # 프레임 버퍼에 새 프레임 추가
            new_samples = new_frame.flatten().astype(np.float32)
            self.frame_buffer = np.append(self.frame_buffer, new_samples)
            # print(f"self.frame_buffer: {self.frame_buffer}")
            
            # 충분한 데이터가 모일 때까지 대기
            if len(self.frame_buffer) < self.input_samples_needed:
                return None

            # 필요한 만큼만 추출
            audio_chunk = self.frame_buffer[:self.input_samples_needed]
            self.frame_buffer = self.frame_buffer[self.input_samples_needed:]

            # 에너지 기반 체크 (리샘플링 전에 수행하여 성능 향상)
            if self.enable_energy_check:
                frame_energy = np.sum(np.abs(audio_chunk))
                if frame_energy < self.noise_threshold * len(audio_chunk):
                    # 에너지가 낮은 경우 모델 호출 없이 바로 반환 -> cpu 절약
                    return (False, 0.0, 0.0)

            # 입력 준비 (리샘플링 포함)
            resampled = self.prepare_input(audio_chunk)

            # 텐서 변환
            tensor = torch.from_numpy(resampled.reshape(1, -1))

            # VAD 추론
            with torch.no_grad():
                speech_prob = self.model(tensor, self.target_sample_rate).item()

            # 최소 음성 확률 필터링
            if speech_prob < self.min_speech_prob:
                speech_prob = 0.0

            # VAD 점수 히스토리 업데이트 (스무딩)
            self.score_history.append(speech_prob)
            if len(self.score_history) > self.history_size:
                self.score_history.pop(0)

            # 스무딩된 확률 계산
            if len(self.score_history) > 0:
                smoothed_prob = np.mean(self.score_history)
            else:
                smoothed_prob = speech_prob

            # 음성 감지 판단
            is_speech = smoothed_prob >= self.threshold

            # # 성능 모니터링
            # process_time = time.time() - start_time
            # self._log_performance(process_time)
            # self.processed_frames += 1

            return (is_speech, smoothed_prob, speech_prob)

        except Exception as e:
            self.logger.error(f'Error processing frame: {str(e)}')
            import traceback
            traceback.print_exc()
            return None
