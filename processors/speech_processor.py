import numpy as np
from rclpy.node import Node
from collections import deque
import time
import os
import soundfile as sf
from datetime import datetime


class SpeechProcessor:
    """
    음성 처리기 - 직접 버퍼링 방식
    
    VAD 결과를 받아서 발화 시작/끝을 판단하고,
    오디오 데이터를 직접 버퍼에 저장하여 완성된 발화 구간을 WAV로 저장합니다.
    """
    
    def __init__(self, node: Node, vad_processor, energy_monitor, audio_manager, use_stream_mode=False):
        """음성 처리기 초기화"""
        self.node = node
        self.vad_processor = vad_processor
        self.energy_monitor = energy_monitor
        self.audio_manager = audio_manager
        self.use_stream_mode = use_stream_mode

        # config에서 설정값 로드
        config = node.config
        config_vad = config.get('vad', {})
        
        # VAD 관련 설정
        self.sample_rate = config_vad.get('sample_rate', 48000)
        self.frame_duration = config_vad.get('frame_duration', 10) / 1000.0  # ms to seconds
        self.vad_buffer_size = config_vad.get('buffer_size', 5)        
        self.vad_threshold = config_vad.get('threshold', 0.5)
        self.vad_high_threshold = config_vad.get('high_threshold', 0.98)
        
        # 에너지 레벨 및 음성처리 관련 설정
        self.noise_db_threshold = config.get('noise_db_threshold', -35)
        self.min_db_threshold = config.get('min_db_threshold', -45)

        # 시간 설정
        min_speech_duration = config.get('min_speech_duration', 0.1)     # 최소 발화 길이 (초)
        max_speech_duration = config.get('max_speech_duration', 30.0)   # 최대 발화 길이 (초)
        pause_duration = config.get('pause_duration', 0.75)             # 발화 종료 침묵 시간 (초)
        pre_speech_padding = config.get('padding_duration', 0.3)        # 발화 전 패딩 (초)
        
        # 파일 저장 설정
        self.speech_segments_dir = config.get('speech_segments_dir', '.aeirobot_asr/speech_logs')

        # 샘플 수로 변환
        self.min_speech_samples = int(self.sample_rate * min_speech_duration)
        self.max_speech_samples = int(self.sample_rate * max_speech_duration)
        self.pause_samples = int(self.sample_rate * pause_duration)
        self.pre_speech_samples = int(self.sample_rate * pre_speech_padding)

        # 버퍼링을 위한 버퍼들
        self.frame_buffer = deque(maxlen=500)                        # 모든 프레임 임시 저장 (최근 500개)
        self.vad_buffer = deque(maxlen=self.vad_buffer_size)
        # self.speech_buffer = []                                          # 현재 발화 오디오 버퍼 (float32 샘플)
        self.pre_speech_buffer = deque(maxlen=self.pre_speech_samples)   # 발화 전 패딩용 링버퍼
        self.pending_speech_buffer = []                                  # 발화 감지 대기 중 임시 버퍼
        
        # 시간 측정용
        self.speech_start_time = None         # 발화 시작 시간
        self.last_voice_time = None           # 마지막 음성 시간 (옛날 코드)

        # 상태 변수
        self.is_speaking = False
        self.speech_sample_count = 0          # 연속 발화 샘플 수
        self.silence_sample_count = 0         # 연속 침묵 샘플 수
       
        # 성능 모니터링
        self.last_log_time = time.time()
        self.log_interval = 5.0

        self.node.get_logger().info(
            f'Speech processor initialized (Direct Buffering Mode):\n'
            f'Sample rate: {self.sample_rate} Hz\n'
            f'Frame duration: {self.frame_duration * 1000:.1f} ms\n'
            f'Min speech duration: {min_speech_duration:.3f}s ({self.min_speech_samples} samples)\n'
            f'Max speech duration: {max_speech_duration:.1f}s ({self.max_speech_samples} samples)\n'
            f'Pause duration: {pause_duration:.3f}s ({self.pause_samples} samples)\n'
            f'Pre-speech padding: {pre_speech_padding:.3f}s ({self.pre_speech_samples} samples)\n'
            f'VAD threshold: {self.vad_threshold}\n'
            f'Noise threshold: {self.noise_db_threshold} dB'
        )

    # ===============================================================================================================================================

    def _reset_state(self):
        """상태 초기화"""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_voice_time = None
        self.frame_buffer.clear()
        self.vad_buffer.clear()
        # self.speech_buffer = []
        self.speech_sample_count = 0
        self.silence_sample_count = 0


    def _check_voice_activity(self, smoothed_vad_score, db_level):
        """음성 활성 상태를 체크하는 함수"""
        # 높은 신뢰도 조건 체크 - VAD 점수가 매우 높고 최소 DB level은 넘을 때
        if smoothed_vad_score >= self.vad_high_threshold and db_level >= self.min_db_threshold:
            return True, 'high'

        # DB level이 기준 이하면 음성이 아님
        if db_level < self.noise_db_threshold:
            # 현재 말하는 중이면 매우 관대한 기준 적용
            if self.is_speaking and smoothed_vad_score >= self.vad_threshold * 0.7:
                return True, 'low'
            return False, 'none'

        # 기본 신뢰도 조건 체크 - DB level이 충분하고 VAD도 기준 이상
        if smoothed_vad_score >= self.vad_threshold:
            return True, 'normal'

        return False, 'none'

    def process_frame(self, raw_data, encoding='F32LE'):
        """ROS2 오디오 메시지 처리 - 옛날 코드 방식 (모든 프레임 버퍼링)

        Args:
            raw_data: ROS2 Audio 메시지의 raw data (bytes, list, array 등)
            encoding: 오디오 인코딩 형식 (예: 'F32LE', 'S16LE')

        Returns:
            tuple: (result, db_level, vad_score)
        """
        try:
            # 1) VAD 및 dB 계산용 float32 프레임
            ## preprocess해서 float32로 변환 
            frame = self.audio_manager.preprocess_audio(raw_data, encoding)
            num_samples = len(frame)
            ## AudioDeviceManager의 버퍼에 추가
            self.audio_manager.add_to_buffer(frame)
            db_level = self.energy_monitor.calculate_energy(frame)
            
            # 2) VAD 처리
            vad_result = self.vad_processor.process_frame(frame)
            if vad_result is None:
                return None, db_level, 0.0

            is_speech, smoothed_prob, vad_score = vad_result
            
            # VAD 결과 버퍼에 추가 및 스무딩
            self.vad_buffer.append(vad_score)
            smoothed_vad_score = sum(self.vad_buffer) / len(self.vad_buffer)

            # 음성 활성 상태 체크, confidence는 'none', 'low', 'normal', 'high'
            is_voice, confidence = self._check_voice_activity(smoothed_vad_score, db_level)
            
            if confidence in ['low']:   # db이 너무 낮아서 confidence가 임계값 이하로 낮으면 무시
                return None, db_level, vad_score
            
            result = self._speech_segmentor(raw_data, is_voice, num_samples)
            return result, db_level, vad_score

        except Exception as e:
            self.node.get_logger().error(f'Error processing frame: {str(e)}')
            import traceback
            traceback.print_exc()
            return None, -80.0, 0.0

    
    def _speech_segmentor(self, raw_data, is_voice, num_samples):
        # raw_data = bytes(raw_data)
        result = None
        if not self.is_speaking:
            # ===== 발화 중이 아닐 때 =====
            # 모든 프레임을 frame_buffer에 임시 저장
            self.frame_buffer.append(raw_data)
            
            if is_voice:
                self.speech_sample_count += num_samples
                self.silence_sample_count = 0
                
                # 최소 발화 길이 도달 시 발화 시작
                if self.speech_sample_count >= self.min_speech_samples:
                    self.is_speaking = True
                    
                    # 버퍼 길이만큼 빼서 발화 시작 시간 계산 (패딩 포함)
                    current_time = time.time()
                    self.speech_start_time = current_time - (len(self.frame_buffer) * self.frame_duration * 0.2)
                    self.last_voice_time = current_time  # 초기화 (발화 시작 시점)
                    
                    # # frame_buffer 전체를 speech_buffer로 이동
                    # self.speech_buffer = list(self.frame_buffer)
                    self.frame_buffer.clear()
                    
                    # 디스플레이 이벤트 업데이트 
                    if hasattr(self.node, 'display_lock'):
                        with self.node.display_lock:
                            self.node.last_event = f"🎤 음성 시작)"
                    
                    self.node.get_logger().debug(f'Speech started')
            else:
                self.speech_sample_count = 0

        else:
            # ===== 발화 중일 때 - 시간 기록  =====
            current_time = time.time()
            
            if is_voice:
                self.last_voice_time = current_time  # 마지막 음성 시간 업데이트
                self.silence_sample_count = 0
                
            else:
                self.silence_sample_count += num_samples

                # 침묵이 일정 시간 지속되면 발화 종료
                if self.silence_sample_count >= self.pause_samples:
                    # 마지막 음성 시간을 여유있게 설정 -> 침묵: 0.6초라면 그 절반인 0.3초까지 여유있게 포함
                    self.last_voice_time = current_time - (self.silence_sample_count / self.sample_rate * 0.5)

                    if hasattr(self.node, 'display_lock'):
                        with self.node.display_lock:
                            self.node.last_event = "🔇 음성 종료"
                    
                    result = self._finalize_speech()
                    return result
            
            # 최대 발화 길이 초과 시 강제 종료
            if self.speech_start_time is not None:
                elapsed = current_time - self.speech_start_time
                if elapsed >= (self.max_speech_samples / self.sample_rate):
                    self.last_voice_time = current_time
                    self.node.get_logger().info('Maximum speech duration reached, forcing finalization')
                    result = self._finalize_speech()
                    return result
        
    
    
    def _finalize_speech(self):
        """ROS2 발화 종료 처리 - 시간 기반 추출 (옛날 코드 방식)"""
        try:
            # 발화 지속 시간 체크
            speech_duration = self.last_voice_time - self.speech_start_time
            min_duration = self.min_speech_samples / self.sample_rate
            
            if speech_duration < min_duration:
                self.node.get_logger().info(
                    f'Speech too short ({speech_duration:.3f}s < {min_duration:.3f}s), discarding'
                )
                self._reset_state()
                return None

            if self.use_stream_mode:
                # 스트림 모드: 파일 저장 없이 오디오 데이터만 추출
                result = self.audio_manager.extract_speech_segment_data(
                    self.speech_start_time,
                    self.last_voice_time
                )
                
                if result:
                    speech_segment, sample_rate = result
                    self.node.get_logger().info(f'Speech segment extracted for streaming: {len(speech_segment)} samples')
                    self._reset_state()
                    return True, (speech_segment, sample_rate)
            else:
                # 파일 모드: AudioDeviceManager의 버퍼에서 시간 기반으로 추출하여 파일 저장
                wav_path = self.audio_manager.extract_speech_segment(
                    self.speech_start_time,
                    self.last_voice_time
                )
                
                if wav_path:
                    self.node.get_logger().info(f'Speech segment saved using time-based extraction: {wav_path}')
                    self._reset_state()
                    return True, wav_path
            
            self._reset_state()
            return None

        except Exception as e:
            self.node.get_logger().error(f'Error finalizing speech (ROS): {str(e)}')
            import traceback
            traceback.print_exc()
            self._reset_state()
            return None
