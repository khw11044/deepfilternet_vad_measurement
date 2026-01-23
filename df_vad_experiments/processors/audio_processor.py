#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
from rclpy.node import Node
from scipy import signal
import soundfile as sf
import os
from os.path import join, getsize
from datetime import datetime
import time
import queue

class AudioProcessor:
    def __init__(self, node: Node):
        """
        오디오 디바이스 프로세서 초기화
        """
        self.node = node

        # config에서 설정값 로드
        config = node.config
        self.sample_rate = config.get('sample_rate', 48000)
        self.frame_duration = config.get('frame_duration', 60)
        self.enable_recording = config.get('enable_recording', False)

        # 샘플 관련 설정
        self.samples_per_frame = int(self.sample_rate * self.frame_duration / 1000)  # ms to samples

        # 녹음 관련 변수
        self.recording = False
        self.callback = None
        self.stream = None
        self.recording_thread = None
        self.stop_flag = False

        # 테스트용 녹음 버퍼
        self.test_buffer = []
        self.is_test_recording = False
        self.test_start_time = None

        # 오디오 버퍼 관리를 위한 변수들
        self.audio_buffer = []  # 전체 오디오 데이터를 저장할 버퍼
        self.buffer_duration = 5  # 버퍼에 유지할 오디오 길이 (초)
        self.max_buffer_size = int(self.sample_rate * self.buffer_duration)  # 최대 버퍼 크기
        self.last_buffer_cleanup = time.time()
        self.cleanup_interval = 1.0  # 1초마다 cleanup 수행
        
        # 용량 확인 함수 바인딩
        self.get_capacity = self.check_capacity
        
        # 패딩 설정
        self.padding_duration = 0.3  # 패딩 길이 (초)
        self.padding_samples = int(self.sample_rate * self.padding_duration)

        # 파일 저장 경로 설정
        self.speech_segments_dir = config.get('speech_segments_dir', './aeirobot_asr/speech_logs')
        self.test_recordings_dir = config.get('test_recordings_dir', './aeirobot_asr/audio_test')

        self.node.get_logger().info(
            f'Audio manager initialized:\n'
            f'Sample rate: {self.sample_rate} Hz\n'
            f'Frame duration: {self.frame_duration} ms\n'
            f'Samples per frame: {self.samples_per_frame}'
        )

    # ===============================================================================================================================================

    def preprocess_audio(self, audio_data, encoding='F32LE'):
        """오디오 데이터 전처리

        Args:
            audio_data: 오디오 데이터 (bytes, list, 또는 numpy.ndarray)
            encoding: 오디오 인코딩 형식 (예: 'S16LE', 'F32LE')

        Returns:
            numpy.ndarray: float32 오디오 프레임 (-1.0 ~ 1.0 근처 스케일)
        """
        # array.array, bytes, list 등 다양한 타입 처리 → raw bytes로 통일
        # 실질적으로 sender_node에서 처음부터 bytes로 보내므로 이 부분은 항상 raw_bytes = audio_data 이렇게 됨
        if hasattr(audio_data, 'tobytes'):
            raw_bytes = audio_data.tobytes()
        elif isinstance(audio_data, bytes):
            raw_bytes = audio_data
        else:
            raw_bytes = bytes(audio_data)

        # encoding에 따라 float32 프레임으로 변환
        try:
            if encoding.upper() == 'F32LE':
                # 32비트 float 리틀엔디언
                float_frame = np.frombuffer(raw_bytes, dtype=np.float32)
            else:
                # 기본: 16비트 signed 리틀엔디언 (S16LE)
                int16_frame = np.frombuffer(raw_bytes, dtype=np.int16)
                float_frame = int16_frame.astype(np.float32) / 32768.0

            # # DC offset 제거
            # if float_frame.size > 0:
            #     float_frame = float_frame - np.mean(float_frame)

            return float_frame.astype(np.float32)
        except Exception as e:
            self.node.get_logger().error(f'Error in preprocess_audio: {str(e)}')
            return np.array([], dtype=np.float32)
    

    def _cleanup_buffer(self):
        """버퍼 정리"""
        current_time = time.time()
        if current_time - self.last_buffer_cleanup >= self.cleanup_interval:
            if len(self.audio_buffer) > self.max_buffer_size:
                excess_samples = len(self.audio_buffer) - self.max_buffer_size
                self.audio_buffer = self.audio_buffer[excess_samples:]
                self.node.get_logger().debug(f'Cleaned up {excess_samples} samples from audio buffer')
            self.last_buffer_cleanup = current_time


    def add_to_buffer(self, audio_data):
        """
        오디오 데이터를 버퍼에 추가하고 오래된 데이터 제거

        Args:
            audio_data (numpy.ndarray): 추가할 오디오 데이터
        """
        self.audio_buffer.extend(audio_data)

        # 버퍼 크기가 최대값을 초과하면 오래된 데이터 제거
        if len(self.audio_buffer) > self.max_buffer_size:
            excess = len(self.audio_buffer) - self.max_buffer_size
            self.audio_buffer = self.audio_buffer[excess:]

    def extract_speech_segment(self, start_time, end_time, output_path=None):
        """
        지정된 시작/종료 시간에 해당하는 발화 구간을 패딩과 함께 추출하여 새로운 WAV 파일 생성
        EPD의 pause duration을 고려하여 실제 발화 종료 시점을 계산

        Args:
            start_time (float): 발화 시작 시간 (초)
            end_time (float): 발화 종료 시간 (초, pause duration 포함)
            output_path (str, optional): 저장할 WAV 파일 경로

        Returns:
            str: 생성된 WAV 파일 경로
        """
        try:
            # EPD pause duration 고려하여 실제 발화 종료 시점 계산
            pause_duration = self.node.config.get('pause_duration', 0.225)  # 기본값 225ms
            actual_end_time = end_time - pause_duration * 2

            # 현재 시간 기준으로 상대적인 시작/종료 인덱스 계산
            current_time = time.time()
            buffer_start_time = current_time - (len(self.audio_buffer) / self.sample_rate)

            # 시작/종료 시간을 버퍼 내 인덱스로 변환
            start_idx = int((start_time - buffer_start_time) * self.sample_rate)
            end_idx = int((actual_end_time - buffer_start_time) * self.sample_rate)

            # 패딩을 고려한 인덱스 계산
            padded_start_idx = max(0, start_idx - self.padding_samples)
            padded_end_idx = min(len(self.audio_buffer), end_idx + self.padding_samples)

            # 발화 구간 추출 (패딩 포함)
            speech_segment = np.array(self.audio_buffer[padded_start_idx:padded_end_idx])

            # WAV 파일 저장
            if output_path is None:
                output_dir = os.path.join(os.path.expanduser('~'), self.speech_segments_dir)
                os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f'speech_{int(time.time()*1000)}.wav')
                
                capacities, num_files = self.get_capacity(output_dir)
                
                if self.enable_recording and capacities < 2000:  # 2GB 미만일 때만 저장
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                else:
                    timestamp = "latest"    
                output_path = os.path.join(output_dir, f'speech_{timestamp}.wav')

            sf.write(output_path, speech_segment, self.sample_rate)
            self.node.get_logger().info(f'Speech segment saved to: {output_path}')

            return output_path

        except Exception as e:
            self.node.get_logger().error(f'Error extracting speech segment: {str(e)}')
            return None

    def extract_speech_segment_data(self, start_time, end_time):
        """
        지정된 시작/종료 시간에 해당하는 발화 구간을 패딩과 함께 추출하여 numpy array로 반환
        파일 저장 없이 메모리에서 직접 데이터를 반환합니다.

        Args:
            start_time (float): 발화 시작 시간 (초)
            end_time (float): 발화 종료 시간 (초, pause duration 포함)

        Returns:
            tuple: (speech_segment, sample_rate) 또는 None
                - speech_segment (numpy.ndarray): 추출된 오디오 데이터 (float32)
                - sample_rate (int): 샘플레이트
        """
        try:
            # EPD pause duration 고려하여 실제 발화 종료 시점 계산
            pause_duration = self.node.config.get('pause_duration', 0.225)  # 기본값 225ms
            actual_end_time = end_time - pause_duration * 2

            # 현재 시간 기준으로 상대적인 시작/종료 인덱스 계산
            current_time = time.time()
            buffer_start_time = current_time - (len(self.audio_buffer) / self.sample_rate)

            # 시작/종료 시간을 버퍼 내 인덱스로 변환
            start_idx = int((start_time - buffer_start_time) * self.sample_rate)
            end_idx = int((actual_end_time - buffer_start_time) * self.sample_rate)

            # 패딩을 고려한 인덱스 계산
            padded_start_idx = max(0, start_idx - self.padding_samples)
            padded_end_idx = min(len(self.audio_buffer), end_idx + self.padding_samples)

            # 발화 구간 추출 (패딩 포함)
            speech_segment = np.array(self.audio_buffer[padded_start_idx:padded_end_idx], dtype=np.float32)

            self.node.get_logger().info(f'Speech segment extracted: {len(speech_segment)} samples ({len(speech_segment)/self.sample_rate:.2f}s)')

            return speech_segment, self.sample_rate

        except Exception as e:
            self.node.get_logger().error(f'Error extracting speech segment data: {str(e)}')
            return None

    def check_capacity(self, dir):
        """디렉토리 용량 확인 (MB 단위)"""
        # capacities = 0
        # num_files = 0
        # for root, dirs, files in os.walk(dir):
        #     capacity = sum([getsize(join(root, name)) for name in files])  / (1024.0 * 1024.0)
        #     num_file = len(files)
        #     result = "%s : %.f MB in %d files." % (os.path.abspath(root), capacity, num_file)
        #     print(result)
        #     capacities += capacity
        #     num_files += num_file
            
        # result = "Total: %.f MB in %d files." % (capacities, num_files)
        # print(result)
        
        # return capacities, num_files
        
        capacities = 0
        num_files = 0
        try:
            if os.path.exists(dir):
                for filename in os.listdir(dir):
                    filepath = os.path.join(dir, filename)
                    if os.path.isfile(filepath):
                        capacities += os.path.getsize(filepath)
                        num_files += 1
        except Exception as e:
            self.node.get_logger().warn(f'Error checking capacity: {str(e)}')
        
        return capacities / (1024 * 1024), num_files  # MB 단위

            

######################################################################################################

    def save_test_recording(self):
        """테스트 녹음 저장"""
        try:
            self.is_test_recording = False

            if not self.test_buffer:
                self.node.get_logger().warn('No audio data to save')
                return

            # 버퍼의 데이터를 하나의 배열로 합치기
            audio_data = np.concatenate(self.test_buffer)

            # 파일 저장 경로 설정
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            filename = f'test_recording_{timestamp}.wav'
            save_dir = os.path.join(os.path.expanduser('~'), self.test_recordings_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            # WAV 파일로 저장
            sf.write(save_path, audio_data, self.sample_rate)

            self.node.get_logger().info(f'Test recording saved to: {save_path}')
            self.test_buffer = []

        except Exception as e:
            self.node.get_logger().error(f'Error saving test recording: {str(e)}')
