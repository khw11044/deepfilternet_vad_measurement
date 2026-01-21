from abc import ABC, abstractmethod
import numpy as np
import logging

class BaseVAD(ABC):
    def __init__(self, 
                 sample_rate=48000, 
                 frame_duration=30
                 ):
        """
        VAD(Voice Activity Detection) 기본 클래스

        Args:
            sample_rate (int): 샘플링 레이트 (Hz)
            frame_duration (int): 프레임 길이 (ms)
        """
        self.logger = logging.getLogger(__name__)

        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.samples_per_frame = int(sample_rate * frame_duration / 1000)
        self.frame_buffer = np.array([], dtype=np.int16)
        
    @abstractmethod
    def process_frame(self, new_frame):
        """
        새로운 오디오 프레임에 대한 VAD 처리

        Args:
            new_frame (numpy.ndarray): 입력 오디오 프레임

        Returns:
            tuple: (is_speech, processed_frame) 또는 None
        """
        pass

    def reset_buffer(self):
        """프레임 버퍼 초기화"""
        self.frame_buffer = np.array([], dtype=np.int16)
        self.logger.debug('VAD frame buffer reset')

    def get_frame_info(self):
        """현재 프레임 설정 정보 반환"""
        return {
            'sample_rate': self.sample_rate,
            'frame_duration': self.frame_duration,
            'samples_per_frame': self.samples_per_frame,
            'buffer_size': len(self.frame_buffer)
        }
