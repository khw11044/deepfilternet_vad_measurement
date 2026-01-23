import numpy as np
import webrtcvad
from .base_vad import BaseVAD
import logging

class WebRtcVAD(BaseVAD):
    def __init__(self, 
                 sample_rate=48000, 
                 frame_duration=30, 
                 buffer_size=2, 
                 threshold=0.5, 
                 min_speech_prob=0.01,
                 sensitive_level=3
                 ):
        """
        WebRTC VAD 구현 클래스

        Args:
            sample_rate (int): 샘플링 레이트 (Hz)
            frame_duration (int): 프레임 길이 (ms)
            sensitive_level (int): WebRTC VAD 감도 레벨 (0-3)
        """
        super().__init__(sample_rate, frame_duration)
        self.logger = logging.getLogger(__name__)
        try:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(sensitive_level)
            self.logger.info(
                f'WebRTC VAD initialized:\n'
                f'Mode: {sensitive_level}\n'
                f'Sample rate: {sample_rate} Hz\n'
                f'Frame duration: {frame_duration} ms'
            )
            # 프레임 버퍼 초기화
            self.frame_buffer = np.array([], dtype=np.float32)
            self.sample_rate = sample_rate
            self.frame_duration = frame_duration
        except Exception as e:
            self.logger.error(f'Error initializing WebRTC VAD: {str(e)}')
            raise

    def process_frame(self, new_frame):
        try:
            self.frame_buffer = np.append(self.frame_buffer, new_frame.flatten())

            if len(self.frame_buffer) >= self.samples_per_frame:
                frame = self.frame_buffer[:self.samples_per_frame]
                self.frame_buffer = self.frame_buffer[self.samples_per_frame:]
                frame = np.ascontiguousarray(frame)

                is_speech = self.vad.is_speech(
                    frame.tobytes(),
                    self.sample_rate
                )

                if is_speech:
                    self.logger.debug(
                        f'WebRTC VAD detected speech in frame (size: {len(frame)})'
                    )

                return is_speech, frame

        except Exception as e:
            self.logger.error(f'Error in WebRTC VAD processing: {str(e)}')
            self.logger.debug(
                f'Frame details - size: {len(new_frame)}, '
                f'buffer size: {len(self.frame_buffer)}'
            )

        return None
