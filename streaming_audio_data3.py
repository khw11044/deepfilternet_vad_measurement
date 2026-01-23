"""
실시간 오디오 데이터 분석 및 시각화
- 파형 (Waveform)
- 진폭 (Amplitude Envelope) - RMS 기반
- 음압 (Sound Pressure) - 정규화된 신호 기준
- 강도 (Intensity) - 진폭의 제곱에 비례
- dB (Decibels) - RMS를 dB로 변환
- 멜-스펙트로그램 (Mel Spectrogram)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sounddevice as sd
import librosa
import librosa.onset
from scipy.signal import butter, sosfilt
from collections import deque
import threading

# STFT 설정
N_FFT = 2048
STFT_HOP = 512

# 오디오 설정
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
CHANNELS = 1


class Microphone:
    """마이크 디바이스 관리 클래스"""

    def __init__(self):
        self.device_id = None
        self.device_info = None

    @staticmethod
    def list_devices():
        """사용 가능한 모든 오디오 디바이스 목록 출력"""
        print("\n" + "=" * 60)
        print("Available Audio Devices")
        print("=" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            # 입력 채널이 있는 디바이스만 표시 (마이크)
            if device['max_input_channels'] > 0:
                marker = " [DEFAULT]" if i == sd.default.device[0] else ""
                print(f"  [{i}] {device['name']}{marker}")
                print(f"      Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']:.0f} Hz")
        print("=" * 60 + "\n")

    @staticmethod
    def get_input_devices():
        """입력 디바이스(마이크) 목록 반환"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate'],
                    'is_default': i == sd.default.device[0]
                })
        return input_devices

    def select_device(self, device_id=None):
        """디바이스 선택 (None이면 대화형 선택)"""
        input_devices = self.get_input_devices()

        if not input_devices:
            raise RuntimeError("No input devices found!")

        if device_id is not None:
            # 지정된 디바이스 ID 사용
            for device in input_devices:
                if device['id'] == device_id:
                    self.device_id = device_id
                    self.device_info = device
                    print(f"Selected: [{device_id}] {device['name']}")
                    return self
            raise ValueError(f"Device ID {device_id} not found or not an input device")

        # 대화형 선택
        self.list_devices()
        while True:
            try:
                choice = input("Select device ID (or press Enter for default): ").strip()
                if choice == "":
                    # 기본 디바이스 선택
                    default_id = sd.default.device[0]
                    for device in input_devices:
                        if device['id'] == default_id:
                            self.device_id = default_id
                            self.device_info = device
                            print(f"Using default: [{default_id}] {device['name']}")
                            return self
                else:
                    device_id = int(choice)
                    for device in input_devices:
                        if device['id'] == device_id:
                            self.device_id = device_id
                            self.device_info = device
                            print(f"Selected: [{device_id}] {device['name']}")
                            return self
                    print(f"Invalid device ID: {device_id}")
            except ValueError:
                print("Please enter a valid number")

    def get_default_device(self):
        """기본 입력 디바이스 자동 선택"""
        default_id = sd.default.device[0]
        devices = sd.query_devices()
        if default_id is not None and devices[default_id]['max_input_channels'] > 0:
            self.device_id = default_id
            self.device_info = {
                'id': default_id,
                'name': devices[default_id]['name'],
                'channels': devices[default_id]['max_input_channels'],
                'sample_rate': devices[default_id]['default_samplerate'],
                'is_default': True
            }
            print(f"Using default device: [{default_id}] {self.device_info['name']}")
            return self
        raise RuntimeError("No default input device found!")

# 버퍼 설정 (약 3초 분량)
BUFFER_SECONDS = 3
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)

# RMS 설정
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# 주파수 필터 설정
LOW_FREQ_CUTOFF = 500    # 저주파 상한 (Hz)
HIGH_FREQ_CUTOFF = 2000  # 고주파 하한 (Hz)
FILTER_ORDER = 4         # 버터워스 필터 차수


class AudioAnalyzer:
    def __init__(self):
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=BUFFER_SIZE)
        self.audio_buffer.extend(np.zeros(BUFFER_SIZE))

        # 스레드 동기화
        self.lock = threading.Lock()

        # 버터워스 필터 계수 미리 계산
        nyquist = SAMPLE_RATE / 2
        self.lowpass_sos = butter(FILTER_ORDER, LOW_FREQ_CUTOFF / nyquist, btype='low', output='sos')
        self.highpass_sos = butter(FILTER_ORDER, HIGH_FREQ_CUTOFF / nyquist, btype='high', output='sos')

        # STFT 설정
        self.n_fft = N_FFT
        self.stft_hop = STFT_HOP

        # 주파수 축 계산
        self.freq_bins = np.fft.rfftfreq(N_FFT, 1/SAMPLE_RATE)
        self.n_bins = len(self.freq_bins)

        # 주파수 대역 마스크 (STFT 도메인용)
        self.low_freq_mask = self.freq_bins < LOW_FREQ_CUTOFF   # < 500Hz
        self.high_freq_mask = self.freq_bins > HIGH_FREQ_CUTOFF  # > 2000Hz

    def audio_callback(self, indata, frames, time, status):
        """오디오 입력 콜백"""
        if status:
            print(f"Audio status: {status}")

        with self.lock:
            # 새 오디오 데이터 추가
            self.audio_buffer.extend(indata[:, 0])

    def get_audio_data(self):
        """현재 오디오 버퍼 데이터 반환"""
        with self.lock:
            return np.array(self.audio_buffer)

    def compute_rms(self, signal):
        """RMS 계산"""
        rms = librosa.feature.rms(
            y=signal,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]
        return rms

    def filter_low_freq(self, signal):
        """저주파 필터링 (Low-pass filter)"""
        return sosfilt(self.lowpass_sos, signal)

    def filter_high_freq(self, signal):
        """고주파 필터링 (High-pass filter)"""
        return sosfilt(self.highpass_sos, signal)

    def compute_stft_frame(self, signal):
        """전체 신호에 대한 STFT 계산 (librosa 사용 - 시각화용)"""
        D = np.abs(librosa.stft(signal, n_fft=self.n_fft, hop_length=self.stft_hop))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        return S_db

    def get_masked_stft(self, S_db, freq_mask):
        """특정 주파수 대역만 마스킹"""
        S_masked = S_db.copy()
        S_masked[~freq_mask, :] = -80  # 마스크 외 영역은 최소값
        return S_masked

    def compute_onset_strength(self, signal):
        """Onset strength 계산 (갑자기 커진 소리 감지)"""
        onset_env = librosa.onset.onset_strength(
            y=signal,
            sr=SAMPLE_RATE,
            hop_length=self.stft_hop
        )
        return onset_env


def main():
    # 마이크 디바이스 선택
    mic = Microphone()
    mic.select_device()  # 대화형 선택
    # mic.get_default_device()  # 또는 기본 디바이스 자동 선택
    # mic.select_device(device_id=0)  # 또는 특정 디바이스 ID 지정

    analyzer = AudioAnalyzer()

    # Figure 설정 - GridSpec으로 6행 x 3열 레이아웃
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(6, 3, figure=fig, width_ratios=[10, 10, 0.5], hspace=0.5, wspace=0.3)
    fig.suptitle('Real-time Audio Analysis (with STFT Visualization)', fontsize=14, fontweight='bold')

    # 왼쪽 열: 기존 6개 그래프
    left_axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]

    # 오른쪽 열: 새로운 4개 그래프
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    # colorbar용 축 (상위 3개 스펙트로그램 공유)
    cax = fig.add_subplot(gs[:3, 2])

    # 시간 축 생성
    time_axis = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)

    # ========== 왼쪽 그래프 (기존 6개) ==========

    # 1. 파형 (Waveform)
    line_waveform, = left_axes[0].plot(time_axis, np.zeros(BUFFER_SIZE),
                                        color='steelblue', linewidth=0.5)
    left_axes[0].set_ylabel('Amplitude')
    left_axes[0].set_title('1. Waveform')
    left_axes[0].set_xlim([0, BUFFER_SECONDS])
    left_axes[0].set_ylim([-1, 1])
    left_axes[0].grid(True, alpha=0.3)

    # RMS 시간 축 (초기값)
    rms_frames = int(BUFFER_SIZE / HOP_LENGTH) + 1
    rms_time = np.linspace(0, BUFFER_SECONDS, rms_frames)

    # 2. 진폭 (Amplitude Envelope) - RMS 기반
    line_rms, = left_axes[1].plot(rms_time, np.zeros(rms_frames),
                                   color='orange', linewidth=1.5)
    left_axes[1].set_ylabel('RMS Amplitude')
    left_axes[1].set_title('2. Amplitude Envelope')
    left_axes[1].set_xlim([0, BUFFER_SECONDS])
    left_axes[1].set_ylim([0, 0.5])
    left_axes[1].grid(True, alpha=0.3)

    # 3. 음압 (Sound Pressure) - 정규화된 신호 절대값
    line_pressure, = left_axes[2].plot(time_axis, np.zeros(BUFFER_SIZE),
                                        color='green', linewidth=0.5)
    left_axes[2].set_ylabel('|Pressure|')
    left_axes[2].set_title('3. Sound Pressure')
    left_axes[2].set_xlim([0, BUFFER_SECONDS])
    left_axes[2].set_ylim([0, 1])
    left_axes[2].grid(True, alpha=0.3)

    # 4. 강도 (Intensity) - RMS^2
    line_intensity, = left_axes[3].plot(rms_time, np.zeros(rms_frames),
                                         color='purple', linewidth=1.5)
    left_axes[3].set_ylabel('Intensity (RMS²)')
    left_axes[3].set_title('4. Sound Intensity')
    left_axes[3].set_xlim([0, BUFFER_SECONDS])
    left_axes[3].set_ylim([0, 0.1])
    left_axes[3].grid(True, alpha=0.3)

    # 5. 저주파 (Low Frequency) - 500Hz 이하
    line_low_freq, = left_axes[4].plot(time_axis, np.zeros(BUFFER_SIZE),
                                        color='red', linewidth=0.5)
    left_axes[4].set_ylabel('Amplitude')
    left_axes[4].set_title(f'5. Low Frequency (< {LOW_FREQ_CUTOFF} Hz)')
    left_axes[4].set_xlim([0, BUFFER_SECONDS])
    left_axes[4].set_ylim([-1, 1])
    left_axes[4].grid(True, alpha=0.3)

    # 6. 고주파 (High Frequency) - 2000Hz 이상
    line_high_freq, = left_axes[5].plot(time_axis, np.zeros(BUFFER_SIZE),
                                         color='cyan', linewidth=0.5)
    left_axes[5].set_xlabel('Time (s)')
    left_axes[5].set_ylabel('Amplitude')
    left_axes[5].set_title(f'6. High Frequency (> {HIGH_FREQ_CUTOFF} Hz)')
    left_axes[5].set_xlim([0, BUFFER_SECONDS])
    left_axes[5].set_ylim([-1, 1])
    left_axes[5].grid(True, alpha=0.3)

    # ========== 오른쪽 그래프 (새로운 4개 - STFT 기반) ==========

    # STFT 시각화용 설정
    stft_extent = [0, BUFFER_SECONDS, 0, SAMPLE_RATE / 2]
    empty_stft = np.full((analyzer.n_bins, 10), -80.0)  # 초기 빈 스펙트로그램

    # 7. STFT 전체 스펙트로그램
    stft_img = right_axes[0].imshow(
        empty_stft, aspect='auto', origin='lower', cmap='magma',
        extent=stft_extent, vmin=-80, vmax=0
    )
    right_axes[0].set_title('7. STFT Spectrogram (Full Band)')
    right_axes[0].set_ylabel('Frequency (Hz)')
    right_axes[0].grid(False)

    # 8. 고주파 마스킹 STFT (> 2000Hz)
    high_stft_img = right_axes[1].imshow(
        empty_stft, aspect='auto', origin='lower', cmap='magma',
        extent=stft_extent, vmin=-80, vmax=0
    )
    right_axes[1].set_title(f'8. High Frequency STFT (> {HIGH_FREQ_CUTOFF} Hz)')
    right_axes[1].set_ylabel('Frequency (Hz)')
    right_axes[1].grid(False)

    # 9. 저주파 마스킹 STFT (< 500Hz)
    low_stft_img = right_axes[2].imshow(
        empty_stft, aspect='auto', origin='lower', cmap='magma',
        extent=stft_extent, vmin=-80, vmax=0
    )
    right_axes[2].set_title(f'9. Low Frequency STFT (< {LOW_FREQ_CUTOFF} Hz)')
    right_axes[2].set_ylabel('Frequency (Hz)')
    right_axes[2].grid(False)

    # 10. Onset Detection (갑자기 커진 소리)
    onset_line, = right_axes[3].plot([], [], color='magenta', linewidth=1.5)
    right_axes[3].set_title('10. Onset Detection (Sudden Sound)')
    right_axes[3].set_xlabel('Time (s)')
    right_axes[3].set_ylabel('Onset Strength')
    right_axes[3].set_xlim([0, BUFFER_SECONDS])
    right_axes[3].set_ylim([0, 1])
    right_axes[3].grid(True, alpha=0.3)

    # 공유 colorbar (스펙트로그램용)
    fig.colorbar(stft_img, cax=cax, label='dB')
    
    def update(frame):
        """애니메이션 업데이트 함수"""
        signal = analyzer.get_audio_data()

        # ========== 왼쪽 그래프 업데이트 (기존) ==========

        # 1. 파형 업데이트
        line_waveform.set_ydata(signal)

        # RMS 계산
        rms = analyzer.compute_rms(signal)

        # RMS 시간 축 재계산
        current_rms_time = np.linspace(0, BUFFER_SECONDS, len(rms))

        # 2. 진폭 (RMS) 업데이트
        line_rms.set_data(current_rms_time, rms)

        # y축 자동 스케일링
        if len(rms) > 0 and np.max(rms) > 0:
            left_axes[1].set_ylim([0, max(0.1, np.max(rms) * 1.2)])

        # 3. 음압 (절대값) 업데이트
        line_pressure.set_ydata(np.abs(signal))

        # 4. 강도 (RMS^2) 업데이트
        intensity = rms ** 2
        line_intensity.set_data(current_rms_time, intensity)
        if len(intensity) > 0 and np.max(intensity) > 0:
            left_axes[3].set_ylim([0, max(0.01, np.max(intensity) * 1.2)])

        # 5. 저주파 업데이트
        low_freq_signal = analyzer.filter_low_freq(signal)
        line_low_freq.set_ydata(low_freq_signal)

        # 6. 고주파 업데이트
        high_freq_signal = analyzer.filter_high_freq(signal)
        line_high_freq.set_ydata(high_freq_signal)

        # ========== 오른쪽 그래프 업데이트 (STFT 기반) ==========

        # STFT 계산 (한 번만)
        S_db = analyzer.compute_stft_frame(signal)

        # 7. STFT 전체 스펙트로그램
        stft_img.set_data(S_db)

        # 8. 고주파 마스킹 STFT
        S_db_high = analyzer.get_masked_stft(S_db, analyzer.high_freq_mask)
        high_stft_img.set_data(S_db_high)

        # 9. 저주파 마스킹 STFT
        S_db_low = analyzer.get_masked_stft(S_db, analyzer.low_freq_mask)
        low_stft_img.set_data(S_db_low)

        # 10. Onset Detection
        onset_env = analyzer.compute_onset_strength(signal)
        onset_time = np.linspace(0, BUFFER_SECONDS, len(onset_env))
        onset_line.set_data(onset_time, onset_env)

        # Onset Y축 자동 스케일링
        max_onset = np.max(onset_env) if len(onset_env) > 0 else 1
        if max_onset > 0:
            right_axes[3].set_ylim([0, max(1, max_onset * 1.2)])

        return (line_waveform, line_rms, line_pressure, line_intensity,
                line_low_freq, line_high_freq,
                stft_img, high_stft_img, low_stft_img, onset_line)

    # 오디오 스트림 시작
    print("\nStarting audio stream...")
    print(f"Device: [{mic.device_id}] {mic.device_info['name']}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Block Size: {BLOCK_SIZE}")
    print(f"Buffer Duration: {BUFFER_SECONDS} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        with sd.InputStream(
            device=mic.device_id,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            callback=analyzer.audio_callback
        ):
            # 애니메이션 시작
            ani = FuncAnimation(
                fig,
                update,
                interval=50,  # 50ms (20 FPS)
                blit=False,
                cache_frame_data=False
            )
            plt.show()

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
