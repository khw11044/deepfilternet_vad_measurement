"""
실시간 오디오 데이터 분석 및 시각화
- 파형 (Waveform)
- 진폭 (Amplitude Envelope) - RMS 기반
- 음압 (Sound Pressure) - 정규화된 신호 기준
- 강도 (Intensity) - 진폭의 제곱에 비례
- 저주파/고주파 필터링
- 오디오 특징 추출 (RMS, ZCR, Centroid, Bandwidth, Rolloff, MFCC)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sounddevice as sd
import librosa
from scipy.signal import butter, sosfilt
from collections import deque
import threading

# 오디오 설정
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
CHANNELS = 1

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

# Feature history 설정 (약 3초, 20FPS 기준)
FEATURE_HISTORY_SIZE = 60


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
            for device in input_devices:
                if device['id'] == device_id:
                    self.device_id = device_id
                    self.device_info = device
                    print(f"Selected: [{device_id}] {device['name']}")
                    return self
            raise ValueError(f"Device ID {device_id} not found or not an input device")

        self.list_devices()
        while True:
            try:
                choice = input("Select device ID (or press Enter for default): ").strip()
                if choice == "":
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


def extract_single_audio_feature(y, sr):
    """
    블로그 기반 오디오 특징 추출 (실시간용 요약 벡터)
    입력: y (1D audio), sr
    출력: dict 형태의 feature
    """
    features = {}

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    features["rms"] = np.mean(rms)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr"] = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["centroid"] = np.mean(centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["bandwidth"] = np.mean(bandwidth)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["rolloff"] = np.mean(rolloff)

    # MFCC (13차 평균)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features["mfcc"] = np.mean(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma"] = np.mean(chroma, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["contrast"] = np.mean(contrast, axis=1)

    return features


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

        # Feature history (약 3초, 20FPS 기준)
        self.feature_history = {
            "rms": deque(maxlen=FEATURE_HISTORY_SIZE),
            "zcr": deque(maxlen=FEATURE_HISTORY_SIZE),
            "centroid": deque(maxlen=FEATURE_HISTORY_SIZE),
            "bandwidth": deque(maxlen=FEATURE_HISTORY_SIZE),
            "rolloff": deque(maxlen=FEATURE_HISTORY_SIZE),
        }

        # 초기값으로 채우기
        for key in self.feature_history:
            for _ in range(FEATURE_HISTORY_SIZE):
                self.feature_history[key].append(0.0)

    def audio_callback(self, indata, frames, time, status):
        """오디오 입력 콜백"""
        if status:
            print(f"Audio status: {status}")

        with self.lock:
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

    def update_feature_history(self, features):
        """Feature history 업데이트"""
        for key in self.feature_history:
            if key in features:
                self.feature_history[key].append(features[key])


def main():
    # 마이크 디바이스 선택
    mic = Microphone()
    mic.select_device()

    analyzer = AudioAnalyzer()

    # Figure 설정 - GridSpec으로 6행 x 2열 레이아웃
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1, 1], hspace=0.5, wspace=0.3)
    fig.suptitle('Real-time Audio Analysis (with Feature Extraction)', fontsize=14, fontweight='bold')

    # 왼쪽 열: 기존 6개 그래프
    left_axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]

    # 오른쪽 열: Feature 그래프 6개
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(6)]

    # 시간 축 생성
    time_axis = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)
    feature_time_axis = np.arange(FEATURE_HISTORY_SIZE)

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

    # ========== 오른쪽 그래프 (Feature 6개) ==========

    feature_config = [
        ("rms", "7. RMS (Root Mean Square)", "coral", 0, 0.5),
        ("zcr", "8. Zero Crossing Rate", "lime", 0, 0.5),
        ("centroid", "9. Spectral Centroid (Hz)", "gold", 0, 5000),
        ("bandwidth", "10. Spectral Bandwidth (Hz)", "deepskyblue", 0, 3000),
        ("rolloff", "11. Spectral Rolloff (Hz)", "orchid", 0, 10000),
    ]

    feature_lines = {}
    for i, (key, title, color, ymin, ymax) in enumerate(feature_config):
        line, = right_axes[i].plot(feature_time_axis, np.zeros(FEATURE_HISTORY_SIZE),
                                    color=color, linewidth=2)
        feature_lines[key] = line
        right_axes[i].set_title(title)
        right_axes[i].set_ylabel(key.upper())
        right_axes[i].set_xlim([0, FEATURE_HISTORY_SIZE])
        right_axes[i].set_ylim([ymin, ymax])
        right_axes[i].grid(True, alpha=0.3)

    # 6번째 오른쪽: MFCC 첫 번째 계수 (에너지 관련)
    line_mfcc, = right_axes[5].plot(feature_time_axis, np.zeros(FEATURE_HISTORY_SIZE),
                                     color='magenta', linewidth=2)
    right_axes[5].set_title('12. MFCC[0] (Energy-related)')
    right_axes[5].set_xlabel('Frame')
    right_axes[5].set_ylabel('MFCC[0]')
    right_axes[5].set_xlim([0, FEATURE_HISTORY_SIZE])
    right_axes[5].set_ylim([-50, 50])
    right_axes[5].grid(True, alpha=0.3)

    # MFCC history
    mfcc_history = deque(maxlen=FEATURE_HISTORY_SIZE)
    for _ in range(FEATURE_HISTORY_SIZE):
        mfcc_history.append(0.0)

    def update(frame):
        """애니메이션 업데이트 함수"""
        signal = analyzer.get_audio_data()

        # ========== 왼쪽 그래프 업데이트 (기존) ==========

        # 1. 파형 업데이트
        line_waveform.set_ydata(signal)

        # RMS 계산
        rms = analyzer.compute_rms(signal)
        current_rms_time = np.linspace(0, BUFFER_SECONDS, len(rms))

        # 2. 진폭 (RMS) 업데이트
        line_rms.set_data(current_rms_time, rms)
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

        # ========== 블로그 Feature 추출 ==========
        features = extract_single_audio_feature(signal, SAMPLE_RATE)

        # Feature history 업데이트
        analyzer.update_feature_history(features)

        # MFCC history 업데이트 (첫 번째 계수)
        mfcc_history.append(features["mfcc"][0])

        # ========== 오른쪽 그래프 업데이트 (Feature) ==========

        for key, line in feature_lines.items():
            data = list(analyzer.feature_history[key])
            line.set_ydata(data)

            # Y축 자동 스케일링
            if len(data) > 0:
                data_min = min(data)
                data_max = max(data)
                margin = (data_max - data_min) * 0.1 + 1e-6
                right_axes[list(feature_lines.keys()).index(key)].set_ylim(
                    data_min - margin, data_max + margin
                )

        # MFCC 업데이트
        mfcc_data = list(mfcc_history)
        line_mfcc.set_ydata(mfcc_data)
        if len(mfcc_data) > 0:
            mfcc_min = min(mfcc_data)
            mfcc_max = max(mfcc_data)
            margin = (mfcc_max - mfcc_min) * 0.1 + 1e-6
            right_axes[5].set_ylim(mfcc_min - margin, mfcc_max + margin)

        return (line_waveform, line_rms, line_pressure, line_intensity,
                line_low_freq, line_high_freq, line_mfcc,
                *feature_lines.values())

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
