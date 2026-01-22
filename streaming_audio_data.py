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
import librosa.display
from collections import deque
import threading

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

# RMS/멜 스펙트로그램 설정
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MELS = 128
FMAX = 8000

# 멜 스펙트로그램 히스토리 (시간축)
MEL_HISTORY_FRAMES = 100


class AudioAnalyzer:
    def __init__(self):
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=BUFFER_SIZE)
        self.audio_buffer.extend(np.zeros(BUFFER_SIZE))

        # 멜 스펙트로그램 히스토리
        self.mel_history = deque(maxlen=MEL_HISTORY_FRAMES)
        for _ in range(MEL_HISTORY_FRAMES):
            self.mel_history.append(np.zeros(N_MELS))

        # 스레드 동기화
        self.lock = threading.Lock()

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

    def compute_mel_spectrogram(self, signal):
        """멜 스펙트로그램 계산"""
        S = librosa.feature.melspectrogram(
            y=signal,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            fmax=FMAX,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB


def main():
    # 마이크 디바이스 선택
    mic = Microphone()
    mic.select_device()  # 대화형 선택
    # mic.get_default_device()  # 또는 기본 디바이스 자동 선택
    # mic.select_device(device_id=0)  # 또는 특정 디바이스 ID 지정

    analyzer = AudioAnalyzer()

    # Figure 설정 - GridSpec으로 colorbar 공간 확보
    fig = plt.figure(figsize=(14, 18))
    fig.suptitle('Real-time Audio Analysis', fontsize=14, fontweight='bold')

    # GridSpec: 6행 x 2열 (두 번째 열은 colorbar용, 폭 비율 30:1)
    gs = GridSpec(6, 2, figure=fig, width_ratios=[30, 1], hspace=0.7, wspace=0.05)

    # 1~5번 그래프는 첫 번째 열 전체 사용
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    # 6번 멜 스펙트로그램
    axes.append(fig.add_subplot(gs[5, 0]))
    # colorbar용 축
    cax = fig.add_subplot(gs[5, 1])

    # 시간 축 생성
    time_axis = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)

    # 1. 파형 (Waveform)
    line_waveform, = axes[0].plot(time_axis, np.zeros(BUFFER_SIZE),
                                   color='steelblue', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('1. Waveform')
    axes[0].set_xlim([0, BUFFER_SECONDS])
    axes[0].set_ylim([-1, 1])
    axes[0].grid(True, alpha=0.3)

    # RMS 시간 축 (초기값)
    rms_frames = int(BUFFER_SIZE / HOP_LENGTH) + 1
    rms_time = np.linspace(0, BUFFER_SECONDS, rms_frames)

    # 2. 진폭 (Amplitude Envelope) - RMS 기반
    line_rms, = axes[1].plot(rms_time, np.zeros(rms_frames),
                              color='orange', linewidth=1.5)
    axes[1].set_ylabel('RMS Amplitude')
    axes[1].set_title('2. Amplitude Envelope')
    axes[1].set_xlim([0, BUFFER_SECONDS])
    axes[1].set_ylim([0, 0.5])
    axes[1].grid(True, alpha=0.3)

    # 3. 음압 (Sound Pressure) - 정규화된 신호 절대값
    line_pressure, = axes[2].plot(time_axis, np.zeros(BUFFER_SIZE),
                                   color='green', linewidth=0.5)
    axes[2].set_ylabel('|Pressure|')
    axes[2].set_title('3. Sound Pressure')
    axes[2].set_xlim([0, BUFFER_SECONDS])
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    # 4. 강도 (Intensity) - RMS^2
    line_intensity, = axes[3].plot(rms_time, np.zeros(rms_frames),
                                    color='purple', linewidth=1.5)
    axes[3].set_ylabel('Intensity (RMS²)')
    axes[3].set_title('4. Sound Intensity')
    axes[3].set_xlim([0, BUFFER_SECONDS])
    axes[3].set_ylim([0, 0.1])
    axes[3].grid(True, alpha=0.3)

    # 5. dB (Decibels)
    line_db, = axes[4].plot(rms_time, np.full(rms_frames, -80),
                            color='red', linewidth=1.5)
    axes[4].set_ylabel('dB (ref=max)')
    axes[4].set_title('5. Amplitude in Decibels')
    axes[4].set_xlim([0, BUFFER_SECONDS])
    axes[4].set_ylim([-80, 0])
    axes[4].grid(True, alpha=0.3)

    # 6. 멜 스펙트로그램
    mel_data = np.zeros((N_MELS, MEL_HISTORY_FRAMES))
    mel_img = axes[5].imshow(
        mel_data,
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=-80,
        vmax=0,
        extent=[0, BUFFER_SECONDS, 0, FMAX]
    )
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Frequency (Hz)')
    axes[5].set_title('6. Mel Spectrogram')
    fig.colorbar(mel_img, cax=cax, label='dB')
    
    def update(frame):
        """애니메이션 업데이트 함수"""
        signal = analyzer.get_audio_data()

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
            axes[1].set_ylim([0, max(0.1, np.max(rms) * 1.2)])

        # 3. 음압 (절대값) 업데이트
        line_pressure.set_ydata(np.abs(signal))

        # 4. 강도 (RMS^2) 업데이트
        intensity = rms ** 2
        line_intensity.set_data(current_rms_time, intensity)
        if len(intensity) > 0 and np.max(intensity) > 0:
            axes[3].set_ylim([0, max(0.01, np.max(intensity) * 1.2)])

        # 5. dB 업데이트
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        line_db.set_data(current_rms_time, rms_db)

        # 6. 멜 스펙트로그램 업데이트
        mel_spec = analyzer.compute_mel_spectrogram(signal)
        mel_img.set_data(mel_spec)
        mel_img.set_clim(vmin=-80, vmax=0)

        return line_waveform, line_rms, line_pressure, line_intensity, line_db, mel_img

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
