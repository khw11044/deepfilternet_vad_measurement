import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import librosa.display
import pyaudio
import sys

# 1. 오디오 설정
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050  # 샘플링 레이트
CHUNK = 1024 * 4  # 한 번에 읽을 샘플 수 (실시간 처리를 위해 적절히 설정)

# PyAudio 객체 생성
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 2. 시각화 설정 (6개의 서브플롯)
fig, axes = plt.subplots(6, 1, figsize=(12, 15))
plt.subplots_adjust(hspace=0.5)

# 각 그래프 초기화
x_time = np.linspace(0, CHUNK / RATE, CHUNK)
line_wave, = axes[0].plot(x_time, np.zeros(CHUNK), color='blue')
line_rms, = axes[1].plot([0], [0], color='orange', marker='o')
line_sp, = axes[2].plot(x_time, np.zeros(CHUNK), color='green')
line_int, = axes[3].plot([0], [0], color='purple', marker='s')
line_db, = axes[4].plot([0], [0], color='red', marker='x')

# 멜-스펙트로그램 초기 설정
# 초기 데이터로 스펙트로그램 생성
S = librosa.feature.melspectrogram(y=np.zeros(CHUNK), sr=RATE, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=RATE, ax=axes[5])
fig.colorbar(img, ax=axes[5], format='%+2.0f dB')

# 그래프 제목 및 축 설정
titles = [
    '1. Waveform (Original Signal)', 
    '2. Amplitude Envelope (RMS based)', 
    '3. Sound Pressure (Normalized)', 
    '4. Intensity (Proportional to Power)', 
    '5. Amplitude in Decibels (dB)', 
    '6. Mel-Spectrogram'
]
ylabels = ['Amplitude', 'RMS', 'Pressure', 'Intensity', 'dB', 'Mel']

for i, ax in enumerate(axes):
    ax.set_title(titles[i])
    ax.set_ylabel(ylabels[i])
    if i < 5: ax.set_xlabel('Time (s)')

# 범위 고정
axes[0].set_ylim(-1, 1)
axes[2].set_ylim(-1, 1)
axes[4].set_ylim(-80, 0)

def update(frame):
    try:
        # 오디오 데이터 읽기
        data = stream.read(CHUNK, exception_on_overflow=False)
        y = np.frombuffer(data, dtype=np.float32)

        # 1. 파형 (Waveform)
        line_wave.set_ydata(y)

        # 2. 진폭 (Amplitude Envelope) - RMS 기반
        rms = librosa.feature.rms(y=y)[0, 0]
        line_rms.set_data([CHUNK/(2*RATE)], [rms])
        axes[1].set_xlim(0, CHUNK/RATE)
        axes[1].set_ylim(0, 0.5)

        # 3. 음압 (Sound Pressure) - 정규화된 신호 기준
        # 튜토리얼 로직: sound_pressure = array / np.max(np.abs(array))
        max_val = np.max(np.abs(y))
        sound_pressure = y / max_val if max_val > 0 else y
        line_sp.set_ydata(sound_pressure)

        # 4. 강도 (Intensity) - 진폭의 제곱에 비례
        intensity = rms ** 2
        line_int.set_data([CHUNK/(2*RATE)], [intensity])
        axes[3].set_xlim(0, CHUNK/RATE)
        axes[3].set_ylim(0, 0.25)

        # 5. dB (Decibels) - RMS를 dB로 변환
        # 튜토리얼 로직: rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        # 실시간에서는 이전 값들의 최대값을 ref로 쓰거나 1.0(Full Scale)을 기준으로 함
        rms_db = librosa.amplitude_to_db(np.array([[rms]]), ref=1.0)[0, 0]
        line_db.set_data([CHUNK/(2*RATE)], [rms_db])
        axes[4].set_xlim(0, CHUNK/RATE)

        # 6. 멜-스펙트로그램 시각화
        axes[5].clear()
        S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=RATE, ax=axes[5])
        axes[5].set_title(titles[5])

    except Exception as e:
        print(f"Error: {e}")
    
    return line_wave, line_rms, line_sp, line_int, line_db

# 애니메이션 실행
print("실시간 오디오 분석을 시작합니다... (Ctrl+C로 종료)")
ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()

# 리소스 해제
stream.stop_stream()
stream.close()
p.terminate()