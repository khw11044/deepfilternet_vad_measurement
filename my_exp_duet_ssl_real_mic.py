import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

c = 343.    # speed of sound
fs = 48000  # sampling frequency (장치에 맞춤)
nfft = 512  # FFT size (48kHz에 맞게 증가)
freq_range = [300, 3500]

# 마이크 간격 (노트북 내장 스테레오 마이크)
mic_distance = 0.15  # 15cm (노트북마다 다름, 조절 필요)

# 사용 가능한 오디오 장치 출력
print("Available audio devices:")
print(sd.query_devices())

# 2채널 스테레오 입력 장치 찾기
input_device = None
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] >= 2 and 'analog-stereo' in dev['name'].lower():
        input_device = i
        print(f"\n>>> Selected device [{i}]: {dev['name']}")
        break

if input_device is None:
    # 기본 입력 장치 사용
    input_device = sd.default.device[0]
    print(f"\n>>> Using default input device: {input_device}")

# 오디오 버퍼 설정
buffer_duration = 0.5  # 0.5초 버퍼
buffer_size = int(fs * buffer_duration)
audio_buffer = np.zeros((buffer_size, 2))  # 2채널 스테레오


def audio_callback(indata, frames, time, status):
    """오디오 콜백 - 버퍼에 데이터 저장"""
    global audio_buffer
    if status:
        print(f"Audio status: {status}")

    # 버퍼 시프트 후 새 데이터 추가
    audio_buffer = np.roll(audio_buffer, -frames, axis=0)
    audio_buffer[-frames:, :] = indata[:, :2] if indata.shape[1] >= 2 else np.column_stack([indata[:, 0], indata[:, 0]])


def compute_stft(signal, nfft, hop):
    """간단한 STFT 계산"""
    num_frames = (len(signal) - nfft) // hop + 1
    if num_frames <= 0:
        return np.zeros((1, nfft // 2 + 1), dtype=complex)

    window = np.hanning(nfft)
    stft = np.zeros((num_frames, nfft // 2 + 1), dtype=complex)

    for i in range(num_frames):
        frame = signal[i * hop:i * hop + nfft] * window
        stft[i] = np.fft.rfft(frame)

    return stft


def compute_duet_histogram(X1, X2, freq_bins, mic_distance,
                           num_angle_bins=180, freq_min=300, freq_max=3500):
    """DUET 스타일 TF-bin별 위상차 히스토그램 계산"""
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)
    angles = np.linspace(-90, 90, num_angle_bins)
    histogram = np.zeros(num_angle_bins)

    for f_idx in np.where(freq_mask)[0]:
        freq = freq_bins[f_idx]
        if freq < 50:
            continue

        for t_idx in range(X1.shape[0]):
            x1 = X1[t_idx, f_idx]
            x2 = X2[t_idx, f_idx]

            energy = np.abs(x1)**2 + np.abs(x2)**2
            if energy < 1e-10:
                continue

            cross = x2 * np.conj(x1)
            phase_diff = np.angle(cross)

            max_phase = 2 * np.pi * freq * mic_distance / c
            if max_phase > 0:
                sin_theta = phase_diff / max_phase
                sin_theta = np.clip(sin_theta, -1, 1)
                theta_deg = np.degrees(np.arcsin(sin_theta))

                bin_idx = np.argmin(np.abs(angles - theta_deg))
                histogram[bin_idx] += energy

    if histogram.max() > 0:
        histogram = histogram / histogram.max()

    return angles, histogram


# 주파수 빈 계산
freq_bins = np.fft.rfftfreq(nfft, 1/fs)

# Figure 설정
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 좌상: 주파수 대역별 DOA 히스토그램
ax1 = axes[0, 0]
line_full, = ax1.plot([], [], 'b-', linewidth=2, label='Full Band (300-3500Hz)')
line_high, = ax1.plot([], [], 'r-', linewidth=2, label='High Freq (>2kHz)')
line_low, = ax1.plot([], [], 'g-', linewidth=2, label='Low Freq (<1kHz)')
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Normalized Energy')
ax1.set_title('Real-time DUET DOA Histogram')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 우상: 시간-각도 scatter plot
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)  # 10초 윈도우
ax2.set_ylim(-90, 90)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Detected Angle (degrees)')
ax2.set_title('DOA Tracking over Time')
ax2.grid(True, alpha=0.3)
scatter = ax2.scatter([], [], c='blue', s=20, alpha=0.5)

# 좌하: 오디오 파형
ax3 = axes[1, 0]
line_wave1, = ax3.plot([], [], 'b-', linewidth=1, alpha=0.7, label='Mic 1')
line_wave2, = ax3.plot([], [], 'r-', linewidth=1, alpha=0.7, label='Mic 2')
ax3.set_xlim(0, buffer_duration)
ax3.set_ylim(-1, 1)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Audio Waveform (Stereo)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 우하: 현재 감지된 방향 표시 (polar)
ax4 = axes[1, 1]
ax4 = fig.add_subplot(2, 2, 4, projection='polar')
ax4.set_theta_zero_location('N')  # 0도가 위쪽
ax4.set_theta_direction(-1)  # 시계방향
ax4.set_ylim(0, 1.5)
ax4.set_title('Detected Direction')

# 화살표 (방향 표시용)
arrow = ax4.annotate('', xy=(0, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=3))

# 상태 텍스트
status_text = fig.suptitle('Initializing...', fontsize=14)

# 누적 데이터
time_counter = [0.0]
time_history = []
angle_history = []


def find_peaks(histogram, angles, threshold=0.5, min_distance=15):
    """히스토그램에서 피크 찾기"""
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > threshold:
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                if not peaks or abs(angles[i] - peaks[-1]) > min_distance:
                    peaks.append(angles[i])
    return peaks


def update(frame):
    global time_counter, time_history, angle_history

    # STFT 계산
    X1 = compute_stft(audio_buffer[:, 0], nfft, nfft // 2)
    X2 = compute_stft(audio_buffer[:, 1], nfft, nfft // 2)

    if X1.shape[0] < 2:
        return

    # 주파수 대역별 DUET 히스토그램
    angles, hist_full = compute_duet_histogram(X1, X2, freq_bins, mic_distance,
                                                freq_min=300, freq_max=3500)
    _, hist_high = compute_duet_histogram(X1, X2, freq_bins, mic_distance,
                                           freq_min=2000, freq_max=7000)
    _, hist_low = compute_duet_histogram(X1, X2, freq_bins, mic_distance,
                                          freq_min=100, freq_max=1000)

    # 1D 히스토그램 업데이트
    line_full.set_data(angles, hist_full)
    line_high.set_data(angles, hist_high)
    line_low.set_data(angles, hist_low)

    # 피크 찾기
    peaks = find_peaks(hist_full, angles)

    # 시간 업데이트
    time_counter[0] += 0.1  # 100ms 간격

    # 피크가 있으면 기록
    for p in peaks:
        time_history.append(time_counter[0])
        angle_history.append(p)

    # 오래된 데이터 제거 (10초 윈도우)
    while time_history and time_history[0] < time_counter[0] - 10:
        time_history.pop(0)
        angle_history.pop(0)

    # Scatter plot 업데이트
    if time_history:
        scatter.set_offsets(np.c_[time_history, angle_history])
        ax2.set_xlim(max(0, time_counter[0] - 10), time_counter[0])

    # 파형 업데이트
    t_wave = np.linspace(0, buffer_duration, len(audio_buffer))
    max_amp = np.max(np.abs(audio_buffer)) + 1e-10
    line_wave1.set_data(t_wave, audio_buffer[:, 0] / max_amp)
    line_wave2.set_data(t_wave, audio_buffer[:, 1] / max_amp)

    # 방향 화살표 업데이트
    if peaks:
        main_angle = peaks[0]  # 가장 강한 피크
        theta_rad = np.radians(main_angle + 90)  # 90도 회전 (위쪽이 0도)
        arrow.xy = (theta_rad, 1.2)
        arrow.set_visible(True)
        direction_str = f"Main: {main_angle:.0f}°"
    else:
        arrow.set_visible(False)
        direction_str = "No detection"

    # 상태 텍스트 업데이트
    rms = np.sqrt(np.mean(audio_buffer**2))
    db = 20 * np.log10(rms + 1e-10)
    status_text.set_text(f'Time: {time_counter[0]:.1f}s | RMS: {db:.1f}dB | {direction_str}')


# 오디오 스트림 시작
print(f"\nStarting audio stream... (fs={fs}Hz, 2 channels)")
print("Press Ctrl+C to stop\n")

try:
    stream = sd.InputStream(
        device=input_device,
        samplerate=fs,
        channels=2,
        callback=audio_callback,
        blocksize=int(fs * 0.05)  # 50ms 블록
    )
    stream.start()
    print(f"Audio stream started: {fs}Hz, 2ch, device={input_device}")

    # 애니메이션 실행
    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    plt.tight_layout()
    plt.show()

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
    print("\nTip: Make sure you have a stereo microphone or try:")
    print("  - Check 'sounddevice.query_devices()' for available devices")
    print("  - Set specific device: sd.default.device = [device_id, None]")
finally:
    if 'stream' in dir():
        stream.stop()
        stream.close()
