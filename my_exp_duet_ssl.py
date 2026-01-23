import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
import pyroomacoustics as pra

c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]
chunk_frames = 20  # 청크당 STFT 프레임 수


# Location of sources
azimuth = np.array([61, 180]) / 180. * np.pi
distance = 2.  # meters

fs1, signal1 = wavfile.read("/home/khw/workspace/Audio/모닥불.wav")
fs2, signal2 = wavfile.read("/home/khw/workspace/Audio/arctic_a0010.wav")

signals = [signal1, signal2]

snr_db = 5.
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

# 방 생성
room_dim = np.r_[10.,10.]
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

# 방 정 중앙에 위치 - 2채널 마이크만 사용 (DUET용)
echo = pra.circular_2D_array(center=room_dim/2, M=2, phi0=0, radius=37.5e-3)
aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))

# 마이크 간격
mic_distance = 0.075  # 7.5cm (37.5mm * 2)


for i, ang in enumerate(azimuth):
    source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
    aroom.add_source(source_location, signal=signals[i])

# Run the simulation
aroom.simulate()

# 2채널 신호
mic_signals = aroom.mic_array.signals  # shape: (2, samples)

# 전체 STFT 계산
X1 = pra.transform.stft.analysis(mic_signals[0], nfft, nfft // 2)  # (frames, freq_bins)
X2 = pra.transform.stft.analysis(mic_signals[1], nfft, nfft // 2)

total_frames = X1.shape[0]
freq_bins = np.fft.rfftfreq(nfft, 1/fs)

print(f"Total frames: {total_frames}")
print(f"Freq bins: {len(freq_bins)}")


def compute_duet_histogram(X1_chunk, X2_chunk, freq_bins, fs, mic_distance,
                           num_angle_bins=360, freq_min=300, freq_max=3500):
    """
    DUET 스타일 TF-bin별 위상차 히스토그램 계산

    Returns:
        angles: 각도 배열 (degrees)
        histogram: 각 각도에 대한 에너지 히스토그램
    """
    # 주파수 범위 필터링
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)

    # 각도 빈 설정 (-180 ~ 180도)
    angles = np.linspace(-180, 180, num_angle_bins)
    histogram = np.zeros(num_angle_bins)

    # 각 TF bin에 대해 위상차 계산
    for f_idx in np.where(freq_mask)[0]:
        freq = freq_bins[f_idx]
        if freq < 50:  # 너무 낮은 주파수 제외
            continue

        for t_idx in range(X1_chunk.shape[0]):
            x1 = X1_chunk[t_idx, f_idx]
            x2 = X2_chunk[t_idx, f_idx]

            # 에너지가 너무 낮으면 스킵
            energy = np.abs(x1)**2 + np.abs(x2)**2
            if energy < 1e-10:
                continue

            # 위상차 계산 (cross-spectrum)
            cross = x2 * np.conj(x1)
            phase_diff = np.angle(cross)

            # 위상차 -> 시간차 -> 각도 변환
            # τ = d * sin(θ) / c
            # phase_diff = 2π * f * τ
            # sin(θ) = phase_diff * c / (2π * f * d)

            max_phase = 2 * np.pi * freq * mic_distance / c
            if max_phase > 0:
                sin_theta = phase_diff / max_phase
                sin_theta = np.clip(sin_theta, -1, 1)
                theta_rad = np.arcsin(sin_theta)
                theta_deg = np.degrees(theta_rad)

                # 히스토그램에 에너지 가중치로 추가
                bin_idx = np.argmin(np.abs(angles - theta_deg))
                histogram[bin_idx] += energy

    # 정규화
    if histogram.max() > 0:
        histogram = histogram / histogram.max()

    return angles, histogram


def compute_duet_2d_histogram(X1_chunk, X2_chunk, freq_bins,
                               num_atten_bins=50, num_delay_bins=50,
                               freq_min=300, freq_max=3500):
    """
    DUET 2D 히스토그램 (감쇠비 vs 위상차)
    각 TF bin의 (α, δ) 값을 2D 공간에 플로팅
    """
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)

    # 감쇠비 범위 (log scale)
    atten_range = np.linspace(-3, 3, num_atten_bins)  # log10(α)
    delay_range = np.linspace(-np.pi, np.pi, num_delay_bins)

    histogram_2d = np.zeros((num_atten_bins, num_delay_bins))

    for f_idx in np.where(freq_mask)[0]:
        for t_idx in range(X1_chunk.shape[0]):
            x1 = X1_chunk[t_idx, f_idx]
            x2 = X2_chunk[t_idx, f_idx]

            if np.abs(x1) < 1e-10:
                continue

            # 감쇠비 (amplitude ratio)
            alpha = np.abs(x2) / np.abs(x1)
            log_alpha = np.log10(alpha + 1e-10)

            # 위상차
            delta = np.angle(x2 / x1)

            # 에너지 가중치
            energy = np.abs(x1)**2 + np.abs(x2)**2

            # 히스토그램에 추가
            a_idx = np.argmin(np.abs(atten_range - log_alpha))
            d_idx = np.argmin(np.abs(delay_range - delta))

            if 0 <= a_idx < num_atten_bins and 0 <= d_idx < num_delay_bins:
                histogram_2d[a_idx, d_idx] += energy

    if histogram_2d.max() > 0:
        histogram_2d = histogram_2d / histogram_2d.max()

    return atten_range, delay_range, histogram_2d


# Figure 설정
fig = plt.figure(figsize=(14, 10))

# 1행: DUET 1D 각도 히스토그램 (주파수 대역별)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

# 2행: DUET 2D 히스토그램 + SRP 비교
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4, projection='polar')

# 초기 플롯
line_full, = ax1.plot([], [], 'b-', linewidth=2, label='Full Band')
line_high, = ax1.plot([], [], 'r-', linewidth=2, label='High Freq (>2kHz)')
line_low, = ax1.plot([], [], 'g-', linewidth=2, label='Low Freq (<1kHz)')

ax1.set_xlim(-90, 90)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Normalized Energy')
ax1.set_title('DUET: TF-bin Phase Difference Histogram')
ax1.legend()
ax1.grid(True, alpha=0.3)

# True locations 표시 (degree로 변환, -90~90 범위로)
true_angles_deg = []
for ang in azimuth:
    deg = np.degrees(ang)
    if deg > 180:
        deg = deg - 360
    if deg > 90:
        deg = 180 - deg
    elif deg < -90:
        deg = -180 - deg
    true_angles_deg.append(deg)

for true_ang in true_angles_deg:
    ax1.axvline(x=true_ang, color='k', linestyle='--', alpha=0.5, label=f'True: {true_ang:.0f}°')

# 시간에 따른 각도 추적
ax2.set_xlim(0, total_frames * (nfft // 2) / fs)
ax2.set_ylim(-90, 90)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Detected Angle (degrees)')
ax2.set_title('DOA Tracking over Time')
ax2.grid(True, alpha=0.3)

scatter_full = ax2.scatter([], [], c='blue', s=10, alpha=0.5, label='Full Band')
scatter_high = ax2.scatter([], [], c='red', s=10, alpha=0.5, label='High Freq')
scatter_low = ax2.scatter([], [], c='green', s=10, alpha=0.5, label='Low Freq')
ax2.legend()

# 2D 히스토그램 (빈 이미지로 시작)
img = ax3.imshow(np.zeros((50, 50)), extent=[-np.pi, np.pi, -3, 3],
                  aspect='auto', origin='lower', cmap='hot')
ax3.set_xlabel('Phase Difference (rad)')
ax3.set_ylabel('Log Amplitude Ratio')
ax3.set_title('DUET 2D Histogram (α vs δ)')
plt.colorbar(img, ax=ax3)

# SRP-PHAT 결과 (polar plot)
doa_srp = pra.doa.algorithms['SRP'](echo, fs, nfft, c=c, num_src=2, max_four=4)
phi_plt = doa_srp.grid.azimuth
c_phi_plt = np.r_[phi_plt, phi_plt[0]]
line_srp, = ax4.plot(c_phi_plt, np.ones_like(c_phi_plt), 'b-', linewidth=2)
ax4.set_title('SRP-PHAT')
ax4.set_ylim([0, 1.5])

# True locations on polar
for ang in azimuth:
    ax4.axvline(x=ang, color='k', linestyle='--', alpha=0.5)

# 시간 표시
time_text = fig.suptitle('Time: 0.00s', fontsize=14)

# 누적 데이터 저장 (각 대역별로 시간도 따로 저장)
time_history_full = []
time_history_high = []
time_history_low = []
angle_history_full = []
angle_history_high = []
angle_history_low = []


def find_peaks(histogram, angles, threshold=0.3, min_distance=10):
    """히스토그램에서 피크 찾기"""
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > threshold:
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                # 인접 피크와 거리 확인
                if not peaks or abs(angles[i] - peaks[-1]) > min_distance:
                    peaks.append(angles[i])
    return peaks


def update(frame_idx):
    global time_history_full, time_history_high, time_history_low
    global angle_history_full, angle_history_high, angle_history_low

    start_frame = frame_idx * chunk_frames
    end_frame = min(start_frame + chunk_frames, total_frames)

    if start_frame >= total_frames:
        return

    # 현재 청크
    X1_chunk = X1[start_frame:end_frame, :]
    X2_chunk = X2[start_frame:end_frame, :]

    current_time = start_frame * (nfft // 2) / fs

    # 주파수 대역별 DUET 히스토그램
    angles, hist_full = compute_duet_histogram(X1_chunk, X2_chunk, freq_bins,
                                                fs, mic_distance, freq_min=300, freq_max=3500)
    _, hist_high = compute_duet_histogram(X1_chunk, X2_chunk, freq_bins,
                                           fs, mic_distance, freq_min=2000, freq_max=7000)
    _, hist_low = compute_duet_histogram(X1_chunk, X2_chunk, freq_bins,
                                          fs, mic_distance, freq_min=100, freq_max=1000)

    # 1D 히스토그램 업데이트
    line_full.set_data(angles, hist_full)
    line_high.set_data(angles, hist_high)
    line_low.set_data(angles, hist_low)

    # 피크 찾아서 시간 기록에 추가 (각 대역별로 시간도 따로)
    peaks_full = find_peaks(hist_full, angles)
    peaks_high = find_peaks(hist_high, angles)
    peaks_low = find_peaks(hist_low, angles)

    for p in peaks_full:
        time_history_full.append(current_time)
        angle_history_full.append(p)
    for p in peaks_high:
        time_history_high.append(current_time)
        angle_history_high.append(p)
    for p in peaks_low:
        time_history_low.append(current_time)
        angle_history_low.append(p)

    # Scatter plot 업데이트 (각 대역별로)
    N = 500
    if len(angle_history_full) > 0:
        t_full = np.array(time_history_full[-N:])
        a_full = np.array(angle_history_full[-N:])
        scatter_full.set_offsets(np.c_[t_full, a_full])

    if len(angle_history_high) > 0:
        t_high = np.array(time_history_high[-N:])
        a_high = np.array(angle_history_high[-N:])
        scatter_high.set_offsets(np.c_[t_high, a_high])

    if len(angle_history_low) > 0:
        t_low = np.array(time_history_low[-N:])
        a_low = np.array(angle_history_low[-N:])
        scatter_low.set_offsets(np.c_[t_low, a_low])

    # 2D 히스토그램 업데이트
    _, _, hist_2d = compute_duet_2d_histogram(X1_chunk, X2_chunk, freq_bins)
    img.set_array(hist_2d.T)
    img.set_clim(0, 1)

    # SRP-PHAT 업데이트
    X_chunk = np.stack([X1_chunk.T, X2_chunk.T], axis=0)  # (2, freq_bins, frames)
    doa_srp.locate_sources(X_chunk, freq_range=freq_range)
    resp = doa_srp.grid.values
    resp = (resp - resp.min()) / (resp.max() - resp.min() + 1e-10)
    c_resp = np.r_[resp, resp[0]]
    line_srp.set_ydata(c_resp + 0.5)

    time_text.set_text(f'Time: {current_time:.2f}s')


# 애니메이션 실행
num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
ani = FuncAnimation(fig, update, frames=num_chunks, interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()
