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


# dB 임계값 설정 (이 값 이상의 소리만 감지)
db_threshold = -10  # dB

# Location of sources
azimuth = np.array([61, 180]) / 180. * np.pi
distance = 2.  # meters

fs1, signal1 = wavfile.read("/home/khw/workspace/Audio/모닥불.wav")
fs2, signal2 = wavfile.read("/home/khw/workspace/Audio/벨소리.wav")

signals = [signal1, signal2]

snr_db = 5.    # signal-to-noise ratiob
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

# 방 생성
room_dim = np.r_[10.,10.]
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

# 방 정 중앙에 위치
echo = pra.circular_2D_array(center=room_dim/2, M=2, phi0=0, radius=37.5e-3)
echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))

for i, ang in enumerate(azimuth):
    source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
    aroom.add_source(source_location, signal=signals[i])

# Run the simulation
aroom.simulate()

# 전체 STFT 계산
X_full = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
X_full = X_full.transpose([2, 1, 0])  # [channels, freq_bins, frames]
total_frames = X_full.shape[2]

# 주파수 bin 계산
freq_bins = np.fft.rfftfreq(nfft, 1/fs)
high_freq_cutoff = 3000
low_freq_cutoff = 1000
low_freq_mask = freq_bins < high_freq_cutoff
high_freq_mask = freq_bins > low_freq_cutoff


# DOA 알고리즘 초기화
algo_names = ['SRP', 'MUSIC', 'TOPS']
doa_algos = {}
for algo_name in algo_names:
    doa_algos[algo_name] = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=2, max_four=4)

# plotting param
base = 1.
height = 10.
true_col = [0, 0, 0]

# Figure 설정 (4x3: 전체, 고주파, 저주파, 큰소리)
fig, axes = plt.subplots(4, 3, figsize=(12, 12), subplot_kw={'projection': 'polar'})
phi_plt = doa_algos['SRP'].grid.azimuth
c_phi_plt = np.r_[phi_plt, phi_plt[0]]

# 플롯 라인 저장
lines = {}
for row_idx in range(4):
    for col_idx, algo_name in enumerate(algo_names):
        ax = axes[row_idx, col_idx]
        line, = ax.plot(c_phi_plt, np.ones_like(c_phi_plt) * base, linewidth=3,
                        alpha=0.55, linestyle='-', label="spatial spectrum")
        lines[(row_idx, col_idx)] = line

        if row_idx == 0:
            ax.set_title(algo_name, fontsize=14)

        # plot true loc
        for angle in azimuth:
            ax.plot([angle, angle], [base, base + height], linewidth=3, linestyle='--',
                color=true_col, alpha=0.6)

        K = len(azimuth)
        ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col, (K, 1)),
                   s=500, alpha=0.75, marker='*', linewidths=0, label='true locations')

        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 * (base + height)])

# Row labels
row_labels = [
    f'Full Band ({freq_range[0]}-{freq_range[1]}Hz)',
    f'High Freq (>{high_freq_cutoff}Hz)',
    f'Low Freq (<{low_freq_cutoff}Hz)',
    f'Loud (>{db_threshold}dB)',
]
for row_idx in range(4):
    axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=11, labelpad=30)

# 시간 표시 텍스트
time_text = fig.suptitle('Time: 0.00s', fontsize=14)

def normalize(values):
    min_val = values.min()
    max_val = values.max()
    if max_val > min_val:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)

def update(frame_idx):
    start_frame = frame_idx * chunk_frames
    end_frame = min(start_frame + chunk_frames, total_frames)

    if start_frame >= total_frames:
        return

    # 현재 청크의 STFT 데이터
    X_chunk = X_full[:, :, start_frame:end_frame]

    # 고주파/저주파 필터링
    X_high = X_chunk.copy()
    X_high[:, low_freq_mask, :] = 0
    X_low = X_chunk.copy()
    X_low[:, high_freq_mask, :] = 0

    # dB 임계값 이상의 소리만 필터링
    X_loud = X_chunk.copy()
    # 각 프레임의 에너지 계산 (모든 채널, 모든 주파수 bin의 평균 파워)
    frame_power = np.mean(np.abs(X_chunk)**2, axis=(0, 1))  # shape: (frames,)
    frame_db = 10 * np.log10(frame_power + 1e-10)  # dB 변환
    # 임계값 이하 프레임은 0으로
    quiet_frames = frame_db < db_threshold
    X_loud[:, :, quiet_frames] = 0

    X_variants = [X_chunk, X_high, X_low, X_loud]
    freq_ranges = [freq_range, [high_freq_cutoff, 7000], [300, low_freq_cutoff], freq_range]

    for row_idx in range(4):
        for col_idx, algo_name in enumerate(algo_names):
            doa = doa_algos[algo_name]
            doa.locate_sources(X_variants[row_idx], freq_range=freq_ranges[row_idx])
            resp = normalize(doa.grid.values)
            c_dirty_img = np.r_[resp, resp[0]]
            lines[(row_idx, col_idx)].set_ydata(base + height * c_dirty_img)

    # 시간 업데이트 (현재 dB 표시)
    current_time = start_frame * (nfft // 2) / fs
    current_db = np.max(frame_db) if len(frame_db) > 0 else -100
    time_text.set_text(f'Time: {current_time:.2f}s | Max dB: {current_db:.1f}')

# 애니메이션 실행
num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
ani = FuncAnimation(fig, update, frames=num_chunks, interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()