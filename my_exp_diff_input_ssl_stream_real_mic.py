import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyroomacoustics as pra
import sounddevice as sd
from collections import deque
import threading

c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]
chunk_frames = 20  # 청크당 STFT 프레임 수
hop_size = nfft // 2

# dB 임계값 설정 (이 값 이상의 소리만 감지)
db_threshold = -10  # dB

# 마이크 설정
n_channels = 2  # 2채널 마이크
mic_distance = 0.075  # 마이크 간 거리 (75mm, 노트북 기준 조정 필요)

# 버퍼 설정
buffer_size = nfft * chunk_frames  # 한 번에 처리할 샘플 수
audio_buffer = deque(maxlen=buffer_size * 2)
buffer_lock = threading.Lock()

# 2채널 마이크 배열 설정 (선형 배열)
mic_array = np.array([
    [-mic_distance/2, 0],  # 왼쪽 마이크
    [mic_distance/2, 0],   # 오른쪽 마이크
]).T  # shape: (2, 2) - [x,y] x n_mics

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
    doa_algos[algo_name] = pra.doa.algorithms[algo_name](mic_array, fs, nfft, c=c, num_src=2, max_four=4)

# 오디오 콜백 함수
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    with buffer_lock:
        audio_buffer.extend(indata.copy())

# 오디오 스트림 시작
stream = sd.InputStream(
    samplerate=fs,
    channels=n_channels,
    dtype='float32',
    blocksize=hop_size,
    callback=audio_callback
)
stream.start()
print(f"마이크 스트림 시작 (fs={fs}Hz, {n_channels}채널)")

# plotting param
base = 1.
height = 10.

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
    # 버퍼에서 데이터 가져오기
    with buffer_lock:
        if len(audio_buffer) < buffer_size:
            return list(lines.values())
        # 버퍼에서 필요한 만큼 데이터 추출
        data = np.array(list(audio_buffer)[-buffer_size:])  # shape: (buffer_size, n_channels)

    # STFT 계산
    X = pra.transform.stft.analysis(data, nfft, hop_size)
    X_chunk = X.transpose([2, 1, 0])  # [channels, freq_bins, frames]

    if X_chunk.shape[2] < 2:
        return list(lines.values())

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
            try:
                doa.locate_sources(X_variants[row_idx], freq_range=freq_ranges[row_idx])
                resp = normalize(doa.grid.values)
                c_dirty_img = np.r_[resp, resp[0]]
                lines[(row_idx, col_idx)].set_ydata(base + height * c_dirty_img)
            except Exception as e:
                pass  # 데이터 부족 시 무시

    # 시간 업데이트 (현재 dB 표시)
    current_db = np.max(frame_db) if len(frame_db) > 0 else -100
    time_text.set_text(f'Real-time | Max dB: {current_db:.1f}')

    return list(lines.values())

# 애니메이션 실행 (실시간이므로 무한 반복)
ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

plt.tight_layout()

# 종료 시 스트림 정리
def on_close(event):
    stream.stop()
    stream.close()
    print("마이크 스트림 종료")

fig.canvas.mpl_connect('close_event', on_close)

plt.show()