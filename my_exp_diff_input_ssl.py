import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]



# Location of sources
azimuth = np.array([61, 180]) / 180. * np.pi
distance = 2.  # meters


fs1, signal1 = wavfile.read("/home/khw/workspace/Audio/모닥불.wav")
fs2, signal2 = wavfile.read("/home/khw/workspace/Audio/천둥소리2.wav")

signals = [signal1, signal2]


snr_db = 5.    # signal-to-noise ratio
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

# 방 생성
room_dim = np.r_[10.,10.]
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)


# 방 정 중앙에 위치
echo = pra.circular_2D_array(center=room_dim/2, M=2, phi0=0, radius=37.5e-3)
# 2채널 마이크의 위치 추가
echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))




# Add sources of 1 second duration
rng = np.random.RandomState(23)
duration_samples = int(fs)

for i, ang in enumerate(azimuth):
    source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
    aroom.add_source(source_location, signal=signals[i])
    
# Run the simulation
aroom.simulate()

X = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
X = X.transpose([2, 1, 0])

# 주파수 bin 계산
freq_bins = np.fft.rfftfreq(nfft, 1/fs)

# 고주파 필터링된 X 생성 (3kHz 이상만)
high_freq_cutoff = 3000
low_freq_cutoff = 1000
X_high = X.copy()
low_freq_mask = freq_bins < high_freq_cutoff
X_high[:, low_freq_mask, :] = 0

# 저주파 필터링된 X 생성 (1kHz 이하만)
X_low = X.copy()
high_freq_mask = freq_bins > low_freq_cutoff
X_low[:, high_freq_mask, :] = 0

algo_names = ['SRP', 'MUSIC', 'TOPS']
spatial_resp = dict()       # 전체 대역
spatial_resp_high = dict()  # 고주파 대역
spatial_resp_low = dict()   # 저주파 대역

# loop through algos
for algo_name in algo_names:
    doa = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=2, max_four=4)

    # 전체 대역 DOA
    doa.locate_sources(X, freq_range=freq_range)
    spatial_resp[algo_name] = doa.grid.values.copy()

    # 고주파 대역 DOA
    doa.locate_sources(X_high, freq_range=[high_freq_cutoff, 7000])
    spatial_resp_high[algo_name] = doa.grid.values.copy()

    # 저주파 대역 DOA
    doa.locate_sources(X_low, freq_range=[300, low_freq_cutoff])
    spatial_resp_low[algo_name] = doa.grid.values.copy()

    # normalize
    for resp in [spatial_resp, spatial_resp_high, spatial_resp_low]:
        min_val = resp[algo_name].min()
        max_val = resp[algo_name].max()
        if max_val > min_val:
            resp[algo_name] = (resp[algo_name] - min_val) / (max_val - min_val)
        else:
            resp[algo_name] = np.zeros_like(resp[algo_name])
    
    
# plotting param
base = 1.
height = 10.
true_col = [0, 0, 0]

# 3x3 subplot: 1행=전체대역, 2행=고주파대역, 3행=저주파대역
fig, axes = plt.subplots(3, 3, figsize=(12, 10), subplot_kw={'projection': 'polar'})
phi_plt = doa.grid.azimuth

row_configs = [
    (spatial_resp, f'Full Band ({freq_range[0]}-{freq_range[1]}Hz)'),
    (spatial_resp_high, f'High Freq (>{high_freq_cutoff}Hz)'),
    (spatial_resp_low, f'Low Freq (<{low_freq_cutoff}Hz)'),
]

for row_idx, (resp_dict, row_label) in enumerate(row_configs):
    for col_idx, algo_name in enumerate(algo_names):
        ax = axes[row_idx, col_idx]
        c_phi_plt = np.r_[phi_plt, phi_plt[0]]
        c_dirty_img = np.r_[resp_dict[algo_name], resp_dict[algo_name][0]]
        ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=3,
                alpha=0.55, linestyle='-',
                label="spatial spectrum")

        if row_idx == 0:
            ax.set_title(algo_name, fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=12, labelpad=30)

        # plot true loc
        for angle in azimuth:
            ax.plot([angle, angle], [base, base + height], linewidth=3, linestyle='--',
                color=true_col, alpha=0.6)

        K = len(azimuth)
        ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
                   (K, 1)), s=500, alpha=0.75, marker='*',
                   linewidths=0,
                   label='true locations')

        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 * (base + height)])

# add legend to last subplot only
handles, labels = axes[-1, -1].get_legend_handles_labels()
axes[-1, -1].legend(handles, labels, framealpha=0.5,
                    scatterpoints=1, loc='center right', fontsize=12,
                    ncol=1, bbox_to_anchor=(1.4, 0.5),
                    handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

plt.tight_layout()
plt.show()