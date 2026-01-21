import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 음성 파일 로드
file_path = 'before_deepfilter.wav'  # 여기에 변환할 wav 파일 경로를 넣으세요
y, sr = librosa.load(file_path)

# 2. 전체 그래프 크기 설정
plt.figure(figsize=(14, 7))

# --- 상단: 파형 (Waveform) ---
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, color='#800080') # 짙은 보라색
plt.title('Waveform Analysis')
plt.ylabel('Amplitude')

# --- 하단: 스펙트로그램 (Spectrogram) ---
plt.subplot(2, 1, 2)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 이미지와 가장 유사한 'magma' 색상 테마 사용
img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.title('Spectrogram (Frequency Analysis)')
plt.ylabel('Hz')

# 3. 레이아웃 조정
plt.tight_layout()

# 4. 파일 저장 (중요!)
# dpi=300은 고해상도 설정을 의미합니다.
output_filename = "audio_visualization.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')



plt.show()