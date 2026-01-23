import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

def visualize_and_save(original_path, enhanced_path, output_png):
    """두 개의 오디오 파일을 읽어 파형과 스펙트로그램을 비교 저장합니다."""
    
    # 데이터 로드
    y1, sr1 = librosa.load(original_path)
    y2, sr2 = librosa.load(enhanced_path)
    
    plt.figure(figsize=(16, 10))

    # --- 1. 원본 (Noisy) 시각화 ---
    # 파형
    plt.subplot(2, 2, 1)
    librosa.display.waveshow(y1, sr=sr1, color='gray', alpha=0.8)
    plt.title('Original (Noisy) Waveform')
    
    # 스펙트로그램
    plt.subplot(2, 2, 3)
    D1 = librosa.stft(y1)
    S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    librosa.display.specshow(S_db1, sr=sr1, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('Original Spectrogram')

    # --- 2. 보정본 (Enhanced) 시각화 ---
    # 파형
    plt.subplot(2, 2, 2)
    librosa.display.waveshow(y2, sr=sr2, color='#800080') # 보라색
    plt.title('Enhanced (DeepFilterNet2) Waveform')
    
    # 스펙트로그램
    plt.subplot(2, 2, 4)
    D2 = librosa.stft(y2)
    S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
    librosa.display.specshow(S_db2, sr=sr2, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('Enhanced Spectrogram')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📊 시각화 이미지가 '{output_png}'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    # 1. DeepFilterNet 모델 초기화
    model, df_state, _ = init_df()
    
    # 2. 오디오 파일 준비 (사용자의 파일 경로로 수정 가능)
    # 예시를 위해 원격 파일을 다운로드하지만, 본인의 파일이 있다면 경로를 직접 입력하세요.
    audio_path = "before_deepfilter.wav"
    
    # 3. 오디오 로드 및 강화(Denoise) 처리
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    
    # 4. 결과 파일 저장
    enhanced_wav_path = "enhanced_result.wav"
    save_audio(enhanced_wav_path, enhanced, df_state.sr())
    print(f"✅ 강화된 오디오가 '{enhanced_wav_path}'로 저장되었습니다.")
    
    # 5. 시각화 및 PNG 저장 실행
    visualize_and_save(audio_path, enhanced_wav_path, "comparison_result.png")