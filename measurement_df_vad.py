import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import sys
sys.path.insert(0, '/home/khw/workspace/Audio')
from processors.vad_engines.silero_vad import SileroVAD


def compute_vad_scores(audio, sr, vad_model):
    """
    오디오에서 프레임별 VAD 스코어를 계산합니다.
    
    Args:
        audio: 오디오 데이터 (numpy array)
        sr: 샘플레이트
        vad_model: SileroVAD 인스턴스
        
    Returns:
        times: 시간축 배열
        scores: VAD 스코어 배열 (0~1)
    """
    # VAD 모델 상태 초기화
    vad_model.frame_buffer = np.array([], dtype=np.float32)
    vad_model.score_history = []
    if hasattr(vad_model.model, 'reset_states'):
        vad_model.model.reset_states()
    
    # 프레임 크기 계산 (SileroVAD는 input_samples_needed 사용)
    frame_size = vad_model.input_samples_needed
    hop_size = frame_size  # non-overlapping frames
    
    scores = []
    times = []
    
    # 프레임 단위로 처리
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        result = vad_model.process_frame(frame)
        
        if result is not None:
            is_speech, smoothed_prob, raw_prob = result
            scores.append(smoothed_prob)
            # 시간 계산 (프레임 중심 기준)
            time_sec = (i + frame_size / 2) / sr
            times.append(time_sec)
    
    return np.array(times), np.array(scores)


def visualize_and_save(original_path, enhanced_path, output_png, vad_model=None):
    """두 개의 오디오 파일을 읽어 파형, 스펙트로그램, VAD 스코어를 비교 저장합니다."""
    
    # 데이터 로드 (VAD용 48kHz와 시각화용 기본 sr)
    y1, sr1 = librosa.load(original_path)
    y2, sr2 = librosa.load(enhanced_path)
    
    # VAD 스코어 계산 (Enhanced만)
    vad_times2, vad_scores2 = None, None
    
    if vad_model is not None:
        # VAD는 48kHz 입력 필요
        y2_48k, _ = librosa.load(enhanced_path, sr=48000)
        
        print("🎤 Enhanced 오디오 VAD 스코어 계산 중...")
        vad_times2, vad_scores2 = compute_vad_scores(y2_48k, 48000, vad_model)
    
    # 레이아웃 설정 (VAD가 있으면 3x2, 없으면 2x2)
    rows = 3 if vad_model is not None else 2
    plt.figure(figsize=(16, 5 * rows))

    # --- 1. 원본 (Noisy) 시각화 ---
    # 파형
    plt.subplot(rows, 2, 1)
    librosa.display.waveshow(y1, sr=sr1, color='gray', alpha=0.8)
    plt.title('Original (Noisy) Waveform')
    
    # 스펙트로그램
    plt.subplot(rows, 2, 3)
    D1 = librosa.stft(y1)
    S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    librosa.display.specshow(S_db1, sr=sr1, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('Original Spectrogram')

    # --- 2. 보정본 (Enhanced) 시각화 ---
    # 파형
    plt.subplot(rows, 2, 2)
    librosa.display.waveshow(y2, sr=sr2, color='#800080') # 보라색
    plt.title('Enhanced (DeepFilterNet2) Waveform')
    
    # 스펙트로그램
    plt.subplot(rows, 2, 4)
    D2 = librosa.stft(y2)
    S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
    librosa.display.specshow(S_db2, sr=sr2, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('Enhanced Spectrogram')

    # --- 3. VAD 스코어 시각화 (Enhanced만, 3행 2열) ---
    if vad_model is not None and vad_times2 is not None:
        # 3행 1열은 비워둠 (subplot 5 스킵)
        
        # Enhanced VAD (3행 2열, 위치 6)
        ax6 = plt.subplot(rows, 2, 6)
        ax6.fill_between(vad_times2, 0, vad_scores2, alpha=0.5, color='purple', label='Speech Probability')
        ax6.plot(vad_times2, vad_scores2, color='#800080', linewidth=0.8)
        ax6.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
        ax6.set_xlim(0, len(y2) / sr2)  # Enhanced 오디오와 동일한 time 스케일
        ax6.set_ylim(0, 1)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Speech Probability')
        ax6.set_title('Enhanced VAD Score (Silero VAD)')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)

    # 레이아웃 간격 조정 (제목과 축 라벨 겹침 방지)
    plt.tight_layout(pad=3.0, h_pad=3.0)
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
    
    # 5. Silero VAD 모델 초기화
    print("🎤 Silero VAD 모델 로딩 중...")
    vad_model = SileroVAD(
        sample_rate=48000,
        frame_duration=30,
        buffer_size=2,
        threshold=0.5
    )
    
    # 6. 시각화 및 PNG 저장 실행 (VAD 포함)
    visualize_and_save(audio_path, enhanced_wav_path, "comparison_result.png", vad_model)
