from .webrtc_vad import WebRtcVAD
from .silero_vad import SileroVAD


class VADFactory:
    @staticmethod
    def create_vad(node):
        """
        VAD 인스턴스 생성

        node.config에서 'vad' 섹션을 읽어 VAD 타입에 맞는 인스턴스를 생성합니다.
        config에 없는 인자는 각 VAD 클래스의 기본값이 사용됩니다.

        Args:
            node (Node): ROS2 노드 (node.config에서 설정을 읽음)

        Returns:
            BaseVAD: VAD 인스턴스
        """
        config = node.config
        vad_config = config.get('vad', {}).copy()
        vad_type = vad_config.pop('type', 'silero')  # type 추출 및 제거
        
        if vad_type.lower() == 'webrtc':
            return WebRtcVAD(**vad_config)
        elif vad_type.lower() == 'silero':
            return SileroVAD(**vad_config)
        else:
            raise ValueError(f'Unknown VAD type: {vad_type}')
