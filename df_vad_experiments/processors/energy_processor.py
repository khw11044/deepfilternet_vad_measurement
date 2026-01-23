import numpy as np
from rclpy.node import Node
from std_msgs.msg import String

class EnergyProcessor:
    def __init__(self, node: Node, is_monitoring=True):
        """
        오디오 에너지 프로세서 클래스 - dB 기반

        Args:
            node (Node): ROS2 노드
            is_monitoring (bool): 모니터링 모드 여부
        """
        self.node = node
        self.is_monitoring = is_monitoring
        self.latest_db = -100.0  # 최근 dB 값 저장
        
        if is_monitoring:
            self.energy_publisher = node.create_publisher(String, 'audio_energy', 10)

        self.node.get_logger().info('Energy monitor initialized (dB-based)')

    # ===============================================================================================================================================
    
    def calculate_energy(self, frame):
        """
        프레임의 데시벨 값을 계산

        Args:
            frame (numpy.ndarray): 오디오 프레임 (int16 또는 float32)

        Returns:
            float: 데시벨 값 (dB)
        """
        try:
            # 입력 데이터 타입에 따라 처리
            if frame.dtype == np.int16:
                # int16을 float32로 변환 (-1.0 ~ 1.0 범위)
                float_frame = frame.astype(np.float32) / 32768.0
            else:
                # 이미 float32인 경우 그대로 사용 (이미 -1.0 ~ 1.0 범위)
                float_frame = frame.astype(np.float32)
            
            # RMS 계산
            rms = np.sqrt(np.mean(np.square(float_frame)))
            
            # 데시벨 계산 (기준: 1.0 = 0dB)
            # -inf를 방지하기 위해 작은 값 추가
            db = 20 * np.log10(rms + 1e-10)

            # 모니터링 모드에서 dB 값 발행
            if self.is_monitoring:
                self.publish_energy(f"{db:.1f}")
            
            # 최근 dB 값 업데이트
            self.latest_db = db
            
            # 기본 레벨 값 반환
            return db

        except Exception as e:
            self.node.get_logger().error(f'Error calculating energy: {str(e)}')
            return -100.0  # 매우 낮은 dB값을 반환

    def publish_energy(self, energy_str):
        """dB 레벨 발행"""
        msg = String()
        msg.data = energy_str
        self.energy_publisher.publish(msg)

    def get_latest_db(self):
        """최근 계산된 dB 값 반환"""
        return self.latest_db

    def start_monitoring(self):
        """모니터링 시작"""
        self.is_monitoring = True

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
