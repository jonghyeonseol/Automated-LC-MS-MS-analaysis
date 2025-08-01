def calculate_improved_partition_coefficient(mobile_peak_area, stationary_peak_area, 
                                           phase_ratio, temperature=298.15):
    """
    개선된 분할 계수 계산 (온도 보정 포함)
    """
    # 기본 분할 계수
    K = (stationary_peak_area / mobile_peak_area) * phase_ratio
    
    # 온도 보정 팩터 (van't Hoff equation 기반)
    temp_correction = np.exp((298.15 - temperature) / (298.15 * temperature) * 1000)
    
    # 활동도 계수 보정 (농도 의존성)
    activity_correction = 1.0  # 필요시 농도에 따른 보정
    
    corrected_K = K * temp_correction * activity_correction
    
    return corrected_K