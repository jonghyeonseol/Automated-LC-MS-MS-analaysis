"""
질량분석 데이터 자동화 시스템 - 유틸리티 패키지
"""

# 각 모듈에서 필요한 함수들을 import
from .data_parser import (
    parse_name,
    parse_prefix_components,
    count_oac,
    validate_data,
    find_header_row
)

from .regression import (
    perform_regression,
    check_priority_series,
    apply_regression_rules,
    detect_outliers,
    safe_float_conversion
)

from .visualization import (
    create_scatter_plot,
    create_histogram,
    plot_to_base64
)

# 패키지 버전
__version__ = '1.0.0'

# 공통 상수들
REQUIRED_COLUMNS = ['Name', 'RT', 'Volume', 'Log P', 'Anchor']

PRIORITY_GROUPS = {
    'GD_series': ['GD1+dHex', 'GD1', 'GD2', 'GD3'],
    'GT3_series': ['GM3', 'GD3', 'GT3'],
    'GP1_series': ['GP1', 'GQ1', 'GT1', 'GD1a']
}

SUGAR_COUNT_MAP = {
    'A': 0, 
    'M': 1, 
    'D': 2, 
    'T': 3, 
    'Q': 4, 
    'P': 5
}

OUTLIER_REASONS = {
    'high_residual': '높은 잔차',
    'OAc_rule_violation': 'OAc 규칙 위반',
    'sugar_count_rule': '당 개수 규칙',
    'no_linear_relationship': '선형관계 없음'
}

# 공통 설정값들
R_SQUARED_THRESHOLD = 0.99
RESIDUAL_THRESHOLD = 3
RT_CLUSTER_THRESHOLD = 0.1

# 편의를 위한 __all__ 정의
__all__ = [
    # data_parser
    'parse_name',
    'parse_prefix_components',
    'count_oac',
    'validate_data',
    'find_header_row',
    
    # regression
    'perform_regression',
    'check_priority_series',
    'apply_regression_rules',
    'detect_outliers',
    'safe_float_conversion',
    
    # visualization
    'create_scatter_plot',
    'create_histogram',
    'plot_to_base64',
    
    # constants
    'REQUIRED_COLUMNS',
    'PRIORITY_GROUPS',
    'SUGAR_COUNT_MAP',
    'OUTLIER_REASONS',
    'R_SQUARED_THRESHOLD',
    'RESIDUAL_THRESHOLD',
    'RT_CLUSTER_THRESHOLD'
]

# NaN 및 Infinity 안전 변환 함수 (공통 사용)
def safe_json_value(value):
    """JSON 직렬화가 가능한 값으로 변환"""
    import math
    
    if isinstance(value, float):
        if math.isnan(value):
            return None
        elif math.isinf(value):
            return None
        else:
            return float(value)
    return value

# 딕셔너리 내의 모든 float 값을 안전하게 변환
def sanitize_dict_for_json(data):
    """딕셔너리 내의 NaN/Infinity를 None으로 변환"""
    if isinstance(data, dict):
        return {k: sanitize_dict_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict_for_json(item) for item in data]
    elif isinstance(data, float):
        return safe_json_value(data)
    else:
        return data

# 로깅 설정 (필요시 사용)
import logging

def setup_logger(name, level=logging.INFO):
    """로거 설정 헬퍼 함수"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger