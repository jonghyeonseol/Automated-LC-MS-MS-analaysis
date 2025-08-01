import numpy as np
from scipy.optimize import curve_fit

class Config:
    DEBUG = True
    PORT = 5000
    UPLOAD_FOLDER = "uploads"
    RESULTS_FOLDER = "results"

def multi_gaussian_model(x, *params):
    """
    다중 가우시안 모델 (오버랩 피크 처리 개선)
    """
    n_peaks = len(params) // 3
    result = np.zeros_like(x)
    
    for i in range(n_peaks):
        amplitude = params[3*i]
        center = params[3*i + 1]
        sigma = params[3*i + 2]
        result += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    return result

def fit_overlapping_peaks(x_data, y_data, initial_peaks):
    """
    오버랩된 피크들에 대한 개선된 피팅
    """
    # 초기 파라미터 추정 개선
    initial_params = []
    bounds_lower = []
    bounds_upper = []
    
    for peak_idx in initial_peaks:
        # amplitude 추정
        amplitude = y_data[peak_idx]
        initial_params.extend([amplitude, x_data[peak_idx], 2.0])
        
        # 바운드 설정
        bounds_lower.extend([0.1 * amplitude, x_data[peak_idx] - 5, 0.5])
        bounds_upper.extend([2.0 * amplitude, x_data[peak_idx] + 5, 10.0])
    
    try:
        popt, pcov = curve_fit(
            multi_gaussian_model, 
            x_data, 
            y_data, 
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        return popt, pcov
    except:
        return None, None