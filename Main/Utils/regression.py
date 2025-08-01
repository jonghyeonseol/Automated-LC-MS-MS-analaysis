"""
회귀분석 유틸리티 모듈
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Optional, Dict


class RegressionAnalyzer:
    """선형 회귀분석 수행 및 이상치 탐지 클래스"""

    def __init__(self, r_squared_threshold: float = 0.99):
        self.r_squared_threshold = r_squared_threshold
        self.regression_results: List[Dict] = []

    def perform_regression(
        self,
        group_data: pd.DataFrame,
        min_points: int = 2
    ) -> Optional[Dict]:
        """
        주어진 데이터프레임에 대해 선형 회귀분석 수행.
        Returns regression stats dict or None if 분석 조건 미충족.
        """
        # NaN 제거 및 데이터 포인트 수 확인
        df = group_data.dropna(subset=["Log P", "RT"])
        if len(df) < min_points:
            return None

        X = df["Log P"].values
        y = df["RT"].values

        # X가 모두 동일하면 분석 불가
        if np.std(X) < 1e-10:
            return None

        # 절편(intercept) 추가
        X_const = sm.add_constant(X)

        try:
            model = OLS(y, X_const).fit()
            return self._extract_regression_results(model)
        except Exception as e:
            print(f"[RegressionAnalyzer] Regression error: {e}")
            return None
    
        def _extract_regression_results(self, model) -> Dict:
            """Extract and format regression results from the fitted model."""
            r2 = 0.0 if np.isnan(model.rsquared) else float(model.rsquared)
            slope = 0.0 if len(model.params) <= 1 or np.isnan(model.params[1]) else float(model.params[1])
            intercept = 0.0 if np.isnan(model.params[0]) else float(model.params[0])
            p_value = 1.0 if len(model.pvalues) <= 1 or np.isnan(model.pvalues[1]) else float(model.pvalues[1])
            residuals = model.resid
            std_resid = model.get_influence().resid_studentized_internal
    
            return {
                "model": model,
                "r_squared": r2,
                "slope": slope,
                "intercept": intercept,
                "p_value": p_value,
                "residuals": residuals,
                "std_residuals": std_resid
            }

    def detect_outliers(
        self,
        regression_result: Dict,
        threshold: float = 3.0
    ) -> List[int]:
        """
        표준화 잔차(std_residuals) 기반 이상치 인덱스 반환
        """
        std_resid = regression_result.get("std_residuals")
        if std_resid is None or len(std_resid) == 0:
            return []
        return [i for i, r in enumerate(std_resid) if abs(r) > threshold]

    def add_result(
        self,
        prefix: str,
        suffix: str,
        group_type: str,
        group_name: str,
        stats: Dict
    ) -> None:
        """
        회귀분석 결과를 regression_results에 추가
        """
        entry = {
            "prefix": prefix,
            "suffix": suffix,
            "group_type": group_type,
            "group_name": group_name,
            **stats
        }
        self.regression_results.append(entry)

    def get_summary(self) -> Dict[str, int]:
        """
        저장된 회귀분석 결과의 통계 정보를 반환
        """
        total = len(self.regression_results)
        high_r2 = len([
            r for r in self.regression_results
            if r.get("r_squared", 0.0) >= self.r_squared_threshold
        ])
        return {
            "total_groups": total,
            "high_r_squared": high_r2,
            "priority_series": len([r for r in self.regression_results if r.get("group_type") == "priority_series"]),
            "a_variation": len([r for r in self.regression_results if r.get("group_type") == "a_variation"]),
            "b_variation": len([r for r in self.regression_results if r.get("group_type") == "b_variation"]),
            "ab_variation": len([r for r in self.regression_results if r.get("group_type") == "ab_variation"])
        }

def improved_peak_detection(spectrum, height_threshold=0.05, prominence=0.02, width_range=(2, 50)):
    """
    개선된 피크 검출 알고리즘
    """
    # 노이즈 제거를 위한 스무딩
    smoothed = signal.savgol_filter(spectrum, window_length=5, polyorder=2)
    
    # 적응적 베이스라인 보정
    baseline = signal.medfilt(smoothed, kernel_size=51)
    corrected_spectrum = smoothed - baseline
    
    # 피크 검출 with 개선된 파라미터
    peaks, properties = signal.find_peaks(
        corrected_spectrum,
        height=height_threshold * np.max(corrected_spectrum),
        prominence=prominence * np.max(corrected_spectrum),
        width=width_range,
        distance=5  # 최소 피크 간격
    )
    
    return peaks, properties, corrected_spectrum

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

def calculate_improved_partition_coefficient(mobile_peak_area, stationary_peak_area, 
                                           phase_ratio, temperature=298.15):
    """
    개선된 분할 계수 계산 (온도 보정 포함)
    """
    # 기본 분할 계수
    K = (stationary_peak_area / mobile_peak_area) * phase_ratio
    
    # 온도 보정 팩터 (van't Hoff equation 기반)
    temp_correction = np.exp((298.15 - temperature) / (298.15 * temperature) * 1000)
    
    # 활동도 계수 보정 (농도 의존성) - 필요시 농도에 따른 보정
    return K * temp_correction * 1.0

class ImprovedRegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    
    def prepare_features(self, molecular_descriptors, experimental_conditions):
        """
        특성 준비 및 전처리 개선
        """
        # 분자 서술자 정규화
        normalized_descriptors = self.scaler.fit_transform(molecular_descriptors)
        
        # 상호작용 특성 추가
        interaction_features = self._create_interaction_features(
            normalized_descriptors, experimental_conditions
        )
        
        return np.hstack([
            normalized_descriptors, 
            experimental_conditions,
            interaction_features
        ])
    
    def _create_interaction_features(self, descriptors, conditions):
        """
        상호작용 특성 생성
        """
        # 온도와 분자 크기의 상호작용
        temp_size_interaction = descriptors[:, 0] * conditions[:, 0]  # 예시
        
        # 극성과 pH의 상호작용
        polarity_ph_interaction = descriptors[:, 1] * conditions[:, 1]  # 예시
        
        return np.column_stack([temp_size_interaction, polarity_ph_interaction])
    
    def fit_with_validation(self, X, y):
        """
        교차 검증을 포함한 모델 훈련
        """
        # 교차 검증 점수
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        # 전체 데이터로 최종 훈련
        self.model.fit(X, y)
        
        return cv_scores

def improved_data_preprocessing(raw_spectra, reference_standards=None):
    """
    개선된 데이터 전처리 파이프라인
    """
    processed_spectra = []
    
    for spectrum in raw_spectra:
        # 1. 노이즈 제거
        denoised = signal.wiener(spectrum)
        
        # 2. 베이스라인 보정
        baseline_corrected = baseline_correction_als(denoised)
        
        # 3. 정규화
        if reference_standards is not None:
            normalized = normalize_with_internal_standard(
                baseline_corrected, reference_standards
            )
        else:
            normalized = baseline_corrected / np.max(baseline_corrected)
        
        # 4. 드리프트 보정
        drift_corrected = correct_instrumental_drift(normalized)
        
        processed_spectra.append(drift_corrected)
    
    return np.array(processed_spectra)

def normalize_with_internal_standard(spectrum, reference_standards):
    """
    내부 표준물질을 사용한 스펙트럼 정규화
    """
    if reference_standards is None or len(reference_standards) == 0:
        return spectrum / np.max(spectrum)
    
    # 참조 표준의 평균 피크 강도 계산
    ref_intensity = np.mean([np.max(ref) for ref in reference_standards])
    
    # 현재 스펙트럼의 최대 강도로 정규화
    spectrum_max = np.max(spectrum)
    
    return spectrum * (ref_intensity / spectrum_max) if spectrum_max > 0 else spectrum

def correct_instrumental_drift(spectrum, drift_factor=0.001):
    """
    기기 드리프트 보정
    """
    # 선형 드리프트 보정 (시간에 따른 신호 감소 보정)
    time_points = np.arange(len(spectrum))
    drift_correction = 1 + drift_factor * (time_points / len(spectrum))
    
    return spectrum * drift_correction

def baseline_correction_als(spectrum, lam=1e4, p=0.01, niter=10):
    """
    Asymmetric Least Squares 베이스라인 보정
    """
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * spectrum)
        w = p * (spectrum > z) + (1-p) * (spectrum < z)
    
    return spectrum - z

def detect_outliers_iqr(data):
    """
    IQR 방법을 사용한 이상치 검출
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return [i for i, value in enumerate(data) 
            if value < lower_bound or value > upper_bound]

def calculate_confidence_intervals(predicted_values, confidence_threshold=0.95):
    """
    신뢰 구간 계산
    """
    from scipy import stats
    
    # 표준 오차 추정
    std_error = np.std(predicted_values) / np.sqrt(len(predicted_values))
    
    # t-분포를 사용한 신뢰 구간 계산
    t_value = stats.t.ppf((1 + confidence_threshold) / 2, len(predicted_values) - 1)
    margin_error = t_value * std_error
    
    lower_bounds = predicted_values - margin_error
    upper_bounds = predicted_values + margin_error
    
    return list(zip(lower_bounds, upper_bounds))

def validate_results(predicted_values, experimental_values, confidence_threshold=0.95):
    """
    결과 검증 및 품질 관리
    """
    # 통계적 메트릭 계산
    r2 = r2_score(experimental_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(experimental_values, predicted_values))
    mae = mean_absolute_error(experimental_values, predicted_values)
    
    # 이상치 검출
    residuals = experimental_values - predicted_values
    outliers = detect_outliers_iqr(residuals)
    
    # 신뢰 구간 계산
    confidence_intervals = calculate_confidence_intervals(
        predicted_values, confidence_threshold
    )
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'outliers': outliers,
        'confidence_intervals': confidence_intervals,
        'pass_threshold': r2 > 0.90 and len(outliers) < 0.05 * len(predicted_values)
    }

class GangliodisideAnalysisPipeline:
    def __init__(self):
        self.preprocessor = improved_data_preprocessing
        self.peak_detector = improved_peak_detection
        self.peak_fitter = fit_overlapping_peaks
        self.regression_model = ImprovedRegressionModel()
    
    def analyze(self, raw_data, experimental_conditions):
        """
        전체 분석 파이프라인 실행
        """
        # 1. 데이터 전처리
        processed_data = self.preprocessor(raw_data)
        
        # 2. 피크 검출 및 정량
        peak_areas = []
        for spectrum in processed_data:
            peaks, properties, corrected = self.peak_detector(spectrum)
            peak_params, _ = self.peak_fitter(
                np.arange(len(spectrum)), spectrum, peaks
            )
            peak_areas.append(self._calculate_areas(peak_params))
        
        # 3. 분할 계수 계산
        partition_coeffs = [
            calculate_improved_partition_coefficient(
                area[0], area[1], experimental_conditions[i, 2]
            ) for i, area in enumerate(peak_areas)
        ]
        
        # 4. 회귀 모델 적용
        features = self.regression_model.prepare_features(
            self._extract_molecular_descriptors(), experimental_conditions
        )
        
        predictions = self.regression_model.model.predict(features)
        
        # 5. 결과 검증
        validation = validate_results(predictions, partition_coeffs)
        
        return {
            'predictions': predictions,
            'experimental': partition_coeffs,
            'validation': validation,
            'peak_data': peak_areas
        }