import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import base64
import warnings

warnings.filterwarnings("ignore")

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class MassSpecAnalyzer:
    """질량분석 데이터 자동 분석 클래스"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = []
        self.regression_results = []
        self.groups = {}
        self.priority_groups = {
            "GD_series": ["GD1+dHex", "GD1", "GD2", "GD3"],
            "GT3_series": ["GM3", "GD3", "GT3"],
            "GP1_series": ["GP1", "GQ1", "GT1", "GD1a"],
        }
        # 디버깅 정보 저장
        self.debug_info = {
            'priority_series_matches': 0,
            'rule1_groups': 0,
            'rule5_violations': 0,
            'rule6_clusters': 0,
            'isomer_pairs': 0,
            'total_true': 0,
            'total_outliers': 0
        }

    def validate_data(self):
        """데이터 유효성 검증"""
        required_columns = ["Name", "RT", "Volume", "Log P", "Anchor"]
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"필수 열이 없습니다: {', '.join(missing_cols)}")

        # 숫자형 데이터 확인 및 변환
        numeric_cols = ["RT", "Volume", "Log P"]
        for col in numeric_cols:
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            except Exception as e:
                raise ValueError(f"{col} 열에 숫자가 아닌 값이 있습니다: {str(e)}")

        # NaN 값 처리
        self.data = self.data.dropna(subset=["Name", "RT", "Log P"])
        
        # 데이터가 비어있는지 확인
        if len(self.data) == 0:
            raise ValueError("유효한 데이터가 없습니다")

        # Anchor 열 기본값 설정
        self.data["Anchor"] = self.data["Anchor"].fillna("")

        return True

    def parse_name(self, name):
        """Name을 접두사와 접미사로 분리"""
        # 괄호를 기준으로 분리
        match = re.match(r"(.+?)(\([^)]+\))$", str(name))
        if match:
            prefix = match.group(1)
            suffix = match.group(2)

            # 접미사에서 a, b, c 성분 추출
            suffix_match = re.match(r"\((\d+):(\d+);(O\d+)\)", suffix)
            if suffix_match:
                a = int(suffix_match.group(1))
                b = int(suffix_match.group(2))
                c = suffix_match.group(3)
                return prefix, suffix, a, b, c

        return str(name), "", None, None, None

    def parse_prefix_components(self, prefix):
        if not prefix or len(prefix) < 2:
            return "", "", "", 0
        
        d = prefix[0]
        e = prefix[1]
        f = prefix[2] if len(prefix) >= 3 else ""

        # 규칙3: e의 당 개수 계산
        sugar_count_e = {"A": 0, "M": 1, "D": 2, "T": 3, "Q": 4, "P": 5}.get(e, 0)

        # 규칙4: f의 당 개수 계산
        sugar_count_f = 0
        if f.isdigit() and 1 <= int(f) <= 4:
            sugar_count_f = 5 - int(f)

        total_sugar = sugar_count_e + sugar_count_f

        # 추가 당 처리
        if "+dHex" in prefix:
            total_sugar += prefix.count("+dHex")
        if "+HexNAc" in prefix:
            total_sugar += prefix.count("+HexNAc")

        return d, e, f, total_sugar

    def count_oac(self, prefix):
        """OAc 개수 계산"""
        if "2OAc" in prefix:
            return 2
        elif "OAc" in prefix:
            return 1
        return 0

    def perform_regression(self, group_data, min_points=2):
        """선형 회귀분석 수행"""
        if len(group_data) < min_points:
            return None

        # NaN 값 제거
        group_data = group_data.dropna(subset=["Log P", "RT"])
        
        if len(group_data) < min_points:
            return None

        X = group_data["Log P"].values
        y = group_data["RT"].values
        
        # 모든 X값이 동일한지 확인
        if np.std(X) < 1e-10:
            return None

        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        try:
            model = self._fit_ols_model(y, X_with_const)
            if model is None:
                return None
                
            return self._extract_regression_results(model)
        except Exception as e:
            print(f"Regression error: {e}")
            return None

    def _fit_ols_model(self, y, X):
        """Helper method to fit OLS model"""
        return OLS(y, X).fit()
        
    def _extract_regression_results(self, model):
        """Extract regression results from fitted model"""
        r_squared = model.rsquared if not np.isnan(model.rsquared) else 0.0
        slope = model.params[1] if len(model.params) > 1 and not np.isnan(model.params[1]) else 0.0
        intercept = model.params[0] if not np.isnan(model.params[0]) else 0.0
        p_value = 1.0 if len(model.pvalues) <= 1 or np.isnan(model.pvalues[1]) else model.pvalues[1]
        
        return {
            "model": model,
            "r_squared": r_squared,
            "slope": slope,
            "intercept": intercept,
            "p_value": p_value,
            "residuals": model.resid,
            "std_residuals": model.get_influence().resid_studentized_internal,
        }

    def check_priority_series(self):
        """우선순위 그룹 간의 선형성 검증"""
        print("\n=== 우선순위 시리즈 검증 시작 ===")
        
        for series_name, prefixes in self.priority_groups.items():
            print(f"\n{series_name} 검증 중: {prefixes}")
            
            # 해당 시리즈의 데이터 수집
            series_data = []
            for prefix in prefixes:
                # 정확한 매칭을 위해 == 사용
                prefix_data = self.data[self.data["prefix"] == prefix]
                if not prefix_data.empty:
                    print(f"  - {prefix}: {len(prefix_data)}개 데이터 발견")
                    series_data.append(prefix_data)
                else:
                    print(f"  - {prefix}: 데이터 없음")

            if len(series_data) >= 2:  # 최소 2개 이상의 그룹이 있어야 함
                print(f"  {series_name}: {len(series_data)}개 그룹으로 분석 진행")
                
                # 동일 접미사별로 그룹화하여 검증
                all_suffixes = set()
                for df in series_data:
                    all_suffixes.update(df["suffix"].unique())

                for suffix in all_suffixes:
                    suffix_data = []
                    for df in series_data:
                        suffix_df = df[df["suffix"] == suffix]
                        if not suffix_df.empty:
                            suffix_data.append(suffix_df)

                    if len(suffix_data) >= 2:  # 2개로 완화
                        # 모든 데이터를 합쳐서 회귀분석
                        combined_data = pd.concat(suffix_data)
                        regression = self.perform_regression(combined_data)

                        if regression and regression["r_squared"] >= 0.99:
                            print(f"    접미사 {suffix}: R²={regression['r_squared']:.4f} - 우선순위 시리즈 확인!")
                            # 해당 시리즈의 모든 데이터를 참값으로 분류
                            for df in suffix_data:
                                self.data.loc[df.index, "classification"] = "true"
                                self.data.loc[df.index, "priority_series"] = series_name
                                self.data.loc[df.index, "regression_group"] = f"{series_name}_{suffix}"
                            
                            # 회귀분석 결과 저장
                            self.regression_results.append({
                                "prefix": series_name,
                                "suffix": suffix,
                                "group_type": "priority_series",
                                "group_name": f"{series_name}_{suffix}",
                                **regression
                            })
                            
                            self.debug_info['priority_series_matches'] += len(combined_data)
            else:
                print(f"  {series_name}: 데이터 부족 (최소 2개 그룹 필요)")

    def apply_rule1(self):
        """규칙1 적용: 동일 접두사 그룹의 회귀분석"""
        print("\n=== 규칙1: 동일 접두사 그룹 회귀분석 ===")
        
        # 접두사별 그룹화
        prefix_groups = self.data.groupby("prefix")
        rule1_count = 0

        for prefix, group in prefix_groups:
            if len(group) < 2:
                continue

            # 이미 priority_series로 분류된 데이터는 건너뛰기
            if any(group["classification"] == "true"):
                continue

            # Anchor='T'인 데이터는 항상 포함
            anchor_indices = group[group["Anchor"] == "T"].index

            # 1) a 성분만 변화하는 경우 (b, c 동일)
            for (b, c), subgroup in group.groupby(["b", "c"]):
                if pd.notna(b) and pd.notna(c) and len(subgroup) >= 2:
                    # Anchor 데이터 포함
                    test_indices = subgroup.index.tolist()
                    for idx in anchor_indices:
                        if (idx not in test_indices and 
                            idx in group.index and
                            group.loc[idx, "b"] == b and 
                            group.loc[idx, "c"] == c):
                            test_indices.append(idx)

                    test_data = self.data.loc[test_indices]
                    regression = self.perform_regression(test_data)

                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_bc_{b}_{c}"
                        self.data.loc[test_indices, "classification"] = "true"
                        self.data.loc[test_indices, "regression_group"] = group_name
                        rule1_count += len(test_indices)

                        self.regression_results.append({
                            "prefix": prefix,
                            "suffix": f"(varied:{b};{c})",
                            "group_type": "a_variation",
                            "group_name": group_name,
                            **regression,
                        })

            # 2) b 성분만 변화하는 경우 (a, c 동일)
            for (a, c), subgroup in group.groupby(["a", "c"]):
                if pd.notna(a) and pd.notna(c) and len(subgroup) >= 2:
                    test_indices = subgroup.index.tolist()
                    for idx in anchor_indices:
                        if (idx not in test_indices and 
                            idx in group.index and
                            group.loc[idx, "a"] == a and 
                            group.loc[idx, "c"] == c):
                            test_indices.append(idx)

                    test_data = self.data.loc[test_indices]
                    regression = self.perform_regression(test_data)

                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_ac_{a}_{c}"
                        self.data.loc[test_indices, "classification"] = "true"
                        self.data.loc[test_indices, "regression_group"] = group_name
                        rule1_count += len(test_indices)

                        self.regression_results.append({
                            "prefix": prefix,
                            "suffix": f"({a}:varied;{c})",
                            "group_type": "b_variation",
                            "group_name": group_name,
                            **regression,
                        })

            # 3) a, b 성분이 동시에 변화하는 경우 (c 동일, 3개 이상)
            for c, subgroup in group.groupby("c"):
                if pd.notna(c) and len(subgroup) >= 3:
                    test_indices = subgroup.index.tolist()
                    for idx in anchor_indices:
                        if idx not in test_indices and idx in group.index and group.loc[idx, "c"] == c:
                            test_indices.append(idx)

                    test_data = self.data.loc[test_indices]
                    regression = self.perform_regression(test_data, min_points=3)

                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_c_{c}"
                        self.data.loc[test_indices, "classification"] = "true"
                        self.data.loc[test_indices, "regression_group"] = group_name
                        rule1_count += len(test_indices)

                        self.regression_results.append({
                            "prefix": prefix,
                            "suffix": f"(varied:varied;{c})",
                            "group_type": "ab_variation",
                            "group_name": group_name,
                            **regression,
                        })

        print(f"규칙1 적용 완료: {rule1_count}개 물질 분류")
        self.debug_info['rule1_groups'] = len([r for r in self.regression_results if r.get('group_type') in ['a_variation', 'b_variation', 'ab_variation']])

    def apply_rule5(self):
        """규칙5: OAc 규칙 적용"""
        print("\n=== 규칙5: OAc 규칙 적용 ===")
        violations = 0
        
        # 접두사별로 그룹화
        for base_prefix in self.data["prefix"].unique():
            # OAc가 없는 기본 형태 찾기
            base_prefix_clean = (base_prefix.replace("+2OAc", "")
                               .replace("+OAc", "")
                               .replace("2OAc", "")
                               .replace("OAc", ""))

            # 같은 base를 가진 모든 변형 찾기
            related_data = self.data[self.data["prefix"].str.contains(base_prefix_clean, regex=False)]

            if len(related_data) > 1:
                # OAc 개수별로 정렬
                related_data = related_data.copy()
                related_data["oac_count"] = related_data["prefix"].apply(self.count_oac)

                # 동일 접미사별로 확인
                for suffix in related_data["suffix"].unique():
                    suffix_data = related_data[related_data["suffix"] == suffix].sort_values("oac_count")

                    if len(suffix_data) > 1:
                        # OAc 개수가 증가할수록 RT가 증가해야 함
                        for i in range(len(suffix_data) - 1):
                            if suffix_data.iloc[i]["RT"] >= suffix_data.iloc[i + 1]["RT"]:
                                # 규칙 위반 - 이상치로 분류
                                self.data.loc[suffix_data.iloc[i + 1].name, "classification"] = "outlier"
                                self.data.loc[suffix_data.iloc[i + 1].name, "outlier_reason"] = "OAc_rule_violation"
                                violations += 1
        
        print(f"규칙5 적용 완료: {violations}개 OAc 규칙 위반")
        self.debug_info['rule5_violations'] = violations

    def apply_rule6(self):
        """규칙6: 당 개수에 따른 RT 규칙"""
        print("\n=== 규칙6: 당 개수 규칙 적용 ===")
        clusters_processed = 0
        
        # 동일 접미사별로 그룹화
        for suffix in self.data["suffix"].unique():
            if not suffix:  # 빈 접미사 건너뛰기
                continue

            suffix_data = self.data[self.data["suffix"] == suffix].copy()

            if len(suffix_data) > 1:
                # RT가 ±0.1 범위 내인 그룹 찾기
                suffix_data = suffix_data.sort_values("RT")

                # RT 클러스터 찾기
                clusters = []
                current_cluster = [suffix_data.iloc[0].name]

                for i in range(1, len(suffix_data)):
                    if abs(suffix_data.iloc[i]["RT"] - suffix_data.iloc[i - 1]["RT"]) <= 0.1:
                        current_cluster.append(suffix_data.iloc[i].name)
                    else:
                        if len(current_cluster) > 1:
                            clusters.append(current_cluster)
                        current_cluster = [suffix_data.iloc[i].name]

                if len(current_cluster) > 1:
                    clusters.append(current_cluster)

                # 각 클러스터 처리
                for cluster in clusters:
                    cluster_data = self.data.loc[cluster].copy()

                    # Anchor='T'가 있으면 건너뛰기
                    if any(cluster_data["Anchor"] == "T"):
                        continue

                    # 당 개수가 가장 많은 것 찾기
                    max_sugar_idx = cluster_data["sugar_count"].idxmax()

                    # 나머지를 이상치로 분류하고 Volume 합산
                    total_volume = cluster_data["Volume"].sum()

                    for idx in cluster:
                        if idx != max_sugar_idx:
                            self.data.loc[idx, "classification"] = "outlier"
                            self.data.loc[idx, "outlier_reason"] = "sugar_count_rule"

                    # Volume 합산
                    self.data.loc[max_sugar_idx, "Volume"] = total_volume
                    clusters_processed += 1
        
        print(f"규칙6 적용 완료: {clusters_processed}개 클러스터 처리")
        self.debug_info['rule6_clusters'] = clusters_processed

    def apply_isomer_rule(self):
        """규칙7: f=1 이성질체 규칙 적용"""
        print("\n=== 규칙7: 이성질체 규칙 적용 ===")
        isomer_pairs = 0
        
        # f=1인 데이터만 추출
        f1_data = self.data[self.data["prefix_f"] == "1"].copy()
        
        if f1_data.empty:
            print("f=1인 데이터가 없습니다.")
            return
        
        # 동일한 prefix, suffix, Log P를 가진 그룹 찾기
        grouped = f1_data.groupby(["prefix", "suffix", "Log P"])
        
        for (prefix, suffix, log_p), group in grouped:
            if len(group) >= 2:
                # RT 차이 확인
                rt_values = group["RT"].values
                rt_diff = rt_values.max() - rt_values.min()
                
                if rt_diff > 0.1:
                    print(f"  이성질체 발견: {prefix}{suffix} (Log P={log_p:.2f}), RT 차이={rt_diff:.2f}")
                    
                    # RT가 더 큰 것을 참값으로, 작은 것을 이상치로 분류
                    max_rt_idx = group["RT"].idxmax()
                    
                    for idx in group.index:
                        if idx != max_rt_idx:
                            self.data.loc[idx, "classification"] = "outlier"
                            self.data.loc[idx, "outlier_reason"] = "isomer_rule"
                    
                    # 참값은 유지 (이미 다른 규칙으로 true가 되었을 수 있음)
                    if self.data.loc[max_rt_idx, "classification"] != "true":
                        self.data.loc[max_rt_idx, "classification"] = "true"
                    
                    isomer_pairs += 1
        
        print(f"규칙7 적용 완료: {isomer_pairs}개 이성질체 쌍 처리")
        self.debug_info['isomer_pairs'] = isomer_pairs

    def detect_outliers(self):
        """잔차 기반 이상치 탐지"""
        print("\n=== 잔차 기반 이상치 탐지 ===")
        outliers_detected = 0
        
        for reg_result in self.regression_results:
            if "std_residuals" in reg_result and reg_result["model"] is not None:
                # regression_group이 일치하는 데이터만 필터링
                group_name = reg_result.get("group_name", "")
                group_data = self.data[self.data["regression_group"] == group_name]
                
                if not group_data.empty and len(reg_result["std_residuals"]) == len(group_data):
                    # 표준화 잔차가 3 이상인 경우 이상치로 분류
                    outlier_mask = np.abs(reg_result["std_residuals"]) > 3
                    outlier_indices = group_data.index[outlier_mask]
                    
                    for idx in outlier_indices:
                        if self.data.loc[idx, "classification"] != "outlier":
                            self.data.loc[idx, "classification"] = "outlier"
                            self.data.loc[idx, "outlier_reason"] = "high_residual"
                            outliers_detected += 1
        
        print(f"잔차 기반 이상치 탐지 완료: {outliers_detected}개 발견")

    def apply_rules(self):
        """모든 규칙 적용"""
        # 데이터 검증
        self.validate_data()
        
        # 초기화
        self.data['classification'] = 'unclassified'
        self.data['regression_group'] = ''
        self.data['outlier_reason'] = ''
        self.data['priority_series'] = ''
        
        # Name 파싱
        parsed_data = self.data['Name'].apply(self.parse_name)
        self.data[['prefix', 'suffix', 'a', 'b', 'c']] = pd.DataFrame(
            parsed_data.tolist(), index=self.data.index
        )
        
        # 접두사 성분 파싱
        prefix_components = self.data['prefix'].apply(self.parse_prefix_components)
        self.data[['prefix_d', 'prefix_e', 'prefix_f', 'sugar_count']] = pd.DataFrame(
            prefix_components.tolist(), index=self.data.index
        )
        
        print("\n========== 질량분석 규칙 적용 시작 ==========")
        print(f"총 데이터 수: {len(self.data)}개")
        
        # 우선순위 시리즈 확인 (최우선)
        self.check_priority_series()
        
        # 규칙1 적용
        self.apply_rule1()
        
        # 규칙5 적용 (OAc 규칙)
        self.apply_rule5()
        
        # 규칙6 적용 (당 개수 규칙)
        self.apply_rule6()
        
        # 규칙7 적용 (이성질체 규칙)
        self.apply_isomer_rule()
        
        # 잔차 기반 이상치 탐지
        self.detect_outliers()
        
        # 분류되지 않은 데이터 처리
        unclassified_mask = self.data['classification'] == 'unclassified'
        self.data.loc[unclassified_mask, 'classification'] = 'outlier'
        self.data.loc[unclassified_mask, 'outlier_reason'] = 'no_linear_relationship'
        
        # 최종 통계
        self.debug_info['total_true'] = len(self.data[self.data['classification'] == 'true'])
        self.debug_info['total_outliers'] = len(self.data[self.data['classification'] == 'outlier'])
        
        print("\n========== 규칙 적용 완료 ==========")
        print(f"참값: {self.debug_info['total_true']}개")
        print(f"이상치: {self.debug_info['total_outliers']}개")
        print(f"회귀분석 그룹: {len(self.regression_results)}개")

    def _add_bar_labels(self, ax, bars):
        """Helper method to add value labels on top of bars"""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

    def _create_bar_plot(self, ax, x, heights, **kwargs):
        """Helper method to create a bar plot with value labels"""
        bars = ax.bar(x, heights, **kwargs)
        self._add_bar_labels(ax, bars)
        return bars

    def create_scatter_plot(self):
        """산점도 생성"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 접두사별 산점도
        prefixes = sorted(self.data["prefix"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(prefixes), 20)))
        prefix_color_map = dict(zip(prefixes[:20], colors))

        for i, prefix in enumerate(prefixes[:20]):  # 최대 20개 접두사만 표시
            prefix_data = self.data[self.data["prefix"] == prefix]

            # 참값
            true_data = prefix_data[prefix_data["classification"] == "true"]
            if not true_data.empty:
                ax1.scatter(
                    true_data["Log P"],
                    true_data["RT"],
                    color=prefix_color_map.get(prefix, 'gray'),
                    label=f"{prefix}",
                    alpha=0.8,
                    s=60,
                )

            # 이상치
            outlier_data = prefix_data[prefix_data["classification"] == "outlier"]
            if not outlier_data.empty:
                ax1.scatter(
                    outlier_data["Log P"],
                    outlier_data["RT"],
                    facecolors="none",
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.8,
                    s=60,
                )

            # 회귀선 그리기
            for reg in self.regression_results:
                if reg["prefix"] == prefix and reg["r_squared"] >= 0.99:
                    x_values = prefix_data["Log P"].dropna()
                    if len(x_values) > 0:
                        x_min, x_max = x_values.min(), x_values.max()
                        x_range = np.linspace(x_min - 0.5, x_max + 0.5, 100)
                        y_pred = reg["slope"] * x_range + reg["intercept"]
                        ax1.plot(
                            x_range,
                            y_pred,
                            "--",
                            color=prefix_color_map.get(prefix, 'gray'),
                            alpha=0.5,
                            linewidth=1.5,
                        )

        ax1.set_xlabel("Log P", fontsize=12)
        ax1.set_ylabel("RT", fontsize=12)
        ax1.set_title("Mass Spec Data Scatter Plot - By Prefix", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)

        # 2. 접미사별 산점도
        suffixes = sorted([s for s in self.data["suffix"].unique() if s])[:20]  # 상위 20개만
        if suffixes:
            suffix_colors = plt.cm.viridis(np.linspace(0, 1, len(suffixes)))
            suffix_color_map = dict(zip(suffixes, suffix_colors))

            for suffix in suffixes:
                suffix_data = self.data[self.data["suffix"] == suffix]
                true_data = suffix_data[suffix_data["classification"] == "true"]

                if not true_data.empty:
                    ax2.scatter(
                        true_data["Log P"],
                        true_data["RT"],
                        color=suffix_color_map[suffix],
                        label=suffix,
                        alpha=0.8,
                        s=60,
                    )

        ax2.set_xlabel("Log P", fontsize=12)
        ax2.set_ylabel("RT", fontsize=12)
        ax2.set_title("Mass Spec Data Scatter Plot - By Suffix", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        if suffixes:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=1)

        plt.tight_layout()

        # 이미지를 base64로 인코딩
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        scatter_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{scatter_plot}"

    def create_histogram(self):
        """히스토그램 생성"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 1. 분류 결과 히스토그램
        classification_counts = self.data["classification"].value_counts()

        colors = ["#667eea", "#f56565"]
        bars1 = self._create_bar_plot(
            ax1,
            ["True Values", "Outliers"],
            [classification_counts.get("true", 0), classification_counts.get("outlier", 0)],
            color=colors,
            alpha=0.8
        )

        ax1.set_ylabel("Number of Substances", fontsize=12)
        ax1.set_title("Classification Results", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # 2. 이상치 원인별 분포
        outlier_reasons = self.data[self.data["classification"] == "outlier"]["outlier_reason"].value_counts()

        if not outlier_reasons.empty:
            reason_labels = {
                "high_residual": "High Residual",
                "OAc_rule_violation": "OAc Rule Violation",
                "sugar_count_rule": "Sugar Count Rule",
                "no_linear_relationship": "No Linear Relationship",
                "isomer_rule": "Isomer Rule (f=1)",
            }

            labels = [reason_labels.get(r, r) for r in outlier_reasons.index]
            bars2 = self._create_bar_plot(
                ax2,
                range(len(labels)),
                outlier_reasons.values,
                color=plt.cm.Reds(np.linspace(0.4, 0.8, len(labels)))
            )
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha="right")

        ax2.set_ylabel("Number of Outliers", fontsize=12)
        ax2.set_title("Outlier Reasons Distribution", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # 이미지를 base64로 인코딩
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        histogram = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{histogram}"

    def create_debug_plot(self):
        """디버그 정보 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # 1. 규칙별 처리 결과
        rule_results = {
            'Priority Series': self.debug_info['priority_series_matches'],
            'Rule 1 Groups': self.debug_info['rule1_groups'],
            'Rule 5 Violations': self.debug_info['rule5_violations'],
            'Rule 6 Clusters': self.debug_info['rule6_clusters'],
            'Isomer Pairs': self.debug_info['isomer_pairs']
        }
        bars1 = ax1.bar(rule_results.keys(), rule_results.values(), 
                        color=plt.cm.Set3(np.linspace(0, 1, len(rule_results))))
        
        # 값 표시
        self._add_bar_labels(ax1, bars1)

        ax1.set_title('Rules Application Results', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. 전체 분류 결과
        classification_data = {
            'True Values': self.debug_info['total_true'],
            'Outliers': self.debug_info['total_outliers']
        }

        bars2 = ax2.bar(classification_data.keys(), classification_data.values(),
                        color=['#2ecc71', '#e74c3c'])
        
        # 값 표시
        self._add_bar_labels(ax2, bars2)

        ax2.set_title('Overall Classification Results', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count')

        plt.tight_layout()

        # 이미지를 base64로 인코딩
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        debug_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f'data:image/png;base64,{debug_plot}'

    def save_results(self, filename):
        """결과를 CSV 파일로 저장"""
        # 결과 데이터프레임 생성
        result_df = self.data.copy()

        # 회귀분석 정보 추가
        result_df["regression_r_squared"] = ""
        result_df["regression_equation"] = ""
        result_df["regression_p_value"] = ""

        # 각 행에 대해 해당하는 회귀분석 결과 찾기
        for idx, row in result_df.iterrows():
            # regression_group이 비어있지 않은 경우에만 찾기
            if row["regression_group"] and row["regression_group"] != "":
                # 해당 regression_group의 결과 찾기
                for reg in self.regression_results:
                    if row["regression_group"] == reg.get("group_name", ""):
                        result_df.loc[idx, "regression_r_squared"] = f"{reg['r_squared']:.4f}"
                        result_df.loc[idx, "regression_equation"] = f"RT = {reg['slope']:.4f} * LogP + {reg['intercept']:.4f}"
                        result_df.loc[idx, "regression_p_value"] = f"{reg['p_value']:.4e}"
                        break

        # 필요한 열 순서 정리
        columns_to_save = [
            "Name", "RT", "Volume", "Log P", "Anchor",  # 원본 데이터
            "prefix", "suffix", "a", "b", "c",  # 파싱 결과
            "prefix_d", "prefix_e", "prefix_f", "sugar_count",  # 접두사 분석
            "classification", "outlier_reason",  # 분류 결과
            "regression_group", "priority_series",  # 그룹 정보
            "regression_r_squared", "regression_equation", "regression_p_value",  # 회귀분석
        ]

        # 존재하는 열만 선택
        columns_to_save = [col for col in columns_to_save if col in result_df.columns]

        # CSV 저장
        result_df[columns_to_save].to_csv(filename, index=False, encoding="utf-8-sig")

        return filename

    def get_analysis_summary(self):
        """분석 결과 요약 반환"""
        total_count = len(self.data)
        true_count = len(self.data[self.data["classification"] == "true"])
        outlier_count = len(self.data[self.data["classification"] == "outlier"])
        group_count = len(self.data["prefix"].unique())
        
        # 디버깅 정보 포함
        summary = {
            "total_count": total_count,
            "true_count": true_count,
            "outlier_count": outlier_count,
            "group_count": group_count,
            "debug_info": self.debug_info
        }
        
        return summary

    def get_regression_table(self):
        """회귀분석 결과 테이블 반환"""
        regression_table = []
        for reg in self.regression_results:
            if "model" in reg:
                # NaN 값 체크 및 변환
                regression_table.append({
                    "prefix": reg["prefix"],
                    "suffix": reg["suffix"],
                    "slope": float(reg["slope"]) if not np.isnan(reg["slope"]) else 0.0,
                    "intercept": float(reg["intercept"]) if not np.isnan(reg["intercept"]) else 0.0,
                    "r_squared": float(reg["r_squared"]) if not np.isnan(reg["r_squared"]) else 0.0,
                    "p_value": float(reg["p_value"]) if not np.isnan(reg["p_value"]) else 1.0,
                    "group_type": reg.get("group_type", ""),
                    "group_name": reg.get("group_name", "")
                })
        return regression_table