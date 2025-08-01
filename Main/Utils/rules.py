"""
규칙 적용 모듈

질량분석 데이터에 대한 다양한 규칙을 적용하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from regression import RegressionAnalyzer


class RuleApplier:
    """규칙 적용 클래스"""
    
    def __init__(self, data: pd.DataFrame, regression_analyzer: RegressionAnalyzer):
        """
        Args:
            data: 분석할 데이터
            regression_analyzer: 회귀분석 수행 객체
        """
        self.data = data
        self.regression_analyzer = regression_analyzer
        self.priority_groups = {
            "GD_series": ["GD1+dHex", "GD1", "GD2", "GD3"],
            "GT3_series": ["GM3", "GD3", "GT3"],
            "GP1_series": ["GP1", "GQ1", "GT1", "GD1a"],
        }
        self.debug_info = {
            'priority_series_matches': 0,
            'rule1_groups': 0,
            'rule5_violations': 0,
            'rule6_clusters': 0,
            'isomer_pairs': 0,
        }
    
    def check_priority_series(self) -> None:
        """우선순위 그룹 간의 선형성 검증"""
        print("\n=== 우선순위 시리즈 검증 시작 ===")
        
        for series_name, prefixes in self.priority_groups.items():
            print(f"\n{series_name} 검증 중: {prefixes}")
            
            # 해당 시리즈의 데이터 수집
            series_data = []
            for prefix in prefixes:
                prefix_data = self.data[self.data["prefix"] == prefix]
                if not prefix_data.empty:
                    print(f"  - {prefix}: {len(prefix_data)}개 데이터 발견")
                    series_data.append(prefix_data)
                else:
                    print(f"  - {prefix}: 데이터 없음")

            if len(series_data) >= 2:
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

                    if len(suffix_data) >= 2:
                        # 선형성 검증
                        is_linear, regression = self.regression_analyzer.check_linearity(suffix_data)
                        
                        if is_linear:
                            print(f"    접미사 {suffix}: R²={regression['r_squared']:.4f} - 우선순위 시리즈 확인!")
                            
                            # 해당 시리즈의 모든 데이터를 참값으로 분류
                            for df in suffix_data:
                                self.data.loc[df.index, "classification"] = "true"
                                self.data.loc[df.index, "priority_series"] = series_name
                                self.data.loc[df.index, "regression_group"] = f"{series_name}_{suffix}"
                            
                            # 회귀분석 결과 저장
                            self.regression_analyzer.add_regression_result(
                                series_name, suffix, "priority_series", 
                                f"{series_name}_{suffix}", regression
                            )
                            
                            self.debug_info['priority_series_matches'] += sum(len(df) for df in suffix_data)
            else:
                print(f"  {series_name}: 데이터 부족 (최소 2개 그룹 필요)")
    
    def apply_rule1(self) -> None:
        """규칙1 적용: 동일 접두사 그룹의 회귀분석"""
        print("\n=== 규칙1: 동일 접두사 그룹 회귀분석 ===")
        
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
                    test_indices = self._get_test_indices(subgroup, anchor_indices, group, 'bc', b, c)
                    test_data = self.data.loc[test_indices]
                    
                    regression = self.regression_analyzer.perform_regression(test_data)
                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_bc_{b}_{c}"
                        self._apply_classification(test_indices, group_name)
                        rule1_count += len(test_indices)
                        
                        self.regression_analyzer.add_regression_result(
                            prefix, f"(varied:{b};{c})", "a_variation", 
                            group_name, regression
                        )

            # 2) b 성분만 변화하는 경우 (a, c 동일)
            for (a, c), subgroup in group.groupby(["a", "c"]):
                if pd.notna(a) and pd.notna(c) and len(subgroup) >= 2:
                    test_indices = self._get_test_indices(subgroup, anchor_indices, group, 'ac', a, c)
                    test_data = self.data.loc[test_indices]
                    
                    regression = self.regression_analyzer.perform_regression(test_data)
                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_ac_{a}_{c}"
                        self._apply_classification(test_indices, group_name)
                        rule1_count += len(test_indices)
                        
                        self.regression_analyzer.add_regression_result(
                            prefix, f"({a}:varied;{c})", "b_variation", 
                            group_name, regression
                        )

            # 3) a, b 성분이 동시에 변화하는 경우 (c 동일, 3개 이상)
            for c, subgroup in group.groupby("c"):
                if pd.notna(c) and len(subgroup) >= 3:
                    test_indices = self._get_test_indices(subgroup, anchor_indices, group, 'c', None, c)
                    test_data = self.data.loc[test_indices]
                    
                    regression = self.regression_analyzer.perform_regression(test_data, min_points=3)
                    if regression and regression["r_squared"] >= 0.99:
                        group_name = f"{prefix}_c_{c}"
                        self._apply_classification(test_indices, group_name)
                        rule1_count += len(test_indices)
                        
                        self.regression_analyzer.add_regression_result(
                            prefix, f"(varied:varied;{c})", "ab_variation", 
                            group_name, regression
                        )

        print(f"규칙1 적용 완료: {rule1_count}개 물질 분류")
        self.debug_info['rule1_groups'] = len([
            r for r in self.regression_analyzer.regression_results 
            if r.get('group_type') in ['a_variation', 'b_variation', 'ab_variation']
        ])
    
    def apply_rule5(self) -> None:
        """규칙5: OAc 규칙 적용"""
        print("\n=== 규칙5: OAc 규칙 적용 ===")
        violations = 0
        
        for base_prefix in self.data["prefix"].unique():
            # OAc가 없는 기본 형태 찾기
            base_prefix_clean = self._clean_prefix(base_prefix)
            
            # 같은 base를 가진 모든 변형 찾기
            related_data = self.data[self.data["prefix"].str.contains(base_prefix_clean, regex=False)]
            
            if len(related_data) > 1:
                violations += self._check_oac_rule(related_data)
        
        print(f"규칙5 적용 완료: {violations}개 OAc 규칙 위반")
        self.debug_info['rule5_violations'] = violations
    
    def apply_rule6(self) -> None:
        """규칙6: 당 개수에 따른 RT 규칙"""
        print("\n=== 규칙6: 당 개수 규칙 적용 ===")
        clusters_processed = 0
        
        for suffix in self.data["suffix"].unique():
            if not suffix:
                continue
            
            suffix_data = self.data[self.data["suffix"] == suffix].copy()
            if len(suffix_data) > 1:
                clusters = self._find_rt_clusters(suffix_data)
                
                for cluster in clusters:
                    if self._process_sugar_cluster(cluster):
                        clusters_processed += 1
        
        print(f"규칙6 적용 완료: {clusters_processed}개 클러스터 처리")
        self.debug_info['rule6_clusters'] = clusters_processed
    
    def apply_isomer_rule(self) -> None:
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
                    
                    if self.data.loc[max_rt_idx, "classification"] != "true":
                        self.data.loc[max_rt_idx, "classification"] = "true"
                    
                    isomer_pairs += 1
        
        print(f"규칙7 적용 완료: {isomer_pairs}개 이성질체 쌍 처리")
        self.debug_info['isomer_pairs'] = isomer_pairs
    
    def detect_outliers_by_residual(self) -> None:
        """잔차 기반 이상치 탐지"""
        print("\n=== 잔차 기반 이상치 탐지 ===")
        outliers_detected = 0
        
        for reg_result in self.regression_analyzer.regression_results:
            if "std_residuals" in reg_result and reg_result["model"] is not None:
                group_name = reg_result.get("group_name", "")
                group_data = self.data[self.data["regression_group"] == group_name]
                
                if not group_data.empty and len(reg_result["std_residuals"]) == len(group_data):
                    outlier_indices = self.regression_analyzer.detect_outliers_by_residual(reg_result)
                    
                    for i, idx in enumerate(group_data.index):
                        if i in outlier_indices and self.data.loc[idx, "classification"] != "outlier":
                            self.data.loc[idx, "classification"] = "outlier"
                            self.data.loc[idx, "outlier_reason"] = "high_residual"
                            outliers_detected += 1
        
        print(f"잔차 기반 이상치 탐지 완료: {outliers_detected}개 발견")
    
    # Helper methods
    def _get_test_indices(self, subgroup, anchor_indices, group, type, a_or_b, c):
        """테스트 인덱스 수집"""
        test_indices = subgroup.index.tolist()
        
        for idx in anchor_indices:
            if idx not in test_indices and idx in group.index:
                if type == 'bc' and group.loc[idx, "b"] == a_or_b and group.loc[idx, "c"] == c:
                    test_indices.append(idx)
                elif type == 'ac' and group.loc[idx, "a"] == a_or_b and group.loc[idx, "c"] == c:
                    test_indices.append(idx)
                elif type == 'c' and group.loc[idx, "c"] == c:
                    test_indices.append(idx)
        
        return test_indices
    
    def _apply_classification(self, indices, group_name):
        """분류 적용"""
        self.data.loc[indices, "classification"] = "true"
        self.data.loc[indices, "regression_group"] = group_name
    
    def _clean_prefix(self, prefix):
        """접두사에서 수식어 제거"""
        return (prefix.replace("+2OAc", "")
                     .replace("+OAc", "")
                     .replace("2OAc", "")
                     .replace("OAc", ""))
    
    def _check_oac_rule(self, related_data):
        """OAc 규칙 확인"""
        violations = 0
        related_data = related_data.copy()
        related_data["oac_count"] = related_data["prefix"].apply(self._count_oac)
        
        for suffix in related_data["suffix"].unique():
            suffix_data = related_data[related_data["suffix"] == suffix].sort_values("oac_count")
            
            if len(suffix_data) > 1:
                for i in range(len(suffix_data) - 1):
                    if suffix_data.iloc[i]["RT"] >= suffix_data.iloc[i + 1]["RT"]:
                        self.data.loc[suffix_data.iloc[i + 1].name, "classification"] = "outlier"
                        self.data.loc[suffix_data.iloc[i + 1].name, "outlier_reason"] = "OAc_rule_violation"
                        violations += 1
        
        return violations
    
    def _count_oac(self, prefix):
        """OAc 개수 계산"""
        if "2OAc" in prefix:
            return 2
        elif "OAc" in prefix:
            return 1
        return 0
    
    def _find_rt_clusters(self, suffix_data):
        """RT 클러스터 찾기"""
        suffix_data = suffix_data.sort_values("RT")
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
        
        return clusters
    
    def _process_sugar_cluster(self, cluster):
        """당 개수 클러스터 처리"""
        cluster_data = self.data.loc[cluster].copy()
        
        # Anchor='T'가 있으면 건너뛰기
        if any(cluster_data["Anchor"] == "T"):
            return False
        
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
        
        return True