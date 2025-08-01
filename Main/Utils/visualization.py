"""
시각화 유틸리티 모듈

질량분석 데이터의 산점도, 히스토그램 등 시각화 기능을 제공합니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """질량분석 데이터 시각화 클래스"""
    
    def __init__(self):
        self.dpi = 150
        self.scatter_figsize = (16, 8)
        self.histogram_figsize = (12, 6)
    
    def create_scatter_plot(self, 
                          data: pd.DataFrame,
                          regression_results: List[Dict]) -> str:
        """
        산점도 생성
        
        Args:
            data: 분석된 데이터
            regression_results: 회귀분석 결과 리스트
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.scatter_figsize)

        # 1. 접두사별 산점도
        self._plot_by_prefix(data, regression_results, ax1)
        
        # 2. 접미사별 산점도
        self._plot_by_suffix(data, ax2)

        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_by_prefix(self, 
                       data: pd.DataFrame,
                       regression_results: List[Dict],
                       ax: plt.Axes) -> None:
        """접두사별 산점도 그리기"""
        prefixes = sorted(data["prefix"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(prefixes), 20)))
        prefix_color_map = dict(zip(prefixes[:20], colors))

        for i, prefix in enumerate(prefixes[:20]):  # 최대 20개 접두사만 표시
            prefix_data = data[data["prefix"] == prefix]

            # 참값
            true_data = prefix_data[prefix_data["classification"] == "true"]
            if not true_data.empty:
                ax.scatter(
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
                ax.scatter(
                    outlier_data["Log P"],
                    outlier_data["RT"],
                    facecolors="none",
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.8,
                    s=60,
                )

            # 회귀선 그리기
            for reg in regression_results:
                if reg["prefix"] == prefix and reg["r_squared"] >= 0.99:
                    x_values = prefix_data["Log P"].dropna()
                    if len(x_values) > 0:
                        x_min, x_max = x_values.min(), x_values.max()
                        x_range = np.linspace(x_min - 0.5, x_max + 0.5, 100)
                        y_pred = reg["slope"] * x_range + reg["intercept"]
                        ax.plot(
                            x_range,
                            y_pred,
                            "--",
                            color=prefix_color_map.get(prefix, 'gray'),
                            alpha=0.5,
                            linewidth=1.5,
                        )

        ax.set_xlabel("Log P", fontsize=12)
        ax.set_ylabel("RT", fontsize=12)
        ax.set_title("Mass Spec Data Scatter Plot - By Prefix", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    
    def _plot_by_suffix(self, 
                       data: pd.DataFrame,
                       ax: plt.Axes) -> None:
        """접미사별 산점도 그리기"""
        suffixes = sorted([s for s in data["suffix"].unique() if s])[:20]  # 상위 20개만
        if suffixes:
            suffix_colors = plt.cm.viridis(np.linspace(0, 1, len(suffixes)))
            suffix_color_map = dict(zip(suffixes, suffix_colors))

            for suffix in suffixes:
                suffix_data = data[data["suffix"] == suffix]
                true_data = suffix_data[suffix_data["classification"] == "true"]

                if not true_data.empty:
                    ax.scatter(
                        true_data["Log P"],
                        true_data["RT"],
                        color=suffix_color_map[suffix],
                        label=suffix,
                        alpha=0.8,
                        s=60,
                    )

        ax.set_xlabel("Log P", fontsize=12)
        ax.set_ylabel("RT", fontsize=12)
        ax.set_title("Mass Spec Data Scatter Plot - By Suffix", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if suffixes:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=1)
    
    def create_histogram(self, data: pd.DataFrame) -> str:
        """
        히스토그램 생성
        
        Args:
            data: 분석된 데이터
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.histogram_figsize)

        # 1. 분류 결과 히스토그램
        self._plot_classification_histogram(data, ax1)
        
        # 2. 이상치 원인별 분포
        self._plot_outlier_reasons(data, ax2)

        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_classification_histogram(self, 
                                     data: pd.DataFrame,
                                     ax: plt.Axes) -> None:
        """분류 결과 히스토그램"""
        classification_counts = data["classification"].value_counts()

        colors = ["#667eea", "#f56565"]
        bars = ax.bar(
            ["True Values", "Outliers"],
            [classification_counts.get("true", 0), classification_counts.get("outlier", 0)],
            color=colors,
            alpha=0.8,
        )

        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_ylabel("Number of Substances", fontsize=12)
        ax.set_title("Classification Results", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    def _plot_outlier_reasons(self, 
                            data: pd.DataFrame,
                            ax: plt.Axes) -> None:
        """이상치 원인별 분포"""
        outlier_reasons = data[data["classification"] == "outlier"]["outlier_reason"].value_counts()

        if not outlier_reasons.empty:
            reason_labels = {
                "high_residual": "High Residual",
                "OAc_rule_violation": "OAc Rule Violation",
                "sugar_count_rule": "Sugar Count Rule",
                "no_linear_relationship": "No Linear Relationship",
                "isomer_rule": "Isomer Rule (f=1)",
            }

            labels = [reason_labels.get(r, r) for r in outlier_reasons.index]
            bars = ax.bar(
                range(len(labels)),
                outlier_reasons.values,
                color=plt.cm.Reds(np.linspace(0.4, 0.8, len(labels))),
            )
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.set_ylabel("Number of Outliers", fontsize=12)
            ax.set_title("Outlier Reasons Distribution", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        matplotlib figure를 base64 문자열로 변환
        
        Args:
            fig: matplotlib figure 객체
            
        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=self.dpi, bbox_inches="tight")
        img.seek(0)
        plot_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_base64}"
    
    def create_debug_plot(self, debug_info: Dict) -> str:
        """
        디버깅 정보 시각화
        
        Args:
            debug_info: 디버깅 정보 딕셔너리
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 디버깅 정보를 막대 그래프로 표시
        labels = ['Priority Series', 'Rule1 Groups', 'OAc Violations', 
                 'Sugar Clusters', 'Isomer Pairs']
        values = [
            debug_info.get('priority_series_matches', 0),
            debug_info.get('rule1_groups', 0),
            debug_info.get('rule5_violations', 0),
            debug_info.get('rule6_clusters', 0),
            debug_info.get('isomer_pairs', 0)
        ]
        
        bars = ax.bar(labels, values, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Analysis Debug Information', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)