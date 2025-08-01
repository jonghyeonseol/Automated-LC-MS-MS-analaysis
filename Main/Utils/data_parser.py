"""
데이터 파싱 유틸리티 모듈

질량분석 데이터의 Name 필드를 파싱하고
접두사/접미사 성분을 분석하는 기능을 제공합니다.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


class DataParser:
    """질량분석 데이터 파싱 클래스"""
    
    def __init__(self):
        # 당 개수 매핑 테이블
        self.sugar_mapping = {
            'A': 0, 'M': 1, 'D': 2, 
            'T': 3, 'Q': 4, 'P': 5
        }
    
    def parse_name(self, name: str) -> Tuple[str, str, Optional[int], Optional[int], Optional[str]]:
        """
        Name을 접두사와 접미사로 분리
        
        Args:
            name: 분석할 물질명 (예: "GD1+OAc+HexNAc(36:1;O2)")
            
        Returns:
            tuple: (prefix, suffix, a, b, c)
                - prefix: 접두사 (예: "GD1+OAc+HexNAc")
                - suffix: 접미사 (예: "(36:1;O2)")
                - a, b, c: 접미사 성분
        """
        try:
            # 괄호를 기준으로 분리
            match = re.match(r'(.+?)(\([^)]+\))$', str(name))
            if match:
                prefix = match.group(1)
                suffix = match.group(2)
                
                # 접미사에서 a, b, c 성분 추출
                suffix_match = re.match(r'\((\d+):(\d+);(O\d+)\)', suffix)
                if suffix_match:
                    a = int(suffix_match.group(1))
                    b = int(suffix_match.group(2))
                    c = suffix_match.group(3)
                    return prefix, suffix, a, b, c
            
            return str(name), '', None, None, None
            
        except Exception as e:
            print(f"Name 파싱 오류 ({name}): {e}")
            return str(name), '', None, None, None
    
    def parse_prefix_components(self, prefix: str) -> Tuple[str, str, str, int]:
        """
        접두사에서 d, e, f 성분 추출 및 당 개수 계산
        
        Args:
            prefix: 접두사 문자열
            
        Returns:
            tuple: (d, e, f, sugar_count)
                - d, e, f: 접두사의 첫 3문자
                - sugar_count: 총 당 개수
        """
        if not prefix or len(prefix) < 2:
            return '', '', '', 0
        
        d = prefix[0] if len(prefix) >= 1 else ''
        e = prefix[1] if len(prefix) >= 2 else ''
        f = prefix[2] if len(prefix) >= 3 else ''
        
        # 규칙3: e의 당 개수 계산
        sugar_count_e = self.sugar_mapping.get(e, 0)
        
        # 규칙4: f의 당 개수 계산 (5-f)
        sugar_count_f = 0
        if f.isdigit() and 1 <= int(f) <= 4:
            sugar_count_f = 5 - int(f)
        
        total_sugar = sugar_count_e + sugar_count_f
        
        # 추가 당 처리
        if '+dHex' in prefix:
            total_sugar += prefix.count('+dHex')
        if '+HexNAc' in prefix:
            total_sugar += prefix.count('+HexNAc')
        
        return d, e, f, total_sugar
    
    def count_oac(self, prefix: str) -> int:
        """
        OAc 개수 계산
        
        Args:
            prefix: 접두사 문자열
            
        Returns:
            int: OAc 개수 (0, 1, 또는 2)
        """
        if '2OAc' in prefix:
            return 2
        elif 'OAc' in prefix:
            return 1
        return 0
    
    def extract_base_prefix(self, prefix: str) -> str:
        """
        접두사에서 OAc 등의 수식어를 제거한 기본형 추출
        
        Args:
            prefix: 전체 접두사
            
        Returns:
            str: 기본 접두사
        """
        base = prefix
        # OAc 제거
        base = base.replace('+2OAc', '').replace('+OAc', '')
        base = base.replace('2OAc', '').replace('OAc', '')
        # 다른 수식어 제거
        base = base.replace('+dHex', '').replace('+HexNAc', '')
        
        return base
    
    def parse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터프레임의 Name 열을 파싱하여 새 열 추가
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 파싱된 열이 추가된 데이터프레임
        """
        # Name 파싱
        parsed_data = df['Name'].apply(self.parse_name)
        df[['prefix', 'suffix', 'a', 'b', 'c']] = pd.DataFrame(
            parsed_data.tolist(), index=df.index
        )
        
        # 접두사 성분 파싱
        prefix_components = df['prefix'].apply(self.parse_prefix_components)
        df[['prefix_d', 'prefix_e', 'prefix_f', 'sugar_count']] = pd.DataFrame(
            prefix_components.tolist(), index=df.index
        )
        
        # OAc 개수 추가
        df['oac_count'] = df['prefix'].apply(self.count_oac)
        
        # 기본 접두사 추가
        df['base_prefix'] = df['prefix'].apply(self.extract_base_prefix)
        
        return df
    
    def get_isomer_candidates(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        이성질체 후보 그룹 찾기
        
        Args:
            df: 파싱된 데이터프레임
            
        Returns:
            dict: {그룹키: [인덱스 리스트]}
        """
        isomer_groups = {}
        
        # f=1인 경우만 이성질체 후보
        f1_data = df[df['prefix_f'] == '1'].copy()
        
        # 접두사, 접미사, Log P가 모두 동일한 그룹 찾기
        for (prefix, suffix, log_p), group in f1_data.groupby(['prefix', 'suffix', 'Log P']):
            if len(group) > 1:
                # RT 차이가 0.1 이상인 경우 확인
                rt_values = group['RT'].values
                if (rt_values.max() - rt_values.min()) > 0.1:
                    key = f"{prefix}_{suffix}_{log_p}"
                    isomer_groups[key] = group.index.tolist()
        
        return isomer_groups
    
    def validate_sugar_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        당 개수 계산의 일관성 검증
        
        Args:
            df: 파싱된 데이터프레임
            
        Returns:
            dict: 검증 결과 (경고 메시지들)
        """
        warnings = {
            'invalid_e': [],
            'invalid_f': [],
            'negative_sugar': []
        }
        
        for idx, row in df.iterrows():
            # e 성분 검증
            if row['prefix_e'] and row['prefix_e'] not in self.sugar_mapping:
                warnings['invalid_e'].append(
                    f"행 {idx}: '{row['prefix_e']}'는 유효하지 않은 e 성분입니다"
                )
            
            # f 성분 검증
            if row['prefix_f']:
                if not row['prefix_f'].isdigit():
                    warnings['invalid_f'].append(
                        f"행 {idx}: '{row['prefix_f']}'는 숫자가 아닙니다"
                    )
                elif not (1 <= int(row['prefix_f']) <= 4):
                    warnings['invalid_f'].append(
                        f"행 {idx}: f={row['prefix_f']}는 1-4 범위를 벗어났습니다"
                    )
            
            # 음수 당 개수 확인
            if row['sugar_count'] < 0:
                warnings['negative_sugar'].append(
                    f"행 {idx}: 음수 당 개수 ({row['sugar_count']})"
                )
        
        return warnings