from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
from datetime import datetime
import traceback

from analyzer import MassSpecAnalyzer
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Create directories if not exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    """메인 페이지"""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """CSV 파일 분석"""
    try:
        # 파일 업로드 검증
        if "file" not in request.files:
            return jsonify({"error": "파일이 없습니다."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "CSV 파일만 업로드 가능합니다."}), 400

        # 파일 저장
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # CSV 읽기
        df = read_csv_with_header_detection(filepath)
        
        if df is None:
            return jsonify({"error": "CSV 파일을 읽을 수 없습니다."}), 400
        
        if len(df) == 0:
            return jsonify({"error": "유효한 데이터가 없습니다."}), 400

        # 분석 실행
        analyzer = MassSpecAnalyzer(df)
        try:
            analyzer.apply_rules()
        except Exception as e:
            return jsonify({"error": f"분석 중 오류 발생: {str(e)}"}), 500

        # 시각화 생성
        scatter_plot = analyzer.create_scatter_plot()
        histogram = analyzer.create_histogram()
        
        # 디버그 플롯 생성 (Phase 2 추가)
        debug_plot = analyzer.create_debug_plot()

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"analysis_result_{timestamp}.csv"
        result_filepath = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
        analyzer.save_results(result_filepath)

        # 분석 요약 가져오기 (개선된 버전)
        summary = analyzer.get_analysis_summary()
        
        # 회귀분석 결과 테이블 가져오기
        regression_table = analyzer.get_regression_table()

        # 임시 파일 삭제
        try:
            os.remove(filepath)
        except:
            pass

        # 응답 데이터 구성
        response_data = {
            "success": True,
            "total_count": summary["total_count"],
            "true_count": summary["true_count"],
            "outlier_count": summary["outlier_count"],
            "group_count": summary["group_count"],
            "regression_results": regression_table,
            "scatter_plot": scatter_plot,
            "histogram": histogram,
            "result_file": result_filename,
        }
        
        # Phase 2: 디버깅 정보 추가
        if "debug_info" in summary:
            response_data["debug_info"] = summary["debug_info"]
            response_data["debug_plot"] = debug_plot
        
        if "regression_stats" in summary:
            response_data["regression_stats"] = summary["regression_stats"]

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500


@app.route("/download/<filename>")
def download(filename):
    """결과 파일 다운로드"""
    try:
        # 보안 검증
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({"error": "잘못된 파일명입니다."}), 400
            
        filepath = os.path.join(app.config["RESULTS_FOLDER"], filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
            
        return send_file(
            filepath, 
            as_attachment=True, 
            download_name=filename, 
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/<filename>")
def debug_info(filename):
    """디버그 정보 조회 (Phase 2 추가)"""
    try:
        # 디버그 파일명 생성
        debug_filename = filename.replace('.csv', '_debug.csv')
        filepath = os.path.join(app.config["RESULTS_FOLDER"], debug_filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "디버그 파일을 찾을 수 없습니다."}), 404
        
        # 디버그 정보 읽기
        debug_df = pd.read_csv(filepath)
        debug_data = debug_df.to_dict('records')[0] if len(debug_df) > 0 else {}
        
        return jsonify({
            "success": True,
            "debug_info": debug_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def read_csv_with_header_detection(filepath):
    """CSV 파일 읽기 (헤더 자동 감지)"""
    try:
        print(f"파일 읽기 시도: {filepath}")
        
        # 파일 존재 여부 확인
        if not os.path.exists(filepath):
            print(f"파일이 존재하지 않습니다: {filepath}")
            return None

        # 파일 크기 확인
        if os.path.getsize(filepath) == 0:
            print("파일이 비어있습니다.")
            return None

        # pandas의 read_csv 옵션들
        csv_options = {
            'sep': [',', ';', '\t'],  # 구분자 옵션
            'encoding': ['utf-8', 'cp949', 'euc-kr', 'utf-16', 'ascii'],
            'skiprows': [0, 1, 2, 3]  # 건너뛸 행 수 옵션
        }

        # 모든 조합 시도
        for sep in csv_options['sep']:
            for encoding in csv_options['encoding']:
                for skiprow in csv_options['skiprows']:
                    try:
                        print(f"시도: sep={sep}, encoding={encoding}, skiprows={skiprow}")
                        
                        # 파일 읽기 시도
                        df = pd.read_csv(
                            filepath,
                            sep=sep,
                            encoding=encoding,
                            skiprows=skiprow,
                            engine='python'  # 더 유연한 파싱
                        )
                        
                        # 필수 컬럼 확인
                        required_columns = ["Name", "RT", "Volume", "Log P", "Anchor"]
                        found_columns = []
                        
                        # 컬럼명 매핑
                        column_mapping = {
                            'name': 'Name',
                            'rt': 'RT',
                            'volume': 'Volume',
                            'log p': 'Log P',
                            'logp': 'Log P',
                            'anchor': 'Anchor'
                        }
                        
                        # 컬럼명 정리 및 매핑
                        df.columns = df.columns.str.strip()
                        df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
                        
                        # 필수 컬럼 존재 확인
                        for req_col in required_columns:
                            if req_col in df.columns:
                                found_columns.append(req_col)
                        
                        if len(found_columns) == len(required_columns):
                            print("모든 필수 컬럼을 찾았습니다!")
                            
                            # 데이터 형식 변환
                            df["RT"] = pd.to_numeric(df["RT"], errors='coerce')
                            df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
                            df["Log P"] = pd.to_numeric(df["Log P"], errors='coerce')
                            df["Anchor"] = df["Anchor"].astype(str)
                            
                            # 결측치가 있는 행 제거
                            df = df.dropna(subset=["RT", "Volume", "Log P"])
                            
                            if len(df) > 0:
                                print(f"성공적으로 데이터를 읽었습니다. 행 수: {len(df)}")
                                return df
                            else:
                                print("유효한 데이터가 없습니다.")
                                continue
                                
                    except Exception as e:
                        print(f"시도 실패: {str(e)}")
                        continue
        
        print("모든 시도 실패")
        return None
        
    except Exception as e:
        print(f"CSV 읽기 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("질량분석 데이터 자동화 시스템")
    print("=" * 50)
    print("서버가 시작되었습니다.")
    print(f"브라우저에서 http://localhost:{Config.PORT} 으로 접속하세요.")
    print("종료하려면 Ctrl+C를 누르세요.")
    print("=" * 50)
    
    app.run(debug=Config.DEBUG, port=Config.PORT)