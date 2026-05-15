# Data Folder

원자료 엑셀 파일은 KIRD 데이터 사용자 서약 및 활용 제한이 있을 수 있어 GitHub에 업로드하지 않습니다.

로컬에서 분석을 재현하려면 다음 중 하나를 사용하세요.

1. 원자료 파일을 `data/raw_data.xlsx`로 복사합니다.
2. 환경변수 `KIRD_RAW_DATA`에 원자료 엑셀 경로를 지정합니다.
3. 기본 경로 `C:\Users\rhksd\Desktop\Data\활동조사_재직자_데이터\활동조사_재직자_데이터_어승수.xlsx`에 원자료를 둡니다.

PowerShell 예시:

```powershell
$env:KIRD_RAW_DATA = "C:\path\to\활동조사_재직자_데이터_어승수.xlsx"
python scripts\01_load_and_preprocess.py
```

주의: GitHub 웹 업로드로 폴더를 드래그할 경우 `.gitignore`가 자동으로 적용되지 않을 수 있으므로, `data/` 폴더에는 이 `README.md`만 남겨두는 것을 권장합니다.
