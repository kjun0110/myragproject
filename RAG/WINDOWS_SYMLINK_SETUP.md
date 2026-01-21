****# Windows 심볼릭 링크 권한 설정 가이드

## 방법 1: Windows Developer Mode 활성화 (권장)

### GUI 방법:
1. **Windows 설정 열기**
   - `Win + I` 키를 누르거나
   - 시작 메뉴에서 "설정" 검색

2. **개발자용 설정으로 이동**
   - "개인 정보 보호 및 보안" → "개발자용" 클릭
   - 또는 "업데이트 및 보안" → "개발자용" (Windows 버전에 따라 다름)

3. **개발자 모드 활성화**
   - "개발자 모드" 토글을 **켜기(ON)**로 설정
   - 확인 대화상자가 나타나면 "예" 클릭
   - 재부팅이 필요할 수 있습니다

### PowerShell 방법 (관리자 권한 필요):
```powershell
# 관리자 권한으로 PowerShell 실행 후:
New-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" -Name AllowDevelopmentWithoutDevLicense -PropertyType DWord -Value 1 -Force
```

---

## 방법 2: 관리자 권한으로 Python 실행

Developer Mode를 활성화하지 않으려면:

1. **PowerShell을 관리자 권한으로 실행**
   - 시작 메뉴에서 "PowerShell" 검색
   - 우클릭 → "관리자 권한으로 실행"

2. **가상 환경 활성화 및 명령 실행**
   ```powershell
   cd C:\Users\123\Documents\my-project\filter\RAG
   conda activate torch313
   # 또는
   .\venv\Scripts\Activate.ps1

   # 그 다음 HuggingFace 명령 실행
   hf download monologg/koelectra-small-v3-discriminator
   ```

---

## 방법 3: 환경 변수로 심볼릭 링크 비활성화 (가장 간단)

Developer Mode를 활성화하지 않고도 사용할 수 있는 방법:

### PowerShell에서 환경 변수 설정:
```powershell
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### 영구적으로 설정하려면:
```powershell
# 사용자 환경 변수로 설정
[System.Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS", "1", "User")
[System.Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS_WARNING", "1", "User")
```

### 또는 .env 파일에 추가:
```env
HF_HUB_DISABLE_SYMLINKS=1
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## 권장 사항

**가장 간단한 방법**: 방법 3 (환경 변수 설정)
- Developer Mode 활성화 불필요
- 관리자 권한 불필요
- 즉시 적용 가능
- 다만 캐시 공간이 약간 더 필요할 수 있음 (중복 파일 저장)

**가장 권장하는 방법**: 방법 1 (Developer Mode)
- 한 번만 설정하면 영구적으로 해결
- 다른 개발 도구에서도 유용
- 심볼릭 링크로 캐시 공간 절약
