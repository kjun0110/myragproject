# MCP 'FunctionTool' object is not callable 해결

## 원인
- fastmcp 서버가 툴 실행 시 툴 객체를 잘못 호출하거나, 툴 등록/조회 불일치로 인해 발생할 수 있음.
- fastmcp 이슈 #2642: 커스텀 이름 툴 + docket 등록 시 `function.__name__`과 툴 이름 불일치로 인한 실패가 보고됨 (2.14.5 근처에서 패치).

## 적용한 조치
1. **fastmcp 버전**: `requirements.txt`에서 `fastmcp>=2.14.5` 사용 (2.14.4 → 2.14.5 이상).
2. **업그레이드 실행**:
   ```bash
   pip install "fastmcp>=2.14.5"
   ```
   또는
   ```bash
   pip install -r api/requirements.txt
   ```

## 그래도 에러가 나는 경우

### 서버 실행 경로 패치 (적용됨)
- **위치**: `api/app/domains/v10/soccer/spokes/mcp/mcp_functiontool_patch.py`
- **동작**: `player_server` 로드 시 툴 클래스(FunctionTool 등)에 `__call__`을 추가해, 서버가 툴을 호출할 때 내부 실제 함수가 실행되도록 함. HTTP 방식은 그대로 유지.
- **제거**: fastmcp/MCP 쪽에서 툴 실행 버그가 수정되면, `player_server.py`에서 `mcp_functiontool_patch` import 한 줄을 제거하면 됨.

### 기타 시도
1. **decorator_mode 시도** (임시): 환경 변수로 `FASTMCP_DECORATOR_MODE=object` 설정 후 서버 재시작.
2. **fastmcp 최신 버전 확인**: [PyPI - fastmcp](https://pypi.org/project/fastmcp/) 에서 최신 2.x 설치.
3. **mcp 패키지 버전** (선택): `pip install -U mcp` 후 서버 재시작.
