# PWA (Progressive Web App) 구성 가이드

## 📱 PWA 기능 개요

이 프로젝트는 Next.js와 next-pwa를 사용하여 완전한 PWA 기능을 제공합니다.

## ✨ 주요 기능

### 1. 오프라인 지원
- Service Worker를 통한 자동 캐싱
- 네트워크 우선 전략으로 최신 콘텐츠 제공
- 오프라인 페이지 제공 (`/offline`)

### 2. 홈 화면 추가
- 모바일 및 데스크톱에서 앱처럼 설치 가능
- 자동 설치 프롬프트 제공
- 독립형(standalone) 모드로 실행

### 3. 캐싱 전략
- **NetworkFirst**: 일반 요청 (최신 데이터 우선)
- **CacheFirst**: 이미지 파일 (오프라인 우선)
- **StaleWhileRevalidate**: 정적 파일 (빠른 로딩)

### 4. 앱 메타데이터
- 아이콘 (192x192, 512x512)
- 테마 색상
- 앱 이름 및 설명
- 바로가기(Shortcuts) 지원

## 🚀 사용 방법

### 개발 환경
```bash
npm run dev
```
- 개발 모드에서는 Service Worker가 비활성화됩니다 (성능 향상)
- PWA 기능은 프로덕션 빌드에서만 활성화됩니다

### 프로덕션 빌드
```bash
npm run build
npm start
```
- 빌드 후 `public/sw.js`와 `public/workbox-*.js` 파일이 생성됩니다
- Service Worker가 자동으로 등록됩니다

## 📋 PWA 설치 방법

### 모바일 (iOS)
1. Safari에서 사이트 접속
2. 공유 버튼(⬆️) 클릭
3. "홈 화면에 추가" 선택

### 모바일 (Android)
1. Chrome에서 사이트 접속
2. 메뉴(⋮) → "앱 설치" 또는 자동 프롬프트 확인
3. 또는 주소창의 설치 아이콘 클릭

### 데스크톱 (Chrome/Edge)
1. 주소창 오른쪽의 설치 아이콘 클릭
2. 또는 자동으로 표시되는 설치 프롬프트 확인

## 🔧 설정 파일

### `next.config.js`
- PWA 플러그인 설정
- 캐싱 전략 정의
- Service Worker 옵션

### `public/manifest.json`
- 앱 메타데이터
- 아이콘 설정
- 바로가기(Shortcuts) 정의
- 테마 색상

### `app/layout.tsx`
- PWA 메타 태그
- 아이콘 링크
- 테마 색상 설정

## 📁 파일 구조

```
frontend/
├── app/
│   ├── components/
│   │   └── PWAInstallPrompt.tsx  # 설치 프롬프트 컴포넌트
│   ├── offline/
│   │   └── page.tsx              # 오프라인 페이지
│   └── layout.tsx               # PWA 메타 태그
├── public/
│   ├── manifest.json           # PWA 매니페스트
│   ├── icon-192x192.png        # 앱 아이콘 (192x192)
│   └── icon-512x512.png        # 앱 아이콘 (512x512)
└── next.config.js              # PWA 설정
```

## 🎨 커스터마이징

### 아이콘 변경
1. `public/icon-192x192.png`와 `public/icon-512x512.png` 교체
2. `public/manifest.json`에서 아이콘 경로 확인

### 테마 색상 변경
1. `public/manifest.json`의 `theme_color` 수정
2. `app/layout.tsx`의 `viewport.themeColor` 수정

### 앱 이름 변경
1. `public/manifest.json`의 `name`과 `short_name` 수정
2. `app/layout.tsx`의 `metadata.title` 수정

## ⚠️ 주의사항

1. **HTTPS 필수**: 프로덕션 환경에서는 HTTPS가 필요합니다 (localhost는 예외)
2. **Service Worker**: 개발 모드에서는 비활성화되어 있습니다
3. **캐시 관리**: 브라우저 개발자 도구에서 Service Worker와 캐시를 관리할 수 있습니다
4. **업데이트**: Service Worker는 자동으로 업데이트되며, `skipWaiting: true`로 즉시 적용됩니다

## 🐛 문제 해결

### Service Worker가 작동하지 않는 경우
1. 브라우저 개발자 도구 → Application → Service Workers 확인
2. 캐시 삭제 후 재시도
3. `public/sw.js` 파일이 생성되었는지 확인

### 설치 프롬프트가 나타나지 않는 경우
1. 이미 설치된 경우 프롬프트가 표시되지 않습니다
2. HTTPS 환경인지 확인
3. 브라우저가 PWA를 지원하는지 확인

### 오프라인 페이지가 표시되지 않는 경우
1. Service Worker가 등록되었는지 확인
2. 네트워크를 완전히 차단한 상태에서 테스트
3. `app/offline/page.tsx` 파일이 존재하는지 확인

## 📚 참고 자료

- [Next.js PWA 문서](https://github.com/shadowwalker/next-pwa)
- [Web App Manifest](https://developer.mozilla.org/en-US/docs/Web/Manifest)
- [Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
