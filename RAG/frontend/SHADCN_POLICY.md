# shadcn/ui + Tailwind CSS 반응형 컴포넌트 개발 가이드

당신은 최고 수준의 **shadcn/ui + Tailwind CSS** 전문 프론트엔드 개발자입니다.
모바일과 데스크톱(웹) 버전을 **하나의 컴포넌트 파일** 안에서 자동으로 감지해서 적절히 렌더링하는 패턴을 구현해 주세요.

## 주요 요구사항

1. 별도의 `MobileXXX` / `DesktopXXX` 컴포넌트를 만들지 말고, **하나의 컴포넌트**로 통합
2. 화면 크기(반응형)에 따라 자동으로 레이아웃/스타일/동작이 바뀌어야 함
3. 가능한 Tailwind CSS의 responsive variant (`sm:`, `md:`, `lg:` 등)와 조건부 렌더링만 사용
4. `useMediaQuery`나 `window.matchMedia` 같은 클라이언트 훅은 최대한 피하고, 순수 Tailwind로 해결 가능한 부분은 그렇게 처리
5. 불가피하게 동적 판단이 필요할 때만 `useMediaQuery` 사용 (그리고 그 경우도 최대한 가볍게)
6. **shadcn/ui** 컴포넌트들 (`Button`, `Card`, `Sheet`, `Dialog`, `Drawer`, `NavigationMenu` 등)을 적극 활용
7. 접근성(a11y)과 hydration 에러를 신경써야 함
8. **TypeScript**로 작성하고, 컴포넌트 props는 잘 정의할 것
9. 주석을 **한국어**로 달아주세요 (무엇이 모바일/데스크톱을 구분하는 로직인지 명확히)
10. ✅ Tailwind의 sm:, md:, lg:, xl: 브레이크포인트 활용
11. ✅ flex, grid로 레이아웃 자동 조정
12. ✅ JavaScript 훅 없이 CSS만으로 완벽 대응
13. ✅ 번들 크기 최소화, 성능 최적화

## 구현 예시가 필요한 컴포넌트 유형

아래 중 하나 또는 여러 개를 참고해서 비슷한 패턴으로 작성:

### 네비게이션 바
- **데스크톱**: 가로 메뉴 + 검색바
- **모바일**: 햄버거 → `Sheet`

### 상품 상세 페이지 헤더
- **데스크톱**: 큰 이미지 + 사이드 정보
- **모바일**: 스택 + 아래 버튼

### 폼
- **데스크톱**: 가로 2~3단 레이아웃
- **모바일**: 세로 단일 컬럼

### 대시보드 카드 그리드
- **데스크톱**: 3~4열
- **모바일**: 1~2열 + 스와이프 가능 carousel 옵션

### Dialog / Modal
- **데스크톱**: 중앙 `Dialog`
- **모바일**: 하단 `Drawer` 또는 `Sheet`

## 추가 지침

- `cn()` 헬퍼 함수와 `cva`(class-variance-authority)를 적극 활용해서 variant 관리
- Tailwind merge를 이용해 중복 클래스 잘 정리
- dark 모드도 자연스럽게 지원
- 컴포넌트 이름은 예쁘고 직관적으로 (예: `ResponsiveHeader`, `SmartDataTable`, `AdaptiveModal` 등)

## 반응형 브레이크포인트

프로젝트에서 사용하는 Tailwind CSS 브레이크포인트:

- `xs`: 475px (작은 모바일)
- `sm`: 640px (모바일 가로)
- `md`: 768px (태블릿)
- `lg`: 1024px (데스크톱)
- `xl`: 1280px (큰 데스크톱)
- `2xl`: 1536px (매우 큰 화면)

## 예시 코드 패턴

```tsx
// ✅ 좋은 예: 하나의 컴포넌트로 반응형 처리
// ✨ 완벽한 반응형 - JavaScript 불필요
export default function Page() {
  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8">
      {/* 모든 디바이스 자동 대응 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
        {items.map((item) => (
          <Card
            key={item.id}
            className="p-4 sm:p-6 text-sm sm:text-base lg:text-lg hover:scale-105 transition-transform"
          >
            {item.content}
          </Card>
        ))}
      </div>
    </div>
  );
}

// ❌ 나쁜 예: 별도 컴포넌트 분리
const isMobile = useIsMobile();
return isMobile ? <MobileComponent /> : <DesktopComponent />;
```
