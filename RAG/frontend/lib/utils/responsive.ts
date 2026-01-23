/**
 * 반응형 유틸리티 함수
 * 웹과 모바일을 하나의 컴포넌트로 공유하기 위한 헬퍼 함수
 */

export type Breakpoint = 'mobile' | 'tablet' | 'desktop';

export const breakpoints = {
  mobile: 640,
  tablet: 768,
  desktop: 1024,
} as const;

/**
 * 현재 화면 크기에 따른 브레이크포인트 반환
 */
export function getBreakpoint(width: number): Breakpoint {
  if (width < breakpoints.tablet) {
    return 'mobile';
  } else if (width < breakpoints.desktop) {
    return 'tablet';
  }
  return 'desktop';
}

/**
 * 모바일 여부 확인
 */
export function isMobile(width: number): boolean {
  return width < breakpoints.tablet;
}

/**
 * 태블릿 여부 확인
 */
export function isTablet(width: number): boolean {
  return width >= breakpoints.tablet && width < breakpoints.desktop;
}

/**
 * 데스크톱 여부 확인
 */
export function isDesktop(width: number): boolean {
  return width >= breakpoints.desktop;
}

/**
 * 서버 사이드에서 기본값 반환
 */
export function getDefaultResponsive() {
  return {
    breakpoint: 'desktop' as Breakpoint,
    isMobile: false,
    isTablet: false,
    isDesktop: true,
    width: 1920,
    height: 1080,
  };
}
