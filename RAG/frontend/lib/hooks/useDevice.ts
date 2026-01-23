'use client'

import { useEffect, useState } from 'react'

/**
 * 디바이스 타입 정의
 * 프로젝트의 Tailwind 브레이크포인트와 일치시킴
 */
export type DeviceType = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl'

/**
 * 브레이크포인트 상수
 * tailwind.config.js의 screens 설정과 일치
 * Tailwind 기본값: sm(640), md(768), lg(1024), xl(1280), 2xl(1536)
 * xs(475)는 커스텀 추가
 */
const BREAKPOINTS = {
    xs: 475,    // 작은 모바일 (커스텀)
    sm: 640,    // 모바일 가로 (Tailwind 기본)
    md: 768,    // 태블릿 (Tailwind 기본)
    lg: 1024,   // 데스크톱 (Tailwind 기본)
    xl: 1280,   // 큰 데스크톱 (Tailwind 기본)
    '2xl': 1536, // 매우 큰 화면 (Tailwind 기본)
} as const

/**
 * 화면 너비에 따라 디바이스 타입을 결정하는 함수
 *
 * @param width - 화면 너비 (px)
 * @returns 디바이스 타입
 *
 * @remarks
 * 큰 화면부터 작은 화면 순서로 체크하여
 * 가장 먼저 매칭되는 브레이크포인트를 반환합니다.
 */
function getDeviceType(width: number): DeviceType {
    if (width >= BREAKPOINTS['2xl']) return '2xl'
    if (width >= BREAKPOINTS.xl) return 'xl'
    if (width >= BREAKPOINTS.lg) return 'lg'
    if (width >= BREAKPOINTS.md) return 'md'
    if (width >= BREAKPOINTS.sm) return 'sm'
    return 'xs'
}
/**
 * 디바이스 타입을 감지하는 커스텀 훅
 *
 * @remarks
 * - SSR 안전성을 위해 초기값은 'lg' (데스크톱)로 설정
 * - 리사이즈 이벤트는 debounce 처리하여 성능 최적화 (150ms)
 * - SHADCN_POLICY에 따라 불가피한 경우에만 사용 (대부분은 Tailwind CSS로 처리 권장)
 * - orientationchange 이벤트도 처리하여 모바일 회전 대응
 *
 * @returns 현재 디바이스 타입
 *
 * @example
 * ```tsx
 * const device = useDevice()
 * if (device === 'xs' || device === 'sm') {
 *   // 모바일 처리
 * }
 * ```
 */
export function useDevice(): DeviceType {
    // SSR 안전성: 서버에서는 기본값으로 'lg' (데스크톱) 반환
    const [device, setDevice] = useState<DeviceType>(() => {
        if (typeof window === 'undefined') return 'lg'
        return getDeviceType(window.innerWidth)
    })

    useEffect(() => {
        let timeoutId: NodeJS.Timeout

        // 디바이스 타입 확인 함수
        const checkDevice = () => {
            const width = window.innerWidth
            const newDevice = getDeviceType(width)

            // 상태가 실제로 변경된 경우에만 업데이트
            setDevice((prevDevice) => {
                if (prevDevice !== newDevice) {
                    return newDevice
                }
                return prevDevice
            })
        }

        // 리사이즈 이벤트 debounce 처리 (성능 최적화)
        const handleResize = () => {
            clearTimeout(timeoutId)
            timeoutId = setTimeout(checkDevice, 150)
        }

        // 모바일 회전 대응
        const handleOrientationChange = () => {
            // orientationchange 후 약간의 지연을 두고 업데이트
            clearTimeout(timeoutId)
            timeoutId = setTimeout(checkDevice, 100)
        }

        // 초기 체크
        checkDevice()

        // 이벤트 리스너 등록
        window.addEventListener('resize', handleResize)
        window.addEventListener('orientationchange', handleOrientationChange)

        // Cleanup
        return () => {
            clearTimeout(timeoutId)
            window.removeEventListener('resize', handleResize)
            window.removeEventListener('orientationchange', handleOrientationChange)
        }
    }, [])

    return device
}
/**
 * 모바일 디바이스 여부를 확인하는 훅
 * xs, sm을 모바일로 간주
 *
 * @returns 모바일 여부
 *
 * @example
 * ```tsx
 * const isMobile = useIsMobile()
 * if (isMobile) {
 *   return <MobileView />
 * }
 * ```
 */
export function useIsMobile(): boolean {
    const device = useDevice()
    return device === 'xs' || device === 'sm'
}

/**
 * 태블릿 디바이스 여부를 확인하는 훅
 * md를 태블릿으로 간주
 *
 * @returns 태블릿 여부
 *
 * @example
 * ```tsx
 * const isTablet = useIsTablet()
 * if (isTablet) {
 *   return <TabletView />
 * }
 * ```
 */
export function useIsTablet(): boolean {
    const device = useDevice()
    return device === 'md'
}

/**
 * 데스크톱 디바이스 여부를 확인하는 훅
 * lg 이상을 데스크톱으로 간주
 *
 * @returns 데스크톱 여부
 *
 * @example
 * ```tsx
 * const isDesktop = useIsDesktop()
 * if (isDesktop) {
 *   return <DesktopView />
 * }
 * ```
 */
export function useIsDesktop(): boolean {
    const device = useDevice()
    return device === 'lg' || device === 'xl' || device === '2xl'
}

/**
 * 작은 화면 여부를 확인하는 훅
 * xs, sm, md를 작은 화면으로 간주 (모바일 + 태블릿)
 *
 * @returns 작은 화면 여부
 *
 * @example
 * ```tsx
 * const isSmallScreen = useIsSmallScreen()
 * if (isSmallScreen) {
 *   return <CompactLayout />
 * }
 * ```
 */
export function useIsSmallScreen(): boolean {
    const device = useDevice()
    return device === 'xs' || device === 'sm' || device === 'md'
}

/**
 * 큰 화면 여부를 확인하는 훅
 * xl, 2xl을 큰 화면으로 간주
 *
 * @returns 큰 화면 여부
 *
 * @example
 * ```tsx
 * const isLargeScreen = useIsLargeScreen()
 * if (isLargeScreen) {
 *   return <WideLayout />
 * }
 * ```
 */
export function useIsLargeScreen(): boolean {
    const device = useDevice()
    return device === 'xl' || device === '2xl'
}
