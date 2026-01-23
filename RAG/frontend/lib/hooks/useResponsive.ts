"use client";

import { useState, useEffect } from "react";
import { Breakpoint, getBreakpoint, isMobile, isTablet, isDesktop } from "@/lib/utils/responsive";

interface UseResponsiveReturn {
  breakpoint: Breakpoint;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  width: number;
  height: number;
}

/**
 * 반응형 훅
 * 웹과 모바일을 하나의 컴포넌트로 공유하기 위한 React 훅
 */
export function useResponsive(): UseResponsiveReturn {
  const [dimensions, setDimensions] = useState({
    width: typeof window !== "undefined" ? window.innerWidth : 1920,
    height: typeof window !== "undefined" ? window.innerHeight : 1080,
  });

  useEffect(() => {
    function handleResize() {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }

    // 초기 설정
    handleResize();

    window.addEventListener("resize", handleResize);
    window.addEventListener("orientationchange", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("orientationchange", handleResize);
    };
  }, []);

  const { width, height } = dimensions;
  const breakpoint = getBreakpoint(width);

  return {
    breakpoint,
    isMobile: isMobile(width),
    isTablet: isTablet(width),
    isDesktop: isDesktop(width),
    width,
    height,
  };
}
