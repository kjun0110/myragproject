"use client";

import { ReactNode, useEffect, useState } from "react";
import { useResponsive } from "@/lib/hooks/useResponsive";

interface ResponsiveContainerProps {
  children: ReactNode;
  mobile?: ReactNode;
  tablet?: ReactNode;
  desktop?: ReactNode;
  className?: string;
}

/**
 * 반응형 컨테이너 컴포넌트
 * 웹과 모바일을 하나의 컴포넌트로 공유하면서
 * 화면 크기에 따라 다른 콘텐츠를 표시할 수 있습니다.
 */
export default function ResponsiveContainer({
  children,
  mobile,
  tablet,
  desktop,
  className = "",
}: ResponsiveContainerProps) {
  const { isMobile, isTablet, isDesktop } = useResponsive();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className={className}>{children}</div>;
  }

  // 특정 화면 크기에 맞는 콘텐츠가 제공된 경우
  if (isMobile && mobile) {
    return <div className={className}>{mobile}</div>;
  }
  if (isTablet && tablet) {
    return <div className={className}>{tablet}</div>;
  }
  if (isDesktop && desktop) {
    return <div className={className}>{desktop}</div>;
  }

  // 기본: 하나의 컴포넌트를 반응형으로 사용
  return <div className={className}>{children}</div>;
}
