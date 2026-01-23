const path = require('path');

const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === 'development',
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 워크스페이스 루트 경고 해결: 현재 디렉토리를 루트로 명시
  outputFileTracingRoot: path.join(__dirname),
  // 워크스페이스 루트 경고 해결
  outputFileTracingRoot: require('path').join(__dirname),
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    // 반응형 디자인: 웹과 모바일을 하나의 컴포넌트로 공유
    // Tailwind CSS의 반응형 브레이크포인트 사용:
    // - sm: 640px (모바일 가로)
    // - md: 768px (태블릿)
    // - lg: 1024px (데스크톱)
    // - xl: 1280px (큰 데스크톱)
    // - 2xl: 1536px (매우 큰 화면)
    NEXT_PUBLIC_RESPONSIVE_MODE: 'unified', // 'unified': 웹/모바일 공유 컴포넌트
  },
  // next-pwa는 webpack 기반이므로 Turbopack과 호환되지 않음
  // 빈 turbopack 설정으로 에러 방지
  turbopack: undefined,
  // 반응형 이미지 최적화 설정
  images: {
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    formats: ['image/avif', 'image/webp'],
  },
  // 반응형 폰트 최적화
  experimental: {
    optimizeCss: true,
  },
};

module.exports = withPWA(nextConfig);
