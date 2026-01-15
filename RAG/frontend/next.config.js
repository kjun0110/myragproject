const withPWA = require("next-pwa")({
  dest: "public",
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === "development",
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Turbopack 경고 해결 (next-pwa는 webpack 사용, 하지만 빌드 시 turbopack 설정 필요)
  turbopack: {},
};

module.exports = withPWA(nextConfig);

