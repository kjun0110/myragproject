const withPWA = require("next-pwa")({
  dest: "public",
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === "development",
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Next.js 16에서 Turbopack 비활성화 (next-pwa와 호환을 위해)
  experimental: {
    turbo: false,
  },
};

module.exports = withPWA(nextConfig);

