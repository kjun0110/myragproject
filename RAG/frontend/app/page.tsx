"use client";

import { useRouter, usePathname } from "next/navigation";
import { useState, useEffect } from "react";

export default function Home() {
  const router = useRouter();
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  const menuItems = [
    {
      id: "openai",
      label: "🌐 OpenAI",
      path: "/v1/openai",
      description: "OpenAI 모델 사용",
    },
    {
      id: "chain",
      label: "🖥️ 로컬(chain)",
      path: "/v1/chain",
      description: "로컬 Chain 모델",
    },
    {
      id: "graph",
      label: "🔗 로컬(graph)",
      path: "/v1/graph",
      description: "로컬 Graph 모델",
    },
    {
      id: "gri_env_mcp",
      label: "🌿 gri_env_mcp",
      path: "/v1/gri_env_mcp",
      description: "GRI 환경 컨텐츠",
    },
    {
      id: "spam",
      label: "📧 스팸메일판독기",
      path: "/v1/spam",
      description: "스팸 메일 분석",
    },
    {
      id: "study",
      label: "📚 study",
      path: "/v10/admin",
      description: "어드민 대시보드",
    },
    {
      id: "study_main",
      label: "💬 study main",
      path: "/v10/main",
      description: "채팅 화면",
    },
  ];

  const handleMenuClick = (path: string) => {
    router.push(path);
    setIsMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-white relative">
      {/* 햄버거 메뉴 버튼 - 데스크톱: 좌측 상단 고정, 모바일: 좌측 상단 */}
      <button
        className="fixed top-4 left-4 z-50 p-3 rounded-lg bg-white shadow-lg hover:bg-gray-50 transition-colors lg:top-6 lg:left-6"
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        aria-label="메뉴 열기"
      >
        <div className="w-6 h-6 flex flex-col justify-center gap-1.5">
          <span
            className={`block h-0.5 w-full bg-gray-800 transition-all duration-300 ${isMenuOpen ? "rotate-45 translate-y-2" : ""
              }`}
          />
          <span
            className={`block h-0.5 w-full bg-gray-800 transition-all duration-300 ${isMenuOpen ? "opacity-0" : ""
              }`}
          />
          <span
            className={`block h-0.5 w-full bg-gray-800 transition-all duration-300 ${isMenuOpen ? "-rotate-45 -translate-y-2" : ""
              }`}
          />
        </div>
      </button>

      {/* 사이드바 메뉴 - 데스크톱: 좌측에서 슬라이드, 모바일: 전체 화면 오버레이 */}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-80 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out lg:w-72 ${isMenuOpen ? "translate-x-0" : "-translate-x-full"
          }`}
      >
        <div className="h-full flex flex-col">
          {/* 사이드바 헤더 */}
          <div className="p-6 bg-gradient-to-br from-purple-600 to-purple-800 text-white">
            <h2 className="text-xl font-bold">🤖 LangChain</h2>
            <p className="text-sm opacity-90 mt-1">AI 챗봇 서비스</p>
          </div>

          {/* 메뉴 리스트 */}
          <nav className="flex-1 overflow-y-auto p-4">
            <div className="space-y-2">
              {menuItems.map((item) => {
                const isActive = pathname === item.path;
                return (
                  <button
                    key={item.id}
                    className={`w-full text-left p-4 rounded-lg transition-all ${isActive
                      ? "bg-gradient-to-r from-purple-600 to-purple-800 text-white shadow-lg"
                      : "bg-gray-50 text-gray-700 hover:bg-gray-100"
                      }`}
                    onClick={() => handleMenuClick(item.path)}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{item.label.split(" ")[0]}</span>
                      <div className="flex-1">
                        <div className="font-semibold">{item.label.replace(/^[^\s]+\s/, "")}</div>
                        <div className={`text-xs mt-0.5 ${isActive ? "opacity-90" : "opacity-60"}`}>
                          {item.description}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </nav>
        </div>
      </div>

      {/* 오버레이 - 모바일에서 메뉴 열릴 때 배경 어둡게 */}
      {isMenuOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsMenuOpen(false)}
        />
      )}

      {/* 메인 콘텐츠 - 화면 중앙에 "kjun develop" 텍스트만 */}
      <main className="min-h-screen flex items-center justify-center">
        <h1 className="text-6xl font-bold text-gray-800 md:text-7xl lg:text-8xl">kjun develop</h1>
      </main>
    </div>
  );
}
