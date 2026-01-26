"use client";

import { useRouter } from "next/navigation";
import UploadSidebar from "@/components/v10/UploadSidebar";

export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();

  return (
    <div className="upload-container">
      <header className="upload-header">
        <div className="header-content">
          <h1>📁 JSONL 파일 업로드</h1>
          <button
            className="back-button"
            onClick={() => router.push("/v10/admin")}
          >
            ← 대시보드로 돌아가기
          </button>
        </div>
      </header>

      <main className="upload-main">
        <div className="upload-layout">
          <UploadSidebar />
          {children}
        </div>
      </main>

      <style jsx>{`
        .upload-container {
          min-height: 100vh;
          background: #f5f7fa;
        }

        .upload-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 2rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .header-content {
          max-width: 1400px;
          margin: 0 auto;
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .upload-header h1 {
          margin: 0;
          font-size: 2rem;
        }

        .back-button {
          padding: 0.75rem 1.5rem;
          background: rgba(255, 255, 255, 0.2);
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 0.5rem;
          color: white;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .back-button:hover {
          background: rgba(255, 255, 255, 0.3);
          border-color: rgba(255, 255, 255, 0.5);
        }

        .upload-main {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
        }

        .upload-layout {
          display: flex;
          gap: 2rem;
          align-items: flex-start;
        }

        @media (max-width: 1024px) {
          .upload-layout {
            flex-direction: column;
          }
        }

        @media (max-width: 768px) {
          .upload-header {
            padding: 1rem;
          }

          .upload-header h1 {
            font-size: 1.5rem;
          }

          .upload-main {
            padding: 1rem;
          }
        }
      `}</style>
    </div>
  );
}
