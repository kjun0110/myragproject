"use client";

import Link from "next/link";
import UploadContent from "@/components/v10/UploadContent";

export default function ScheduleUploadPage() {
  return (
    <section className="schedule-upload-page">
      <nav className="page-nav" aria-label="업로드 메뉴">
        <Link href="/v10/admin/upload" className="back-link" prefetch={false}>
          ← 데이터 타입 선택
        </Link>
      </nav>
      <UploadContent itemType="schedule" />
      <style jsx>{`
        .schedule-upload-page {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .page-nav {
          flex-shrink: 0;
        }

        .back-link {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          color: #4b5563;
          text-decoration: none;
          font-size: 0.95rem;
          transition: color 0.2s;
        }

        .back-link:hover {
          color: #667eea;
        }
      `}</style>
    </section>
  );
}
