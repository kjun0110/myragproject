import Link from "next/link";
import styles from "./upload-layout.module.css";

export const dynamic = "force-static";
export const revalidate = false;

export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={styles.uploadContainer}>
      <header className={styles.uploadHeader}>
        <div className={styles.headerContent}>
          <h1>📁 JSONL 파일 업로드</h1>
          <Link href="/v10/admin" className={styles.backButton} prefetch={false}>
            ← 대시보드로 돌아가기
          </Link>
        </div>
      </header>

      <main className={styles.uploadMain}>
        <div className={styles.uploadLayout}>{children}</div>
      </main>
    </div>
  );
}
