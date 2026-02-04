import { NextResponse } from "next/server";

export async function POST() {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    const endpoint = `${backendUrl.replace(/\/$/, "")}/api/v10/soccer/player/embed`;

    const response = await fetch(endpoint, { method: "POST" });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || data.detail || "임베딩 작업 등록에 실패했습니다." },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("[NEXT_API] Player embed 오류:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다." },
      { status: 500 }
    );
  }
}
