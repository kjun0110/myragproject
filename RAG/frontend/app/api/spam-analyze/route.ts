import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email_text } = body;

    if (!email_text || typeof email_text !== "string") {
      return NextResponse.json(
        { error: "email_text는 필수이며 문자열이어야 합니다." },
        { status: 400 }
      );
    }

    // 백엔드 API 호출
    const response = await fetch(`${BACKEND_URL}/api/mcp/spam-analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email_text }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.detail || errorData.message || "스팸 분석 중 오류가 발생했습니다." },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("[ERROR] 스팸 분석 API 라우트 오류:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다." },
      { status: 500 }
    );
  }
}
