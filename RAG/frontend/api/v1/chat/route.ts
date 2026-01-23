import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, history, model_type } = body;

    if (!message) {
      return NextResponse.json(
        { error: "메시지가 필요합니다." },
        { status: 400 }
      );
    }

    // LangChain 백엔드 API 호출
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

    // 디버깅: 백엔드 URL과 model_type 확인
    console.log("[DEBUG] 백엔드 URL:", backendUrl);
    console.log("[DEBUG] Next.js API route에서 받은 model_type:", model_type);

    const response = await fetch(`${backendUrl}/api/chain`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        history: history || [],
        model_type: model_type,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        {
          error: errorData.detail || errorData.message || "백엔드 서버 오류가 발생했습니다.",
          status: response.status,
        },
        { status: response.status }
      );
    }

    // Content-Type 확인하여 스트리밍 여부 판단
    const contentType = response.headers.get("content-type");

    if (contentType && contentType.includes("text/plain")) {
      // 스트리밍 응답 처리
      if (!response.body) {
        return NextResponse.json(
          { error: "스트리밍 응답을 받을 수 없습니다." },
          { status: 500 }
        );
      }

      // 백엔드의 텍스트 스트림을 그대로 전달
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });
    } else {
      // 기존 JSON 응답 처리
      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch (error) {
    console.error("[ERROR] API 라우트 오류:", error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.",
      },
      { status: 500 }
    );
  }
}
