import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, history } = body;

    if (!message) {
      return NextResponse.json(
        { error: "메시지가 필요합니다." },
        { status: 400 }
      );
    }

    // LangGraph 백엔드 API 호출
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

    // 디버깅: 백엔드 URL 확인
    console.log("[DEBUG] LangGraph 백엔드 URL:", backendUrl);

    // 타임아웃 설정 (로컬 모델은 시간이 오래 걸릴 수 있으므로 120초)
    const timeout = 120000; // 120초
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${backendUrl}/api/graph`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          history: history || [],
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || errorData.message || response.statusText;

        return NextResponse.json(
          {
            error: errorMessage || "백엔드 서버 오류가 발생했습니다.",
            detail: errorData.detail || errorMessage,
            status: response.status,
          },
          { status: response.status }
        );
      }

      // 스트리밍 응답 처리
      if (!response.body) {
        return NextResponse.json(
          { error: "스트리밍 응답을 받을 수 없습니다." },
          { status: 500 }
        );
      }

      // 백엔드의 text/plain 스트림을 그대로 프론트엔드에 전달
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);

      // 타임아웃 에러
      if (fetchError instanceof Error && fetchError.name === "AbortError") {
        return NextResponse.json(
          {
            error: "요청 시간이 초과되었습니다. 모델이 로딩 중이거나 응답 생성에 시간이 오래 걸리고 있습니다.",
            status: 504,
          },
          { status: 504 }
        );
      }

      // 네트워크 오류
      if (fetchError instanceof TypeError) {
        return NextResponse.json(
          {
            error: "백엔드 서버에 연결할 수 없습니다. 백엔드 서비스가 실행 중인지 확인해주세요.",
            status: 503,
          },
          { status: 503 }
        );
      }

      return NextResponse.json(
        {
          error: `백엔드 서버와 통신 중 오류가 발생했습니다: ${fetchError instanceof Error ? fetchError.message : "알 수 없는 오류"}`,
          status: 500,
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      {
        error: "서버 오류가 발생했습니다.",
        details: error instanceof Error ? error.message : "알 수 없는 오류",
      },
      { status: 500 }
    );
  }
}
