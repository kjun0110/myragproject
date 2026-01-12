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
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
    try {
      // 디버깅: 백엔드 URL 확인
      console.log("[DEBUG] LangGraph 백엔드 URL:", backendUrl);

      // 타임아웃 설정 (로컬 모델은 시간이 오래 걸릴 수 있으므로 120초)
      const timeout = 120000; // 120초
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      let response: Response;
      try {
        response = await fetch(`${backendUrl}/api/graph`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message,
            history: history || [],
          }),
          signal: controller.signal,
          // Next.js fetch 타임아웃 설정
          next: { revalidate: 0 },
        } as RequestInit);
        clearTimeout(timeoutId);
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === "AbortError") {
          throw new Error("요청 시간이 초과되었습니다. 모델이 로딩 중이거나 응답 생성에 시간이 오래 걸리고 있습니다.");
        }
        throw fetchError;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || errorData.message || response.statusText;

        // 환경 불일치 에러 (400)
        if (response.status === 400) {
          return NextResponse.json(
            {
              error: errorMessage,
              detail: errorData.detail || errorMessage,
              status: response.status,
            },
            { status: response.status }
          );
        }

        // 기타 백엔드 오류
        return NextResponse.json(
          {
            error: errorMessage || "백엔드 서버 오류가 발생했습니다.",
            detail: errorData.detail || errorMessage,
            status: response.status,
          },
          { status: response.status }
        );
      }

      const data = await response.json();
      return NextResponse.json({ response: data.response || data.answer });
    } catch (backendError) {
      // 백엔드 연결 실패 (서버가 실행되지 않음)
      console.error("Backend connection error:", backendError);

      // 타임아웃 에러 확인
      if (
        backendError instanceof Error &&
        (backendError.message.includes("시간이 초과") ||
          backendError.message.includes("AbortError") ||
          backendError.cause?.code === "UND_ERR_HEADERS_TIMEOUT")
      ) {
        return NextResponse.json(
          {
            error: "요청 시간이 초과되었습니다. 모델이 로딩 중이거나 응답 생성에 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해주세요.",
            status: 504,
          },
          { status: 504 }
        );
      }

      // 네트워크 오류인지 확인
      if (
        backendError instanceof TypeError &&
        (backendError.message.includes("fetch") ||
          backendError.message.includes("HeadersTimeoutError"))
      ) {
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
          error: `백엔드 서버와 통신 중 오류가 발생했습니다: ${backendError instanceof Error ? backendError.message : "알 수 없는 오류"}`,
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
