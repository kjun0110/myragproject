import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  console.log("[NEXT_API] Player 업로드 API Route 시작");
  
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    console.log("[NEXT_API] 파일 수신:", file ? file.name : "없음");

    if (!file) {
      console.log("[NEXT_API] 파일이 없음 - 400 에러 반환");
      return NextResponse.json(
        { error: "파일이 필요합니다." },
        { status: 400 }
      );
    }

    if (!file.name.endsWith(".jsonl")) {
      console.log("[NEXT_API] JSONL 파일이 아님 - 400 에러 반환");
      return NextResponse.json(
        { error: "JSONL 파일만 업로드 가능합니다." },
        { status: 400 }
      );
    }

    // 백엔드 API 호출
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    const backendEndpoint = `${backendUrl}/api/v10/soccer/player/upload`;
    
    console.log("[NEXT_API] 백엔드 URL:", backendUrl);
    console.log("[NEXT_API] 백엔드 엔드포인트:", backendEndpoint);
    console.log("[NEXT_API] 백엔드로 요청 전송 시작...");
    
    const backendFormData = new FormData();
    backendFormData.append("file", file);

    const response = await fetch(backendEndpoint, {
      method: "POST",
      body: backendFormData,
    });

    console.log("[NEXT_API] 백엔드 응답 상태:", response.status);
    console.log("[NEXT_API] 백엔드 응답 OK:", response.ok);

    if (!response.ok) {
      console.log("[NEXT_API] 백엔드 응답 실패 - 상태 코드:", response.status);
      const errorData = await response.json().catch(() => ({}));
      console.log("[NEXT_API] 에러 데이터:", errorData);
      return NextResponse.json(
        {
          error: errorData.detail || errorData.message || "백엔드 서버 오류가 발생했습니다.",
          status: response.status,
        },
        { status: response.status }
      );
    }

    console.log("[NEXT_API] 백엔드 응답 성공 - 데이터 파싱 중...");
    const data = await response.json();
    console.log("[NEXT_API] 백엔드 응답 데이터 수신 완료");
    console.log("[NEXT_API] 응답 데이터:", JSON.stringify(data).substring(0, 200) + "...");
    
    return NextResponse.json(data);
  } catch (error) {
    console.error("[NEXT_API] Player 업로드 API 라우트 오류:", error);
    console.error("[NEXT_API] 에러 상세:", error instanceof Error ? error.stack : error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.",
      },
      { status: 500 }
    );
  }
}
