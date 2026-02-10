"use client";

import { useState, useCallback, DragEvent } from "react";
import { itemConfigs, ItemType } from "./UploadSidebar";

interface UploadState {
  file: File | null;
  isDragging: boolean;
  isUploading: boolean;
  uploadResult: {
    success: boolean;
    message: string;
    savedCount?: number;
    totalRecords?: number;
    displayedRecords?: number;
    data?: any[];
  } | null;
  error: string | null;
}

interface EmbedState {
  isEmbedding: boolean;
  message: string | null;
  error: string | null;
}

interface UploadContentProps {
  itemType: ItemType;
}

export default function UploadContent({ itemType }: UploadContentProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    isDragging: false,
    isUploading: false,
    uploadResult: null,
    error: null,
  });
  const [embedState, setEmbedState] = useState<EmbedState>({
    isEmbedding: false,
    message: null,
    error: null,
  });

  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState((prev) => ({ ...prev, isDragging: true }));
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState((prev) => ({ ...prev, isDragging: false }));
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();

      setUploadState((prev) => ({ ...prev, isDragging: false }));

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (file.name.endsWith(".jsonl")) {
          setUploadState((prev) => ({
            ...prev,
            file,
            error: null,
            uploadResult: null,
          }));
        } else {
          setUploadState((prev) => ({
            ...prev,
            error: "JSONL 파일만 업로드 가능합니다.",
          }));
        }
      }
    },
    []
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.name.endsWith(".jsonl")) {
          setUploadState((prev) => ({
            ...prev,
            file,
            error: null,
            uploadResult: null,
          }));
        } else {
          setUploadState((prev) => ({
            ...prev,
            error: "JSONL 파일만 업로드 가능합니다.",
          }));
        }
      }
    },
    []
  );

  // job_id로 임베딩 상태 폴링 (5초 간격). 업로드 후 자동 발동 또는 "임베딩" 버튼 클릭 시 사용
  const POLL_INTERVAL_MS = 5000;
  const MAX_POLL_ATTEMPTS = 240; // 5초 × 240 ≈ 20분

  const pollEmbeddingStatus = useCallback(async (jobId: string) => {
    setEmbedState((prev) => ({
      ...prev,
      isEmbedding: true,
      message: "임베딩을 시작했습니다. 완료될 때까지 잠시 기다려주세요.",
      error: null,
    }));
    let attempts = 0;
    while (attempts < MAX_POLL_ATTEMPTS) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      const statusRes = await fetch(`/v10/api/player/embed/status/${jobId}`);
      const statusData = await statusRes.json().catch(() => ({}));
      if (!statusRes.ok) {
        setEmbedState((prev) => ({
          ...prev,
          isEmbedding: false,
          message: null,
          error: statusData.error || "상태 조회에 실패했습니다.",
        }));
        return;
      }
      const status = statusData.status;
      if (status === "completed") {
        const msg = statusData.result?.message || "임베딩 작업이 완료되었습니다.";
        setEmbedState((prev) => ({
          ...prev,
          isEmbedding: false,
          message: msg,
          error: null,
        }));
        return;
      }
      if (status === "failed") {
        setEmbedState((prev) => ({
          ...prev,
          isEmbedding: false,
          message: null,
          error: statusData.error || "임베딩 작업이 실패했습니다.",
        }));
        return;
      }
      attempts += 1;
    }
    setEmbedState((prev) => ({
      ...prev,
      isEmbedding: false,
      message: null,
      error: "상태 조회 시간이 초과되었습니다. (최대 20분 대기) 백엔드에서 작업이 계속 진행 중일 수 있으니 잠시 후 페이지를 새로고침해 보세요.",
    }));
  }, []);

  const handleEmbed = useCallback(async () => {
    if (itemType !== "player") return;
    setEmbedState((prev) => ({ ...prev, isEmbedding: true, message: null, error: null }));
    try {
      const createRes = await fetch("/v10/api/player/embed", { method: "POST" });
      const createData = await createRes.json().catch(() => ({}));
      if (!createRes.ok) {
        setEmbedState((prev) => ({
          ...prev,
          isEmbedding: false,
          message: null,
          error: createData.error || "임베딩 작업 등록에 실패했습니다.",
        }));
        return;
      }
      const jobId = createData.job_id;
      if (!jobId) {
        setEmbedState((prev) => ({
          ...prev,
          isEmbedding: false,
          message: null,
          error: "job_id를 받지 못했습니다.",
        }));
        return;
      }
      await pollEmbeddingStatus(jobId);
    } catch (err) {
      setEmbedState((prev) => ({
        ...prev,
        isEmbedding: false,
        message: null,
        error: err instanceof Error ? err.message : "임베딩 요청 중 오류가 발생했습니다.",
      }));
    }
  }, [itemType, pollEmbeddingStatus]);

  const handleUpload = useCallback(async () => {
    if (!uploadState.file) {
      setUploadState((prev) => ({
        ...prev,
        error: "파일을 선택해주세요.",
      }));
      return;
    }

    setUploadState((prev) => ({
      ...prev,
      isUploading: true,
      error: null,
      uploadResult: null,
    }));

    // 타임아웃 120초 (대용량 업로드·백엔드 처리 시간 고려)
    const uploadTimeoutMs = 120_000;

    try {
      const formData = new FormData();
      formData.append("file", uploadState.file);

      // Next.js API Route 사용
      let apiEndpoint = "";
      if (itemType === "player") {
        apiEndpoint = "/v10/api/player/upload";
      } else if (itemType === "schedule") {
        apiEndpoint = "/v10/api/schedule/upload";
      } else if (itemType === "stadium") {
        apiEndpoint = "/v10/api/stadium/upload";
      } else if (itemType === "team") {
        apiEndpoint = "/v10/api/team/upload";
      } else {
        throw new Error(`지원하지 않는 아이템 타입: ${itemType}`);
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), uploadTimeoutMs);

      let response: Response;
      try {
        response = await fetch(apiEndpoint, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeoutId);
      }

      const result = await response.json();

      if (response.ok) {
        setUploadState((prev) => ({
          ...prev,
          isUploading: false,
          uploadResult: {
            success: true,
            message: result.message,
            savedCount: result.saved_count || result.displayed_records,
            totalRecords: result.total_records,
            displayedRecords: result.displayed_records,
            data: result.data,
          },
        }));

        // Player 업로드이고 job_id가 있으면 임베딩 상태 폴링 자동 발동 (5초 간격)
        if (itemType === "player" && result.job_id) {
          setEmbedState((prev) => ({
            ...prev,
            isEmbedding: true,
            message: "임베딩을 시작했습니다. 완료될 때까지 잠시 기다려주세요.",
            error: null,
          }));
          pollEmbeddingStatus(result.job_id);
        }

        // 첫 5개 행 데이터가 있으면 콘솔에 출력
        if (result.data && Array.isArray(result.data)) {
          console.log("첫 5개 행 데이터:", result.data);
        }
      } else {
        setUploadState((prev) => ({
          ...prev,
          isUploading: false,
          error: result.detail || "업로드 실패",
        }));
      }
    } catch (error) {
      const isAbort = error instanceof Error && error.name === "AbortError";
      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
        error: isAbort
          ? `요청 시간이 초과되었습니다 (${uploadTimeoutMs / 1000}초). 대용량 파일이면 시간이 더 걸릴 수 있습니다. 백엔드(port 8000) 실행 여부를 확인한 뒤 다시 시도해주세요.`
          : error instanceof Error
            ? error.message
            : "업로드 중 오류가 발생했습니다.",
      }));
    }
  }, [uploadState.file, itemType, pollEmbeddingStatus]);

  const handleRemoveFile = useCallback(() => {
    setUploadState((prev) => ({
      ...prev,
      file: null,
      error: null,
      uploadResult: null,
    }));
  }, []);

  const currentConfig = itemConfigs[itemType];

  return (
    <div className="upload-content">
      <div className="upload-header-section">
        <div className="upload-title">
          <span className="title-icon">{currentConfig.icon}</span>
          <h2>{currentConfig.label}</h2>
        </div>
        <p className="upload-description">{currentConfig.description}</p>
      </div>

      <div
        className={`drop-zone ${uploadState.isDragging ? "dragging" : ""} ${
          uploadState.file ? "has-file" : ""
        }`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {uploadState.file ? (
          <div className="file-info">
            <span className="file-icon">📄</span>
            <div className="file-details">
              <p className="file-name">{uploadState.file.name}</p>
              <p className="file-size">
                {(uploadState.file.size / 1024).toFixed(2)} KB
              </p>
            </div>
            <button
              className="remove-button"
              onClick={handleRemoveFile}
              disabled={uploadState.isUploading}
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <span className="drop-icon">📁</span>
            <p className="drop-text">파일을 드래그 앤 드롭하거나 클릭하여 선택</p>
            <input
              type="file"
              accept=".jsonl"
              onChange={handleFileSelect}
              className="file-input"
              id={`file-input-${itemType}`}
            />
            <label htmlFor={`file-input-${itemType}`} className="file-input-label">
              파일 선택
            </label>
          </div>
        )}
      </div>

      {uploadState.error && (
        <div className="error-message">{uploadState.error}</div>
      )}

      {uploadState.uploadResult && (
        <div
          className={`result-message ${
            uploadState.uploadResult.success ? "success" : "error"
          }`}
        >
          <p>{uploadState.uploadResult.message}</p>
          {uploadState.uploadResult.displayedRecords !== undefined && (
            <p className="result-details">
              출력된 레코드: {uploadState.uploadResult.displayedRecords} /{" "}
              {uploadState.uploadResult.totalRecords}
            </p>
          )}
          {uploadState.uploadResult.savedCount !== undefined && (
            <p className="result-details">
              저장된 레코드: {uploadState.uploadResult.savedCount} /{" "}
              {uploadState.uploadResult.totalRecords}
            </p>
          )}
          {uploadState.uploadResult.data && Array.isArray(uploadState.uploadResult.data) && uploadState.uploadResult.data.length > 0 && (
            <div className="data-preview">
              <p className="data-preview-title">첫 5개 행 데이터:</p>
              <pre className="data-preview-content">
                {JSON.stringify(uploadState.uploadResult.data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      <button
        type="button"
        className="upload-button"
        onClick={handleUpload}
        disabled={!uploadState.file || uploadState.isUploading}
      >
        {uploadState.isUploading ? "업로드 중..." : "업로드"}
      </button>

      <button
        type="button"
        className="embed-button"
        onClick={handleEmbed}
        disabled={embedState.isEmbedding || itemType !== "player"}
      >
        {embedState.isEmbedding ? "임베딩 중..." : "임베딩"}
      </button>
      {itemType !== "player" && (
        <p className="embed-hint">임베딩은 선수(Player) 타입에서만 사용할 수 있습니다.</p>
      )}
      {embedState.message && (
        <div
          className={`result-message embed-completion-banner ${
            embedState.isEmbedding ? "info" : "success"
          }`}
        >
          <p className="embed-completion-title">
            {embedState.isEmbedding ? "임베딩 시작" : "임베딩 완료"}
          </p>
          <p>{embedState.message}</p>
        </div>
      )}
      {embedState.error && (
        <div className="embed-result error">{embedState.error}</div>
      )}

      <style jsx>{`
        .upload-content {
          flex: 1;
          background: white;
          border-radius: 0.5rem;
          padding: 2rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .upload-header-section {
          margin-bottom: 2rem;
        }

        .upload-title {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-bottom: 0.5rem;
        }

        .title-icon {
          font-size: 2.5rem;
        }

        .upload-title h2 {
          margin: 0;
          font-size: 2rem;
          color: #333;
        }

        .upload-description {
          color: #6b7280;
          font-size: 1.1rem;
          margin: 0;
        }

        .drop-zone {
          border: 3px dashed #d1d5db;
          border-radius: 0.5rem;
          padding: 4rem 2rem;
          text-align: center;
          transition: all 0.2s;
          margin-bottom: 1.5rem;
          min-height: 300px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .drop-zone.dragging {
          border-color: #667eea;
          background: #f0f4ff;
        }

        .drop-zone.has-file {
          border-color: #10b981;
          background: #f0fdf4;
        }

        .drop-zone-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1.5rem;
        }

        .drop-icon {
          font-size: 5rem;
        }

        .drop-text {
          font-size: 1.2rem;
          color: #6b7280;
          margin: 0;
        }

        .file-input {
          display: none;
        }

        .file-input-label {
          padding: 1rem 2rem;
          background: #667eea;
          color: white;
          border-radius: 0.5rem;
          cursor: pointer;
          transition: background 0.2s;
          font-size: 1.1rem;
          font-weight: 600;
        }

        .file-input-label:hover {
          background: #5568d3;
        }

        .file-info {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          width: 100%;
          max-width: 600px;
        }

        .file-icon {
          font-size: 3rem;
        }

        .file-details {
          flex: 1;
          text-align: left;
        }

        .file-name {
          margin: 0 0 0.5rem 0;
          font-weight: 600;
          color: #333;
          font-size: 1.2rem;
        }

        .file-size {
          margin: 0;
          color: #6b7280;
          font-size: 1rem;
        }

        .remove-button {
          padding: 0.75rem;
          background: #ef4444;
          color: white;
          border: none;
          border-radius: 0.5rem;
          cursor: pointer;
          font-size: 1.5rem;
          transition: background 0.2s;
          width: 48px;
          height: 48px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .remove-button:hover:not(:disabled) {
          background: #dc2626;
        }

        .remove-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .error-message {
          padding: 1rem;
          background: #fef2f2;
          border: 1px solid #fecaca;
          border-radius: 0.5rem;
          color: #dc2626;
          margin-bottom: 1rem;
        }

        .result-message {
          padding: 1rem;
          border-radius: 0.5rem;
          margin-bottom: 1rem;
        }

        .result-message.success {
          background: #f0fdf4;
          border: 1px solid #86efac;
          color: #166534;
        }

        .result-message.info {
          background: #eff6ff;
          border: 1px solid #93c5fd;
          color: #1e40af;
        }

        .result-message.error {
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #dc2626;
        }

        .embed-completion-banner {
          margin-top: 1rem;
        }

        .embed-completion-title {
          font-weight: 700;
          margin-bottom: 0.25rem;
        }

        .result-details {
          margin: 0.5rem 0 0 0;
          font-size: 0.9rem;
          opacity: 0.8;
        }

        .data-preview {
          margin-top: 1rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.5);
          border-radius: 0.5rem;
        }

        .data-preview-title {
          margin: 0 0 0.5rem 0;
          font-weight: 600;
          font-size: 0.95rem;
        }

        .data-preview-content {
          margin: 0;
          padding: 1rem;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 0.25rem;
          overflow-x: auto;
          font-size: 0.85rem;
          line-height: 1.5;
          max-height: 400px;
          overflow-y: auto;
        }

        .upload-button {
          width: 100%;
          padding: 1rem;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 0.5rem;
          font-size: 1.2rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .upload-button:hover:not(:disabled) {
          background: #5568d3;
        }

        .upload-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .embed-button {
          width: 100%;
          margin-top: 1rem;
          padding: 1rem;
          background: #10b981;
          color: white;
          border: none;
          border-radius: 0.5rem;
          font-size: 1.2rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .embed-button:hover:not(:disabled) {
          background: #059669;
        }

        .embed-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .embed-hint {
          margin: 0.5rem 0 0 0;
          font-size: 0.9rem;
          color: #64748b;
        }

        .embed-result {
          margin-top: 0.75rem;
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          font-size: 0.95rem;
        }

        .embed-result.success {
          background: #f0fdf4;
          border: 1px solid #86efac;
          color: #166534;
        }

        .embed-result.error {
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #dc2626;
        }

        @media (max-width: 768px) {
          .upload-content {
            padding: 1.5rem;
          }

          .drop-zone {
            padding: 2rem 1rem;
            min-height: 250px;
          }

          .drop-icon {
            font-size: 3rem;
          }

          .drop-text {
            font-size: 1rem;
          }
        }
      `}</style>
    </div>
  );
}
