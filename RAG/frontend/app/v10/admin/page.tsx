"use client";

import { useState } from "react";
import Link from "next/link";

interface Match {
  id: string;
  homeTeam: string;
  awayTeam: string;
  date: string;
  time: string;
  venue: string;
  availableSeats: number;
  price: number;
}

interface BettingOdds {
  matchId: string;
  homeWin: number;
  draw: number;
  awayWin: number;
  homeWinProbability: number;
  drawProbability: number;
  awayWinProbability: number;
}

interface Product {
  id: string;
  name: string;
  type: "ticket" | "merchandise" | "experience";
  price: number;
  stock: number;
  description: string;
}

interface Member {
  id: string;
  name: string;
  email: string;
  phone: string;
  membershipLevel: "bronze" | "silver" | "gold" | "platinum";
  joinDate: string;
  totalSpent: number;
}

export default function StudyPage() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "tickets" | "betting" | "products" | "members">("dashboard");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  // 파일 삭제 핸들러
  const handleFileRemove = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // 샘플 데이터
  const [matches, setMatches] = useState<Match[]>([
    {
      id: "1",
      homeTeam: "맨체스터 유나이티드",
      awayTeam: "리버풀",
      date: "2024-03-15",
      time: "20:00",
      venue: "올드 트래퍼드",
      availableSeats: 150,
      price: 85000,
    },
    {
      id: "2",
      homeTeam: "첼시",
      awayTeam: "아스날",
      date: "2024-03-16",
      time: "19:30",
      venue: "스탬포드 브리지",
      availableSeats: 200,
      price: 75000,
    },
    {
      id: "3",
      homeTeam: "토트넘",
      awayTeam: "맨체스터 시티",
      date: "2024-03-17",
      time: "21:00",
      venue: "토트넘 홋스퍼 스타디움",
      availableSeats: 100,
      price: 90000,
    },
  ]);

  const [bettingOdds, setBettingOdds] = useState<BettingOdds[]>([
    {
      matchId: "1",
      homeWin: 2.5,
      draw: 3.2,
      awayWin: 2.8,
      homeWinProbability: 40,
      drawProbability: 31,
      awayWinProbability: 29,
    },
    {
      matchId: "2",
      homeWin: 2.1,
      draw: 3.4,
      awayWin: 3.1,
      homeWinProbability: 48,
      drawProbability: 29,
      awayWinProbability: 23,
    },
    {
      matchId: "3",
      homeWin: 3.2,
      draw: 3.0,
      awayWin: 2.2,
      homeWinProbability: 31,
      drawProbability: 33,
      awayWinProbability: 36,
    },
  ]);

  const [products, setProducts] = useState<Product[]>([
    {
      id: "1",
      name: "유니폼 세트",
      type: "merchandise",
      price: 120000,
      stock: 50,
      description: "홈 유니폼 + 원정 유니폼 세트",
    },
    {
      id: "2",
      name: "VIP 경기 관람 패키지",
      type: "experience",
      price: 500000,
      stock: 10,
      description: "VIP 좌석 + 선수 사인회 + 경기 후 파티",
    },
    {
      id: "3",
      name: "시즌 패스",
      type: "ticket",
      price: 2000000,
      stock: 5,
      description: "전 시즌 홈 경기 무제한 관람",
    },
  ]);

  const [members, setMembers] = useState<Member[]>([
    {
      id: "1",
      name: "홍길동",
      email: "hong@example.com",
      phone: "010-1234-5678",
      membershipLevel: "gold",
      joinDate: "2023-01-15",
      totalSpent: 2500000,
    },
    {
      id: "2",
      name: "김철수",
      email: "kim@example.com",
      phone: "010-2345-6789",
      membershipLevel: "platinum",
      joinDate: "2022-06-20",
      totalSpent: 5000000,
    },
    {
      id: "3",
      name: "이영희",
      email: "lee@example.com",
      phone: "010-3456-7890",
      membershipLevel: "silver",
      joinDate: "2023-08-10",
      totalSpent: 1200000,
    },
  ]);

  const [selectedMatch, setSelectedMatch] = useState<string | null>(null);
  const [ticketQuantity, setTicketQuantity] = useState(1);
  const [newProduct, setNewProduct] = useState<Partial<Product>>({
    name: "",
    type: "merchandise",
    price: 0,
    stock: 0,
    description: "",
  });
  const [newMember, setNewMember] = useState<Partial<Member>>({
    name: "",
    email: "",
    phone: "",
    membershipLevel: "bronze",
  });

  const handleTicketPurchase = (matchId: string) => {
    const match = matches.find((m) => m.id === matchId);
    if (match && match.availableSeats >= ticketQuantity) {
      setMatches((prev) =>
        prev.map((m) =>
          m.id === matchId
            ? { ...m, availableSeats: m.availableSeats - ticketQuantity }
            : m
        )
      );
      alert(`${match.homeTeam} vs ${match.awayTeam} 경기 티켓 ${ticketQuantity}매 예매 완료!`);
      setTicketQuantity(1);
      setSelectedMatch(null);
    } else {
      alert("예매 가능한 좌석이 부족합니다.");
    }
  };

  const handleAddProduct = () => {
    if (newProduct.name && newProduct.price && newProduct.stock) {
      const product: Product = {
        id: Date.now().toString(),
        name: newProduct.name!,
        type: newProduct.type || "merchandise",
        price: newProduct.price!,
        stock: newProduct.stock!,
        description: newProduct.description || "",
      };
      setProducts((prev) => [...prev, product]);
      setNewProduct({
        name: "",
        type: "merchandise",
        price: 0,
        stock: 0,
        description: "",
      });
      alert("상품이 추가되었습니다.");
    }
  };

  const handleAddMember = () => {
    if (newMember.name && newMember.email && newMember.phone) {
      const member: Member = {
        id: Date.now().toString(),
        name: newMember.name!,
        email: newMember.email!,
        phone: newMember.phone!,
        membershipLevel: newMember.membershipLevel || "bronze",
        joinDate: new Date().toISOString().split("T")[0],
        totalSpent: 0,
      };
      setMembers((prev) => [...prev, member]);
      setNewMember({
        name: "",
        email: "",
        phone: "",
        membershipLevel: "bronze",
      });
      alert("회원이 추가되었습니다.");
    }
  };

  const getMembershipColor = (level: string) => {
    switch (level) {
      case "platinum":
        return "#e5e7eb";
      case "gold":
        return "#fbbf24";
      case "silver":
        return "#9ca3af";
      case "bronze":
        return "#cd7f32";
      default:
        return "#6b7280";
    }
  };

  return (
    <div className="admin-container">
      <header className="admin-header">
        <div className="header-content">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold">⚽ 축구 경기 관리 어드민 대시보드</h1>
          <div className="flex items-center gap-2 sm:gap-3">
            <Link
              href="/v10/admin/upload"
              prefetch={false}
              className="upload-button flex items-center gap-2 px-3 sm:px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm sm:text-base"
              aria-label="파일 업로드"
            >
              <span className="text-lg sm:text-xl">📁</span>
              <span className="hidden sm:inline">파일 업로드</span>
            </Link>
            <Link
              href="/"
              prefetch={false}
              className="home-button flex items-center gap-2 px-3 sm:px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm sm:text-base"
            >
              <span className="text-lg sm:text-xl">🏠</span>
              <span className="hidden sm:inline">홈으로</span>
            </Link>
          </div>
        </div>
        <div className="tab-selector">
          <button
            type="button"
            className={`tab-button ${activeTab === "dashboard" ? "active" : ""}`}
            onClick={() => setActiveTab("dashboard")}
          >
            📊 대시보드
          </button>
          <button
            type="button"
            className={`tab-button ${activeTab === "tickets" ? "active" : ""}`}
            onClick={() => setActiveTab("tickets")}
          >
            🎫 경기 표 예매
          </button>
          <button
            type="button"
            className={`tab-button ${activeTab === "betting" ? "active" : ""}`}
            onClick={() => setActiveTab("betting")}
          >
            🎲 배팅 시스템
          </button>
          <button
            type="button"
            className={`tab-button ${activeTab === "products" ? "active" : ""}`}
            onClick={() => setActiveTab("products")}
          >
            🛍️ 상품 관리
          </button>
          <button
            type="button"
            className={`tab-button ${activeTab === "members" ? "active" : ""}`}
            onClick={() => setActiveTab("members")}
          >
            👥 멤버 관리
          </button>
        </div>
      </header>

      <main className="admin-main">
        {activeTab === "dashboard" && (
          <div className="tab-content">
            <h2>대시보드</h2>

            {/* 통계 카드 */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">⚽</div>
                <div className="stat-info">
                  <h3>예정된 경기</h3>
                  <p className="stat-value">{matches.length}경기</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">👥</div>
                <div className="stat-info">
                  <h3>총 회원 수</h3>
                  <p className="stat-value">{members.length}명</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">🛍️</div>
                <div className="stat-info">
                  <h3>등록된 상품</h3>
                  <p className="stat-value">{products.length}개</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">💰</div>
                <div className="stat-info">
                  <h3>총 매출</h3>
                  <p className="stat-value">
                    {members.reduce((sum, m) => sum + m.totalSpent, 0).toLocaleString()}원
                  </p>
                </div>
              </div>
            </div>

            {/* 최근 경기 및 빠른 액세스 */}
            <div className="dashboard-grid">
              <div className="dashboard-section">
                <div className="section-header">
                  <h3>📅 최근 경기</h3>
                  <button type="button" className="view-all-button" onClick={() => setActiveTab("tickets")}>
                    전체 보기 →
                  </button>
                </div>
                <div className="recent-matches">
                  {matches.slice(0, 3).map((match) => {
                    const odds = bettingOdds.find((o) => o.matchId === match.id);
                    return (
                      <div key={match.id} className="recent-match-card">
                        <div className="recent-match-header">
                          <h4>{match.homeTeam} vs {match.awayTeam}</h4>
                          <span className="match-date">{match.date} {match.time}</span>
                        </div>
                        <div className="recent-match-info">
                          <span>💺 {match.availableSeats}석 남음</span>
                          <span>💰 {match.price.toLocaleString()}원</span>
                        </div>
                        {odds && (
                          <div className="recent-odds">
                            <span>홈 {odds.homeWinProbability}%</span>
                            <span>무 {odds.drawProbability}%</span>
                            <span>원정 {odds.awayWinProbability}%</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="dashboard-section">
                <div className="section-header">
                  <h3>🛍️ 인기 상품</h3>
                  <button type="button" className="view-all-button" onClick={() => setActiveTab("products")}>
                    전체 보기 →
                  </button>
                </div>
                <div className="popular-products">
                  {products.slice(0, 3).map((product) => (
                    <div key={product.id} className="popular-product-card">
                      <div className="product-name-row">
                        <h4>{product.name}</h4>
                        <span className={`product-type-badge ${product.type}`}>
                          {product.type === "merchandise" ? "상품" : product.type === "ticket" ? "티켓" : "체험"}
                        </span>
                      </div>
                      <p className="product-price">{product.price.toLocaleString()}원</p>
                      <p className="product-stock">재고: {product.stock}개</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 최근 회원 및 빠른 액세스 */}
            <div className="dashboard-grid">
              <div className="dashboard-section">
                <div className="section-header">
                  <h3>👥 최근 가입 회원</h3>
                  <button type="button" className="view-all-button" onClick={() => setActiveTab("members")}>
                    전체 보기 →
                  </button>
                </div>
                <div className="recent-members">
                  {members.slice(0, 5).map((member) => (
                    <div key={member.id} className="recent-member-card">
                      <div className="member-info-row">
                        <span className="member-name">{member.name}</span>
                        <span
                          className="membership-badge-small"
                          style={{
                            backgroundColor: getMembershipColor(member.membershipLevel),
                          }}
                        >
                          {member.membershipLevel.toUpperCase()}
                        </span>
                      </div>
                      <div className="member-details">
                        <span>{member.email}</span>
                        <span>총 구매액: {member.totalSpent.toLocaleString()}원</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="dashboard-section">
                <div className="section-header">
                  <h3>🎲 활성 배팅</h3>
                  <button type="button" className="view-all-button" onClick={() => setActiveTab("betting")}>
                    전체 보기 →
                  </button>
                </div>
                <div className="active-betting">
                  {matches.slice(0, 3).map((match) => {
                    const odds = bettingOdds.find((o) => o.matchId === match.id);
                    if (!odds) return null;
                    return (
                      <div key={match.id} className="betting-summary-card">
                        <h4>{match.homeTeam} vs {match.awayTeam}</h4>
                        <div className="betting-odds-summary">
                          <div className="odds-summary-item">
                            <span>홈 승</span>
                            <span className="odds-value-small">{odds.homeWin.toFixed(2)}</span>
                          </div>
                          <div className="odds-summary-item">
                            <span>무</span>
                            <span className="odds-value-small">{odds.draw.toFixed(2)}</span>
                          </div>
                          <div className="odds-summary-item">
                            <span>원정 승</span>
                            <span className="odds-value-small">{odds.awayWin.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "tickets" && (
          <div className="tab-content">
            <h2>경기 표 예매</h2>
            <div className="matches-grid">
              {matches.map((match) => {
                const odds = bettingOdds.find((o) => o.matchId === match.id);
                return (
                  <div key={match.id} className="match-card">
                    <div className="match-header">
                      <h3>{match.homeTeam} vs {match.awayTeam}</h3>
                      <span className="venue">{match.venue}</span>
                    </div>
                    <div className="match-info">
                      <p>📅 {match.date} {match.time}</p>
                      <p>💺 남은 좌석: {match.availableSeats}석</p>
                      <p>💰 가격: {match.price.toLocaleString()}원</p>
                    </div>
                    {odds && (
                      <div className="odds-preview">
                        <p>승률 추론:</p>
                        <div className="odds-bar">
                          <div
                            className="odds-segment home"
                            style={{ width: `${odds.homeWinProbability}%` }}
                          >
                            홈 {odds.homeWinProbability}%
                          </div>
                          <div
                            className="odds-segment draw"
                            style={{ width: `${odds.drawProbability}%` }}
                          >
                            무 {odds.drawProbability}%
                          </div>
                          <div
                            className="odds-segment away"
                            style={{ width: `${odds.awayWinProbability}%` }}
                          >
                            원정 {odds.awayWinProbability}%
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="match-actions">
                      {selectedMatch === match.id ? (
                        <div className="purchase-form">
                          <label>
                            수량:
                            <input
                              type="number"
                              min="1"
                              max={match.availableSeats}
                              value={ticketQuantity}
                              onChange={(e) =>
                                setTicketQuantity(parseInt(e.target.value) || 1)
                              }
                            />
                          </label>
                          <div className="button-group">
                            <button
                              type="button"
                              className="purchase-button"
                              onClick={() => handleTicketPurchase(match.id)}
                            >
                              예매하기 ({(match.price * ticketQuantity).toLocaleString()}원)
                            </button>
                            <button
                              type="button"
                              className="cancel-button"
                              onClick={() => {
                                setSelectedMatch(null);
                                setTicketQuantity(1);
                              }}
                            >
                              취소
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="select-button"
                          onClick={() => setSelectedMatch(match.id)}
                        >
                          예매하기
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === "betting" && (
          <div className="tab-content">
            <h2>승률 추론 기반 배팅 시스템</h2>
            <div className="betting-grid">
              {matches.map((match) => {
                const odds = bettingOdds.find((o) => o.matchId === match.id);
                if (!odds) return null;
                return (
                  <div key={match.id} className="betting-card">
                    <div className="betting-header">
                      <h3>{match.homeTeam} vs {match.awayTeam}</h3>
                      <p>{match.date} {match.time}</p>
                    </div>
                    <div className="betting-odds">
                      <div className="odds-item">
                        <span className="team">{match.homeTeam} 승</span>
                        <span className="odds-value">{odds.homeWin.toFixed(2)}</span>
                        <div className="probability-bar">
                          <div
                            className="probability-fill home"
                            style={{ width: `${odds.homeWinProbability}%` }}
                          />
                        </div>
                        <span className="probability">{odds.homeWinProbability}%</span>
                      </div>
                      <div className="odds-item">
                        <span className="team">무승부</span>
                        <span className="odds-value">{odds.draw.toFixed(2)}</span>
                        <div className="probability-bar">
                          <div
                            className="probability-fill draw"
                            style={{ width: `${odds.drawProbability}%` }}
                          />
                        </div>
                        <span className="probability">{odds.drawProbability}%</span>
                      </div>
                      <div className="odds-item">
                        <span className="team">{match.awayTeam} 승</span>
                        <span className="odds-value">{odds.awayWin.toFixed(2)}</span>
                        <div className="probability-bar">
                          <div
                            className="probability-fill away"
                            style={{ width: `${odds.awayWinProbability}%` }}
                          />
                        </div>
                        <span className="probability">{odds.awayWinProbability}%</span>
                      </div>
                    </div>
                    <div className="betting-actions">
                      <button type="button" className="bet-button">배팅하기</button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === "products" && (
          <div className="tab-content">
            <h2>파생 상품 관리</h2>
            <div className="products-section">
              <div className="add-product-form">
                <h3>새 상품 추가</h3>
                <div className="form-grid">
                  <label>
                    상품명:
                    <input
                      type="text"
                      value={newProduct.name || ""}
                      onChange={(e) =>
                        setNewProduct({ ...newProduct, name: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    타입:
                    <select
                      value={newProduct.type || "merchandise"}
                      onChange={(e) =>
                        setNewProduct({
                          ...newProduct,
                          type: e.target.value as "ticket" | "merchandise" | "experience",
                        })
                      }
                    >
                      <option value="merchandise">상품</option>
                      <option value="ticket">티켓</option>
                      <option value="experience">체험</option>
                    </select>
                  </label>
                  <label>
                    가격:
                    <input
                      type="number"
                      value={newProduct.price || 0}
                      onChange={(e) =>
                        setNewProduct({
                          ...newProduct,
                          price: parseInt(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                  <label>
                    재고:
                    <input
                      type="number"
                      value={newProduct.stock || 0}
                      onChange={(e) =>
                        setNewProduct({
                          ...newProduct,
                          stock: parseInt(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                  <label className="full-width">
                    설명:
                    <textarea
                      value={newProduct.description || ""}
                      onChange={(e) =>
                        setNewProduct({ ...newProduct, description: e.target.value })
                      }
                      rows={3}
                    />
                  </label>
                </div>
                <button type="button" className="add-button" onClick={handleAddProduct}>
                  상품 추가
                </button>
              </div>

              <div className="products-list">
                <h3>상품 목록</h3>
                <div className="products-grid">
                  {products.map((product) => (
                    <div key={product.id} className="product-card">
                      <div className="product-header">
                        <h4>{product.name}</h4>
                        <span className={`product-type ${product.type}`}>
                          {product.type === "merchandise"
                            ? "상품"
                            : product.type === "ticket"
                              ? "티켓"
                              : "체험"}
                        </span>
                      </div>
                      <p className="product-description">{product.description}</p>
                      <div className="product-info">
                        <p>💰 가격: {product.price.toLocaleString()}원</p>
                        <p>📦 재고: {product.stock}개</p>
                      </div>
                      <div className="product-actions">
                        <button type="button" className="edit-button">수정</button>
                        <button type="button" className="delete-button">삭제</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "members" && (
          <div className="tab-content">
            <h2>멤버 관리</h2>
            <div className="members-section">
              <div className="add-member-form">
                <h3>새 회원 추가</h3>
                <div className="form-grid">
                  <label>
                    이름:
                    <input
                      type="text"
                      value={newMember.name || ""}
                      onChange={(e) =>
                        setNewMember({ ...newMember, name: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    이메일:
                    <input
                      type="email"
                      value={newMember.email || ""}
                      onChange={(e) =>
                        setNewMember({ ...newMember, email: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    전화번호:
                    <input
                      type="tel"
                      value={newMember.phone || ""}
                      onChange={(e) =>
                        setNewMember({ ...newMember, phone: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    멤버십 등급:
                    <select
                      value={newMember.membershipLevel || "bronze"}
                      onChange={(e) =>
                        setNewMember({
                          ...newMember,
                          membershipLevel: e.target.value as
                            | "bronze"
                            | "silver"
                            | "gold"
                            | "platinum",
                        })
                      }
                    >
                      <option value="bronze">브론즈</option>
                      <option value="silver">실버</option>
                      <option value="gold">골드</option>
                      <option value="platinum">플래티넘</option>
                    </select>
                  </label>
                </div>
                <button type="button" className="add-button" onClick={handleAddMember}>
                  회원 추가
                </button>
              </div>

              <div className="members-list">
                <h3>회원 목록</h3>
                <div className="members-table">
                  <table>
                    <thead>
                      <tr>
                        <th>이름</th>
                        <th>이메일</th>
                        <th>전화번호</th>
                        <th>멤버십</th>
                        <th>가입일</th>
                        <th>총 구매액</th>
                        <th>액션</th>
                      </tr>
                    </thead>
                    <tbody>
                      {members.map((member) => (
                        <tr key={member.id}>
                          <td>{member.name}</td>
                          <td>{member.email}</td>
                          <td>{member.phone}</td>
                          <td>
                            <span
                              className="membership-badge"
                              style={{
                                backgroundColor: getMembershipColor(member.membershipLevel),
                              }}
                            >
                              {member.membershipLevel.toUpperCase()}
                            </span>
                          </td>
                          <td>{member.joinDate}</td>
                          <td>{member.totalSpent.toLocaleString()}원</td>
                          <td>
                            <button type="button" className="edit-button">수정</button>
                            <button type="button" className="delete-button">삭제</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <style jsx>{`
        .admin-container {
          min-height: 100vh;
          background: #f5f7fa;
        }

        .admin-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 2rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .header-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .admin-header h1 {
          margin: 0;
          font-size: 1.8rem;
        }

        /* 파일 업로드 / 홈으로 - Link로 이동(hydration 방지) */
        .upload-button,
        .home-button {
          display: inline-flex;
          text-decoration: none;
        }
        .upload-button {
          min-height: 44px;
          min-width: 44px;
        }

        .home-button {
          padding: 0.5rem 1rem;
          border: 2px solid rgba(255, 255, 255, 0.5);
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.15);
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 44px;
          min-width: 44px;
        }

        .home-button:hover {
          background: rgba(255, 255, 255, 0.25);
          border-color: rgba(255, 255, 255, 0.7);
        }

        .tab-selector {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }

        .tab-button {
          padding: 0.75rem 1.5rem;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 0.95rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .tab-button:hover {
          background: rgba(255, 255, 255, 0.2);
          border-color: rgba(255, 255, 255, 0.5);
        }

        .tab-button.active {
          background: rgba(255, 255, 255, 0.3);
          border-color: white;
          font-weight: 600;
        }

        .admin-main {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
        }

        .tab-content {
          background: white;
          border-radius: 0.5rem;
          padding: 2rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .tab-content h2 {
          margin-top: 0;
          margin-bottom: 2rem;
          color: #333;
        }

        .matches-grid,
        .betting-grid,
        .products-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1.5rem;
        }

        .match-card,
        .betting-card,
        .product-card {
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1.5rem;
          background: white;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .match-card:hover,
        .betting-card:hover,
        .product-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .match-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .match-header h3 {
          margin: 0;
          font-size: 1.2rem;
          color: #333;
        }

        .venue {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .match-info p {
          margin: 0.5rem 0;
          color: #6b7280;
        }

        .odds-preview {
          margin: 1rem 0;
          padding: 1rem;
          background: #f9fafb;
          border-radius: 0.5rem;
        }

        .odds-bar {
          display: flex;
          height: 30px;
          border-radius: 0.25rem;
          overflow: hidden;
          margin-top: 0.5rem;
        }

        .odds-segment {
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .odds-segment.home {
          background: #3b82f6;
        }

        .odds-segment.draw {
          background: #6b7280;
        }

        .odds-segment.away {
          background: #ef4444;
        }

        .match-actions {
          margin-top: 1rem;
        }

        .select-button,
        .purchase-button,
        .bet-button {
          width: 100%;
          padding: 0.75rem;
          border: none;
          border-radius: 0.5rem;
          background: #667eea;
          color: white;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .select-button:hover,
        .purchase-button:hover,
        .bet-button:hover {
          background: #5568d3;
        }

        .purchase-form {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .purchase-form label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .purchase-form input {
          flex: 1;
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
        }

        .button-group {
          display: flex;
          gap: 0.5rem;
        }

        .cancel-button {
          padding: 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 0.5rem;
          background: white;
          color: #6b7280;
          cursor: pointer;
          transition: background 0.2s;
        }

        .cancel-button:hover {
          background: #f9fafb;
        }

        .betting-header {
          margin-bottom: 1rem;
        }

        .betting-header h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.2rem;
        }

        .betting-header p {
          margin: 0;
          color: #6b7280;
          font-size: 0.9rem;
        }

        .betting-odds {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .odds-item {
          display: grid;
          grid-template-columns: 1fr auto auto;
          gap: 0.5rem;
          align-items: center;
        }

        .team {
          font-weight: 600;
        }

        .odds-value {
          font-size: 1.2rem;
          font-weight: 700;
          color: #667eea;
        }

        .probability-bar {
          width: 100%;
          height: 8px;
          background: #e5e7eb;
          border-radius: 4px;
          overflow: hidden;
          grid-column: 1 / -1;
        }

        .probability-fill {
          height: 100%;
          transition: width 0.3s;
        }

        .probability-fill.home {
          background: #3b82f6;
        }

        .probability-fill.draw {
          background: #6b7280;
        }

        .probability-fill.away {
          background: #ef4444;
        }

        .probability {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .products-section,
        .members-section {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .add-product-form,
        .add-member-form {
          padding: 1.5rem;
          background: #f9fafb;
          border-radius: 0.5rem;
        }

        .add-product-form h3,
        .add-member-form h3 {
          margin-top: 0;
        }

        .form-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .form-grid label {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .form-grid label.full-width {
          grid-column: 1 / -1;
        }

        .form-grid input,
        .form-grid select,
        .form-grid textarea {
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          font-size: 0.95rem;
        }

        .add-button {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 0.5rem;
          background: #10b981;
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .add-button:hover {
          background: #059669;
        }

        .product-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .product-header h4 {
          margin: 0;
          font-size: 1.1rem;
        }

        .product-type {
          padding: 0.25rem 0.75rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .product-type.merchandise {
          background: #dbeafe;
          color: #1e40af;
        }

        .product-type.ticket {
          background: #fef3c7;
          color: #92400e;
        }

        .product-type.experience {
          background: #fce7f3;
          color: #9f1239;
        }

        .product-description {
          color: #6b7280;
          margin-bottom: 1rem;
        }

        .product-info p {
          margin: 0.5rem 0;
          color: #6b7280;
        }

        .product-actions {
          display: flex;
          gap: 0.5rem;
          margin-top: 1rem;
        }

        .edit-button,
        .delete-button {
          flex: 1;
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background: white;
          cursor: pointer;
          transition: background 0.2s;
        }

        .edit-button {
          color: #3b82f6;
        }

        .edit-button:hover {
          background: #eff6ff;
        }

        .delete-button {
          color: #ef4444;
        }

        .delete-button:hover {
          background: #fef2f2;
        }

        .members-table {
          overflow-x: auto;
        }

        .members-table table {
          width: 100%;
          border-collapse: collapse;
        }

        .members-table th,
        .members-table td {
          padding: 1rem;
          text-align: left;
          border-bottom: 1px solid #e5e7eb;
        }

        .members-table th {
          background: #f9fafb;
          font-weight: 600;
          color: #374151;
        }

        .membership-badge {
          padding: 0.25rem 0.75rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
          color: white;
        }

        /* 대시보드 스타일 */
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .stat-card {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1.5rem;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 0.5rem;
          color: white;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .stat-icon {
          font-size: 3rem;
        }

        .stat-info h3 {
          margin: 0 0 0.5rem 0;
          font-size: 0.9rem;
          opacity: 0.9;
        }

        .stat-value {
          margin: 0;
          font-size: 2rem;
          font-weight: 700;
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 2rem;
          margin-bottom: 2rem;
        }

        .dashboard-section {
          background: #f9fafb;
          border-radius: 0.5rem;
          padding: 1.5rem;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .section-header h3 {
          margin: 0;
          font-size: 1.2rem;
          color: #333;
        }

        .view-all-button {
          padding: 0.5rem 1rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background: white;
          color: #667eea;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-all-button:hover {
          background: #667eea;
          color: white;
          border-color: #667eea;
        }

        .recent-matches,
        .popular-products,
        .recent-members,
        .active-betting {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .recent-match-card,
        .popular-product-card,
        .recent-member-card,
        .betting-summary-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1rem;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .recent-match-card:hover,
        .popular-product-card:hover,
        .recent-member-card:hover,
        .betting-summary-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .recent-match-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .recent-match-header h4 {
          margin: 0;
          font-size: 1rem;
          color: #333;
        }

        .match-date {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .recent-match-info {
          display: flex;
          gap: 1rem;
          margin-bottom: 0.5rem;
          font-size: 0.85rem;
          color: #6b7280;
        }

        .recent-odds {
          display: flex;
          gap: 0.5rem;
          font-size: 0.8rem;
          color: #6b7280;
        }

        .recent-odds span {
          padding: 0.25rem 0.5rem;
          background: #f3f4f6;
          border-radius: 0.25rem;
        }

        .product-name-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .product-name-row h4 {
          margin: 0;
          font-size: 1rem;
          color: #333;
        }

        .product-type-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .product-type-badge.merchandise {
          background: #dbeafe;
          color: #1e40af;
        }

        .product-type-badge.ticket {
          background: #fef3c7;
          color: #92400e;
        }

        .product-type-badge.experience {
          background: #fce7f3;
          color: #9f1239;
        }

        .product-price {
          margin: 0.5rem 0;
          font-size: 1.1rem;
          font-weight: 600;
          color: #667eea;
        }

        .product-stock {
          margin: 0;
          font-size: 0.85rem;
          color: #6b7280;
        }

        .member-info-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .member-name {
          font-weight: 600;
          color: #333;
        }

        .membership-badge-small {
          padding: 0.2rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.7rem;
          font-weight: 600;
          color: white;
        }

        .member-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          font-size: 0.85rem;
          color: #6b7280;
        }

        .betting-summary-card h4 {
          margin: 0 0 0.75rem 0;
          font-size: 1rem;
          color: #333;
        }

        .betting-odds-summary {
          display: flex;
          gap: 1rem;
        }

        .odds-summary-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.25rem;
          flex: 1;
          padding: 0.5rem;
          background: #f9fafb;
          border-radius: 0.25rem;
        }

        .odds-summary-item span:first-child {
          font-size: 0.75rem;
          color: #6b7280;
        }

        .odds-value-small {
          font-size: 1.1rem;
          font-weight: 700;
          color: #667eea;
        }

        @media (max-width: 768px) {
          .admin-header {
            padding: 1rem;
          }

          .admin-header h1 {
            font-size: 1.3rem;
          }

          .header-content {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
          }

          .header-content > div {
            width: 100%;
            justify-content: flex-start;
          }

          .tab-selector {
            flex-direction: column;
          }

          .tab-button {
            width: 100%;
          }

          .admin-main {
            padding: 1rem;
          }

          .matches-grid,
          .betting-grid,
          .products-grid {
            grid-template-columns: 1fr;
          }

          .form-grid {
            grid-template-columns: 1fr;
          }

          .members-table {
            font-size: 0.85rem;
          }

          .stats-grid {
            grid-template-columns: 1fr;
          }

          .dashboard-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
